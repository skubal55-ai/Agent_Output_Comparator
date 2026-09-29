"""Compute every statistic reported in Section VI of the paper (Tables III-VI) from docs/examples/results_raw.jsonl.
Uses the unrounded metric scores stored in metrics_detail. Writes v3_stats.json."""
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from source.evalkit.metrics.semantic_accuracy import SemanticAccuracyMetric  # noqa: E402

HERE = Path(__file__).parent
rows = [json.loads(l) for l in (APP / "docs" / "examples" / "results_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows.sort(key=lambda r: r["id"])
n = len(rows)
AG = ("copilot", "claude")


def md(r, a):
    return r[a]["scores"]["metrics_detail"]


def sc(r, a, k):
    return md(r, a)[k]["score"]


def overall(r, a):
    return md(r, a)["overall"]


def wo_eff(r, a):
    return (0.30 * sc(r, a, "quality") + 0.35 * sc(r, a, "semantic_accuracy") + 0.20 * sc(r, a, "length_fit")) / 0.85


def kerby_rb(diff):
    d = np.array([x for x in diff if x != 0])
    ranks = stats.rankdata(np.abs(d))
    rp, rm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((rp - rm) / (rp + rm)) if (rp + rm) else 0.0


def boot_ci(diff, B=20000, seed=7):
    rng = np.random.default_rng(seed)
    d = np.array(diff)
    means = rng.choice(d, size=(B, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(fn):
    c = [fn(r, "copilot") for r in rows]
    l = [fn(r, "claude") for r in rows]
    diff = [b - a for a, b in zip(c, l)]
    out = {"copilot": st.mean(c), "claude": st.mean(l), "diff": st.mean(diff), "ci": boot_ci(diff)}
    if any(x != 0 for x in diff):
        w = stats.wilcoxon(l, c)
        out.update(p=float(w.pvalue), W=float(w.statistic), rb=kerby_rb(diff))
    else:
        out.update(p=None, W=None, rb=None)
    return out


R = {"n": n}
R["overall"] = paired(overall)
R["overall_rounded_ui"] = paired(lambda r, a: r[a]["scores"]["overall"])
R["wo_eff"] = paired(wo_eff)
for k in ("quality", "semantic_accuracy", "efficiency", "length_fit"):
    R[k] = paired(lambda r, a, k=k: sc(r, a, k))
R["latency_s"] = paired(lambda r, a: r[a]["latency_ms"] / 1000)
R["words"] = paired(lambda r, a: r[a]["scores"]["word_count"])
# Holm across the 5 metric-level tests (quality, SA, efficiency, words, latency) -- length fit identical
fam = [(k, R[k]["p"]) for k in ("quality", "semantic_accuracy", "efficiency", "latency_s", "words") if R[k]["p"] is not None]
order = sorted(fam, key=lambda x: x[1])
m = len(order)
adj, running = {}, 0.0
for i, (k, p) in enumerate(order):
    running = max(running, min(1.0, (m - i) * p))
    adj[k] = running
R["holm"] = adj

# winners
def winner(fn, r, band=3.0):
    c, l = fn(r, "copilot"), fn(r, "claude")
    return "tie" if abs(c - l) <= band else ("copilot" if c > l else "claude")

R["wins_unrounded"] = {k: sum(winner(overall, r) == k for r in rows) for k in ("copilot", "claude", "tie")}
R["wins_ui"] = {k: sum(winner(lambda r, a: r[a]["scores"]["overall"], r) == k for r in rows) for k in ("copilot", "claude", "tie")}
R["wins_wo_eff"] = {k: sum(winner(wo_eff, r) == k for r in rows) for k in ("copilot", "claude", "tie")}
faster = [("claude" if r["claude"]["latency_ms"] < r["copilot"]["latency_ms"] else "copilot") for r in rows]
R["winner_is_faster"] = sum(1 for r, f in zip(rows, faster) if winner(overall, r) == f)
R["clopper_pearson_54_55"] = [float(stats.beta.ppf(0.025, 54, 2)), float(stats.beta.ppf(0.975, 55, 1))]

# latency ratio
ratio = [r["copilot"]["latency_ms"] / r["claude"]["latency_ms"] for r in rows]
lr = np.log(ratio)
rng = np.random.default_rng(11)
bm = rng.choice(lr, size=(20000, n), replace=True).mean(axis=1)
R["latency_ratio_geo"] = {"gm": float(math.exp(lr.mean())), "ci": [float(math.exp(np.percentile(bm, 2.5))), float(math.exp(np.percentile(bm, 97.5)))],
                          "min": min(ratio), "max": max(ratio)}
gaps = sorted(abs(r["copilot"]["latency_ms"] - r["claude"]["latency_ms"]) / 1000 for r in rows)
R["gap"] = {"min": gaps[0], "max": gaps[-1], "under2s": sum(g < 2 for g in gaps), "under1s": sum(g < 1 for g in gaps)}

# sensitivity to the Efficiency weight and a ratio-based Efficiency
def with_w(we, eff=lambda r, a: sc(r, a, "efficiency")):
    return lambda r, a: (1 - we) * wo_eff(r, a) + we * eff(r, a)

def ratio_eff(r, a):
    lo = min(r["copilot"]["latency_ms"], r["claude"]["latency_ms"])
    return 100.0 * lo / r[a]["latency_ms"]

sens = []
for label, fn in [("0", with_w(0.0)), ("0.05", with_w(0.05)), ("0.10", with_w(0.10)), ("0.15 (default)", with_w(0.15)),
                  ("0.25", with_w(0.25)), ("ratio E, 0.15", with_w(0.15, ratio_eff))]:
    c = st.mean(fn(r, "copilot") for r in rows)
    l = st.mean(fn(r, "claude") for r in rows)
    w = {k: sum(winner(fn, r) == k for r in rows) for k in ("copilot", "claude", "tie")}
    diff = [fn(r, "claude") - fn(r, "copilot") for r in rows]
    p = float(stats.wilcoxon([fn(r, "claude") for r in rows], [fn(r, "copilot") for r in rows]).pvalue)
    sens.append({"w": label, "copilot": c, "claude": l, "diff": l - c, "ci": boot_ci(diff), "wins": w, "p": p})
R["sensitivity"] = sens
# check: default row equals unrounded overall
assert abs(sens[3]["copilot"] - R["overall"]["copilot"]) < 0.01, (sens[3]["copilot"], R["overall"]["copilot"])

# drift over run position
pos = np.arange(1, n + 1)
for a in AG:
    lat = np.array([r[a]["latency_ms"] / 1000 for r in rows])
    rho, p = stats.spearmanr(pos, lat)
    R[f"drift_{a}"] = {"rho": float(rho), "p": float(p), "first_half": float(lat[:27].mean()), "second_half": float(lat[27:].mean())}

# ceilings / structure / word limit
R["quality100"] = {a: sum(sc(r, a, "quality") == 100 for r in rows) for a in AG}
R["length100"] = {a: sum(sc(r, a, "length_fit") == 100 for r in rows) for a in AG}
R["sa_range"] = [min(sc(r, a, "semantic_accuracy") for r in rows for a in AG), max(sc(r, a, "semantic_accuracy") for r in rows for a in AG)]
R["overall_max"] = max(overall(r, a) for r in rows for a in AG)
R["headings"] = {a: sum(1 for r in rows if re.search(r"(?m)^#{1,6}\s", r[a]["output"])) for a in AG}
R["over250_ws"] = {a: sum(r[a]["scores"]["word_count"] > 250 for r in rows) for a in AG}
R["over250_alnum"] = {a: sum(len([t for t in r[a]["output"].split() if re.search(r"[A-Za-z0-9]", t)]) > 250 for r in rows) for a in AG}
api = next(r for r in rows if "non-technical manager" in r["prompt"])
R["api100"] = {a: len([t for t in api[a]["output"].split() if re.search(r"[A-Za-z0-9]", t)]) for a in AG}
# grades
def grade(s): return "Excellent" if s >= 80 else "Good" if s >= 60 else "Fair" if s >= 40 else "Poor"
R["grades"] = {a: {g: sum(grade(r[a]["scores"]["overall"]) == g for r in rows) for g in ("Excellent", "Good", "Fair", "Poor")} for a in AG}

# Semantic Accuracy against the user prompt only
sa = SemanticAccuracyMetric()
R["sa_user_only"] = paired(lambda r, a: sa.compute(output=r[a]["output"], prompt=r["prompt"], system_prompt="").score)

# file extractor
R["files"] = {a: sum(len(r[a].get("files") or []) for r in rows) for a in AG}
R["file_runs"] = [(r["id"], a, [f.get("name") for f in r[a]["files"]], [f.get("exists") for f in r[a]["files"]])
                  for r in rows for a in AG if r[a].get("files")]
R["failures"] = sum(1 for r in rows for a in AG if r[a].get("error"))

# cost
costs = [r["claude"].get("total_cost_usd") or 0 for r in rows]
R["cost"] = {"total": sum(costs), "mean": st.mean(costs), "min": min(costs), "max": max(costs)}
R["claude_models"] = sorted({r["claude"].get("model") or "" for r in rows})

# worked example: git merge vs rebase
g = next(r for r in rows if "git merge and git rebase" in r["prompt"])
R["worked"] = {"id": g["id"], "prompt": g["prompt"], **{a: {"lat": g[a]["latency_ms"] / 1000, "words": g[a]["scores"]["word_count"],
               "q": sc(g, a, "quality"), "sa": sc(g, a, "semantic_accuracy"), "e": sc(g, a, "efficiency"),
               "l": sc(g, a, "length_fit"), "overall": overall(g, a)} for a in AG}}
# first prompt of each category for Table V
cats = list(dict.fromkeys(r["category"] for r in rows))
R["table5"] = []
for c in cats:
    r = next(x for x in rows if x["category"] == c)
    R["table5"].append({"id": r["id"], "cat": c, "prompt": r["prompt"],
                        **{a: {"overall": overall(r, a), "lat": r[a]["latency_ms"] / 1000} for a in AG}})
cw = next(r for r in rows if winner(overall, r) == "copilot")
R["copilot_win"] = {"id": cw["id"], "prompt": cw["prompt"], **{a: {"overall": overall(cw, a), "lat": cw[a]["latency_ms"] / 1000} for a in AG}}
R["by_category"] = [{"cat": c, **{a: st.mean(overall(r, a) for r in rows if r["category"] == c) for a in AG},
                     "wins": {k: sum(1 for r in rows if r["category"] == c and winner(overall, r) == k) for k in ("copilot", "claude", "tie")}}
                    for c in cats]
(APP / "docs" / "examples" / "analysis_stats.json").write_text(json.dumps(R, indent=1), encoding="utf-8")

f = lambda x: f"{x:.2f}"
print("overall", f(R["overall"]["copilot"]), f(R["overall"]["claude"]), "diff", f(R["overall"]["diff"]), [f(x) for x in R["overall"]["ci"]], "p", R["overall"]["p"], "rb", f(R["overall"]["rb"]))
print("UI-rounded overall p", R["overall_rounded_ui"]["p"], "wo_eff", f(R["wo_eff"]["copilot"]), f(R["wo_eff"]["claude"]), [f(x) for x in R["wo_eff"]["ci"]], "p", R["wo_eff"]["p"], "rb", R["wo_eff"]["rb"])
print("wins unrounded", R["wins_unrounded"], "ui", R["wins_ui"], "wo_eff", R["wins_wo_eff"], "winner=faster", R["winner_is_faster"])
for k in ("quality", "semantic_accuracy", "efficiency", "length_fit", "latency_s", "words", "sa_user_only"):
    x = R[k]; print(k, f(x["copilot"]), f(x["claude"]), "diff", f(x["diff"]), [f(v) for v in x["ci"]], "p", x["p"], "rb", x["rb"])
print("holm", R["holm"]); print("CP", R["clopper_pearson_54_55"]); print("ratio", R["latency_ratio_geo"]); print("gap", R["gap"])
for s in sens: print("sens", s["w"], f(s["copilot"]), f(s["claude"]), f(s["diff"]), [f(v) for v in s["ci"]], s["wins"], s["p"])
print("drift", R["drift_copilot"], R["drift_claude"])
print("q100", R["quality100"], "l100", R["length100"], "sa range", R["sa_range"], "max overall", R["overall_max"], "grades", R["grades"])
print("headings", R["headings"], "over250 ws", R["over250_ws"], "alnum", R["over250_alnum"], "api100", R["api100"])
print("files", R["files"], R["file_runs"], "failures", R["failures"]); print("cost", R["cost"], R["claude_models"])
print("worked", R["worked"]); print("copilot win", R["copilot_win"])
