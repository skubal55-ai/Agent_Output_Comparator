"""
Run the metric-validation study against a human-labeled gold set
(docs/methodology.md Section 4): inter-rater agreement (when exactly two
raters are present) and metric-vs-human correlation for each of the
framework's default metrics. Prints a summary and can write a markdown
report.

Usage:
    python scripts/run_validation.py --gold-set docs/gold_set.csv
    python scripts/run_validation.py --gold-set docs/gold_set.csv --report docs/validation_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.evalkit.metrics.base import Metric
from source.evalkit.metrics.structure import QualityHeuristicMetric, LengthFitMetric
from source.evalkit.metrics.semantic_accuracy import SemanticAccuracyMetric
from source.evalkit.validation.correlate import metric_correlation, inter_rater_agreement
from source.evalkit.validation.gold_set import GoldExample, load_gold_set_csv

# Efficiency is excluded — it scores latency, not output content, and a gold
# set of (prompt, output, human_score) triples has no latency to compare against.
DEFAULT_METRICS: dict[str, Metric] = {
    "quality": QualityHeuristicMetric(),
    "semantic_accuracy": SemanticAccuracyMetric(),
    "length_fit": LengthFitMetric(),
}

KAPPA_RELIABLE_THRESHOLD = 0.4
CORRELATION_MEANINGFUL_THRESHOLD = 0.3


def generate_report(examples: list[GoldExample], metrics: dict[str, Metric]) -> tuple[list[str], dict]:
    """Builds the report as markdown lines plus a machine-readable summary
    dict, so this logic is testable independent of CLI printing/writing."""
    rater_ids = sorted({ex.rater_id for ex in examples})
    example_ids = sorted({ex.id for ex in examples})

    lines = [
        "# Metric Validation Report",
        "",
        f"Examples: {len(example_ids)}  ",
        f"Raters: {', '.join(rater_ids) if rater_ids else '(none)'}",
        "",
    ]
    summary: dict = {"n_examples": len(example_ids), "raters": rater_ids, "agreement": None, "correlations": {}}

    if len(rater_ids) == 2:
        try:
            agreement = inter_rater_agreement(examples)
            reliable = agreement["cohens_kappa"] >= KAPPA_RELIABLE_THRESHOLD
            summary["agreement"] = {**agreement, "reliable": reliable}
            lines += [
                "## Inter-Rater Agreement",
                "",
                f"Cohen's κ = {agreement['cohens_kappa']:.3f} (n={agreement['n']})",
                "",
                "Reliable (κ ≥ 0.4)." if reliable
                else "**Below the 0.4 reliability threshold — the human rubric itself may be "
                     "unreliable. Fix rater agreement before trusting the correlations below.**",
                "",
            ]
        except ValueError as exc:
            lines += ["## Inter-Rater Agreement", "", f"Skipped: {exc}", ""]
    else:
        lines += [
            "## Inter-Rater Agreement",
            "",
            f"Skipped — requires exactly 2 raters, found {len(rater_ids)}.",
            "",
        ]

    lines += ["## Metric Correlation", "", "| Metric | n | Pearson r | Spearman ρ | Verdict |", "|---|---|---|---|---|"]
    for name, metric in metrics.items():
        try:
            result = metric_correlation(examples, metric)
        except ValueError as exc:
            lines.append(f"| {name} | — | — | — | skipped: {exc} |")
            continue
        meaningful = abs(result["spearman_r"]) >= CORRELATION_MEANINGFUL_THRESHOLD
        verdict = "keep current weight" if meaningful else "reduce/remove weight (weak correlation)"
        summary["correlations"][name] = {**result, "meaningful": meaningful}
        lines.append(
            f"| {name} | {result['n']} | {result['pearson_r']:.3f} | {result['spearman_r']:.3f} | {verdict} |"
        )

    return lines, summary


def main() -> None:
    # Windows consoles often default to cp1252, which can't encode κ/ρ; make
    # stdout UTF-8-safe so the report always prints instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-set", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None, help="Optional path to write a markdown report")
    args = parser.parse_args()

    examples = load_gold_set_csv(args.gold_set)
    if not examples:
        print(f"No examples found in {args.gold_set}.")
        return

    lines, summary = generate_report(examples, DEFAULT_METRICS)
    print("\n".join(lines))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
