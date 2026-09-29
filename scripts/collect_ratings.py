"""
Interactive CLI to build a human-labeled gold set for evalkit metric
validation (docs/methodology.md Section 4).

Runs each prompt against the selected agents (reusing the same evalkit
Agent wrappers the rest of the framework uses), shows you the prompt and
output, asks you to enter a 0-100 human score, and appends each rating to
the gold set CSV as you go. Safe to Ctrl+C (or type 'q') and resume later.

Multi-rater correctness: the *same* generated output must be shown to every
rater for a given (prompt, agent) pair, since agents are non-deterministic —
re-running the agent per rater would let each rater score a different
output, invalidating both inter-rater agreement and metric correlation.
This tool handles that: the first rater to reach a (prompt, agent) pair
triggers the agent call and caches prompt/output in the CSV; every
subsequent rater (even in a separate invocation) is shown that exact same
cached output, never a fresh one.

This tool does not invent scores — it is a data-collection aid. The actual
judgment is yours or your raters'.

Usage:
    python scripts/collect_ratings.py --prompts docs/gold_set_prompts.example.json \\
        --agents copilot claude --rater-id rater1 --out docs/gold_set.csv

    # A second rater, later, rating the SAME cached outputs:
    python scripts/collect_ratings.py --prompts docs/gold_set_prompts.example.json \\
        --agents copilot claude --rater-id rater2 --out docs/gold_set.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.evalkit.experiment.runner import build_agents
from source.evalkit.validation.gold_set import GoldExample, load_gold_set_csv, save_gold_set_csv

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def load_prompts(path: Path) -> list[dict]:
    """Load prompts from a JSON or YAML file. Accepts a list of plain
    strings, or a list of {id, system_prompt, user_prompt} objects — same
    shape as ExperimentConfig's prompts."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml") and yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    prompts = []
    for i, p in enumerate(data):
        if isinstance(p, str):
            prompts.append({"id": f"p{i}", "system_prompt": "", "user_prompt": p})
        else:
            prompts.append(
                {
                    "id": p.get("id", f"p{i}"),
                    "system_prompt": p.get("system_prompt", ""),
                    "user_prompt": p.get("user_prompt", p.get("prompt", "")),
                }
            )
    return prompts


def existing_keys(examples: list[GoldExample]) -> set[tuple[str, str]]:
    """(example_id, rater_id) pairs already rated — lets a rerun resume
    without re-asking for scores already given."""
    return {(ex.id, ex.rater_id) for ex in examples}


def cached_outputs(examples: list[GoldExample]) -> dict[str, GoldExample]:
    """First example seen per id, used to reuse a cached (prompt, output)
    pair for a second rater instead of re-invoking the agent."""
    cache: dict[str, GoldExample] = {}
    for ex in examples:
        cache.setdefault(ex.id, ex)
    return cache


def prompt_for_score(agent: str, prompt_id: str, prompt_text: str, output_text: str) -> float | None:
    print("\n" + "=" * 70)
    print(f"[{prompt_id} / {agent}]")
    print("-" * 70)
    print("PROMPT:\n" + prompt_text)
    print("-" * 70)
    print("OUTPUT:\n" + (output_text or "(empty output)"))
    print("-" * 70)
    while True:
        raw = input("Score 0-100 (blank to skip, 'q' to quit and save): ").strip()
        if raw == "":
            return None
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            score = float(raw)
        except ValueError:
            print("Enter a number 0-100, blank to skip, or 'q' to quit.")
            continue
        if not (0 <= score <= 100):
            print("Score must be between 0 and 100.")
            continue
        return score


def main() -> None:
    # Windows consoles often default to cp1252; agent output can contain
    # arbitrary Unicode, so make stdout UTF-8-safe rather than crashing mid-rating.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prompts", required=True, type=Path,
        help="JSON or YAML file: list of strings, or {id, system_prompt, user_prompt} objects",
    )
    parser.add_argument("--agents", nargs="+", default=["copilot", "claude"], help="Agent names to run and rate")
    parser.add_argument(
        "--rater-id", required=True,
        help="Your rater identifier (e.g. 'rater1'). Use a distinct id per human rater, "
        "consistent across sessions for the same person.",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/gold_set.csv"), help="Gold set CSV to append to")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    agents = build_agents(args.agents)

    existing = load_gold_set_csv(args.out) if args.out.exists() else []
    done = existing_keys(existing)
    cache = cached_outputs(existing)

    collected: list[GoldExample] = list(existing)
    try:
        for prompt in prompts:
            for agent_name, agent in agents.items():
                example_id = f"{prompt['id']}::{agent_name}"
                if (example_id, args.rater_id) in done:
                    continue  # this rater already scored it

                cached = cache.get(example_id)
                if cached is not None:
                    prompt_text, system_prompt_text, output_text = cached.prompt, cached.system_prompt, cached.output
                else:
                    print(f"\nRunning {agent_name} on {prompt['id']}...")
                    result = agent.run(prompt["system_prompt"], prompt["user_prompt"])
                    prompt_text, system_prompt_text, output_text = prompt["user_prompt"], prompt["system_prompt"], result.output

                score = prompt_for_score(agent_name, prompt["id"], prompt_text, output_text)
                if score is None:
                    continue

                example = GoldExample(
                    id=example_id,
                    prompt=prompt_text,
                    system_prompt=system_prompt_text,
                    output=output_text,
                    human_score=score,
                    rater_id=args.rater_id,
                )
                collected.append(example)
                cache.setdefault(example_id, example)
    except KeyboardInterrupt:
        print("\nStopping early — saving progress so far.")
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_gold_set_csv(collected, args.out)
        print(f"\nSaved {len(collected)} ratings to {args.out}")


if __name__ == "__main__":
    main()
