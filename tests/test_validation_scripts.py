import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.collect_ratings import load_prompts, existing_keys, cached_outputs
from scripts.run_validation import generate_report, DEFAULT_METRICS
from source.evalkit.validation.gold_set import GoldExample


class TestLoadPrompts:
    def test_loads_plain_string_list(self, tmp_path):
        path = tmp_path / "prompts.json"
        path.write_text('["Explain recursion.", "Reverse a string."]', encoding="utf-8")
        prompts = load_prompts(path)
        assert len(prompts) == 2
        assert prompts[0] == {"id": "p0", "system_prompt": "", "user_prompt": "Explain recursion."}
        assert prompts[1]["id"] == "p1"

    def test_loads_object_list_with_ids(self, tmp_path):
        path = tmp_path / "prompts.json"
        path.write_text(
            '[{"id": "recursion", "system_prompt": "Be terse.", "user_prompt": "Explain recursion."}]',
            encoding="utf-8",
        )
        prompts = load_prompts(path)
        assert prompts[0]["id"] == "recursion"
        assert prompts[0]["system_prompt"] == "Be terse."

    def test_object_accepts_prompt_key_as_alias(self, tmp_path):
        path = tmp_path / "prompts.json"
        path.write_text('[{"id": "x", "prompt": "hello"}]', encoding="utf-8")
        prompts = load_prompts(path)
        assert prompts[0]["user_prompt"] == "hello"


class TestExistingKeysAndCache:
    def test_existing_keys_pairs_id_and_rater(self):
        examples = [
            GoldExample(id="p0::copilot", prompt="p", output="o", human_score=80.0, rater_id="rater1"),
            GoldExample(id="p0::copilot", prompt="p", output="o", human_score=70.0, rater_id="rater2"),
        ]
        keys = existing_keys(examples)
        assert keys == {("p0::copilot", "rater1"), ("p0::copilot", "rater2")}

    def test_cached_outputs_keeps_first_seen_per_id(self):
        examples = [
            GoldExample(id="p0::copilot", prompt="p", output="first-output", human_score=80.0, rater_id="rater1"),
            GoldExample(id="p0::copilot", prompt="p", output="second-output", human_score=70.0, rater_id="rater2"),
        ]
        cache = cached_outputs(examples)
        # A second rater must see the SAME output the first rater saw, not a fresh one.
        assert cache["p0::copilot"].output == "first-output"

    def test_resuming_a_rater_skips_already_rated_examples(self):
        existing = [GoldExample(id="p0::copilot", prompt="p", output="o", human_score=80.0, rater_id="rater1")]
        done = existing_keys(existing)
        assert ("p0::copilot", "rater1") in done
        assert ("p0::copilot", "rater2") not in done  # rater2 still needs to rate it


def _gold_set_with_known_correlation() -> list[GoldExample]:
    # human_score scales with word count, same construction as test_correlate.py,
    # so semantic_accuracy/length_fit/quality all see *some* signal to correlate against.
    examples = []
    for i in range(1, 7):
        output = " ".join(["explain recursion clearly"] * i)
        examples.append(
            GoldExample(
                id=f"ex{i}",
                prompt="explain recursion",
                output=output,
                human_score=min(i * 15.0, 100.0),
                rater_id="rater1",
            )
        )
    return examples


class TestGenerateReport:
    def test_reports_skip_agreement_without_two_raters(self):
        lines, summary = generate_report(_gold_set_with_known_correlation(), DEFAULT_METRICS)
        assert summary["agreement"] is None
        assert any("Skipped — requires exactly 2 raters" in line for line in lines)

    def test_reports_agreement_with_two_raters(self):
        examples = [
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="rater1"),
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="rater2"),
            GoldExample(id="ex2", prompt="p", output="o", human_score=50.0, rater_id="rater1"),
            GoldExample(id="ex2", prompt="p", output="o", human_score=50.0, rater_id="rater2"),
            GoldExample(id="ex3", prompt="p", output="o", human_score=90.0, rater_id="rater1"),
            GoldExample(id="ex3", prompt="p", output="o", human_score=90.0, rater_id="rater2"),
        ]
        lines, summary = generate_report(examples, DEFAULT_METRICS)
        assert summary["agreement"]["cohens_kappa"] == 1.0
        assert summary["agreement"]["reliable"] is True

    def test_reports_correlation_for_each_metric(self):
        lines, summary = generate_report(_gold_set_with_known_correlation(), DEFAULT_METRICS)
        assert set(summary["correlations"]) == set(DEFAULT_METRICS)
        for name in DEFAULT_METRICS:
            assert "spearman_r" in summary["correlations"][name]

    def test_too_few_examples_reports_skip_not_crash(self):
        examples = _gold_set_with_known_correlation()[:2]
        lines, summary = generate_report(examples, DEFAULT_METRICS)
        assert "skipped" in "\n".join(lines).lower()
        assert summary["correlations"] == {}
