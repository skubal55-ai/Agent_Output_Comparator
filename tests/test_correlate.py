import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.evalkit.metrics.base import Metric, MetricResult
from source.evalkit.validation.gold_set import GoldExample
from source.evalkit.validation.correlate import metric_correlation, inter_rater_agreement


class _WordCountMetric(Metric):
    """Deterministic metric whose score is proportional to word count,
    used so the correlation with a matching human-score gold set is known.
    """

    name = "word_count"

    def compute(self, *, output, prompt, system_prompt="", **context):
        return MetricResult(score=min(len(output.split()) * 10.0, 100.0))


def _gold_set_perfectly_correlated() -> list[GoldExample]:
    # human_score set to match word-count-derived score exactly (r ~= 1.0)
    examples = []
    for i in range(1, 7):
        output = " ".join(["word"] * i)
        examples.append(
            GoldExample(
                id=f"ex{i}",
                prompt="say something",
                output=output,
                human_score=min(i * 10.0, 100.0),
                rater_id="rater1",
            )
        )
    return examples


class TestMetricCorrelation:
    def test_strong_correlation_detected(self):
        result = metric_correlation(_gold_set_perfectly_correlated(), _WordCountMetric())
        assert result["pearson_r"] > 0.95
        assert result["spearman_r"] > 0.95
        assert result["n"] == 6

    def test_multi_rater_scores_are_averaged(self):
        examples = [
            GoldExample(id="ex1", prompt="p", output="one two three", human_score=80.0, rater_id="r1"),
            GoldExample(id="ex1", prompt="p", output="one two three", human_score=60.0, rater_id="r2"),
            GoldExample(id="ex2", prompt="p", output="a b c d e f", human_score=90.0, rater_id="r1"),
            GoldExample(id="ex2", prompt="p", output="a b c d e f", human_score=90.0, rater_id="r2"),
            GoldExample(id="ex3", prompt="p", output="x", human_score=10.0, rater_id="r1"),
            GoldExample(id="ex3", prompt="p", output="x", human_score=10.0, rater_id="r2"),
        ]
        # Should not raise, and should run on 3 unique example ids
        result = metric_correlation(examples, _WordCountMetric())
        assert result["n"] == 3

    def test_too_few_examples_raises(self):
        examples = _gold_set_perfectly_correlated()[:2]
        try:
            metric_correlation(examples, _WordCountMetric())
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestInterRaterAgreement:
    def test_perfect_agreement_kappa_is_one(self):
        examples = [
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="r1"),
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="r2"),
            GoldExample(id="ex2", prompt="p", output="o", human_score=50.0, rater_id="r1"),
            GoldExample(id="ex2", prompt="p", output="o", human_score=50.0, rater_id="r2"),
            GoldExample(id="ex3", prompt="p", output="o", human_score=90.0, rater_id="r1"),
            GoldExample(id="ex3", prompt="p", output="o", human_score=90.0, rater_id="r2"),
        ]
        result = inter_rater_agreement(examples)
        assert result["cohens_kappa"] == 1.0

    def test_requires_exactly_two_raters(self):
        examples = [
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="r1"),
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="r2"),
            GoldExample(id="ex1", prompt="p", output="o", human_score=10.0, rater_id="r3"),
        ]
        try:
            inter_rater_agreement(examples)
            assert False, "expected ValueError"
        except ValueError:
            pass
