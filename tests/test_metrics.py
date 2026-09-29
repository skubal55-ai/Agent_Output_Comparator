import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.evalkit.metrics.structure import QualityHeuristicMetric, LengthFitMetric
from source.evalkit.metrics.semantic_accuracy import SemanticAccuracyMetric
from source.evalkit.metrics.efficiency import EfficiencyMetric
from source.evalkit.metrics.aggregate import AggregateScorer, WeightedMetric, default_scorer
from source.evalkit.metrics.base import Metric, MetricResult


class TestQualityHeuristicMetric:
    def test_empty_output_scores_zero(self):
        result = QualityHeuristicMetric().compute(output="", prompt="x")
        assert result.score == 0.0

    def test_well_structured_output_scores_high(self):
        output = (
            "Here is a detailed explanation of the topic.\n\n"
            "- First point about the topic.\n"
            "- Second point about the topic.\n\n"
            "This covers the essentials. It should be enough context. "
            "Let me know if you need more."
        )
        result = QualityHeuristicMetric().compute(output=output, prompt="explain the topic")
        assert result.score >= 80

    def test_error_phrase_lowers_score(self):
        clean = "This is a normal response with enough words to count as substantial content here."
        erroring = "Error: command not found. This is a normal response with enough words to count."
        clean_score = QualityHeuristicMetric().compute(output=clean, prompt="x").score
        error_score = QualityHeuristicMetric().compute(output=erroring, prompt="x").score
        assert error_score < clean_score

    def test_truncated_output_scores_lower(self):
        full = "This is a complete sentence with a clear ending point right here."
        truncated = "This is an incomplete sentence that just cuts off mid way..."
        full_score = QualityHeuristicMetric().compute(output=full, prompt="x").score
        truncated_score = QualityHeuristicMetric().compute(output=truncated, prompt="x").score
        assert truncated_score < full_score


class TestLengthFitMetric:
    def test_very_short_scores_low(self):
        result = LengthFitMetric().compute(output="Hi", prompt="x")
        assert result.score <= 10

    def test_ideal_length_scores_100(self):
        output = " ".join(["word"] * 100)
        result = LengthFitMetric().compute(output=output, prompt="x")
        assert result.score == 100

    def test_excessively_long_scores_lower_than_ideal(self):
        ideal = " ".join(["word"] * 100)
        long_output = " ".join(["word"] * 800)
        ideal_score = LengthFitMetric().compute(output=ideal, prompt="x").score
        long_score = LengthFitMetric().compute(output=long_output, prompt="x").score
        assert long_score < ideal_score


class TestSemanticAccuracyMetric:
    def test_no_keyword_overlap_scores_low(self):
        result = SemanticAccuracyMetric().compute(
            output="bananas apples oranges grapefruit",
            prompt="explain quantum computing algorithms",
        )
        assert result.score < 30

    def test_high_keyword_overlap_scores_higher(self):
        prompt = "explain how binary search trees balance rotation"
        matching = "Binary search trees balance rotation using rotation operations on the tree."
        unrelated = "The weather today is sunny with a chance of rain in the afternoon."
        match_score = SemanticAccuracyMetric().compute(output=matching, prompt=prompt).score
        unrelated_score = SemanticAccuracyMetric().compute(output=unrelated, prompt=prompt).score
        assert match_score > unrelated_score

    def test_empty_output_scores_zero(self):
        result = SemanticAccuracyMetric().compute(output="", prompt="something with real words")
        assert result.score == 0.0


class TestEfficiencyMetric:
    def test_no_peer_data_is_neutral(self):
        result = EfficiencyMetric().compute(output="x", prompt="x", latency_ms=1000, peer_latencies_ms=[])
        assert result.score == 70.0

    def test_fastest_scores_highest(self):
        fast = EfficiencyMetric().compute(
            output="x", prompt="x", latency_ms=100, peer_latencies_ms=[100, 500, 900]
        )
        slow = EfficiencyMetric().compute(
            output="x", prompt="x", latency_ms=900, peer_latencies_ms=[100, 500, 900]
        )
        assert fast.score > slow.score
        assert fast.score == 100.0
        assert slow.score == 20.0


class _ConstantMetric(Metric):
    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = value

    def compute(self, *, output, prompt, system_prompt="", **context):
        return MetricResult(score=self.value)


class TestAggregateScorer:
    def test_weighted_combination(self):
        scorer = AggregateScorer(
            [
                WeightedMetric("a", _ConstantMetric(100.0), 1.0),
                WeightedMetric("b", _ConstantMetric(0.0), 1.0),
            ]
        )
        result = scorer.score(output="anything", prompt="x")
        assert result["overall"] == 50.0

    def test_weights_are_normalized(self):
        scorer = AggregateScorer(
            [
                WeightedMetric("a", _ConstantMetric(100.0), 3.0),
                WeightedMetric("b", _ConstantMetric(0.0), 1.0),
            ]
        )
        result = scorer.score(output="anything", prompt="x")
        assert result["overall"] == 75.0

    def test_requires_at_least_one_metric(self):
        try:
            AggregateScorer([])
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_default_scorer_produces_overall_and_word_count(self):
        result = default_scorer().score(output="A reasonably long response here.", prompt="respond")
        assert 0 <= result["overall"] <= 100
        assert result["word_count"] == 5

    def test_default_scorer_excludes_task_success_by_default(self):
        result = default_scorer().score(output="anything", prompt="x")
        assert "task_success" not in result

    def test_default_scorer_includes_task_success_when_weighted(self):
        result = default_scorer(task_success_weight=0.5).score(
            output="anything", prompt="x", pass_rate=1.0
        )
        assert "task_success" in result
        assert result["task_success"]["score"] == 100.0


class TestSemanticAccuracyEmbeddingHook:
    def test_defaults_to_tfidf_backend(self):
        result = SemanticAccuracyMetric().compute(output="some words here", prompt="some words")
        assert result.details["backend"] == "tfidf"

    def test_uses_embedding_fn_when_supplied(self):
        # Deterministic fake embedding: identical text -> identical vector -> similarity 1.0
        def fake_embed(text: str):
            return [float(len(text)), float(text.count("a")), 1.0]

        metric = SemanticAccuracyMetric(embedding_fn=fake_embed)
        result = metric.compute(output="aaa", prompt="aaa")
        assert result.details["backend"] == "embedding"
        assert result.score > 90  # identical text -> near-identical embedding -> high score

    def test_embedding_similarity_maps_negative_cosine_to_low_score(self):
        def opposite_embed(text: str):
            return [1.0, 0.0] if text == "prompt-side" else [-1.0, 0.0]

        metric = SemanticAccuracyMetric(embedding_fn=opposite_embed)
        result = metric.compute(output="output-side", prompt="prompt-side")
        assert result.score < 10  # cosine similarity -1 -> mapped near 0

    def test_mismatched_embedding_dims_raises(self):
        def bad_embed(text: str):
            return [1.0, 2.0] if "prompt" in text else [1.0]

        metric = SemanticAccuracyMetric(embedding_fn=bad_embed)
        try:
            metric.compute(output="output", prompt="prompt")
            assert False, "expected ValueError"
        except ValueError:
            pass
