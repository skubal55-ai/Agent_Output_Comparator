import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.evalkit.stats.aggregate_stats import mean_stdev, confidence_interval_95, summarize
from source.evalkit.stats.significance import paired_significance, holm_bonferroni, all_pairs_significance


class TestMeanStdev:
    def test_empty_returns_zeros(self):
        assert mean_stdev([]) == (0.0, 0.0)

    def test_single_value_zero_stdev(self):
        assert mean_stdev([42.0]) == (42.0, 0.0)

    def test_known_values(self):
        mean, stdev = mean_stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(mean - 5.0) < 1e-9
        assert abs(stdev - 2.13809) < 1e-3


class TestConfidenceInterval95:
    def test_single_value_zero_width_interval(self):
        lo, hi = confidence_interval_95([10.0])
        assert lo == hi == 10.0

    def test_interval_contains_mean(self):
        values = [70.0, 72.0, 68.0, 75.0, 71.0]
        lo, hi = confidence_interval_95(values)
        mean, _ = mean_stdev(values)
        assert lo <= mean <= hi

    def test_more_variance_widens_interval(self):
        tight = confidence_interval_95([50.0, 51.0, 49.0, 50.0, 50.0])
        wide = confidence_interval_95([10.0, 90.0, 20.0, 80.0, 50.0])
        assert (wide[1] - wide[0]) > (tight[1] - tight[0])

    def test_bounds_truncate_interval_to_metric_range(self):
        values = [100.0, 100.0, 20.0, 100.0, 100.0, 100.0]  # t-interval would exceed 100
        assert confidence_interval_95(values)[1] > 100.0
        lo, hi = confidence_interval_95(values, bounds=(0.0, 100.0))
        assert 0.0 <= lo <= hi == 100.0


class TestSummarize:
    def test_includes_all_fields(self):
        result = summarize([60.0, 70.0, 80.0])
        assert result["n"] == 3
        assert result["mean"] == 70.0
        assert "stdev" in result
        assert "ci95_low" in result and "ci95_high" in result


class TestPairedSignificance:
    def test_identical_samples_not_significant(self):
        result = paired_significance([50.0, 60.0, 70.0], [50.0, 60.0, 70.0])
        assert result["p_value"] == 1.0

    def test_consistently_higher_sample_detected(self):
        a = [90.0, 92.0, 88.0, 91.0, 89.0, 93.0]
        b = [40.0, 42.0, 38.0, 41.0, 39.0, 43.0]
        result = paired_significance(a, b)
        assert result["p_value"] < 0.05
        assert result["effect_size"] > 0

    def test_effect_size_is_matched_pairs_rank_biserial(self):
        # diffs +1, +2, -3, +4: ranks 1,2,3,4 -> R+ = 7, R- = 3 -> (7-3)/10 = 0.4
        # (a sign count would give (3-1)/4 = 0.5, which is the old, mislabelled statistic)
        a = [11.0, 22.0, 27.0, 44.0]
        b = [10.0, 20.0, 30.0, 40.0]
        result = paired_significance(a, b)
        assert abs(result["effect_size"] - 0.4) < 1e-9

    def test_ttest_method(self):
        a = [90.0, 92.0, 88.0, 91.0, 89.0, 93.0]
        b = [40.0, 42.0, 38.0, 41.0, 39.0, 43.0]
        result = paired_significance(a, b, method="ttest")
        assert result["method"] == "ttest"
        assert result["p_value"] < 0.05

    def test_mismatched_lengths_raises(self):
        try:
            paired_significance([1.0, 2.0], [1.0])
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_too_few_observations_raises(self):
        try:
            paired_significance([1.0], [2.0])
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestHolmBonferroni:
    def test_empty_input_returns_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_pvalue_uses_full_alpha(self):
        result = holm_bonferroni([0.03], alpha=0.05)
        assert result[0]["threshold"] == 0.05
        assert result[0]["significant"] is True

    def test_smallest_pvalue_gets_strictest_threshold(self):
        # 3 tests, alpha=0.05: thresholds are 0.05/3, 0.05/2, 0.05/1 in rank order
        result = holm_bonferroni([0.20, 0.01, 0.04], alpha=0.05)
        assert abs(result[1]["threshold"] - 0.05 / 3) < 1e-9  # smallest p (0.01) -> rank 1
        assert result[1]["significant"] is True

    def test_step_down_stops_at_first_failure(self):
        # p=0.01 passes 0.05/3; p=0.02 passes 0.05/2; p=0.20 fails 0.05/1 -> all after it fail too
        result = holm_bonferroni([0.01, 0.02, 0.20], alpha=0.05)
        assert result[0]["significant"] is True
        assert result[1]["significant"] is True
        assert result[2]["significant"] is False

    def test_uncorrected_significant_pvalue_can_fail_after_correction(self):
        # 0.04 < 0.05 uncorrected, but with 4 comparisons Holm's strictest threshold is 0.05/4
        result = holm_bonferroni([0.04, 0.5, 0.6, 0.7], alpha=0.05)
        assert result[0]["significant"] is False


class TestAllPairsSignificance:
    def test_two_agents_no_correction_applied(self):
        scores = {
            "agent_a": [90.0, 92.0, 88.0, 91.0],
            "agent_b": [40.0, 42.0, 38.0, 41.0],
        }
        result = all_pairs_significance(scores)
        assert len(result["pairs"]) == 1
        assert "holm_bonferroni" not in result["pairs"][0]

    def test_three_agents_gets_holm_bonferroni_on_all_pairs(self):
        scores = {
            "agent_a": [90.0, 92.0, 88.0, 91.0, 89.0],
            "agent_b": [40.0, 42.0, 38.0, 41.0, 39.0],
            "agent_c": [91.0, 93.0, 89.0, 92.0, 90.0],
        }
        result = all_pairs_significance(scores)
        assert len(result["pairs"]) == 3  # C(3,2)
        for pair in result["pairs"]:
            assert "holm_bonferroni" in pair

    def test_single_agent_returns_no_pairs(self):
        result = all_pairs_significance({"agent_a": [1.0, 2.0, 3.0]})
        assert result["pairs"] == []
