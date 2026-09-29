import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.evalkit.agents.base import Agent, AgentResult
from source.evalkit.experiment.config import ExperimentConfig
from source.evalkit.experiment.runner import run_experiment
from source.evalkit.experiment.storage import ResultStore


class StubAgent(Agent):
    """Deterministic fake agent — no real CLI is invoked in tests."""

    def __init__(self, name: str, latency_ms: int, output: str) -> None:
        self.name = name
        self._latency_ms = latency_ms
        self._output = output

    def run(self, system_prompt: str, user_prompt: str, **kwargs) -> AgentResult:
        return AgentResult(output=self._output, latency_ms=self._latency_ms, error=None, meta={})

    def version(self) -> str:
        return "stub-1.0"


def _config(n_trials: int = 2) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "stub-experiment",
            "agents": ["agent_a", "agent_b"],
            "prompts": [
                {"id": "p1", "user_prompt": "Explain recursion with an example."},
                {"id": "p2", "user_prompt": "Write a function to reverse a string."},
            ],
            "n_trials": n_trials,
        }
    )


def _stub_agents() -> dict:
    return {
        "agent_a": StubAgent(
            "agent_a", 300, "A clear explanation with structure.\n- point one\n- point two"
        ),
        "agent_b": StubAgent("agent_b", 900, "Short answer."),
    }


def test_run_experiment_persists_all_trials(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = _config(n_trials=2)

    experiment_id = run_experiment(config, store=store, agents=_stub_agents())

    experiment = store.get_experiment(experiment_id)
    assert experiment is not None
    # 2 agents x 2 prompts x 2 trials = 8 trial rows
    assert len(experiment["trials"]) == 8


def test_run_experiment_scores_and_reproducibility_recorded(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = _config(n_trials=1)

    experiment_id = run_experiment(config, store=store, agents=_stub_agents())
    trials = store.get_trials(experiment_id)

    for trial in trials:
        assert "overall" in trial["scores"]
        assert 0 <= trial["scores"]["overall"] <= 100
        assert trial["reproducibility"]["agent_version"] == "stub-1.0"
        assert trial["reproducibility"]["agent_name"] == trial["agent"]


def test_faster_agent_gets_higher_efficiency_score(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = _config(n_trials=1)

    experiment_id = run_experiment(config, store=store, agents=_stub_agents())
    trials = store.get_trials(experiment_id)

    fast_scores = [t["scores"]["efficiency"]["score"] for t in trials if t["agent"] == "agent_a"]
    slow_scores = [t["scores"]["efficiency"]["score"] for t in trials if t["agent"] == "agent_b"]
    assert all(f > s for f, s in zip(fast_scores, slow_scores))


class FailingAgent(StubAgent):
    """Returns an error (e.g. an auth failure) very quickly."""

    def run(self, system_prompt: str, user_prompt: str, **kwargs) -> AgentResult:
        return AgentResult(output="", latency_ms=50, error="Not logged in", meta={})


def test_failed_agent_scores_zero_and_is_not_a_latency_peer(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = _config(n_trials=1)
    ok = StubAgent("agent_a", 900, "A clear explanation with structure.\n- point one\n- point two")
    agents = {"agent_a": ok, "agent_b": FailingAgent("agent_b", 0, "")}

    experiment_id = run_experiment(config, store=store, agents=agents)
    trials = store.get_trials(experiment_id)

    failed = [t for t in trials if t["agent"] == "agent_b"]
    assert all(t["scores"]["failed"] and t["scores"]["overall"] == 0 for t in failed)
    assert all(t["scores"]["efficiency"]["score"] == 0 for t in failed)
    # The fast failure must not make the successful agent look slow.
    solo = store.get_trials(run_experiment(config, store=store, agents={"agent_a": ok}))
    ok_eff = sorted(t["scores"]["efficiency"]["score"] for t in trials if t["agent"] == "agent_a")
    assert ok_eff == sorted(t["scores"]["efficiency"]["score"] for t in solo)


def test_empty_answer_counts_as_failed_run(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = _config(n_trials=1)
    agents = {
        "agent_a": StubAgent("agent_a", 300, "A clear explanation with structure.\n- point one\n- point two"),
        "agent_b": StubAgent("agent_b", 100, "   "),
    }

    trials = store.get_trials(run_experiment(config, store=store, agents=agents))

    empty = [t for t in trials if t["agent"] == "agent_b"]
    assert all(t["scores"]["failed"] and t["error"] for t in empty)


def test_claude_agent_is_registered():
    from source.evalkit.experiment.runner import AGENT_REGISTRY

    assert set(AGENT_REGISTRY) == {"copilot", "claude"}


def test_unknown_agent_name_raises():
    config = ExperimentConfig.from_dict(
        {"name": "x", "agents": ["not_a_real_agent"], "prompts": ["hi"], "n_trials": 1}
    )
    try:
        run_experiment(config)
        assert False, "expected ValueError for unknown agent"
    except ValueError:
        pass


def test_task_success_weight_excluded_when_zero(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = _config(n_trials=1)  # task_success_weight defaults to 0.0

    experiment_id = run_experiment(config, store=store, agents=_stub_agents())
    trials = store.get_trials(experiment_id)

    assert all("task_success" not in t["scores"] for t in trials)


def test_task_success_weight_included_with_pass_rate(tmp_path):
    store = ResultStore(db_path=tmp_path / "results.db")
    config = ExperimentConfig.from_dict(
        {
            "name": "task-success-experiment",
            "agents": ["agent_a"],
            "prompts": [
                {"id": "p1", "user_prompt": "Write a function.", "pass_rate": 1.0},
                {"id": "p2", "user_prompt": "Write another function.", "pass_rate": 0.0},
            ],
            "n_trials": 1,
            "task_success_weight": 0.5,
        }
    )
    agents = {"agent_a": StubAgent("agent_a", 300, "def f(): pass")}

    experiment_id = run_experiment(config, store=store, agents=agents)
    trials = store.get_trials(experiment_id)

    by_prompt = {t["prompt_id"]: t for t in trials}
    assert by_prompt["p1"]["scores"]["task_success"]["score"] == 100.0
    assert by_prompt["p2"]["scores"]["task_success"]["score"] == 0.0
    # A prompt with a failing test should score lower overall than one that passes,
    # all else being equal, once task_success carries real weight.
    assert by_prompt["p1"]["scores"]["overall"] > by_prompt["p2"]["scores"]["overall"]
