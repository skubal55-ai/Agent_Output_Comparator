"""Experiment runner: executes agents x prompts x n_trials sequentially,
scores every output, and persists each trial for later statistical
analysis. Sequential execution matches the existing app's behavior of
running agents one at a time per prompt so their latencies aren't skewed
by shared-resource contention.
"""
from __future__ import annotations

from typing import Any

from ..agents.base import Agent, AgentResult
from ..agents.claude_agent import ClaudeCodeAgent
from ..agents.copilot_agent import CopilotAgent
from ..metrics.aggregate import AggregateScorer, default_scorer
from .config import ExperimentConfig
from .reproducibility import capture
from .storage import ResultStore

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "copilot": CopilotAgent,
    "claude": ClaudeCodeAgent,
}


def build_agents(names: list[str]) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for n in names:
        cls = AGENT_REGISTRY.get(n)
        if cls is None:
            raise ValueError(f"Unknown agent '{n}'. Known agents: {sorted(AGENT_REGISTRY)}")
        agents[n] = cls()
    return agents


def run_experiment(
    config: ExperimentConfig,
    *,
    store: ResultStore | None = None,
    scorer: AggregateScorer | None = None,
    agents: dict[str, Agent] | None = None,
) -> str:
    """Runs the full experiment and returns the experiment id.

    ``agents`` may be supplied directly (e.g. test stubs implementing the
    Agent interface) to bypass AGENT_REGISTRY / real CLI invocation.
    """
    store = store or ResultStore()
    scorer = scorer or default_scorer(task_success_weight=config.task_success_weight)
    agents = agents or build_agents(config.agents)

    experiment_id = store.create_experiment(config.name, config.to_dict())

    for prompt in config.prompts:
        for trial_n in range(config.n_trials):
            # Run every agent once per (prompt, trial) so their latencies are peers
            # for the efficiency metric within this trial.
            trial_results: dict[str, Any] = {}
            for agent_name, agent in agents.items():
                kwargs = config.agent_kwargs.get(agent_name, {})
                result = agent.run(prompt.system_prompt, prompt.user_prompt, **kwargs)
                if not result.error and not (result.output or "").strip():
                    # No answer text is a failed run, not a (very short) answer.
                    result = AgentResult(output=result.output, latency_ms=result.latency_ms,
                                         error="Empty response: the agent returned no answer text.",
                                         meta=result.meta)
                trial_results[agent_name] = result

            # Failed runs are excluded as latency peers so a fast failure can't
            # inflate or deflate another agent's efficiency score.
            peer_latencies = [r.latency_ms for r in trial_results.values() if not r.error]
            for agent_name, result in trial_results.items():
                if result.error:
                    scores = scorer.failed_result(result.error)
                else:
                    scores = scorer.score(
                        output=result.output,
                        prompt=prompt.user_prompt,
                        system_prompt=prompt.system_prompt,
                        latency_ms=result.latency_ms,
                        peer_latencies_ms=peer_latencies,
                        test_command=prompt.test_command,
                        cwd=prompt.test_cwd,
                        pass_rate=prompt.pass_rate,
                    )
                repro = capture(agent_name, agents[agent_name].version())
                store.add_trial(
                    experiment_id=experiment_id,
                    agent=agent_name,
                    prompt_id=prompt.id,
                    trial_n=trial_n,
                    output=result.output,
                    latency_ms=result.latency_ms,
                    error=result.error,
                    scores=scores,
                    reproducibility=repro.to_dict(),
                )

    return experiment_id
