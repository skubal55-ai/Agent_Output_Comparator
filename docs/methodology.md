# Evaluation Methodology

This document specifies the evaluation methodology implemented by
`source/evalkit/` — a framework for comparing coding agents (CLI-based AI
coding assistants) on identical prompts. It is written to double as the
Methods section of a paper describing the framework; code references point
at the implementing module for traceability.

## 1. Problem Statement

Comparing coding agents ad hoc — running one prompt once against each tool
and eyeballing the output — does not support a defensible claim that one
agent outperforms another. A comparison is defensible only if:

1. the scoring dimensions are explicitly defined and their limitations
   stated,
2. the scoring dimensions correlate with what they claim to measure
   (validated against human judgment where the dimension is subjective),
3. results come from enough repeated trials to characterize variance, and
4. differences between agents are tested for statistical significance
   rather than read off a single run.

This framework is structured around those four requirements.

## 2. Agent Abstraction

Every evaluated agent implements `evalkit.agents.base.Agent`
(`run(system_prompt, user_prompt, **kwargs) -> AgentResult`,
`version() -> str`). The framework currently ships two concrete agents —
`CopilotAgent` and `ClaudeCodeAgent` — wrapping GitHub Copilot CLI and
Claude Code CLI respectively. Extending the study to a new agent requires only
a new `Agent` implementation; no change to metrics, the experiment runner,
persistence, or statistics.

## 3. Metrics

Each metric is an independent `evalkit.metrics.base.Metric` returning a
`MetricResult(score in [0, 100], details)`. Metrics are combined by
`AggregateScorer` with **named, configurable weights** — not hardcoded
inside a single scoring function — so a weight-sensitivity analysis can be
reported and weights can be revised without touching metric code.

| Metric | Module | Formula summary | Default weight |
|---|---|---|---|
| Quality (structure heuristic) | `metrics/structure.py` `QualityHeuristicMetric` | Additive score for length ≥10/≥30 words, presence of list/code/heading markup, ≥3 sentences, absence of error phrases, no truncation | 0.30 |
| Semantic accuracy | `metrics/semantic_accuracy.py` `SemanticAccuracyMetric` | TF-IDF cosine similarity between (system_prompt+prompt) and output, IDF computed over the 2-document pair | 0.35 |
| Efficiency | `metrics/efficiency.py` `EfficiencyMetric` | Min-max normalized latency across all agents/trials for the same prompt; fastest → 100, slowest → 20 | 0.15 |
| Length fit | `metrics/structure.py` `LengthFitMetric` | Bucketed score penalizing <30 or >400-word outputs | 0.20 |
| Task success (tests) | `metrics/task_success.py` `TestExecutionMetric` | Pass rate of a supplied test command / pre-computed pass rate; **optional**, requires a per-prompt test definition | 0 unless `task_success_weight` is set |
| Task success (LLM judge) | `metrics/task_success.py` `LLMJudgeMetric` | Delegates to a caller-supplied rubric judge function; **disabled by default** (no judge configured) | not in default weights |

`Quality` and `Length fit` are treated as **minor structural signals**, not
correctness measures — they capture surface form (Is there structure? Is it
a plausible length?), not whether the output is right. `Semantic accuracy`
is the primary lexical-overlap signal. Where a per-prompt test suite or an
LLM-judge rubric is available, `Task success` metrics should be added to
the weighted set and should dominate the weighting, since they are the only
metrics in this set that directly measure task correctness.

`TestExecutionMetric` is wired into the experiment harness, not just
available as a standalone class: `PromptSpec` (in
`evalkit.experiment.config`) carries optional `test_command`, `test_cwd`,
and `pass_rate` fields per prompt, and `ExperimentConfig.task_success_weight`
(default `0.0`) turns the metric on in `default_scorer()` and folds it into
the weighted average — `run_experiment` passes each prompt's test info
through automatically. It remains opt-in per experiment because the
framework cannot assume every prompt suite defines runnable tests or
pre-computed pass rates; leaving it at `0.0` (the default) is itself a
methodological choice that must be stated in any published comparison (see
Section 8).

`SemanticAccuracyMetric` defaults to the TF-IDF backend described above, but
accepts an optional `embedding_fn: Callable[[str], Sequence[float]]`
constructor argument to compute cosine similarity in embedding space
instead — see Section 8 for the tradeoffs and Section 9 for how to wire one
in. This is opt-in, not automatic, since it requires the caller to supply an
embedding model/API and accept that dependency.

## 4. Metric Validation

Automated metrics are proxies. Before trusting `Quality`, `Semantic
accuracy`, or an `LLMJudgeMetric` rubric in a published comparison, they
must be validated against human judgment:

1. Collect a gold set of (prompt, output, human_score) triples — see
   `docs/gold_set_template.csv` and `evalkit.validation.gold_set`.
   Recommended size: 30-50 examples, each rated by ≥2 independent raters
   on a 0-100 scale.
2. Compute inter-rater agreement with
   `evalkit.validation.correlate.inter_rater_agreement` (Cohen's κ on
   binned low/medium/high scores). Low κ (< 0.4) means the human rubric
   itself is unreliable — fix the rubric before validating a metric
   against it.
3. Compute metric-vs-human correlation with
   `evalkit.validation.correlate.metric_correlation` (Pearson r and
   Spearman ρ, human scores averaged across raters). Report both; Spearman
   is more robust if the relationship is monotonic but non-linear.
4. A metric with weak correlation (|r| < 0.3) to human judgment should not
   be weighted heavily, or at all, in `AggregateScorer` — its weight should
   be reduced and the change documented.

This validation step, not the metric definitions alone, is what justifies
the default weights in Section 3. Until it is run for a given prompt
domain, the default weights should be reported as **unvalidated priors**,
not established facts.

### 4.1 Running the validation study

Two scripts implement steps 1-3 above end to end — they do not invent or
simulate human judgments; they are data-collection and analysis aids around
a human actually supplying scores:

1. **Collect ratings**: `python scripts/collect_ratings.py --prompts
   docs/gold_set_prompts.example.json --agents copilot claude --rater-id
   rater1 --out docs/gold_set.csv` runs each prompt against the given
   agents (reusing the same `Agent` wrappers the rest of the framework
   uses), shows you the prompt and output in the terminal, and asks for a
   0-100 score. Progress is saved to the CSV as you go — Ctrl+C (or typing
   `q`) is safe, and rerunning resumes where you left off. Run it again
   with `--rater-id rater2` for the second rater: because agents are
   non-deterministic, the script caches the first-generated output per
   (prompt, agent) pair in the CSV and shows every subsequent rater that
   *exact same* cached output rather than a fresh generation — this is
   required for inter-rater agreement and metric correlation to be
   comparing raters against the same artifact, not different ones.
   `docs/gold_set_prompts.example.json` ships as a small starter prompt
   set; swap it for your own domain-representative prompts (Section 5).
2. **Run the analysis**: `python scripts/run_validation.py --gold-set
   docs/gold_set.csv --report docs/validation_report.md` loads the CSV,
   computes Cohen's κ (only when exactly two raters are present) and, for
   each of `Quality`, `Semantic accuracy`, and `Length fit`, Pearson r and
   Spearman ρ against the averaged human score (`Efficiency` is excluded —
   it scores latency, which a static gold set of prompt/output pairs
   doesn't carry). It prints a table and, with `--report`, writes it to a
   markdown file citing the same 0.4 (κ) / 0.3 (|r|) thresholds used in
   this section, flagging low agreement or weak correlation instead of
   silently accepting them.
3. **Act on the result**: if κ is below 0.4, the rubric or rater
   instructions need fixing before the correlation numbers mean anything.
   For any metric whose correlation is weak, reduce or zero its weight in
   the `AggregateScorer` you construct (Section 9) and re-state the
   weighting rationale — the default weights in Section 3 are a starting
   point, not the output of this validation.

## 5. Experiment Design

Experiments are config-driven (`evalkit.experiment.config.ExperimentConfig`)
rather than single ad hoc runs:

- **Agents**: the set under comparison.
- **Prompt suite**: a fixed, versioned list of prompts (with optional
  per-prompt system prompt), not typed in ad hoc per run. Prompt suite
  design should state selection criteria (task domain, difficulty
  distribution, source) so results generalize to a stated scope, not to
  "whatever the tester happened to type."
- **n_trials**: each (agent, prompt) pair is run `n_trials` times to
  characterize run-to-run variance (agent outputs are not deterministic).
  A minimum of 5 trials per (agent, prompt) is recommended before reporting
  a comparison; single-trial results should be labeled as such and treated
  as exploratory, not conclusive.

`evalkit.experiment.runner.run_experiment` executes every (agent, prompt,
trial) combination sequentially — agents for the same prompt+trial are run
one after another rather than concurrently, matching the original
comparator's behavior of avoiding shared-resource contention that would
bias latency comparisons. All trials are persisted to SQLite
(`evalkit.experiment.storage.ResultStore`) as they run.

## 6. Statistical Reporting

- **Aggregate statistics** (`evalkit.stats.aggregate_stats.summarize`):
  mean, standard deviation, and a 95% confidence interval (t-distribution,
  appropriate for the small trial counts typical of agent evaluation)
  per (agent, metric). Scores are bounded (0–100), so the API truncates the
  interval to that range; with few trials an untruncated t-interval can
  otherwise report impossible values such as 120.9.
- **Significance testing**
  (`evalkit.stats.significance.paired_significance`): a paired comparison
  between two agents' scores on the same prompts. Default is the Wilcoxon
  signed-rank test (non-parametric, appropriate for bounded 0-100 scores
  that are not guaranteed to be normally distributed); a paired t-test is
  offered as an alternative. Effect size is reported alongside p-values —
  a statistically significant but tiny effect size should not be reported
  as a meaningful practical difference.
- **Multiple comparisons**: comparing more than two agents pairwise inflates
  the family-wise false-positive rate if each pair's p-value is read against
  the same uncorrected α. `evalkit.stats.significance.all_pairs_significance`
  runs every pairwise `paired_significance` test across the agents in an
  experiment and, once there are more than two, applies a Holm-Bonferroni
  step-down correction (`evalkit.stats.significance.holm_bonferroni`) across
  all pairs — each pair's result carries both the raw p-value and a
  `holm_bonferroni` block with the corrected significance decision. The
  `/api/experiments/<id>` endpoint in `server.py` calls this automatically
  for any number of agents ≥2 with equal trial counts; a 2-agent comparison
  is reported without correction (nothing to correct for).

## 7. Reproducibility

Every trial records (`evalkit.experiment.reproducibility.capture`):
agent name, agent CLI version (queried via `<binary> --version` at run
time), OS platform string, Python version, and a UTC timestamp. This lets a
reported comparison be tied to the exact tool versions it used — agent CLI
behavior changes over time (model routing, prompt handling), so a
comparison run against one version does not necessarily hold for a later
one. Any published result should state the captured versions.

## 8. Threats to Validity

- **Non-determinism**: Coding agents are typically non-deterministic
  (sampling, tool-call ordering, model routing). Single-trial comparisons
  are not reliable; this is why `n_trials` and confidence intervals exist,
  but even multi-trial results only bound variance observed in the
  sampled runs, not all possible runs.
- **Network/service latency**: Latency-based `Efficiency` scores conflate
  agent processing time with network conditions and backend load at the
  time of the run, which the framework does not control for. Latency
  comparisons across different days/times should be treated cautiously.
- **Lexical vs. semantic accuracy**: `SemanticAccuracyMetric` defaults to
  TF-IDF cosine similarity, not embedding-based semantic similarity. Two
  outputs that are semantically equivalent but lexically different (e.g.
  paraphrases, different variable names) will be scored as dissimilar under
  the default backend. An `embedding_fn` hook exists (Section 3) so a caller
  can supply real embeddings, but the framework does not ship one by
  default — doing so would force a model-download or API-key dependency on
  every user regardless of whether they need it. Any comparison using the
  embedding backend must report which embedding model/version was used
  (same reproducibility expectation as agent CLI versions, Section 7) and
  should re-run the Section 4 validation against that specific embedding
  choice — correlation with human judgment is not guaranteed to transfer
  from TF-IDF to a given embedding model.
- **Small gold sets**: A 30-50 example gold set (Section 4) gives limited
  statistical power to detect weak-to-moderate correlations, and results
  may not generalize across prompt domains (a metric validated on coding
  prompts may not validate on documentation-writing prompts).
- **Prompt suite representativeness**: Any conclusion is scoped to the
  prompt suite used. Claims of general agent superiority require a prompt
  suite whose domain coverage and difficulty are explicitly justified, not
  assumed.
- **Task success is optional and often unset**: `task_success_weight`
  defaults to `0.0` (Section 3). When left at 0, the framework's default
  weighting has no direct correctness signal —
  `Quality`/`Length fit`/`Semantic accuracy` are surface proxies. Any
  comparison run without a nonzero `Task success` weight should state this
  limitation explicitly; setting a nonzero weight requires the prompt suite
  to carry real `test_command`/`pass_rate` data per prompt (garbage in,
  garbage out — an unset `test_command` falls back to a neutral 50 score,
  which silently dilutes the aggregate if not noticed).
- **Multiple comparisons**: see Section 6 — `all_pairs_significance`
  applies a Holm-Bonferroni correction automatically for >2 agents, but the
  correction only controls the family-wise error rate for the pairwise
  tests actually run in *that* call. Running the same comparison multiple
  times (e.g. re-querying `/api/experiments/<id>` after appending more
  trials, or comparing subsets of agents across separate calls) and
  reporting the most favorable result is a form of uncorrected multiple
  comparisons that Holm-Bonferroni does not protect against — decide the
  comparison set and trial count in advance.
- **The default weights are unvalidated priors** (restated from Section 4):
  none of the numbers in the Section 3 table have been checked against
  human judgment in this environment. `scripts/collect_ratings.py` and
  `scripts/run_validation.py` (Section 4.1) make running that check a
  two-command workflow, but they cannot make the check happen — nothing in
  the codebase enforces that validation occurs before a weighted score is
  reported. Producing an actual gold set and running the scripts is still
  a step the experimenter must perform; the framework can only verify a
  gold set once one exists.

## 9. Extending the Framework

- **New agent**: implement `evalkit.agents.base.Agent`.
- **New metric**: implement `evalkit.metrics.base.Metric`; add it to a
  `AggregateScorer` with an explicit weight; add it to `DEFAULT_METRICS` in
  `scripts/run_validation.py` and re-run the Section 4.1 workflow before
  trusting its weight.
- **New judge backend**: supply a `judge_fn` to
  `evalkit.metrics.task_success.LLMJudgeMetric`; document the rubric
  prompt, judge model, and judge model version alongside any results that
  use it (the judge model version is part of the reproducibility
  statement, same as agent versions).
- **Enabling task success in an experiment**: set `test_command`,
  `test_cwd`, and/or `pass_rate` on the relevant `PromptSpec` entries and
  set `ExperimentConfig.task_success_weight` to a nonzero value (e.g. via
  the `task_success_weight` field in the `/api/experiments` request body).
- **Enabling embedding-based semantic accuracy**: build a scorer with
  `SemanticAccuracyMetric(embedding_fn=my_embed_fn)` in place of the one
  `default_scorer()` constructs, and pass it as the `scorer=` argument to
  `evalkit.experiment.runner.run_experiment`. `my_embed_fn` must map a
  string to a fixed-length numeric vector (e.g. wrap a sentence-transformers
  model or an embeddings API call); document its identity/version alongside
  results (Section 7, Section 8).
- **UI**: `index.html`'s "Research Mode" panel (below the single-comparison
  view) drives `POST /api/experiments` and renders the aggregates +
  significance from `GET /api/experiments/<id>`, with CSV/JSON export
  links to `GET /api/experiments/<id>/export`. It currently exposes agent
  selection and trial count only — `task_success_weight` and per-prompt
  test definitions are set via the API/Python directly, not yet from the UI.
