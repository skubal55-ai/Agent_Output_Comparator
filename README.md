# Agent Output Comparator

A local tool that sends the same prompt to two command-line AI coding assistants —
**GitHub Copilot CLI** and **Claude Code CLI** — shows their answers side by side, and scores
each answer with a deterministic, judge-free rubric.

- Runs both agents **one after the other** on an identical prompt (so latencies aren't distorted by contention)
- Detects **failed runs** (launch errors, timeouts, CLI-reported errors such as expired logins, empty answers) and scores them 0 instead of scoring the error text
- Scores every answer on four metrics — **Quality**, **Semantic Accuracy**, **Efficiency** and **Length Fit** — shown in the UI as Quality / Accuracy / Speed / Length
- Checks files an agent names in its answer against the local file system, with preview and download
- **Research mode**: N trials per (agent, prompt), stored in SQLite, with means, 95% confidence intervals and paired Wilcoxon tests; CSV/JSON export

See [`docs/methodology.md`](docs/methodology.md) for how every metric and statistic is computed and their known limitations.

## Requirements

- Windows (binary detection is Windows-oriented), Python 3.10+
- [GitHub Copilot CLI](https://github.com/github/copilot-cli), signed in
- [Claude Code CLI](https://code.claude.com/docs/en/overview), signed in — run `claude` once in a terminal and use `/login`

```bash
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

Then open http://localhost:5050 (or use `start.bat` / `start.sh`). The header shows whether both CLIs were found.

1. **Step 1** – load an agent instruction file (`.md`); it becomes the system prompt for both agents.
2. **Step 2** – add prompts (or use *Suggest Prompts*), optionally set the agent working folder.
3. **Run Comparison** – see both answers, per-metric scores, the winner and a results-history table.
4. **Research Mode** – choose trials per prompt and run a multi-trial experiment with statistics.

Environment variables: `COMPARE_COPILOT_CWD`, `COMPARE_COPILOT_TIMEOUT_SEC`, `COMPARE_CLAUDE_TIMEOUT_SEC`,
`COMPARE_CLAUDE_PERMISSION_MODE` (default `acceptEdits`), `COMPARE_CLAUDE_MODEL`, `PORT` (default 5050).

> The backend only accepts cross-origin requests from its own origin (`http://localhost:<PORT>`), so open the UI
> through the server rather than as a local `file://` page.

## Example study (55 prompts)

[`docs/examples/`](docs/examples/) contains a 55-prompt comparison across 11 software-engineering categories:
the prompts, the instruction file, the raw per-run results and an Excel report.

```bash
python scripts/run_example_batch.py     # re-runs all 55 prompts through /api/compare (uses both CLIs)
python scripts/analyze_examples.py      # recomputes every statistic from docs/examples/results_raw.jsonl
```

In that study the faster agent won every prompt; with the Efficiency metric removed the two agents scored the
same — see the methodology notes before reading the scores as a quality ranking.

## Tests

```bash
python -m pytest -q
```

## Project layout

| Path | Contents |
|---|---|
| `server.py` | Flask backend: CLI runners, output normalization, failure detection, file extraction, API |
| `index.html` | Single-page UI |
| `source/evalkit/` | Agent interface, metrics, experiment runner, SQLite storage, statistics, validation |
| `scripts/` | Validation-study helpers and the example-study scripts |
| `tests/` | Unit tests |
| `docs/` | Methodology, architecture diagrams, example study |
