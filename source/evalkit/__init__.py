"""evalkit — a pluggable evaluation framework for comparing coding agents.

Sub-packages:
  agents      Agent interface + concrete wrappers (Copilot CLI, Claude Code CLI, ...)
  metrics     Independent, pluggable scoring metrics + AggregateScorer
  experiment  Config-driven multi-trial experiment runner + persistence
  stats       Aggregate statistics and significance testing across trials
  validation  Tooling to validate automated metrics against human judgment
  reporting   Export experiment results for external analysis / papers
"""
