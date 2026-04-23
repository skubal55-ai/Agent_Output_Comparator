# Architecture Overview

## System Purpose
The **Agent Output Comparator** is a local web application that lets engineers
run the same prompt against multiple AI coding agents (GitHub Copilot CLI and
OpenCode), inspect their file outputs side-by-side, and score the results.

## Key Quality Attributes
- **Observability** – every run is streamed and persisted for diff review.
- **Extensibility** – new agent runners are added by implementing a single
  `run(prompt, repo_path)` interface.
- **Portability** – pure Python / Flask backend; single-page HTML front-end
  with no build step required.

## C4 Diagram Index
| Level | File | Description |
|-------|------|-------------|
| 1 – Context   | diagrams/c4-level1-context.mmd   | System and external actors |
| 2 – Container | diagrams/c4-level2-container.mmd | Major deployable units |
| 3 – Component | diagrams/c4-level3-component.mmd | Internal API components |
| 4 – Code      | diagrams/c4-level4-code.mmd      | Key class relationships |

## Technology Decisions
- **Flask** – lightweight HTTP layer; no ORM needed for a local tool.
- **Mermaid (.mmd)** – diagrams live alongside code and render natively on GitHub.
- **pytest** – standard test runner; no extra plugins required.
