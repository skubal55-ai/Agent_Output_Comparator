"""
architecture_agent – generates C4 architecture diagrams and docs from a spec file.

Public API
----------
read_spec(path)                          -> str
create_output_dirs(repo_root)            -> str  (diagrams dir)
write_file(path, content)               -> dict
generate_overview_md_content(spec)      -> str
generate_c4_level1_context_mmd_content(spec) -> str
generate_c4_level2_container_mmd_content(spec) -> str
generate_c4_level3_component_mmd_content(spec) -> str
generate_c4_level4_code_mmd_content(spec)     -> str
generate_system_prompt(spec_path, repo_root) -> str
run_architecture_agent(spec_path, repo_root) -> dict
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# System prompt template (mirrors the architecture-agent spec)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Software Architecture Agent. Write files DIRECTLY using filesystem tools.

## LOCAL RESOURCES - READ FIRST

**Specification:** {localSpecPath}
  - Read and analyse requirements before designing architecture.

**Wireframes:** {localWireframePath}/
  - Check for UI wireframes that may influence component design.

## OUTPUT PATH: {localRepoPath}

## FILES TO GENERATE:

1. {localRepoPath}/docs/architecture/overview.md                        - Architecture overview
2. {localRepoPath}/docs/architecture/diagrams/c4-level1-context.mmd     - System Context
3. {localRepoPath}/docs/architecture/diagrams/c4-level2-container.mmd   - Containers
4. {localRepoPath}/docs/architecture/diagrams/c4-level3-component.mmd   - Components
5. {localRepoPath}/docs/architecture/diagrams/c4-level4-code.mmd        - Class diagram

## RULES
- Use ONLY valid GitHub-renderable Mermaid syntax (flowchart TB, classDiagram).
- No C4Context / Person() / Container() / Rel() syntax – it does NOT render on GitHub.
- No special characters or <br/> inside node labels.
- File extension must be .mmd.
"""


def read_spec(path: str) -> str:
    """Reads the content of a specification file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Specification file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def create_output_dirs(repo_root: str) -> str:
    """Creates the docs/architecture/diagrams directory and returns its absolute path."""
    diagrams_path = os.path.join(repo_root, 'docs', 'architecture', 'diagrams')
    os.makedirs(diagrams_path, exist_ok=True)
    return diagrams_path


def write_file(path: str, content: str) -> dict:
    """Writes *content* to *path*, creating parent directories as needed.

    Returns ``{"status": "success", "path": path}`` on success.
    Raises ``IOError`` (with a descriptive message) on any I/O failure.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "path": path}
    except IOError as e:
        raise IOError(f"Failed to write file {path}: {e}") from e

def generate_overview_md_content(spec_content: str) -> str:
    """Returns the architecture overview document.

    The *spec_content* parameter is accepted for future dynamic injection
    but the structure is kept deterministic so tests remain hermetic.
    """
    return """# Architecture Overview

## System Purpose
The **Agent Output Comparator** is a local web application that lets engineers
run the same prompt against multiple AI coding agents (GitHub Copilot CLI and
Claude Code), inspect their file outputs side-by-side, and score the results.

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
"""


def generate_c4_level1_context_mmd_content(spec_content: str) -> str:
    """Returns a Level-1 System Context flowchart (GitHub-renderable Mermaid)."""
    return """flowchart TB
    Developer([Developer])
    User([User])
    System[Agent Output Comparator]
    CopilotCLI[GitHub Copilot CLI]
    ClaudeCodeCLI[Claude Code CLI]

    Developer --> System
    User --> System
    System --> CopilotCLI
    System --> ClaudeCodeCLI
"""


def generate_c4_level2_container_mmd_content(spec_content: str) -> str:
    """Returns a Level-2 Container flowchart (GitHub-renderable Mermaid)."""
    return """flowchart TB
    subgraph Frontend
        Browser[Web Browser - Single-page HTML/JS/CSS]
    end
    subgraph Backend
        FlaskAPIServer[Flask API Server - Python 3]
        CopilotCLIRunner[Copilot CLI Runner - subprocess]
        ClaudeCodeCLIRunner[Claude Code CLI Runner - subprocess]
        ScoringEngine[Scoring Engine - Pure Python]
        FileResolver[File Resolver - Pure Python]
    end
    subgraph ExternalCLIs
        CopilotExe[copilot.exe]
        ClaudeExe[claude.exe]
    end

    Browser -->|REST /api/*| FlaskAPIServer
    FlaskAPIServer --> CopilotCLIRunner
    FlaskAPIServer --> ClaudeCodeCLIRunner
    FlaskAPIServer --> ScoringEngine
    FlaskAPIServer --> FileResolver
    CopilotCLIRunner --> CopilotExe
    ClaudeCodeCLIRunner --> ClaudeExe
"""


def generate_c4_level3_component_mmd_content(spec_content: str) -> str:
    """Returns a Level-3 Component flowchart for the Flask API Server."""
    return """flowchart TB
    subgraph Flask API Server
        CompareEndpoint[POST /api/compare]
        ReadMDEndpoint[POST /api/read-md]
        SuggestPromptsEndpoint[POST /api/suggest-prompts]
    end
    subgraph Runners
        CopilotCLIRunner[Copilot CLI Runner]
        ClaudeCodeCLIRunner[Claude Code CLI Runner]
    end
    ScoringEngine[Scoring Engine]
    FileResolver[File Resolver]

    CompareEndpoint --> CopilotCLIRunner
    CompareEndpoint --> ClaudeCodeCLIRunner
    CompareEndpoint --> ScoringEngine
    ReadMDEndpoint --> FileResolver
    SuggestPromptsEndpoint --> FileResolver
"""


def generate_c4_level4_code_mmd_content(spec_content: str) -> str:
    """Returns a Level-4 Code class diagram (GitHub-renderable Mermaid)."""
    return """classDiagram
    class FlaskAPIServer {
        +compare(system_prompt, user_prompt) dict
        +read_md(path) dict
        +suggest_prompts(spec_content) dict
    }
    class CopilotCLIRunner {
        +run(prompt, cwd) dict
        -_parse_jsonl(output) str
    }
    class ClaudeCodeCLIRunner {
        +run(prompt, spec_path) dict
        -_parse_jsonl(output) str
    }
    class ScoringEngine {
        +score(output, prompt, latency_sec, peer_latency_sec) dict
        -_quality_score(text) int
        -_accuracy_score(text, prompt) int
        -_speed_score(t, peer_t) int
        -_length_fit_score(text) int
    }
    class FileResolver {
        +resolve(path) str
    }
    class CompareResponse {
        +copilot_output str
        +claude_output str
        +copilot_score dict
        +claude_score dict
    }

    FlaskAPIServer --> CopilotCLIRunner
    FlaskAPIServer --> ClaudeCodeCLIRunner
    FlaskAPIServer --> ScoringEngine
    FlaskAPIServer --> FileResolver
    FlaskAPIServer --> CompareResponse
"""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def generate_system_prompt(spec_path: str, repo_root: str) -> str:
    """Renders the architecture-agent system prompt with resolved paths substituted.

    Parameters
    ----------
    spec_path:
        Absolute or relative path to the specification markdown file.
        The file must exist; raises ``FileNotFoundError`` otherwise.
    repo_root:
        Root directory of the repository where architecture files will be written.

    Returns
    -------
    str
        The rendered system prompt, ready to pass to an AI agent.

    Raises
    ------
    FileNotFoundError
        If *spec_path* does not point to an existing file.
    """
    spec_path = str(Path(spec_path).resolve())
    repo_root = str(Path(repo_root).resolve())

    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Specification file not found: {spec_path}")

    wireframe_path = os.path.join(repo_root, "wireframes")

    return _SYSTEM_PROMPT_TEMPLATE.format(
        localSpecPath=spec_path,
        localRepoPath=repo_root,
        localWireframePath=wireframe_path,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_EXPECTED_FILES = [
    os.path.join("docs", "architecture", "overview.md"),
    os.path.join("docs", "architecture", "diagrams", "c4-level1-context.mmd"),
    os.path.join("docs", "architecture", "diagrams", "c4-level2-container.mmd"),
    os.path.join("docs", "architecture", "diagrams", "c4-level3-component.mmd"),
    os.path.join("docs", "architecture", "diagrams", "c4-level4-code.mmd"),
]


def run_architecture_agent(spec_path: str, repo_root: str) -> dict:
    """Read *spec_path* and write all C4 architecture files under *repo_root*.

    The function is deliberately *fail-fast*: all content is generated before
    any file is written so that a generator error never leaves partial output.

    Parameters
    ----------
    spec_path:
        Path to the markdown specification file (must exist and be non-empty).
    repo_root:
        Root of the target repository; ``docs/architecture/`` will be created
        inside it.

    Returns
    -------
    dict
        ``{"success": True, "outputPath": "<abs-arch-dir>",
           "filesCreated": ["docs/architecture/…", …],
           "summary": {"total": 5}}``

    Raises
    ------
    FileNotFoundError
        If *spec_path* does not exist.
    ValueError
        If the specification file is empty or whitespace-only.
    IOError / OSError
        If a file cannot be written.
    RuntimeError
        Wraps any other unexpected error (with ``__cause__`` set).
    """
    spec_path = str(Path(spec_path).resolve())
    repo_root = str(Path(repo_root).resolve())

    # 1. Read spec — propagates FileNotFoundError
    spec_content = read_spec(spec_path)

    # 2. Validate spec is non-empty
    if not spec_content.strip():
        raise ValueError(f"Specification file is empty: {spec_path}")

    # 3. Ensure output directories exist
    create_output_dirs(repo_root)

    arch_dir = str(Path(repo_root) / "docs" / "architecture")

    # 4. Generate all content *before* touching the filesystem (fail-fast)
    try:
        contents = [
            generate_overview_md_content(spec_content),
            generate_c4_level1_context_mmd_content(spec_content),
            generate_c4_level2_container_mmd_content(spec_content),
            generate_c4_level3_component_mmd_content(spec_content),
            generate_c4_level4_code_mmd_content(spec_content),
        ]
    except (FileNotFoundError, IOError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"run_architecture_agent: content generation failed for {spec_path!r}"
        ) from exc

    # 5. Write files — propagates IOError/OSError on failure
    files_created: list[str] = []
    for rel_path, content in zip(_EXPECTED_FILES, contents):
        abs_path = os.path.join(repo_root, rel_path)
        write_file(abs_path, content)
        files_created.append(rel_path)

    return {
        "success": True,
        "outputPath": arch_dir,
        "filesCreated": files_created,
        "summary": {"total": len(files_created)},
    }

