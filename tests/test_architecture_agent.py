import os
import pytest
from unittest.mock import patch, mock_open

from source.architecture_agent import read_spec, create_output_dirs, write_file, \
    generate_overview_md_content, generate_c4_level1_context_mmd_content, \
    generate_c4_level2_container_mmd_content, generate_c4_level3_component_mmd_content, \
    generate_c4_level4_code_mmd_content, generate_system_prompt, run_architecture_agent

# Mock data for testing
MOCK_SPEC_CONTENT = """# Agent Output Comparator — Architecture Overview

## Purpose

The **Agent Output Comparator** is a local developer tool that runs identical prompts against two AI coding CLIs — **GitHub Copilot CLI** and **OpenCode CLI** — measures latency, and scores each output across four quality dimensions. Results are displayed side-by-side in a browser UI so developers can evaluate and compare agent behaviour objectively.

---

## System Components

| Component | Technology | Responsibility |
|---|---|---|
| Frontend UI | Single-page HTML / JS / CSS | Prompt entry, file upload, side-by-side diff, score charts |
| Flask API Server | Python 3, Flask, flask-cors | REST endpoints, orchestration, scoring |
| Copilot CLI Runner | `subprocess` (no shell) | Invokes `copilot.exe`, captures JSONL output |
| OpenCode CLI Runner | `subprocess` (shell, stdin redirect) | Invokes `opencode`, captures JSONL output |
| Scoring Engine | Pure Python | 4-dimension heuristic scoring (quality, accuracy, speed, length) |
| File Resolver | Pure Python | Resolves relative paths in agent output across common Windows locations |

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/api/compare` | Run prompt against both CLIs, return scored results |
| POST | `/api/read-md` | Read an agent `.md` spec file from disk |
| POST | `/api/suggest-prompts` | Analyse spec content and return 5 contextual test prompts |

### `/api/compare` request body
```json
{
  "system_prompt": "string",
  "user_prompt": "string",
  "local_spec_path": "optional/path/to/spec.md",
  "copilot_cwd": "optional/working/directory"
}
```

---

## Scoring Dimensions

Each CLI output is scored 0–100 on four axes; the overall score is a weighted average:

| Dimension | Weight | Description |
|---|---|---|
| Quality | 35% | Length, structure (lists/code/headers), completeness, no errors |
| Accuracy | 35% | Keyword overlap between prompt keywords and output |
| Speed | 15% | Relative latency vs the peer CLI |
| Length Fit | 15% | Penalises very short (<30 words) and very long (>700 words) responses |

---

## Data Flow

```
Browser
  │
  ├─ POST /api/compare
  │       │
  │       ├─ run_copilot_cli()  →  copilot.exe  (temp file + subprocess, no shell)
  │       │        └─ parse JSONL  →  extract assistant.message
  │       │
  │       ├─ run_opencode_cli() →  opencode.cmd (shell + stdin redirect)
  │       │        └─ parse JSONL  →  extract type=text parts
  │       │
  │       └─ score_output() × 2  →  quality / accuracy / speed / length
  │
  └─ Response JSON → side-by-side display + radar charts
```

---

## Design Decisions

- **No shell quoting issues:** Copilot prompts are written to a temp file and passed via `-p` flag as a list argument (`shell=False`). OpenCode prompts are written to a temp file and piped via stdin redirect.
- **JSONL output parsing:** Both CLIs emit JSONL. The server extracts only `assistant.message` (Copilot) and `type=text` (OpenCode) events; all tool-use chatter and stats are discarded.
- **No database:** All state is ephemeral. Each comparison request is self-contained.
- **CORS enabled globally:** Allows the UI (`index.html` opened from file system or a different port) to call the Flask server.
- **Configurable via env vars:** `COMPARE_COPILOT_CWD`, `COMPARE_COPILOT_TIMEOUT_SEC`, `COMPARE_OPENCODE_SPEC` override defaults without code changes.

---

## Diagram Index

| File | Level | Description |
|---|---|---|
| [c4-level1-context.mmd](diagrams/c4-level1-context.mmd) | L1 | System context — actors and external systems |
| [c4-level2-container.mmd](diagrams/c4-level2-container.mmd) | L2 | Containers — frontend, backend, external CLIs |
| [c4-level3-component.mmd](diagrams/c4-level3-component.mmd) | L3 | Components inside the Flask backend |
| [c4-level4-code.mmd](diagrams/c4-level4-code.mmd) | L4 | Key data structures and class relationships |
"""

@pytest.fixture
def setup_temp_file(tmp_path):
    test_file = tmp_path / "test_spec.md"
    test_file.write_text(MOCK_SPEC_CONTENT, encoding='utf-8')
    return test_file

def test_read_spec_success(setup_temp_file):
    content = read_spec(str(setup_temp_file))
    assert content == MOCK_SPEC_CONTENT

def test_read_spec_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_spec("non_existent_file.md")

def test_create_output_dirs(tmp_path):
    repo_root = str(tmp_path)
    diagrams_path = create_output_dirs(repo_root)
    expected_path = os.path.join(repo_root, 'docs', 'architecture', 'diagrams')
    assert diagrams_path == expected_path
    assert os.path.exists(diagrams_path)

def test_write_file_success(tmp_path):
    test_file_path = os.path.join(str(tmp_path), "output.txt")
    content = "Hello, world!"
    result = write_file(test_file_path, content)
    assert result["status"] == "success"
    assert result["path"] == test_file_path
    assert os.path.exists(test_file_path)
    with open(test_file_path, 'r') as f:
        assert f.read() == content

def test_write_file_failure(tmp_path):
    test_file_path = os.path.join(str(tmp_path), "fail_output.txt")
    content = "This should fail."
    with patch('os.makedirs', side_effect=IOError("Disk full")):
        with pytest.raises(IOError, match="Failed to write file"):
            write_file(test_file_path, content)



def test_generate_c4_level1_context_mmd_content():
    content = generate_c4_level1_context_mmd_content(MOCK_SPEC_CONTENT)
    assert isinstance(content, str)
    assert "flowchart TB" in content
    assert "System[Agent Output Comparator]" in content

def test_generate_c4_level2_container_mmd_content():
    content = generate_c4_level2_container_mmd_content(MOCK_SPEC_CONTENT)
    assert isinstance(content, str)
    assert "flowchart TB" in content
    assert "subgraph Frontend" in content
    assert "Flask API Server" in content

def test_generate_c4_level3_component_mmd_content():
    content = generate_c4_level3_component_mmd_content(MOCK_SPEC_CONTENT)
    assert isinstance(content, str)
    assert "flowchart TB" in content
    assert "subgraph Flask API Server" in content
    assert "CompareEndpoint[POST /api/compare]" in content

def test_generate_c4_level4_code_mmd_content():
    content = generate_c4_level4_code_mmd_content(MOCK_SPEC_CONTENT)
    assert isinstance(content, str)
    assert "classDiagram" in content
    assert "class FlaskAPIServer" in content
