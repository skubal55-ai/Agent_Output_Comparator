"""
Agent Output Comparator - Flask Backend
Runs prompts against GitHub Copilot CLI and Claude Code CLI,
measures latency, and scores outputs across 4 metrics.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from source.evalkit.experiment.config import ExperimentConfig
from source.evalkit.experiment.runner import run_experiment
from source.evalkit.experiment.storage import ResultStore
from source.evalkit.metrics.aggregate import default_scorer
from source.evalkit.reporting.export import experiment_to_csv, experiment_to_json
from source.evalkit.stats.aggregate_stats import summarize
from source.evalkit.stats.significance import all_pairs_significance

# Serve index.html and static assets from the same folder as server.py
BASE_DIR = Path(__file__).parent.resolve()
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
# Only the comparator's own page may call the API cross-origin. An open CORS policy would let
# any website open in the same browser read local files via /api/read-file or start agent runs.
_PORT = int(os.environ.get("PORT", "5050"))
CORS(app, origins=[f"http://localhost:{_PORT}", f"http://127.0.0.1:{_PORT}"])

EMPTY_ANSWER_ERROR = "Empty response: the CLI finished without returning any answer text."


@app.route("/")
def index():
    """Serve the frontend at http://localhost:5050"""
    return send_from_directory(str(BASE_DIR), "index.html")

# ---------------------------------------------------------------------------
# CLI invocation helpers
# ---------------------------------------------------------------------------

def _copilot_effective_cwd(copilot_cwd_request: str | None = None) -> str:
    """
    Working directory for the Copilot CLI process (write/read tools use this tree).

    Precedence:
      1. ``copilot_cwd_request`` from the API (optional)
      2. Env ``COMPARE_COPILOT_CWD`` (e.g. ``C:\\Users\\you\\source``)
      3. This app's directory (predictable for the comparator UI / path resolution)
      4. ``USERPROFILE`` / home / process cwd
    """
    for cand in ((copilot_cwd_request or "").strip(), os.environ.get("COMPARE_COPILOT_CWD", "").strip()):
        if cand and os.path.isdir(cand):
            return str(Path(cand).resolve())
    app = _app_directory()
    if app.is_dir():
        return str(app.resolve())
    for key in ("USERPROFILE", "HOME"):
        v = os.environ.get(key)
        if v and os.path.isdir(v):
            return v
    home = str(Path.home())
    return home if os.path.isdir(home) else os.getcwd()


def _subprocess_timeout_sec(env_var: str = "COMPARE_COPILOT_TIMEOUT_SEC") -> int:
    """
    Max wall-clock time for one agent CLI run.
    Agent + tools often exceeds 120s; override with ``env_var``
    (COMPARE_COPILOT_TIMEOUT_SEC / COMPARE_CLAUDE_TIMEOUT_SEC, 30–3600).
    """
    raw = os.environ.get(env_var, "").strip()
    default = 600
    if not raw:
        return default
    try:
        sec = int(raw)
    except ValueError:
        return default
    return max(30, min(sec, 3600))


_COPILOT_FILEGEN_HINT_RE = re.compile(
    r"(?i)\b(write|create|generate|implement|save|scaffold|\.md|\.py|\.mmd|file|files|"
    r"directory|folder|pytest|tests?)\b"
)


def _combined_agent_prompt(system_prompt: str, user_prompt: str) -> str:
    """The exact prompt text every agent receives, so comparisons stay like-for-like."""
    combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    if _COPILOT_FILEGEN_HINT_RE.search(user_prompt) or (
        system_prompt and _COPILOT_FILEGEN_HINT_RE.search(system_prompt)
    ):
        combined += (
            "\n\n---\n[Comparator]\n"
            "If this task requires artifacts on disk, use your file tools to write them "
            "under the CLI working directory. Do not only describe paths or file contents "
            "in prose—create the actual files so they can be opened from disk.\n"
        )
    return combined


def _run_copilot_with_file(exe: str, prompt_file: str, cwd_override: str | None = None) -> dict:
    """
    Run copilot.exe with the prompt read from a file, avoiding all shell quoting issues.
    Strategy: use subprocess directly (no shell) — pass exe and args as a list,
    and feed the prompt via stdin so no quoting is needed at all.
    """
    import tempfile
    start = time.perf_counter()
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env["TERM"] = "dumb"
    env.setdefault("CI", "true")
    cli_cwd = _copilot_effective_cwd(cwd_override)
    timeout_sec = _subprocess_timeout_sec("COMPARE_COPILOT_TIMEOUT_SEC")

    try:
        with open(prompt_file, "r", encoding="utf-8") as fh:
            prompt_text = fh.read()

        # Programmatic flags (see GitHub Copilot CLI programmatic reference):
        #   --allow-all       tools + paths + urls (avoids permission stalls vs --allow-all-tools alone)
        #   --no-ask-user     do not pause for follow-up input (common cause of "hangs" without a TTY)
        # stdin=DEVNULL       without a TTY some CLIs wait on stdin indefinitely
        proc = subprocess.run(
            [
                exe,
                "-p",
                prompt_text,
                "--allow-all",
                "--no-ask-user",
                "--no-color",
                "--output-format",
                "json",
            ],
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cli_cwd,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0 and not stdout:
            return {
                "tool": "Copilot CLI",
                "output": "",
                "error": stderr or f"Exit code {proc.returncode}",
                "latency_ms": int(elapsed * 1000),
                "working_directory": cli_cwd,
            }
        return {
            "tool": "Copilot CLI",
            "output": stdout or stderr,
            "error": None,
            "latency_ms": int(elapsed * 1000),
            "working_directory": cli_cwd,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "tool": "Copilot CLI",
            "output": "",
            "error": f"Timed out after {timeout_sec} seconds. "
            f"Set COMPARE_COPILOT_TIMEOUT_SEC (e.g. 900) to allow longer runs.",
            "latency_ms": int(elapsed * 1000),
            "working_directory": cli_cwd,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "tool": "Copilot CLI",
            "output": "",
            "error": str(exc),
            "latency_ms": int(elapsed * 1000),
            "working_directory": cli_cwd,
        }


def run_copilot_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    copilot_cwd: str | None = None,
) -> dict:
    """
    Invoke GitHub Copilot CLI binary directly via shell=True.
    Uses a temp file to avoid Windows shell quoting issues with
    prompts that contain quotes, newlines, or special characters.
    """
    import tempfile
    combined = _combined_agent_prompt(system_prompt, user_prompt)

    # Write prompt to temp file — avoids shell quoting issues
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      delete=False, encoding="utf-8")
    tmp.write(combined)
    tmp.close()
    tmp_path = tmp.name

    binary_path = _find_copilot_binary()
    exe = binary_path if binary_path else "copilot"

    # Pass prompt via -s (system/session file) flag — copilot reads the file directly,
    # no shell quoting or word-splitting involved at all.
    # Fallback: if -s is not supported, use PowerShell with quoted variable.
    result = _run_copilot_with_file(exe, tmp_path, cwd_override=copilot_cwd)

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if result.get("output"):
        raw = result["output"]
        result["output"] = _clean_copilot_json_output(raw)
        # Exit code 0 with only an error event on stdout (e.g. auth failure) is a failed
        # run, not an answer — surface it as an error so it isn't scored.
        err = _jsonl_error_without_answer(raw)
        if err:
            result["output"] = ""
            result["error"] = err
    return result


def _jsonl_error_without_answer(text: str) -> str | None:
    """
    Return the error message from a JSONL agent log that contains an error event
    (``type`` == "error" or ending in ".error") but no ``assistant.message`` answer.
    """
    error_msg = None
    for line in text.splitlines():
        try:
            obj = json.loads(line.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        etype = str(obj.get("type", ""))
        if etype == "assistant.message":
            content = (obj.get("data") or {}).get("content")
            if isinstance(content, str) and content.strip():
                return None
        elif etype == "error" or etype.endswith(".error"):
            err = obj.get("error") or obj.get("data") or obj.get("message")
            if isinstance(err, dict):
                err = err.get("message") or (err.get("data") or {}).get("message") or json.dumps(err)
            error_msg = str(err or etype)
    return error_msg


def _find_copilot_binary() -> str:
    """
    Locate the copilot.exe binary by checking known paths in order.
    Returns the full path string if found, else empty string.
    """
    import os
    user_profile = os.environ.get("USERPROFILE", "")
    local_app_data = os.environ.get("LOCALAPPDATA", "")

    candidates = [
        # Confirmed location on this machine (node_modules)
        os.path.join(user_profile, "node_modules", "@github", "copilot-win32-x64", "copilot.exe"),
        # Common gh download location
        os.path.join(local_app_data, "GitHub CLI", "copilot.exe"),
        # Global node_modules
        os.path.join(os.environ.get("APPDATA", ""), "npm", "node_modules", "@github", "copilot-win32-x64", "copilot.exe"),
        # ARM64 variant
        os.path.join(user_profile, "node_modules", "@github", "copilot-win32-arm64", "copilot.exe"),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path
    # Fall back to PATH, which is also what a run uses when none of the above exist.
    import shutil
    return shutil.which("copilot") or ""


def _clean_copilot_output(text: str) -> str:
    """
    Clean copilot binary output. Strips usage/token stats and
    box-drawing separators, keeps the actual response content.

    Example raw output:
      ● List directory .
        └ 76 files found
      Here are the files...
      Total usage est:   1 Premium request
      API time spent:    8s
      ...
    """
    lines = text.splitlines()
    cleaned = []
    skip_prefixes = [
        "total usage est",
        "api time spent",
        "total session time",
        "total code changes",
        "breakdown by ai model",
        "claude-",
        "gpt-",
        "just copy/paste",
        "if you want accepted",
        "consider installing",
        "https://www.npmjs",
    ]
    for line in lines:
        stripped = line.strip()
        # Skip blank lines at start
        if not stripped and not cleaned:
            continue
        # Skip usage/stats lines
        if any(stripped.lower().startswith(p) for p in skip_prefixes):
            continue
        # Skip pure box-drawing separator lines
        box_chars = sum(1 for c in stripped if c in '─━═–—\u2500\u2501\u2550')
        if stripped and len(stripped) > 3 and box_chars / len(stripped) > 0.5:
            continue
        cleaned.append(line)

    # Strip trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned).strip()


def _clean_copilot_json_output(text: str) -> str:
    """
    Parse Copilot --output-format json JSONL output.

    Key event types (confirmed from real output):
      - "assistant.message"       → data.content has the FULL final answer (use this)
      - "assistant.message_delta" → data.deltaContent has streaming chunks (ignore)
      - "assistant.reasoning"     → internal reasoning (ignore)
      - "result"                  → session stats (ignore)

    We pick the last "assistant.message" with non-empty content as the answer.
    Falls back to plain text cleaner if not valid JSONL.
    """
    import json
    lines = text.strip().splitlines()
    final_answer = ""
    is_jsonl = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            is_jsonl = True
            if not isinstance(obj, dict):
                continue
            event_type = obj.get("type", "")

            # Full assistant message — data.content has the complete response
            if event_type == "assistant.message":
                data = obj.get("data", {})
                content = data.get("content", "")
                if isinstance(content, str) and content.strip():
                    final_answer = content.strip()  # Keep updating — use the last one

        except (json.JSONDecodeError, ValueError):
            pass

    if is_jsonl and final_answer:
        return final_answer

    # Not JSONL or no content extracted — fall back to plain cleaner
    return _clean_copilot_output(text)



def _claude_permission_mode() -> str:
    """
    Claude Code ``--permission-mode`` for comparator runs.
    Default ``acceptEdits`` lets it write files in the working folder (parity with
    Copilot's file tools) without allowing unattended shell commands.
    Override with COMPARE_CLAUDE_PERMISSION_MODE (e.g. ``bypassPermissions``).
    """
    allowed = {"acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"}
    mode = os.environ.get("COMPARE_CLAUDE_PERMISSION_MODE", "").strip()
    return mode if mode in allowed else "acceptEdits"


def run_claude_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    cwd: str | None = None,
) -> dict:
    """
    Invoke Claude Code CLI in print mode: ``claude -p --output-format json``.
    The prompt is fed via stdin (no shell quoting, no command-line length limit)
    and is built exactly like the Copilot prompt so both agents get the same input.
    """
    start = time.perf_counter()
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env["TERM"] = "dumb"
    # Claude Code refuses to start when it thinks it is nested inside another
    # Claude Code session (e.g. when this server was launched from one).
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    cli_cwd = _copilot_effective_cwd(cwd)
    timeout_sec = _subprocess_timeout_sec("COMPARE_CLAUDE_TIMEOUT_SEC")
    exe = _find_claude_binary() or "claude"
    cmd = [
        exe,
        "-p",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--permission-mode",
        _claude_permission_mode(),
    ]
    model = os.environ.get("COMPARE_CLAUDE_MODEL", "").strip()
    if model:
        cmd += ["--model", model]

    base = {"tool": "Claude Code CLI", "working_directory": cli_cwd}
    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            input=_combined_agent_prompt(system_prompt, user_prompt),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cli_cwd,
        )
    except FileNotFoundError:
        return {**base, "output": "", "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": "'claude' not found. Install Claude Code CLI and make sure it is on PATH."}
    except subprocess.TimeoutExpired:
        return {**base, "output": "", "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": f"Timed out after {timeout_sec} seconds. "
                         f"Set COMPARE_CLAUDE_TIMEOUT_SEC (e.g. 900) to allow longer runs."}
    except Exception as exc:
        return {**base, "output": "", "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": str(exc)}

    latency_ms = int((time.perf_counter() - start) * 1000)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        payload = None

    if not isinstance(payload, dict):
        # Not the expected single JSON result — treat as plain text / failure.
        if proc.returncode != 0 or not stdout:
            return {**base, "output": "", "latency_ms": latency_ms,
                    "error": stderr or stdout or f"Exit code {proc.returncode}"}
        return {**base, "output": stdout, "error": None, "latency_ms": latency_ms}

    text = payload.get("result")
    text = text.strip() if isinstance(text, str) else ""
    meta = {
        "model": ", ".join((payload.get("modelUsage") or {}).keys()) or None,
        "num_turns": payload.get("num_turns"),
        "total_cost_usd": payload.get("total_cost_usd"),
    }
    if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
        # e.g. {"is_error": true, "result": "Not logged in · Please run /login"}
        return {**base, **meta, "output": "", "latency_ms": latency_ms,
                "error": text or str(payload.get("subtype") or "Claude Code reported an error")}
    return {**base, **meta, "output": text, "error": None, "latency_ms": latency_ms}


def _find_claude_binary() -> str:
    """
    Resolve the Claude Code CLI executable (PATH first, then common install locations).
    Returns an empty string when it cannot be found.
    """
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    user_profile = os.environ.get("USERPROFILE", "")
    app_data = os.environ.get("APPDATA", "")
    candidates = [
        os.path.join(user_profile, ".local", "bin", "claude.exe"),
        os.path.join(app_data, "npm", "claude.cmd"),
        os.path.join(user_profile, "node_modules", ".bin", "claude.cmd"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _run_cli(name: str, cmd: list) -> dict:
    """Run a CLI command and return timing + output."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0 and not stdout:
            return {
                "tool": name,
                "output": "",
                "error": stderr or f"Exit code {proc.returncode}",
                "latency_ms": int(elapsed * 1000),
            }

        return {
            "tool": name,
            "output": stdout or stderr,
            "error": None,
            "latency_ms": int(elapsed * 1000),
        }

    except FileNotFoundError:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": f"'{cmd[0]}' not found. Is it installed and on your PATH?",
            "latency_ms": int(elapsed * 1000),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": "Timed out after 120 seconds.",
            "latency_ms": int(elapsed * 1000),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": str(exc),
            "latency_ms": int(elapsed * 1000),
        }


def _run_shell(
    name: str,
    cmd: str,
    powershell: bool = False,
    timeout_sec: int = 90,
) -> dict:
    """
    Run a command string via shell.
    powershell=True: runs via `powershell -Command` (needed for & operator, Get-Content etc.)
    powershell=False: runs via cmd.exe shell=True (for .cmd wrappers, stdin redirect etc.)
    Uses UTF-8 encoding to avoid cp1252 errors.
    """
    start = time.perf_counter()
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env["TERM"] = "dumb"

    if powershell:
        actual_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
        use_shell = False
    else:
        actual_cmd = cmd
        use_shell = True

    try:
        proc = subprocess.run(
            actual_cmd,
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0 and not stdout:
            return {
                "tool": name,
                "output": "",
                "error": stderr or f"Exit code {proc.returncode}",
                "latency_ms": int(elapsed * 1000),
            }
        return {
            "tool": name,
            "output": stdout or stderr,
            "error": None,
            "latency_ms": int(elapsed * 1000),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": f"Timed out after {timeout_sec} seconds.",
            "latency_ms": int(elapsed * 1000),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": str(exc),
            "latency_ms": int(elapsed * 1000),
        }


def _run_cli_input(name: str, cmd: list) -> dict:
    """
    Run a CLI command without stdin (stdin=DEVNULL).
    Uses utf-8 encoding explicitly to avoid Windows cp1252 errors.
    Sets NO_COLOR and TERM=dumb to suppress ANSI escape codes.
    Does NOT pipe input="" as that causes some CLIs to think no message was given.
    """
    start = time.perf_counter()
    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,   # Close stdin cleanly without sending data
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            env=env,
            encoding="utf-8",           # Force UTF-8 to avoid Windows cp1252 errors
            errors="replace",           # Replace undecodable chars instead of crashing
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout.strip() if proc.stdout else ""
        stderr = proc.stderr.strip() if proc.stderr else ""

        if proc.returncode != 0 and not stdout:
            return {
                "tool": name,
                "output": "",
                "error": stderr or f"Exit code {proc.returncode}",
                "latency_ms": int(elapsed * 1000),
            }
        return {
            "tool": name,
            "output": stdout or stderr,
            "error": None,
            "latency_ms": int(elapsed * 1000),
        }
    except FileNotFoundError:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": f"'{cmd[0]}' not found. Is it installed and on your PATH?",
            "latency_ms": int(elapsed * 1000),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": "Timed out after 90 seconds.",
            "latency_ms": int(elapsed * 1000),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "tool": name,
            "output": "",
            "error": str(exc),
            "latency_ms": int(elapsed * 1000),
        }


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
#
# Scoring is delegated to evalkit.metrics.AggregateScorer (see
# source/evalkit/metrics/ and docs/methodology.md for metric definitions,
# weighting rationale, and validation methodology). score_for_ui() adapts
# the framework's namespaced metric output into the flat
# {quality, accuracy, speed, length, overall, word_count} shape the existing
# UI renders, so the frontend didn't need to change.

_UI_SCORER = default_scorer()


def score_for_ui(
    output: str,
    prompt: str,
    system_prompt: str,
    latency_ms: int,
    peer_latency_ms: int,
    error: str | None = None,
) -> dict:
    """``error`` set → the run failed; every score is 0 and ``failed`` is True.
    Pass ``peer_latency_ms=0`` when the peer failed so its latency isn't used."""
    if error:
        result = _UI_SCORER.failed_result(error)
    else:
        result = _UI_SCORER.score(
            output=output,
            prompt=prompt,
            system_prompt=system_prompt,
            latency_ms=latency_ms,
            peer_latencies_ms=[peer_latency_ms] if peer_latency_ms else [],
        )
    return {
        "failed": result["failed"],
        "quality": round(result["quality"]["score"]),
        "accuracy": round(result["semantic_accuracy"]["score"]),
        "speed": round(result["efficiency"]["score"]),
        "length": round(result["length_fit"]["score"]),
        "word_count": result["word_count"],
        "overall": round(result["overall"]),
        "metrics_detail": result,
    }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/read-md", methods=["POST"])
def read_md():
    """Read and return contents of the agent .md file via file path."""
    data = request.json or {}
    file_path = data.get("path", "").strip()

    if not file_path:
        return jsonify({"error": "No file path provided."}), 400

    path = Path(file_path).expanduser()
    if not path.exists():
        return jsonify({"error": f"File not found: {file_path}"}), 404
    if not path.suffix.lower() == ".md":
        return jsonify({"error": "File must be a .md (Markdown) file."}), 400

    try:
        content = path.read_text(encoding="utf-8")
        return jsonify({"content": content, "filename": path.name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/suggest-prompts", methods=["POST"])
def suggest_prompts():
    """
    Analyse the agent .md content and return 5 relevant test prompts.
    Uses keyword/section extraction — no external LLM call needed.
    Body: { "content": str, "filename": str }
    """
    data = request.json or {}
    content = data.get("content", "").strip()
    filename = data.get("filename", "agent.md")

    if not content:
        return jsonify({"error": "No content provided."}), 400

    prompts = _generate_prompts_from_md(content, filename)
    return jsonify({"prompts": prompts})


def _generate_prompts_from_md(content: str, filename: str) -> list:
    """
    Parse the agent .md and generate contextual test prompts by:
    1. Extracting headings, capabilities, examples, and key domain nouns
    2. Mapping them to prompt templates suited for that agent type
    """
    lines = content.splitlines()
    headings, bullet_points, code_blocks, examples = [], [], [], []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_blocks.append(stripped)
            continue
        if stripped.startswith("#"):
            clean = re.sub(r"^#+\s*", "", stripped)
            if len(clean) > 3:
                headings.append(clean)
        elif re.match(r"^[-*•]\s+", stripped):
            clean = re.sub(r"^[-*•]\s+", "", stripped)
            if len(clean) > 10:
                bullet_points.append(clean)
        elif re.search(r"\bexample\b|\be\.g\b|\bsample\b", stripped, re.I):
            examples.append(stripped)

    # Detect agent domain from filename + headings + bullets
    all_text = content.lower()
    domain_hints = {
        "architecture": ["architect", "system design", "microservice", "scalab", "infrastructure", "pattern", "deployment"],
        "coding":       ["code", "function", "debug", "refactor", "implement", "class", "algorithm", "test"],
        "data":         ["data", "sql", "query", "pipeline", "etl", "database", "analytics", "schema"],
        "devops":       ["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "cloud", "aws", "terraform"],
        "security":     ["security", "auth", "vulnerab", "encrypt", "token", "permission", "rbac"],
        "product":      ["product", "feature", "roadmap", "user stor", "requirement", "stakeholder", "priorit"],
        "writing":      ["write", "content", "document", "draft", "edit", "blog", "summarize"],
        "research":     ["research", "analys", "investigat", "find", "report", "summarize", "compare"],
    }

    detected_domain = "general"
    best_count = 0
    for domain, keywords in domain_hints.items():
        count = sum(1 for kw in keywords if kw in all_text)
        if count > best_count:
            best_count = count
            detected_domain = domain

    # Domain-specific prompt templates
    templates = {
        "architecture": [
            "Design a scalable {topic} system architecture for 1 million users. Include service boundaries, data flow, and storage strategy.",
            "What architectural patterns would you recommend for a {topic} platform? Compare monolith vs microservices.",
            "How would you handle fault tolerance and high availability in a {topic} system?",
            "Design the authentication and authorization layer for a multi-tenant {topic} application.",
            "What are the trade-offs between event-driven and request-response architecture for {topic}?",
        ],
        "coding": [
            "Write a {topic} function with proper error handling and unit tests.",
            "Refactor this {topic} code for better readability, performance, and maintainability.",
            "Debug and fix common issues in a {topic} implementation.",
            "Implement a {topic} algorithm and explain the time and space complexity.",
            "What are best practices for structuring a {topic} codebase?",
        ],
        "data": [
            "Write an optimized SQL query to {topic} with proper indexing strategy.",
            "Design a data pipeline for {topic} that handles schema evolution and failures.",
            "How would you model the {topic} schema for a high-read, low-write workload?",
            "Explain the ETL process for {topic} and how to handle data quality issues.",
            "Compare different storage solutions for {topic} workloads.",
        ],
        "devops": [
            "Write a CI/CD pipeline for {topic} using GitHub Actions.",
            "How would you containerize and deploy a {topic} application on Kubernetes?",
            "Design a monitoring and alerting strategy for a {topic} service.",
            "What are best practices for {topic} infrastructure as code using Terraform?",
            "How would you handle zero-downtime deployment for {topic}?",
        ],
        "security": [
            "What are the top security vulnerabilities to check for in a {topic} system?",
            "Design a secure authentication flow for {topic} using OAuth2 and JWT.",
            "How would you implement role-based access control for {topic}?",
            "Audit this {topic} implementation for security risks and suggest fixes.",
            "What encryption and data-at-rest strategies should be used for {topic}?",
        ],
        "product": [
            "Write user stories for the {topic} feature with acceptance criteria.",
            "Prioritize these {topic} requirements using the MoSCoW method.",
            "Create a product roadmap for {topic} for the next two quarters.",
            "What metrics would you track to measure success of the {topic} feature?",
            "How would you handle conflicting stakeholder requirements for {topic}?",
        ],
        "writing": [
            "Write a clear and concise {topic} document for a technical audience.",
            "Summarize the key points of {topic} in under 200 words.",
            "Rewrite this {topic} section to be more engaging and structured.",
            "Create a {topic} template that covers all essential sections.",
            "Edit and improve the clarity of this {topic} draft.",
        ],
        "general": [
            "Explain how {topic} works and when you would use it.",
            "What are the best practices for {topic}? Give concrete examples.",
            "Compare different approaches to {topic} and recommend the best one.",
            "What are common mistakes when working with {topic} and how to avoid them?",
            "Give a step-by-step guide for implementing {topic} from scratch.",
        ],
    }

    # Pick the best topic noun from headings > bullets > filename
    topic_candidates = headings[:3] + [b[:40] for b in bullet_points[:3]]
    if not topic_candidates:
        topic_candidates = [re.sub(r"[-_]", " ", filename.replace(".md", ""))]

    # Build 5 prompts by cycling through templates and topic candidates
    selected_templates = templates.get(detected_domain, templates["general"])
    generated = []
    for i, tmpl in enumerate(selected_templates[:5]):
        topic = topic_candidates[i % len(topic_candidates)] if topic_candidates else "the system"
        # Clean up topic (strip markdown noise)
        topic = re.sub(r"[`*_#]", "", topic).strip().lower()
        prompt = tmpl.replace("{topic}", topic)
        generated.append(prompt)

    # If we found real example lines in the doc, replace last prompt with one derived from them
    if examples:
        ex = re.sub(r"[`*_#]", "", examples[0]).strip()
        if len(ex) > 20:
            generated[-1] = f"Based on this example: \"{ex[:120]}\" — explain the concept and provide an alternative approach."

    return generated


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Run a prompt against both CLIs and return scored comparison.
    Body: {
      "system_prompt": str, "user_prompt": str,
      "copilot_cwd" | "copilot_working_directory": optional folder where both agents'
          file tools write (same folder for both so the comparison is fair)
    }
    """
    data = request.json or {}
    system_prompt = data.get("system_prompt", "").strip()
    user_prompt = data.get("user_prompt", "").strip()

    cwd_req = data.get("copilot_cwd") or data.get("copilot_working_directory") or ""
    cwd_req = cwd_req.strip() if isinstance(cwd_req, str) else ""

    if not user_prompt:
        return jsonify({"error": "user_prompt is required."}), 400

    # Run both CLIs (sequentially to avoid interfering with each other's timing)
    copilot_result = run_copilot_cli(
        system_prompt, user_prompt, copilot_cwd=cwd_req or None
    )
    claude_result = run_claude_cli(system_prompt, user_prompt, cwd=cwd_req or None)
    # A run that returns no answer text is a failure, not a (very short) answer.
    for result in (copilot_result, claude_result):
        if not result.get("error") and not (result.get("output") or "").strip():
            result["error"] = EMPTY_ANSWER_ERROR

    # Score outputs. A failed run scores 0 and is not used as the other's latency peer.
    def peer_latency(peer: dict) -> int:
        return 0 if peer.get("error") else peer["latency_ms"]

    copilot_scores = score_for_ui(
        copilot_result["output"],
        user_prompt,
        system_prompt,
        copilot_result["latency_ms"],
        peer_latency(claude_result),
        error=copilot_result.get("error"),
    )
    claude_scores = score_for_ui(
        claude_result["output"],
        user_prompt,
        system_prompt,
        claude_result["latency_ms"],
        peer_latency(copilot_result),
        error=claude_result.get("error"),
    )

    # Extract any file paths mentioned or created in the output
    def files_for(result: dict) -> list:
        wd = result.get("working_directory")
        return _extract_file_paths(
            result.get("output", ""),
            cli_working_dir=wd if isinstance(wd, str) and wd.strip() else None,
        )

    return jsonify({
        "prompt": user_prompt,
        "copilot": {**copilot_result, "scores": copilot_scores, "files": files_for(copilot_result)},
        "claude":  {**claude_result,  "scores": claude_scores,  "files": files_for(claude_result)},
    })


# ---------------------------------------------------------------------------
# Experiment harness API (evalkit) — multi-trial, statistically-backed runs
# ---------------------------------------------------------------------------

def _experiment_aggregates(experiment: dict) -> dict:
    """Per (agent, metric) mean/stdev/95% CI, plus pairwise significance on
    'overall' scores (Wilcoxon by default) when >=2 agents ran, with
    Holm-Bonferroni correction applied once there are more than two agents
    (see docs/methodology.md Section 6/8 on why the correction matters).
    """
    by_agent_metric: dict[str, dict[str, list[float]]] = {}
    for trial in experiment["trials"]:
        agent = trial["agent"]
        by_agent_metric.setdefault(agent, {})
        for metric_key, metric_val in trial["scores"].items():
            if metric_key == "overall":
                values = by_agent_metric[agent].setdefault("overall", [])
                values.append(metric_val)
            elif isinstance(metric_val, dict) and "score" in metric_val:
                values = by_agent_metric[agent].setdefault(metric_key, [])
                values.append(metric_val["score"])

    aggregates = {
        # Every metric score (and overall) is on a 0-100 scale; keep the CI inside it.
        agent: {metric: summarize(values, bounds=(0.0, 100.0)) for metric, values in metrics.items()}
        for agent, metrics in by_agent_metric.items()
    }

    significance = None
    overall_by_agent = {
        agent: metrics["overall"] for agent, metrics in by_agent_metric.items() if "overall" in metrics
    }
    lengths = {len(v) for v in overall_by_agent.values()}
    if len(overall_by_agent) >= 2 and len(lengths) == 1 and next(iter(lengths)) >= 2:
        try:
            significance = all_pairs_significance(overall_by_agent)
        except (ImportError, ValueError) as exc:
            significance = {"error": str(exc)}

    return {"aggregates": aggregates, "significance": significance}


@app.route("/api/experiments", methods=["POST"])
def create_experiment():
    """
    Run a config-driven multi-trial experiment: agents x prompts x n_trials.
    Body (ExperimentConfig-shaped):
    {
      "name": str, "agents": ["copilot", "claude"],
      "prompts": [{"id": str, "system_prompt": str, "user_prompt": str}, ...],
      "n_trials": int, "agent_kwargs": {"copilot": {...}, "claude": {...}}
    }
    """
    data = request.json or {}
    try:
        config = ExperimentConfig.from_dict(data)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        experiment_id = run_experiment(config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"experiment_id": experiment_id})


@app.route("/api/experiments/<experiment_id>", methods=["GET"])
def get_experiment(experiment_id: str):
    """Aggregated stats (mean/stdev/95% CI per agent+metric) and, when
    exactly two agents were compared, a paired significance test."""
    store = ResultStore()
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        return jsonify({"error": f"Experiment not found: {experiment_id}"}), 404

    return jsonify({**experiment, **_experiment_aggregates(experiment)})


@app.route("/api/experiments/<experiment_id>/export", methods=["GET"])
def export_experiment(experiment_id: str):
    """Export raw trial rows as CSV or JSON for external analysis. ?format=csv|json"""
    store = ResultStore()
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        return jsonify({"error": f"Experiment not found: {experiment_id}"}), 404

    fmt = (request.args.get("format") or "json").strip().lower()
    if fmt == "csv":
        body = experiment_to_csv(experiment)
        return Response(
            body,
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{experiment_id}.csv"'},
        )
    return Response(
        experiment_to_json(experiment),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.json"'},
    )


@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    store = ResultStore()
    return jsonify({"experiments": store.list_experiments()})


def _app_directory() -> Path:
    """Directory containing this server module (project / app root)."""
    return Path(__file__).resolve().parent


def _userprofile_glob_roots() -> list[Path]:
    """
    Directories under USERPROFILE where IDE/CLI output files commonly land.
    Used for join + limited ** glob (avoids scanning the entire profile root).
    """
    up = os.environ.get("USERPROFILE")
    if not up:
        return []
    base = Path(up)
    roots: list[Path] = []
    seen: set[str] = set()

    def add(d: Path) -> None:
        try:
            r = str(d.resolve())
        except OSError:
            r = str(d)
        if r not in seen and d.is_dir():
            seen.add(r)
            roots.append(d)

    add(base / "Documents")
    add(base / "Desktop")
    add(base / "Downloads")
    try:
        for od in base.glob("OneDrive*"):
            add(od)
            add(od / "Documents")
            add(od / "Desktop")
    except OSError:
        pass
    return roots


def _path_glob_roots() -> list[Path]:
    """Roots for recursive search: profile subtrees + app dir + process cwd (deduped)."""
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            try:
                if p.is_dir():
                    seen.add(key)
                    out.append(p)
            except OSError:
                pass

    for r in _userprofile_glob_roots():
        add(r)
    add(_app_directory())
    add(Path.cwd())
    return out


def _rglob_match_relative_suffix(roots: list[Path], rel: Path, per_root_cap: int = 6000) -> tuple[str, bool] | None:
    """
    Find a file by basename under roots, keeping paths whose posix path ends with rel.
    Handles deep trees (e.g. OneDrive/.../project/docs/architecture/overview.md) where
    join(root, rel) misses because the real root is a descendant folder.
    """
    if ".." in rel.parts:
        return None
    needle = rel.as_posix().replace("\\", "/").lower()
    if not needle or "*" in needle or "[" in needle:
        return None
    name = rel.name
    if not name or "." not in name:
        return None
    for root in roots:
        if not root.is_dir():
            continue
        n = 0
        try:
            for m in root.rglob(name):
                n += 1
                if n > per_root_cap:
                    break
                try:
                    if not m.is_file():
                        continue
                    s = m.as_posix().replace("\\", "/").lower()
                except (OSError, ValueError):
                    continue
                if s.endswith(needle):
                    try:
                        return str(m.resolve()), True
                    except OSError:
                        return str(m), True
        except OSError:
            continue
    return None


def _resolve_path(
    raw: str,
    extra_bases: list[str] | None = None,
) -> tuple[str, bool]:
    """
    Given a raw path string (possibly relative), try to find the actual file.
    Search order:
      1. As-is (absolute or already correct relative)
      2. Relative to each extra base (e.g. CLI working directory from the run)
      3. Relative to cwd, home, USERPROFILE
      4. Relative to common USERPROFILE subfolders (Documents, Desktop, OneDrive*, …)
      5. Limited ** glob under those folders for paths like docs/architecture/x.md
    Returns (resolved_path_str, exists_bool).
    """
    p = Path(raw)
    if p.is_absolute():
        try:
            return str(p), p.is_file()
        except OSError:
            return str(p), False

    if ".." in p.parts:
        return raw, False

    rel_posix = p.as_posix()

    def try_candidates(candidates: list[Path]) -> tuple[str, bool] | None:
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return str(candidate.resolve()), True
            except OSError:
                continue
        return None

    ordered: list[Path] = []
    if extra_bases:
        for b in extra_bases:
            if b and str(b).strip():
                ordered.append(Path(b) / p)
    ordered.append(_app_directory() / p)
    ordered.extend(
        [
            Path.cwd() / p,
            Path.home() / p,
        ]
    )
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        ordered.append(Path(userprofile) / p)
    for root in _userprofile_glob_roots():
        ordered.append(root / p)

    hit = try_candidates(ordered)
    if hit:
        return hit

    # Last resort: ** glob under profile subtrees + app root + cwd
    glob_roots = _path_glob_roots()
    if rel_posix and "*" not in rel_posix and "[" not in rel_posix:
        max_iter = 400
        n = 0
        for root in glob_roots:
            if not root.is_dir():
                continue
            try:
                for m in root.glob("**/" + rel_posix):
                    n += 1
                    if n > max_iter:
                        break
                    if m.is_file():
                        return str(m.resolve()), True
            except (OSError, ValueError):
                continue
            if n > max_iter:
                break

    suffix_hit = _rglob_match_relative_suffix(glob_roots, p)
    if suffix_hit:
        return suffix_hit

    return raw, False


_QUOTED_PATH_RE = re.compile(r'["\']([\w./\\-]+\.[A-Za-z0-9]{1,8})["\']')
_BARE_PATH_RE = re.compile(
    r"(?im)^[\s\-*>]*([\w./\\-]+\."
    r"(?:md|py|js|ts|tsx|jsx|json|yaml|yml|txt|mmd|html|css|sql|sh|bat|ps1|toml|cfg|ini))\s*$"
)


def _extract_file_paths(output: str, cli_working_dir: str | None = None) -> list:
    """
    Scan CLI output for file paths that were created/modified.
    Looks for:
      - JSON with filesCreated / outputPath arrays
      - Lines with common file extensions
      - Quoted paths like "docs/architecture/overview.md"
    Returns list of dicts: {path, name, ext, exists, resolve_base?}
    """
    import json as _json
    found = []
    seen = set()
    extra = [cli_working_dir] if cli_working_dir else None

    def add(path):
        path = path.strip().strip("'\"")
        if not path or path in seen:
            return
        seen.add(path)
        resolved, exists = _resolve_path(path, extra_bases=extra)
        entry = {
            "path": path,
            "name": Path(path).name,
            "ext": Path(path).suffix.lstrip("."),
            "exists": exists,
        }
        if exists and resolved != path:
            entry["resolve_base"] = str(Path(resolved).parent)
        found.append(entry)

    # 1. JSON blocks with filesCreated / outputPath arrays
    for match in re.finditer(r"\{[^{}]*\}", output):
        try:
            obj = _json.loads(match.group(0))
        except (_json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("filesCreated", "files_created", "outputPath", "output_path", "files"):
            val = obj.get(key)
            if isinstance(val, str):
                add(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        add(item)
                    elif isinstance(item, dict) and isinstance(item.get("path"), str):
                        add(item["path"])

    # 2. Quoted paths like "docs/architecture/overview.md"
    for match in _QUOTED_PATH_RE.finditer(output):
        add(match.group(1))

    # 3. Bare paths on their own line (e.g. bullet-listed file names)
    for match in _BARE_PATH_RE.finditer(output):
        add(match.group(1))

    return found


@app.route("/api/health")
def health():
    """Reports whether each CLI binary is resolvable, for the UI's status dot."""
    copilot_available = bool(_find_copilot_binary())
    claude_available = bool(_find_claude_binary())
    return jsonify({
        "status": "ok",
        "tools": {
            "gh_copilot": "available" if copilot_available else "unavailable",
            "claude": "available" if claude_available else "unavailable",
        },
    })


@app.route("/api/read-file", methods=["POST"])
def read_file():
    """Read a file (produced by a CLI run) for preview in the UI."""
    data = request.json or {}
    raw_path = (data.get("path") or "").strip()
    resolve_base = (data.get("resolve_base") or "").strip()

    if not raw_path:
        return jsonify({"error": "No file path provided."}), 400

    extra_bases = [resolve_base] if resolve_base else None
    resolved, exists = _resolve_path(raw_path, extra_bases=extra_bases)
    if not exists:
        return jsonify({"error": f"File not found: {raw_path}"}), 404

    path = Path(resolved)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return jsonify({"content": content, "ext": path.suffix.lstrip("."), "name": path.name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/download-file", methods=["POST"])
def download_file():
    """Download a file (produced by a CLI run) as an attachment."""
    data = request.json or {}
    raw_path = (data.get("path") or "").strip()
    resolve_base = (data.get("resolve_base") or "").strip()
    download_name = (data.get("name") or "").strip()

    if not raw_path:
        return jsonify({"error": "No file path provided."}), 400

    extra_bases = [resolve_base] if resolve_base else None
    resolved, exists = _resolve_path(raw_path, extra_bases=extra_bases)
    if not exists:
        return jsonify({"error": f"File not found: {raw_path}"}), 404

    path = Path(resolved)
    return send_file(str(path), as_attachment=True, download_name=download_name or path.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)