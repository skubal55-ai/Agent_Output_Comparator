"""
Agent Output Comparator - Flask Backend
Runs prompts against GitHub Copilot CLI and OpenCode CLI,
measures latency, and scores outputs across 4 metrics.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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


def _copilot_subprocess_timeout_sec() -> int:
    """
    Max wall-clock time for one Copilot CLI run.
    Agent + tools often exceeds 120s; override with COMPARE_COPILOT_TIMEOUT_SEC (30–3600).
    """
    raw = os.environ.get("COMPARE_COPILOT_TIMEOUT_SEC", "").strip()
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
    timeout_sec = _copilot_subprocess_timeout_sec()

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
        result["output"] = _clean_copilot_json_output(result["output"])
    return result


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
    return ""


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



# Relative paths (under app / compare project) tried when resolving a spec file
_OPENCODE_DEFAULT_SPEC_FILES: tuple[str, ...] = (
    "openspec.md",
    "OPEN_SPEC.md",
    "open-spec.md",
    "opencode.spec.md",
    "SPEC.md",
    "spec.md",
    "specification.md",
    "docs/specification.md",
    "docs/architecture/overview.md",
    ".opencode/spec.md",
    "openapi.yaml",
    "openapi.yml",
)


def _resolve_opencode_spec_file(project: Path, hint: str | None) -> Path | None:
    """
    Pick a specification file for OpenCode agents that expect `localSpecPath`.
    hint may be absolute, or relative to project; env COMPARE_OPENCODE_SPEC is a fallback.
    """
    raw = (hint or "").strip() or os.environ.get("COMPARE_OPENCODE_SPEC", "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p.resolve()
        rel = project / raw
        if rel.is_file():
            return rel.resolve()
        return None
    for name in _OPENCODE_DEFAULT_SPEC_FILES:
        cand = project / name
        if cand.is_file():
            return cand.resolve()
    return None


def _opencode_context_prefix(project_abs: str, spec: Path | None) -> str:
    """Inject paths so models/tools do not ask for localSpecPath interactively."""
    lines = [
        "[Comparator context] Project root (absolute): " + project_abs,
    ]
    if spec is not None:
        sp = str(spec.resolve())
        lines.append(
            "[Comparator context] Specification file `localSpecPath` (absolute): " + sp
        )
    else:
        lines.append(
            "[Comparator context] No specification file was auto-detected under the project; "
            "if you require `localSpecPath`, use the project root above or paths beneath it."
        )
    return "\n".join(lines) + "\n\n"


def _clean_opencode_output(text: str) -> str:
    """
    OpenCode --format json emits JSONL (one JSON object per line).
    Each line has a "type" field. We extract text from lines where:
      type == "text" and part.text contains the assistant response.

    Example line:
      {"type":"text", "part": {"type":"text", "text": "You can use..."}}
    """
    import json
    lines = text.strip().splitlines()
    parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("type") == "text":
                part = obj.get("part", {})
                t = part.get("text", "")
                if t and t.strip():
                    parts.append(t.strip())
        except (json.JSONDecodeError, ValueError):
            parts.append(line)

    result = "\n".join(parts).strip()
    return result if result else text.strip()


def run_opencode_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    local_spec_path: str | None = None,
) -> dict:
    """
    Invoke OpenCode CLI v1.14+.
    Writes the prompt to a temp file and passes it via stdin redirect
    to avoid Windows shell quoting issues with spaces and special chars.

    Uses ``opencode run --dir <app>`` so the project config loads, optional ``--file``
    for a spec, and a short context prefix so agents receive an absolute ``localSpecPath``.
    """
    import tempfile
    project = _app_directory()
    proj_abs = str(project.resolve())
    spec = _resolve_opencode_spec_file(project, local_spec_path)
    prefix = _opencode_context_prefix(proj_abs, spec)
    combined = prefix + (f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt)

    # Write prompt to a temp file — avoids all shell quoting issues
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      delete=False, encoding="utf-8")
    tmp.write(combined)
    tmp.close()
    tmp_path = tmp.name

    opencode_bin = _find_opencode_binary()
    file_arg = ""
    if spec is not None:
        file_arg = f' --file "{str(spec.resolve())}"'
    # --dir: run with project root (loads .opencode / local agents); --file attaches spec
    cmd = (
        f'"{opencode_bin}" run --dir "{proj_abs}"{file_arg} '
        f'--dangerously-skip-permissions --format json < "{tmp_path}"'
    )

    result = _run_shell("OpenCode CLI", cmd, timeout_sec=600)

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if result.get("output"):
        result["output"] = _clean_opencode_output(result["output"])

    result["opencode_project_dir"] = proj_abs
    if spec is not None:
        result["local_spec_path"] = str(spec.resolve())

    return result


def _find_opencode_binary() -> str:
    """
    Resolve full path to opencode binary.
    Checks common Windows npm install locations so Python subprocess
    can find it even when PATH differs from the shell.
    """
    import shutil
    # First try: let shutil find it via PATH (works if PATH is inherited)
    found = shutil.which("opencode")
    if found:
        return found

    # Fallback: check common npm global install locations on Windows
    user_profile = os.environ.get("USERPROFILE", "")
    app_data = os.environ.get("APPDATA", "")
    candidates = [
        # Confirmed path: C:\Users\a949557\AppData\Roaming\npm\opencode.cmd
        # APPDATA already points to AppData\Roaming on Windows
        os.path.join(app_data, "npm", "opencode.cmd"),
        os.path.join(app_data, "npm", "opencode"),
        os.path.join(user_profile, "node_modules", ".bin", "opencode.cmd"),
        os.path.join(user_profile, "node_modules", ".bin", "opencode"),
        r"C:\Program Files\nodejs\opencode.cmd",
        r"C:\Program Files\nodejs\opencode",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Last resort: return bare name and let it fail with a clear error
    return "opencode"


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

def score_output(output: str, prompt: str, system_prompt: str, latency_ms: int, peer_latency_ms: int) -> dict:
    """
    Score a single output across 4 dimensions (0-100 each).

    Quality     – length, structure, paragraph/list use, no truncation
    Accuracy    – keyword overlap between prompt keywords and output
    Speed       – relative latency vs peer (faster = higher score)
    Length fit  – penalise very short or excessively long responses
    """
    if not output:
        return {"quality": 0, "accuracy": 0, "speed": 0, "length": 0, "overall": 0}

    words = output.split()
    word_count = len(words)

    # --- Quality (0-100) ---
    quality = 0
    # Has meaningful length
    if word_count >= 10:
        quality += 20
    if word_count >= 30:
        quality += 15
    # Has structured content (lists, code blocks, headers)
    if re.search(r"(^[-*•]\s|\d+\.\s|```|#{1,3}\s)", output, re.MULTILINE):
        quality += 20
    # Multiple sentences / paragraphs
    sentences = re.split(r"[.!?]\s+", output)
    if len(sentences) >= 3:
        quality += 15
    # No error-like phrases
    error_phrases = ["error:", "not found", "command not found", "traceback", "exception"]
    if not any(p in output.lower() for p in error_phrases):
        quality += 15
    # Not truncated (doesn't end mid-word or with "...")
    if not output.rstrip().endswith(("...", "…")):
        quality += 15
    quality = min(quality, 100)

    # --- Accuracy (0-100): keyword overlap ---
    # Extract significant words from prompt + system_prompt
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "and", "or", "but", "if", "then", "so", "yet", "nor", "not", "no",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "this", "that",
        "these", "those", "what", "how", "when", "where", "why", "which", "who",
    }
    combined_input = f"{system_prompt} {prompt}".lower()
    input_words = set(re.findall(r"\b[a-z]{4,}\b", combined_input)) - stop_words
    output_lower = output.lower()
    output_word_set = set(re.findall(r"\b[a-z]{4,}\b", output_lower)) - stop_words

    if input_words:
        overlap = len(input_words & output_word_set) / len(input_words)
        accuracy = min(int(overlap * 160), 100)  # scale up — full overlap rare
    else:
        accuracy = 50  # neutral if no keywords to compare

    # --- Speed (0-100): relative comparison ---
    # When equal, both get 70. Faster tool scales up to 100, slower down to 40.
    if peer_latency_ms == 0 or latency_ms == 0:
        speed = 70  # neutral if we can't compare
    else:
        ratio = latency_ms / max(peer_latency_ms, 1)
        # ratio < 1 → faster (score > 70), ratio > 1 → slower (score < 70)
        speed = int(70 + 30 * (1 - ratio))
        speed = max(20, min(speed, 100))

    # --- Length fit (0-100) ---
    # Ideal range: 30-400 words
    if word_count < 5:
        length_score = 10
    elif word_count < 15:
        length_score = 40
    elif word_count < 30:
        length_score = 65
    elif word_count <= 400:
        length_score = 100
    elif word_count <= 700:
        length_score = 80
    else:
        length_score = 60

    overall = int((quality * 0.35) + (accuracy * 0.35) + (speed * 0.15) + (length_score * 0.15))

    return {
        "quality": quality,
        "accuracy": accuracy,
        "speed": speed,
        "length": length_score,
        "word_count": word_count,
        "overall": overall,
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
      "local_spec_path" | "localSpecPath": optional absolute or project-relative spec file
      "copilot_cwd" | "copilot_working_directory": optional folder where Copilot tools write files
    }
    """
    data = request.json or {}
    system_prompt = data.get("system_prompt", "").strip()
    user_prompt = data.get("user_prompt", "").strip()
    spec_hint = (
        data.get("local_spec_path")
        or data.get("localSpecPath")
        or ""
    )
    if isinstance(spec_hint, str):
        spec_hint = spec_hint.strip()
    else:
        spec_hint = ""

    cop_cwd_req = data.get("copilot_cwd") or data.get("copilot_working_directory") or ""
    cop_cwd_req = cop_cwd_req.strip() if isinstance(cop_cwd_req, str) else ""

    if not user_prompt:
        return jsonify({"error": "user_prompt is required."}), 400

    # Run both CLIs (sequentially to avoid interfering with each other's timing)
    copilot_result = run_copilot_cli(
        system_prompt, user_prompt, copilot_cwd=cop_cwd_req or None
    )
    opencode_result = run_opencode_cli(
        system_prompt, user_prompt, local_spec_path=spec_hint or None
    )

    # Score outputs
    copilot_scores = score_output(
        copilot_result["output"],
        user_prompt,
        system_prompt,
        copilot_result["latency_ms"],
        opencode_result["latency_ms"],
    )
    opencode_scores = score_output(
        opencode_result["output"],
        user_prompt,
        system_prompt,
        opencode_result["latency_ms"],
        copilot_result["latency_ms"],
    )

    # Extract any file paths mentioned or created in the output
    cop_cwd = copilot_result.get("working_directory")
    copilot_files = _extract_file_paths(
        copilot_result.get("output", ""),
        cli_working_dir=cop_cwd if isinstance(cop_cwd, str) and cop_cwd.strip() else None,
    )
    opencode_files = _extract_file_paths(opencode_result.get("output", ""))

    return jsonify({
        "prompt": user_prompt,
        "copilot":  {**copilot_result,  "scores": copilot_scores,  "files": copilot_files},
        "opencode": {**opencode_result, "scores": opencode_scores, "files": opencode_files},
    })


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
        path = path.strip().strip('"').strip("'").replace("\\\\", "/").replace("\\", "/")
        if path in seen or len(path) < 3:
            return
        seen.add(path)
        resolved, exists = _resolve_path(path, extra_bases=extra)
        name = Path(resolved).name or path.split("/")[-1]
        ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        entry: dict = {"path": resolved, "name": name, "ext": ext, "exists": exists}
        if cli_working_dir and str(cli_working_dir).strip():
            entry["resolve_base"] = cli_working_dir.strip()
        found.append(entry)

    # 1. Try JSON in output (opencode style)
    json_match = re.search(r'\{[\s\S]*"filesCreated"[\s\S]*\}', output)
    if json_match:
        try:
            obj = _json.loads(json_match.group(0))
            for p in obj.get("filesCreated", []):
                add(p)
            op = obj.get("outputPath", "")
            if op and op != "/":
                add(op)
        except Exception:
            pass

    # 2. Regex: quoted paths with file extensions
    for m in re.finditer(r'["\']([^"\']+\.[a-zA-Z]{1,6})["\']', output):
        add(m.group(1))

    # 3. Bare paths like docs/architecture/overview.md
    for m in re.finditer(r'(?<!\w)([\w./\\-]+/[\w./\\-]+\.(?:md|mmd|json|yaml|yml|ts|js|py|txt|html|svg|png|pdf))', output):
        add(m.group(1))

    return found


def _read_resolve_bases(data: dict) -> list[str] | None:
    """Optional bases from client (same run as /api/compare) for relative paths."""
    rb = (data.get("resolve_base") or data.get("working_directory") or "").strip()
    return [rb] if rb else None


@app.route("/api/read-file", methods=["POST"])
def read_file():
    """Read a generated file for preview. Body: { path: str, resolve_base?: str }"""
    data = request.json or {}
    fpath = data.get("path", "").strip()
    if not fpath:
        return jsonify({"error": "No path provided"}), 400
    resolved, exists = _resolve_path(fpath, extra_bases=_read_resolve_bases(data))
    p = Path(resolved)
    if not exists or not p.is_file():
        return jsonify({"error": f"File not found: {fpath}"}), 404
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        size = p.stat().st_size
        return jsonify({"content": content, "name": p.name, "size": size,
                        "ext": p.suffix.lstrip(".").lower()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-file", methods=["POST"])
def download_file():
    """Download a generated file. Body: { path: str, resolve_base?: str }"""
    from flask import send_file
    data = request.json or {}
    fpath = data.get("path", "").strip()
    if not fpath:
        return jsonify({"error": "No path provided"}), 400
    resolved, exists = _resolve_path(fpath, extra_bases=_read_resolve_bases(data))
    p = Path(resolved)
    if not exists or not p.is_file():
        return jsonify({"error": f"File not found: {fpath}"}), 404
    return send_file(str(p.resolve()), as_attachment=True, download_name=p.name)


# ---------------------------------------------------------------------------
# Git repo helpers
# ---------------------------------------------------------------------------

# Track active cloned repos so we can clean them up
_cloned_repos: dict[str, str] = {}   # token → temp_dir path


def _build_auth_url(repo_url: str, pat: str) -> str:
    """Embed PAT into a GitHub HTTPS URL for authenticated clone."""
    # https://github.com/org/repo  →  https://<pat>@github.com/org/repo
    if repo_url.startswith("https://"):
        return repo_url.replace("https://", f"https://{pat}@", 1)
    return repo_url


def _collect_repo_code(repo_dir: str, max_chars: int = 120_000) -> str:
    """
    Walk the cloned repo and concatenate file contents into one context string.
    Skips binary files, .git folder, node_modules, common build artefacts.
    Stops once max_chars is reached to avoid overwhelming the CLI.
    """
    SKIP_DIRS  = {".git", "node_modules", "__pycache__", ".venv", "venv",
                  "dist", "build", ".next", ".nuxt", "target", "vendor"}
    TEXT_EXTS  = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cs", ".go",
        ".rb", ".php", ".rs", ".cpp", ".c", ".h", ".hpp",
        ".html", ".css", ".scss", ".json", ".yaml", ".yml",
        ".md", ".txt", ".sh", ".bat", ".ps1", ".sql",
        ".tf", ".toml", ".ini", ".env.example", ".dockerfile",
        "dockerfile",
    }
    MAX_FILE_CHARS = 8_000   # cap per individual file

    chunks = []
    total  = 0
    root   = Path(repo_dir)

    for fpath in sorted(root.rglob("*")):
        if total >= max_chars:
            chunks.append("\n\n[... context limit reached — remaining files omitted ...]")
            break
        # Skip directories and hidden/build folders
        if fpath.is_dir():
            continue
        parts = set(fpath.parts)
        if parts & SKIP_DIRS:
            continue
        # Only include text-ish files
        ext = fpath.suffix.lower()
        name_lower = fpath.name.lower()
        if ext not in TEXT_EXTS and name_lower not in TEXT_EXTS:
            continue

        rel = fpath.relative_to(root)
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n[... file truncated ...]"

        chunk = f"\n\n### FILE: {rel}\n```{ext.lstrip('.')}\n{text}\n```"
        chunks.append(chunk)
        total += len(chunk)

    if not chunks:
        return "(no readable source files found in repository)"

    return "".join(chunks)


@app.route("/api/clone-repo", methods=["POST"])
def clone_repo():
    """
    Clone a GitHub HTTPS repo and return repo metadata + file tree summary.
    Body: { url, pat, branch?, subdir? }
    Returns: { token, repo_name, branch, file_count, size_chars, tree_preview }
    """
    import tempfile, shutil, uuid

    data       = request.json or {}
    repo_url   = (data.get("url") or "").strip().rstrip("/")
    pat        = (data.get("pat") or "").strip()
    branch     = (data.get("branch") or "").strip() or None
    subdir     = (data.get("subdir") or "").strip().strip("/") or None

    if not repo_url:
        return jsonify({"error": "repo_url is required"}), 400
    if not repo_url.startswith("https://github.com/"):
        return jsonify({"error": "Only GitHub HTTPS URLs are supported (https://github.com/...)"}), 400

    auth_url = _build_auth_url(repo_url, pat) if pat else repo_url
    tmp_dir  = tempfile.mkdtemp(prefix="aoc_repo_")

    try:
        clone_cmd = ["git", "clone", "--depth", "1", "--single-branch"]
        if branch:
            clone_cmd += ["--branch", branch]
        clone_cmd += [auth_url, tmp_dir]

        proc = subprocess.run(
            clone_cmd,
            capture_output=True, text=True,
            timeout=120, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            err = proc.stderr.strip()
            # Scrub PAT from error message before returning
            if pat:
                err = err.replace(pat, "***")
            return jsonify({"error": f"git clone failed: {err}"}), 400

        # Narrow to subdir if requested
        work_dir = tmp_dir
        if subdir:
            candidate = Path(tmp_dir) / subdir
            if candidate.is_dir():
                work_dir = str(candidate)
            else:
                return jsonify({"error": f"Subdir '{subdir}' not found in repo"}), 400

        # Collect readable source files
        code_context = _collect_repo_code(work_dir)
        file_count   = code_context.count("### FILE:")
        size_chars   = len(code_context)

        # Build a short tree preview (first 40 files)
        root = Path(work_dir)
        tree_lines = []
        for i, f in enumerate(sorted(root.rglob("*"))):
            if f.is_file() and ".git" not in f.parts:
                tree_lines.append(str(f.relative_to(root)))
            if len(tree_lines) >= 40:
                tree_lines.append("… (more files)")
                break
        tree_preview = "\n".join(tree_lines)

        # Detect actual branch
        head = Path(tmp_dir) / ".git" / "HEAD"
        actual_branch = branch or "main"
        if head.exists():
            head_text = head.read_text(encoding="utf-8", errors="replace").strip()
            if head_text.startswith("ref: refs/heads/"):
                actual_branch = head_text.replace("ref: refs/heads/", "")

        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        # Store the code context keyed by a token so /api/compare can use it
        token = str(uuid.uuid4())
        _cloned_repos[token] = {
            "dir": tmp_dir,
            "code_context": code_context,
            "repo_name": repo_name,
            "branch": actual_branch,
        }

        return jsonify({
            "token":        token,
            "repo_name":    repo_name,
            "branch":       actual_branch,
            "file_count":   file_count,
            "size_chars":   size_chars,
            "tree_preview": tree_preview,
        })

    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": "git clone timed out (120 s)"}), 504
    except Exception as exc:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/repo-context", methods=["POST"])
def repo_context():
    """
    Return the collected code context string for a previously cloned repo.
    Body: { token }
    """
    data  = request.json or {}
    token = (data.get("token") or "").strip()
    if not token or token not in _cloned_repos:
        return jsonify({"error": "Invalid or expired repo token. Clone again."}), 404
    info = _cloned_repos[token]
    return jsonify({
        "code_context": info["code_context"],
        "repo_name":    info["repo_name"],
        "branch":       info["branch"],
    })


@app.route("/api/repo-cleanup", methods=["POST"])
def repo_cleanup():
    """Remove a cloned repo temp directory. Body: { token }"""
    import shutil
    data  = request.json or {}
    token = (data.get("token") or "").strip()
    if token in _cloned_repos:
        shutil.rmtree(_cloned_repos[token]["dir"], ignore_errors=True)
        del _cloned_repos[token]
    return jsonify({"ok": True})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — also checks if CLIs are on PATH."""
    tools = {}
    copilot_bin = _find_copilot_binary()
    opencode_bin = _find_opencode_binary()
    cop_ver = f'"{copilot_bin}" --version' if copilot_bin else "copilot --version"
    opc_ver = f'"{opencode_bin}" --version'

    for tool, cmd in [("gh_copilot", cop_ver), ("opencode", opc_ver)]:
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=10,
                                  encoding="utf-8", errors="replace")
            tools[tool] = "available" if proc.returncode == 0 else f"error (exit {proc.returncode})"
        except Exception as e:
            tools[tool] = f"error: {e}"

    return jsonify({"status": "ok", "tools": tools})


if __name__ == "__main__":
    print("=" * 55)
    print("  Agent Output Comparator — Backend")
    print("  Running at http://localhost:5050")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False)
