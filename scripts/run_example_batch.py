"""Reproduce the 55-prompt study of the paper (Section VI).
Run every prompt through the comparator's /api/compare (Flask test client, in-process).
Resumable: results are appended to results.jsonl; already-finished ids are skipped."""
import json
import os
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
HERE = APP / "docs" / "examples"          # prompts.json and software-assistant.md live here
SANDBOX = HERE / "agent_sandbox"          # any stray agent files land here, not in the project
SANDBOX.mkdir(exist_ok=True)
OUT = HERE / "results_raw.jsonl"

os.chdir(APP)
sys.path.insert(0, str(APP))
os.environ.pop("CLAUDECODE", None)
import server  # noqa: E402

system_prompt = (HERE / "software-assistant.md").read_text(encoding="utf-8")
prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))

# Guard: none of the prompts should trigger the "write files to disk" hint.
for cat, p in prompts:
    assert not server._COPILOT_FILEGEN_HINT_RE.search(p), p
assert not server._COPILOT_FILEGEN_HINT_RE.search(system_prompt)

done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["id"])

client = server.app.test_client()
t_start = time.time()
for i, (category, prompt) in enumerate(prompts, start=1):
    if i in done:
        continue
    t0 = time.time()
    resp = client.post("/api/compare", json={
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "copilot_cwd": str(SANDBOX),
    })
    data = resp.get_json()
    row = {"id": i, "category": category, "prompt": prompt, "http": resp.status_code, **data}
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    c, l = data["copilot"], data["claude"]
    print(f"[{i:02d}/{len(prompts)}] {time.time()-t0:5.0f}s  "
          f"copilot={c['scores']['overall']:>3}{' FAIL' if c.get('error') else ''}  "
          f"claude={l['scores']['overall']:>3}{' FAIL' if l.get('error') else ''}  | {prompt[:60]}",
          flush=True)
print(f"DONE in {(time.time()-t_start)/60:.1f} min", flush=True)
