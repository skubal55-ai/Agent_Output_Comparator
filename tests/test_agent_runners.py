"""Tests for the CLI runners in server.py — subprocess is mocked, no real CLI runs."""
import json
import os
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestRunClaudeCli:
    def test_success_returns_result_text(self):
        payload = {"type": "result", "subtype": "success", "is_error": False,
                   "result": "Cache-aside means...", "num_turns": 1,
                   "modelUsage": {"claude-sonnet-4-6": {}}}
        with patch.object(server.subprocess, "run", return_value=_completed(json.dumps(payload))) as run:
            result = server.run_claude_cli("You are an architect.", "Explain cache-aside.")

        assert result["error"] is None
        assert result["output"] == "Cache-aside means..."
        assert result["model"] == "claude-sonnet-4-6"
        # Prompt goes over stdin, identical to what Copilot receives.
        assert run.call_args.kwargs["input"] == server._combined_agent_prompt(
            "You are an architect.", "Explain cache-aside."
        )
        assert run.call_args.kwargs["shell"] is False

    def test_is_error_payload_becomes_error(self):
        payload = {"type": "result", "subtype": "success", "is_error": True,
                   "result": "Not logged in · Please run /login"}
        with patch.object(server.subprocess, "run", return_value=_completed(json.dumps(payload))):
            result = server.run_claude_cli("", "hi")

        assert result["output"] == ""
        assert result["error"] == "Not logged in · Please run /login"

    def test_nested_session_env_vars_are_removed(self):
        payload = {"subtype": "success", "is_error": False, "result": "ok"}
        with patch.dict(os.environ, {"CLAUDECODE": "1"}), \
             patch.object(server.subprocess, "run", return_value=_completed(json.dumps(payload))) as run:
            server.run_claude_cli("", "hi")

        assert "CLAUDECODE" not in run.call_args.kwargs["env"]

    def test_timeout_is_reported_as_error(self):
        with patch.object(server.subprocess, "run", side_effect=subprocess.TimeoutExpired("claude", 5)):
            result = server.run_claude_cli("", "hi")

        assert result["output"] == ""
        assert "Timed out" in result["error"]


class TestCopilotErrorDetection:
    def test_error_event_without_answer_is_error(self):
        raw = '{"type":"error","error":{"name":"UnknownError","data":{"message":"invalid_grant"}}}'
        assert server._jsonl_error_without_answer(raw) == "invalid_grant"

    def test_answer_wins_over_error_event(self):
        raw = "\n".join([
            '{"type":"session.error","data":{"message":"transient"}}',
            '{"type":"assistant.message","data":{"content":"Final answer"}}',
        ])
        assert server._jsonl_error_without_answer(raw) is None


class TestCompareEndpoint:
    def _fake(self, output, latency_ms=1000, error=None):
        return {"tool": "x", "output": output, "error": error, "latency_ms": latency_ms, "working_directory": None}

    def test_empty_answer_is_reported_as_failed_run(self):
        client = server.app.test_client()
        with patch.object(server, "run_copilot_cli", return_value=self._fake("A full answer.\n- one\n- two")), \
             patch.object(server, "run_claude_cli", return_value=self._fake("   ")):
            data = client.post("/api/compare", json={"user_prompt": "hi"}).get_json()

        assert data["claude"]["error"] == server.EMPTY_ANSWER_ERROR
        assert data["claude"]["scores"]["failed"] is True
        assert data["claude"]["scores"]["overall"] == 0
        # The failed peer is not used for Efficiency: the successful agent gets the neutral 70.
        assert data["copilot"]["scores"]["speed"] == 70

    def test_cors_only_allows_own_origin(self):
        client = server.app.test_client()
        own = client.get("/api/health", headers={"Origin": "http://localhost:5050"})
        other = client.get("/api/health", headers={"Origin": "https://evil.example"})
        assert own.headers.get("Access-Control-Allow-Origin") == "http://localhost:5050"
        assert "Access-Control-Allow-Origin" not in other.headers


class TestScoreForUi:
    def test_failed_run_scores_zero(self):
        scores = server.score_for_ui("", "prompt", "", 50, 20000, error="Not logged in")
        assert scores["failed"] is True
        assert scores["overall"] == scores["speed"] == scores["quality"] == 0

    def test_successful_run_is_not_failed(self):
        scores = server.score_for_ui("A structured answer.\n- one\n- two", "prompt", "", 1000, 0)
        assert scores["failed"] is False
        assert scores["overall"] > 0
