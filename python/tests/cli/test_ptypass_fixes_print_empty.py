"""
`-p` says so when the model returns nothing (the ptypass-fixes lane,
2026-09-27, brief item 14; a chat-fixes leftover). The chat printed the
truthful line (`failures.present_empty_reply`), while `-p` printed an empty
line and exited 0, so a script could not tell an empty reply from an answer.
The line goes to stderr; stdout keeps the answer channel's shape. The
TypeScript twin is `typescript/tests/unit/cli/ptypass-fixes-print-empty.test.ts`.

Driven for real: the CLI in a child process against a local stand-in for the
OpenAI API that streams a completion with no content.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from webagents.cli.repl.failures import present_empty_reply

AGENT = "---\nname: quiet\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nSay nothing.\n"


@pytest.fixture
def empty_model():
    class Empty(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's name
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
            base = {"id": "empty-1", "object": "chat.completion.chunk", "created": 0, "model": request.get("model") or "gpt-4o-mini"}
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in (
                    {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ):
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                return
            body = json.dumps({**base, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Empty)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_p_prints_the_truthful_line_for_an_empty_reply(tmp_path, empty_model):
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENT.md").write_text(AGENT)
    home = tmp_path / "home"
    home.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("_API_KEY") and key not in ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "WEBAGENTS_DEBUG")
    }
    env.update(
        HOME=str(home),
        WEBAGENTS_SECRETS_BACKEND="file",
        ROBUTLER_API_URL="http://127.0.0.1:9",
        OPENAI_API_KEY="sk-test-not-a-real-key",
        OPENAI_BASE_URL=f"http://127.0.0.1:{empty_model}/v1",
    )
    done = subprocess.run([sys.executable, "-m", "webagents", "-p", "say nothing"], cwd=project, env=env, capture_output=True, text=True, timeout=180)
    expected = present_empty_reply(None)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""
    assert expected.headline in done.stderr
    if expected.hint:
        assert expected.hint in done.stderr
