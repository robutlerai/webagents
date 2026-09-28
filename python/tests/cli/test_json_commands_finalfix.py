"""
`--json` is honoured by `init`, `templates list`, `models`, `skills add` and
`publish --dry-run`, and with `-p` it is `--output-format json` (2026-09-27,
the final e2e re-run: all of them ignored the flag). The documents are pinned
by `tests/fixtures/cli/json_documents.json` and the refusals by
`cli/json_errors.json`, which the TypeScript suite runs too
(`tests/unit/cli/json-commands-finalfix.test.ts`). A scratch HOME with the
file backend, the platform at a loopback stand-in, and for `-p` a fake OpenAI
server on loopback that answers "You said: hi."
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.init_templates import INIT_TEMPLATES
from webagents.cli.main import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cli"
DOCS = json.loads((FIXTURES / "json_documents.json").read_text())
ERRORS = json.loads((FIXTURES / "json_errors.json").read_text())
TEMPLATES = list(json.loads((FIXTURES / "init_templates.json").read_text())["templates"])
AGENT_MD = "---\nname: my-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n"
runner = CliRunner()


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN", "_BASE_URL")) or var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "WEBAGENTS_DEBUG"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _document(result):
    # ONE document and nothing else on stdout.
    return json.loads(result.stdout)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "AGENT.md").write_text(AGENT_MD)
    return tmp_path


def test_templates_list(scratch):
    result = runner.invoke(app, ["--json", "templates", "list"])
    assert result.exit_code == 0, result.output
    doc = _document(result)
    assert doc["ok"] is True and list(doc["data"]) == DOCS["templates_list"]["data_keys"]
    rows = doc["data"]["templates"]
    assert [row["name"] for row in rows] == TEMPLATES == list(INIT_TEMPLATES)
    for row in rows:
        assert list(row) == DOCS["templates_list"]["row_keys"]


def test_models(scratch, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    result = runner.invoke(app, ["--json", "models"])
    assert result.exit_code == 0, result.output
    doc = _document(result)
    assert doc["ok"] is True and list(doc["data"]) == DOCS["models"]["data_keys"]
    rows = doc["data"]["providers"]
    assert rows
    for row in rows:
        assert list(row) == DOCS["models"]["row_keys"]
        assert isinstance(row["ready"], bool)
    by_id = {row["id"]: row for row in rows}
    assert by_id["openai"]["ready"] is True
    assert by_id["anthropic"]["ready"] is False


def test_init_and_its_refusals(scratch):
    made = runner.invoke(app, ["--json", "init", "proj", "-t", "chatbot"])
    assert made.exit_code == 0, made.output
    doc = _document(made)
    assert doc["ok"] is True and list(doc["data"]) == DOCS["init"]["data_keys"]
    assert doc["data"]["name"] == "proj" and doc["data"]["template"] == "chatbot"
    assert doc["data"]["path"] == str((scratch / "proj").resolve())
    assert doc["data"]["files"] == DOCS["init"]["files"]
    # The model the file names: none with no provider key here (B3, 2026-09-28).
    assert doc["data"]["model"] is None or isinstance(doc["data"]["model"], str)
    assert (scratch / "proj" / "AGENT.md").exists()

    again = runner.invoke(app, ["--json", "init", "proj"])
    assert again.exit_code == ERRORS["directory_exists"]["exit"]
    assert _document(again) == {"ok": False, "error": {"code": "directory_exists", "message": ERRORS["directory_exists"]["message"].replace("{name}", "proj")}}

    typo = runner.invoke(app, ["--json", "init", "other", "-t", "nope"])
    assert typo.exit_code == ERRORS["unknown_template"]["exit"]
    message = ERRORS["unknown_template"]["message"].replace("{template}", "nope").replace("{templates}", ", ".join(TEMPLATES))
    assert _document(typo) == {"ok": False, "error": {"code": "unknown_template", "message": message}}
    assert not (scratch / "other").exists()


def test_skills_add_and_its_refusal(scratch):
    _project(scratch)
    added = runner.invoke(app, ["--json", "skills", "add", "todo"])
    assert added.exit_code == 0, added.output
    doc = _document(added)
    assert doc["ok"] is True and list(doc["data"]) == DOCS["skills_add"]["data_keys"]
    assert doc["data"]["file"] == "AGENT.md" and doc["data"]["added"] == ["todo"] and doc["data"]["already"] == []
    assert doc["data"]["messages"][0] == "Added todo to AGENT.md."
    assert "- todo" in (scratch / "AGENT.md").read_text()

    bad = runner.invoke(app, ["--json", "skills", "add", "nosuchskill"])
    assert bad.exit_code == ERRORS["skills_add_failed"]["exit"]
    refusal = _document(bad)
    assert refusal["ok"] is False and refusal["error"]["code"] == "skills_add_failed"
    assert refusal["error"]["message"].startswith(ERRORS["skills_add_failed"]["unknown_skill_starts"].replace("{name}", "nosuchskill"))
    assert "nosuchskill" not in (scratch / "AGENT.md").read_text()


def test_publish_dry_run(scratch):
    _project(scratch)
    result = runner.invoke(app, ["--json", "publish", "--dry-run"])
    assert result.exit_code == 0, result.output
    doc = _document(result)
    assert doc["ok"] is True and list(doc["data"]) == DOCS["publish_dry_run"]["data_keys"]
    assert doc["data"]["method"] == "POST"
    assert doc["data"]["url"] == "http://127.0.0.1:9/api/agents"
    assert doc["data"]["body"]["name"] == "my-agent"
    assert doc["data"]["created"] is True


def test_publish_with_no_agent_file_is_the_error_envelope(scratch):
    result = runner.invoke(app, ["--json", "publish", "--dry-run"])
    assert result.exit_code == ERRORS["publish_failed"]["exit"]
    doc = _document(result)
    assert doc["ok"] is False and doc["error"]["code"] == "publish_failed"
    assert "No AGENT.md" in doc["error"]["message"]


@pytest.fixture
def fake_openai():
    """A fake OpenAI: streams or answers "You said: <last user message>." """

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802 - http.server's name
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            last = next((m.get("content") for m in reversed(body.get("messages") or []) if m.get("role") == "user"), "")
            text = f"You said: {last if isinstance(last, str) else json.dumps(last)}."
            usage = {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def chunk(delta, finish, **extra):
                    payload = {"id": "cmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())

                chunk({"role": "assistant", "content": ""}, None)
                chunk({"content": text}, None)
                chunk({}, "stop", usage=usage)
                self.wfile.write(b"data: [DONE]\n\n")
                return
            payload = {"id": "cmpl-test", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_json_with_p_is_output_format_json(scratch, monkeypatch, fake_openai):
    _project(scratch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", fake_openai)
    result = runner.invoke(app, DOCS["prompt"]["case"]["argv"])
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["content"] == DOCS["prompt"]["case"]["content"]


def test_an_explicit_output_format_wins(scratch, monkeypatch, fake_openai):
    _project(scratch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", fake_openai)
    result = runner.invoke(app, DOCS["prompt"]["case"]["explicit_wins"])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert len(lines) > 1 and lines[-1]["type"] == "done"
