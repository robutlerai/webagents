"""
Small things the CLI e2e pass found (2026-09-28), in the Python CLI, against
the shared fixture `cli/final_sdk_low_items.json` the TypeScript suite reads
too: tool lines cut at a word boundary, `secrets set NAME VALUE`, `-p`'s
sign-in hint and its failures in JSON, the failover note in `stream-json`,
and a server that prints no emoji.
"""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "final_sdk_low_items.json").read_text()
)
runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    keys = [v for p in LLM_PROVIDERS for v in p.env_vars] + ["GOOGLE_API_KEY"]
    for name in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "WEBAGENTS_AGENT_TOKEN", *keys):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")


@pytest.mark.parametrize("case", FIXTURE["clip"], ids=lambda c: f"{c['limit']}:{c['text'][:20]}")
def test_a_tool_line_is_cut_at_a_word_boundary(case):
    from webagents.cli.repl.render import clip_words

    assert clip_words(case["text"], case["limit"]) == case["clipped"]


def test_a_file_tool_refusal_is_shown_whole():
    from webagents.cli.repl.render import FAILURE_LINE_LIMIT, tool_result_summary

    assert FAILURE_LINE_LIMIT == FIXTURE["failure_line_limit"]
    refusal = FIXTURE["clip"][2]["text"]
    assert tool_result_summary("read_file", refusal, "error") == refusal


def test_secrets_set_with_a_value_says_where_values_come_from():
    result = runner.invoke(app, ["secrets", "set", "SOME_KEY", "some-value-as-arg"])
    assert result.exit_code == 1
    assert result.output.strip() == FIXTURE["secrets_value_as_argument"]
    listed = runner.invoke(app, ["secrets", "list"])
    assert "SOME_KEY" not in listed.output


def test_p_names_the_login_command_not_the_chat_command():
    from webagents.cli.repl.failures import EXPIRED_HINT, for_prompt

    assert for_prompt(EXPIRED_HINT) == FIXTURE["prompt_expired_hint"]
    assert for_prompt("something else") == "something else"


def test_p_json_answers_a_failure_in_json():
    Path("AGENT.md").write_text("---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n")
    result = runner.invoke(app, ["-p", "hi", "--output-format", "json"])
    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert list(document["error"]) == FIXTURE["prompt_failure"]["json_keys"]
    assert document["error"]["code"] == FIXTURE["prompt_failure"]["no_model_code"]
    stream = runner.invoke(app, ["-p", "hi", "--output-format", "stream-json"])
    assert stream.exit_code == 1
    line = json.loads(stream.stdout.strip().splitlines()[-1])
    assert line["type"] == FIXTURE["prompt_failure"]["stream_json_type"]
    assert line["error"]["code"] == FIXTURE["prompt_failure"]["no_model_code"]


def test_stream_json_carries_the_failover_note(monkeypatch):
    from webagents.agents.core.base_agent import BaseAgent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    monkeypatch.setattr("webagents.cli.model_access._client_installed", lambda provider: True)

    async def canned(self, messages, **kwargs):
        yield {"webagents_note": "openai/gpt-4o-mini did not answer; trying anthropic/claude-haiku-4-5"}
        yield {"choices": [{"delta": {"content": "ok"}}]}

    monkeypatch.setattr(BaseAgent, "run_streaming", canned)
    Path("AGENT.md").write_text("---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n")
    result = runner.invoke(app, ["-p", "hi", "--output-format", "stream-json"])
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert lines[0] == {"type": "note", "note": "openai/gpt-4o-mini did not answer; trying anthropic/claude-haiku-4-5"}


def test_the_server_prints_no_emoji(capsys):
    from fastapi.testclient import TestClient

    from webagents.agents.core.base_agent import BaseAgent
    from webagents.server.core.app import create_server

    server = create_server(agents=[BaseAgent(name="plain", instructions="x")], quiet=False)
    with TestClient(server.app):
        pass
    printed = capsys.readouterr().out
    assert "server ready" in printed
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", printed)
