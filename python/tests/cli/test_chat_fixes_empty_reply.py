"""
What the chat says about a turn that produced no answer (2026-09-27, the
chat-fixes lane), against the shared fixture
`tests/fixtures/cli/chat_fixes_empty_reply.json`, which the TypeScript suite
reads too (`chat-fixes-empty-reply.test.ts`):

  * the sentences (`present_empty_reply`), the same in both chats;
  * the agent's canned "content filtering" apology is gone, and nothing is
    added to the conversation in its place;
  * an empty turn prints the truthful line, is not counted as a reply by the
    goodbye line, and leaves no unanswered message in the conversation.
"""

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.failures import EMPTY_REPLY_HINT, present_empty_reply
from webagents.cli.repl.render import Finish, events_from_chunk
from webagents.cli.repl.session import WebAgentsSession

TABLE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_fixes_empty_reply.json").read_text())
AGENT = "---\nname: helper\nskills:\n  - openai\n---\nHelp.\n"


@pytest.mark.parametrize("case", TABLE["cases"], ids=[case["name"] for case in TABLE["cases"]])
def test_the_table_both_sdks_run(case):
    explained = present_empty_reply(
        case["reason"], blocked=case["blocked"], retried=case["retried"], thinking=case["thinking"], rounds=case.get("rounds"),
        tool=case.get("tool"),
    )
    assert (explained.headline, explained.hint) == (case["headline"], TABLE["hint"])


def test_the_hint_is_the_fixtures():
    assert EMPTY_REPLY_HINT == TABLE["hint"]


def test_the_last_chunk_says_why_the_provider_stopped():
    chunk = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "webagents_finish": {"reason": "MALFORMED_FUNCTION_CALL", "blocked": False, "retried": True},
    }
    assert events_from_chunk(chunk) == [Finish("MALFORMED_FUNCTION_CALL", False, True)]
    assert events_from_chunk({"choices": [{"index": 0, "delta": {"content": "hi"}}]})[1:] == []


def test_the_apology_is_gone_from_the_agent():
    source = Path(__file__).resolve().parents[2].joinpath("webagents", "agents", "core", "base_agent.py").read_text()
    assert "I apologize, but I encountered an issue generating a response" not in source
    assert 'error_message = "I apologize' not in source


@pytest.fixture
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    for var in ("ANTHROPIC_API_KEY", "WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return project


def _chat() -> WebAgentsSession:
    Path("AGENT.md").write_text(AGENT)
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    return session


def _stream(*chunks):
    async def run_streaming(_messages, **_kwargs):
        for chunk in chunks:
            yield chunk

    return run_streaming


def test_an_empty_turn_says_so_and_is_not_a_reply(newcomer, monkeypatch):
    session = _chat()
    monkeypatch.setattr(
        session.built.agent,
        "run_streaming",
        _stream({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 0},
            "webagents_finish": {"reason": "MALFORMED_FUNCTION_CALL", "blocked": False, "retried": True},
        }),
    )
    asyncio.run(session.handle_input("list the folder"))
    out = session.console.export_text(clear=False)
    assert "✗ The model returned no answer: the provider reported MALFORMED_FUNCTION_CALL, twice." in out
    assert "⎿  /model <provider/model> tries another model." in out
    assert "apologize" not in out
    assert session.turns == 0
    # The unanswered message leaves no trace.
    assert session.messages == []
    # The tokens were still spent.
    assert session.session_tokens == 120


def test_a_thinking_only_turn_says_so_too(newcomer, monkeypatch):
    session = _chat()
    monkeypatch.setattr(
        session.built.agent,
        "run_streaming",
        _stream(
            {"choices": [{"index": 0, "delta": {"content": "<think>Let me see.</think>"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "webagents_finish": {"reason": "STOP", "blocked": False, "retried": False}},
        ),
    )
    asyncio.run(session.handle_input("hi"))
    out = session.console.export_text(clear=False)
    assert "✗ The model returned no answer: it only produced thinking (the provider reported STOP)." in out
    assert session.turns == 0


def test_a_turn_that_answered_counts_and_prints_no_such_line(newcomer, monkeypatch):
    session = _chat()
    monkeypatch.setattr(
        session.built.agent,
        "run_streaming",
        _stream(
            {"choices": [{"index": 0, "delta": {"content": "Two files."}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "webagents_finish": {"reason": "STOP", "blocked": False, "retried": False}},
        ),
    )
    asyncio.run(session.handle_input("hi"))
    out = session.console.export_text(clear=False)
    assert "Two files." in out and "returned no answer" not in out
    assert session.turns == 1
    assert session.messages[-1] == {"role": "assistant", "content": "Two files."}


def _tool_round(index):
    """One model call that asks for a tool, finished as Gemini finishes it: STOP."""
    call = {"index": 0, "id": f"call-{index}", "type": "function", "function": {"name": "list_directory", "arguments": "{}"}}
    return (
        {"choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "webagents_finish": {"reason": "STOP", "blocked": False, "retried": False}},
        {"type": "tool_result", "id": f"call-{index}", "status": "success", "result": "README.md"},
    )


#: The agent's last chunk when the turn spent its tool rounds (`core/tool_budget.py`).
SPENT = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "webagents_finish": {"reason": "tool_round_limit", "blocked": False, "retried": False, "rounds": 5}}


def _answers(*replies):
    """`_ask` answering `replies` in turn, recording the questions."""
    asked = []

    async def ask(question):
        asked.append(question)
        return replies[len(asked) - 1] if len(asked) <= len(replies) else "n"

    return ask, asked


def test_a_turn_that_spent_its_tool_rounds_says_so_not_the_provider(newcomer, monkeypatch):
    # The owner's turn (2026-09-28): "nice", five rounds of tools, no answer,
    # and the chat said "the provider reported STOP". The last, tool-less
    # call brought nothing here either: the agent's cap is said, then asked.
    session = _chat()
    rounds = [chunk for index in range(1, 6) for chunk in _tool_round(index)]
    monkeypatch.setattr(session.built.agent, "run_streaming", _stream(*rounds, SPENT))
    ask, asked = _answers("n")
    monkeypatch.setattr(session, "_ask", ask)
    asyncio.run(session.handle_input("nice"))
    out = session.console.export_text(clear=False)
    assert "✗ The agent stopped after 5 tool rounds without an answer." in out
    assert "provider reported" not in out
    assert [q.strip() for q in asked] == ["Used 5 tool rounds. Keep going? [Y/n]"]
    assert session.turns == 0
    assert session.messages == []


def test_after_an_answer_at_the_cap_the_chat_asks_and_no_ends_the_turn(newcomer, monkeypatch):
    session = _chat()
    answer = {"choices": [{"index": 0, "delta": {"content": "Two folders so far."}, "finish_reason": None}]}
    monkeypatch.setattr(session.built.agent, "run_streaming", _stream(*_tool_round(1), answer, SPENT))
    ask, asked = _answers("n")
    monkeypatch.setattr(session, "_ask", ask)
    asyncio.run(session.handle_input("nice"))
    out = session.console.export_text(clear=False)
    assert "Two folders so far." in out
    assert "without an answer" not in out
    assert [q.strip() for q in asked] == ["Used 5 tool rounds. Keep going? [Y/n]"]
    assert session.messages[-1] == {"role": "assistant", "content": "Two folders so far."}


def test_yes_goes_on_from_the_conversation_with_a_fresh_budget(newcomer, monkeypatch):
    session = _chat()
    sent = []
    turns = iter([
        [*_tool_round(1), {"choices": [{"index": 0, "delta": {"content": "Two folders so far."}, "finish_reason": None}]}, SPENT],
        [{"choices": [{"index": 0, "delta": {"content": "All done."}, "finish_reason": None}]}],
    ])

    async def run_streaming(messages, **_kwargs):
        sent.append([dict(m) for m in messages])
        for chunk in next(turns):
            yield chunk

    monkeypatch.setattr(session.built.agent, "run_streaming", run_streaming)
    ask, asked = _answers("")  # enter: yes is the default
    monkeypatch.setattr(session, "_ask", ask)
    asyncio.run(session.handle_input("nice"))
    assert len(sent) == 2
    assert sent[1][-2:] == [{"role": "assistant", "content": "Two folders so far."}, {"role": "user", "content": "Keep going."}]
    assert "All done." in session.console.export_text(clear=False)
    assert len(asked) == 1


def test_a_loop_is_said_after_the_answer(newcomer, monkeypatch):
    session = _chat()
    looped = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
              "webagents_finish": {"reason": "tool_loop", "blocked": False, "retried": False, "rounds": 3, "tool": "list_directory"}}
    answer = {"choices": [{"index": 0, "delta": {"content": "The folder is empty."}, "finish_reason": None}]}
    monkeypatch.setattr(session.built.agent, "run_streaming", _stream(*_tool_round(1), answer, looped))
    ask, asked = _answers()
    monkeypatch.setattr(session, "_ask", ask)
    asyncio.run(session.handle_input("nice"))
    out = session.console.export_text(clear=False)
    assert "The folder is empty." in out
    # Joined: the console wraps the sentence at its width.
    assert "✗ The agent stopped early: it called list_directory 3 times in a row with the same arguments and got the same result each time." in " ".join(out.split())
    assert asked == []


# ---------------------------------------------------------------------------
# The budget is settable (2026-09-28): the file, the flag, `/rounds`, `/status`
# ---------------------------------------------------------------------------

BUDGET = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "agent_loop" / "tool_round_budget.json").read_text())


def test_the_agent_file_key_sets_the_budget(newcomer):
    Path("AGENT.md").write_text(BUDGET["agent_file"]["good"]["text"])
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    assert session.built.agent.max_tool_iterations == BUDGET["agent_file"]["good"]["rounds"]
    assert session.rounds_status() == "12 per turn (AGENT.md)"


def test_a_bad_agent_file_value_is_refused_with_the_sentence():
    from webagents.cli.loader.schema import AgentFormatError, AgentMetadata

    with pytest.raises(AgentFormatError) as refused:
        AgentMetadata(name="helper", max_tool_rounds=0)
    assert str(refused.value) == BUDGET["agent_file"]["bad"]["says"]


def test_the_flag_wins_over_the_file_and_the_default_is_fifty():
    from webagents.agents.core.tool_budget import effective_max_tool_rounds

    for case in BUDGET["precedence"]:
        env = {BUDGET["env"]: case["env"]} if case["env"] else {}
        assert effective_max_tool_rounds(case["file"], env) == (case["rounds"], case["source"])


def test_rounds_shows_sets_and_refuses(newcomer, monkeypatch):
    session = _chat()
    asyncio.run(session.handle_input("/rounds"))
    asyncio.run(session.handle_input("/rounds 12"))
    asyncio.run(session.handle_input("/rounds abc"))
    asyncio.run(session.handle_input("/status"))
    out = session.console.export_text(clear=False)
    assert "Tool rounds: 50 per turn (the default)." in out
    assert "Tool rounds set to 12 per turn, for this chat." in out
    assert '/rounds must be a whole number from 1 to 1000, not "abc".' in out
    assert "12 per turn (set in this chat)" in out
    assert session.built.agent.max_tool_iterations == 12
    # A rebuild keeps this chat's value.
    asyncio.run(session.initialize())
    assert session.built.agent.max_tool_iterations == 12


def test_rounds_save_keeps_it_in_the_agent_file(newcomer, monkeypatch):
    session = _chat()

    async def yes(_question):
        return True

    monkeypatch.setattr(session, "_confirm", yes)
    asyncio.run(session.handle_input("/rounds 7 --save"))
    import yaml

    front = yaml.safe_load(Path("AGENT.md").read_text().split("---")[1])
    assert str(front["max_tool_rounds"]) == "7"
    assert "Tool rounds 7 kept in AGENT.md." in session.console.export_text(clear=False)
    assert session.built.agent.max_tool_iterations == 7
    assert session.rounds_status() == "7 per turn (AGENT.md)"


def test_the_root_flag_is_checked_once_with_the_sentence(tmp_path):
    import os
    import subprocess
    import sys

    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop(BUDGET["env"], None)
    done = subprocess.run(
        [sys.executable, "-m", "webagents", "--max-tool-rounds", "0", "-p", "hi"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1
    assert "--max-tool-rounds must be a whole number from 1 to 1000, not \"0\"." in done.stderr
