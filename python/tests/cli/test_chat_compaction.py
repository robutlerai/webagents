"""
Compaction in the chat (2026-09-29), the same in the TypeScript chat
(`typescript/tests/unit/cli/chat-compaction.test.ts`).

The owner asked for the best strategy for compaction, with settings, a command
and an API. Pinned here, with the agent's model replaced by a stub: `/compact`
makes a summary of everything before the latest exchange, the conversation the
model is sent becomes that, and the whole conversation stays in the file
(`transcript`); before a message is sent past `compaction.at` the chat compacts
first and says so; `/context` says how full the context is and when it
compacts; the footer shows it past half; and a run's safety stop leaves the
turn in progress whole.
"""

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.agents.core.context_compaction import WORDS, CompactionPolicy
from webagents.cli.repl.chat_words import CHAT_WORDS, fill
from webagents.cli.repl.session import WebAgentsSession
from webagents.cli.sessions import list_sessions, load_session, sessions_dir

KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY")


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    for var in KEY_VARS[1:] + ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    (project / "AGENT.md").write_text("---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nHelp.\n")
    yield project


def _chat() -> WebAgentsSession:
    session = WebAgentsSession(agent_path=Path("AGENT.md"), interactive=True)
    session.console = Console(file=StringIO(), width=120, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    seen = []

    async def summarize(transcript: str, instructions: str) -> str:
        seen.append(instructions)
        return f"[stub summary of {len(transcript.splitlines())} lines]"

    session.built.agent._summarize_for_compaction = summarize
    session.summaries_asked = seen  # type: ignore[attr-defined]
    return session


def _said(session: WebAgentsSession, action) -> str:
    before = len(session.console.export_text(clear=False))
    result = action()
    if asyncio.iscoroutine(result):
        asyncio.run(result)
    return session.console.export_text(clear=False)[before:]


def _talk(n: int, size: int = 0):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"question {i} " + "x" * size})
        out.append({"role": "assistant", "content": f"answer {i} " + "y" * size})
    return out


def test_compact_summarizes_and_the_file_keeps_the_whole_conversation(newcomer):
    chat = _chat()
    talk = _talk(3)
    chat.messages = list(talk)
    out = _said(chat, lambda: chat.handle_input("/compact the budget"))
    assert CHAT_WORDS["compacting"] in out
    assert "Compacted the conversation: 4 earlier messages became a summary, the last 2 stay as they were." in out
    assert chat.messages[0] == {"role": "system", "content": WORDS["summaryPrefix"] + "[stub summary of 4 lines]"}
    assert chat.messages[1:] == talk[4:]
    assert "Pay particular attention to: the budget" in chat.summaries_asked[0]
    saved = load_session(sessions_dir(newcomer, "helper"), chat.session_id)
    assert saved["messages"] == chat.messages and saved["transcript"] == talk


def test_compact_with_nothing_to_compact(newcomer):
    chat = _chat()
    assert CHAT_WORDS["compactEmpty"] in _said(chat, lambda: chat.handle_input("/compact"))
    chat.messages = _talk(1)
    out = _said(chat, lambda: chat.handle_input("/compact"))
    assert "Nothing to compact yet" in out and chat.transcript is None


def test_before_a_message_past_the_threshold_the_chat_compacts_first(newcomer):
    chat = _chat()
    chat.built.agent.compaction_policy = CompactionPolicy(at=200, keep=40, hard=10000)
    chat.messages = _talk(4, size=200)
    out = _said(chat, lambda: chat._compact_before_sending("and now?"))
    assert "Compacted the conversation" in out
    assert chat.messages[0]["content"].startswith(WORDS["summaryPrefix"])
    assert chat.transcript == _talk(4, size=200)
    # Under the threshold now: the next message is sent as it is.
    kept = list(chat.messages)
    assert _said(chat, lambda: chat._compact_before_sending("thanks")).strip() == ""
    assert chat.messages == kept


def test_off_means_off(newcomer):
    chat = _chat()
    chat.built.agent.compaction_policy = CompactionPolicy(auto=False, at=200, keep=40, hard=10000)
    chat.messages = _talk(4, size=200)
    assert _said(chat, lambda: chat._compact_before_sending("and now?")).strip() == ""
    assert chat.transcript is None


def test_context_says_how_full_it_is_and_the_footer_shows_it_past_half(newcomer):
    chat = _chat()
    chat.messages = _talk(2)
    out = _said(chat, lambda: chat.handle_input("/context"))
    assert fill("contextHeading", model="openai/gpt-4o-mini", window="128k") in out
    assert "The conversation: about" in out and "in 4 messages." in out
    assert fill("contextAuto", at=80, tokens="102k") in out
    assert not any(p.startswith("context ") for p in chat._footer_parts())
    chat.built.agent.compaction_policy = CompactionPolicy(window=40, at=0.8, hard=0.95)
    assert any(p.startswith("context ") and p.endswith("%") for p in chat._footer_parts())
    chat.built.agent.compaction_policy = CompactionPolicy(auto=False)
    assert CHAT_WORDS["contextAutoOff"] in _said(chat, lambda: chat.handle_input("/context"))
    assert "Usage: /context" in _said(chat, lambda: chat.handle_input("/context now"))


def test_a_resumed_conversation_keeps_its_transcript(newcomer):
    chat = _chat()
    chat.messages = _talk(3)
    _said(chat, lambda: chat.handle_input("/compact"))
    compacted, whole = list(chat.messages), list(chat.transcript)
    _said(chat, lambda: chat.handle_input("/new"))
    assert chat.transcript is None
    out = _said(chat, lambda: chat.handle_input("/resume 1"))
    assert chat.messages == compacted and chat.transcript == whole
    # "N messages" is the whole conversation's words, not the compacted view's.
    assert "(6 messages)" in out


def test_a_compacted_conversation_is_listed_whole(newcomer):
    chat = _chat()
    chat.messages = _talk(3)
    _said(chat, lambda: chat.handle_input("/compact"))
    [kept] = list_sessions(sessions_dir(newcomer, "helper"))
    assert (kept.message_count, kept.preview) == (6, "question 0")


def test_the_runs_safety_stop_leaves_the_turn_in_progress_whole(newcomer):
    from webagents.server.context.context_vars import create_context

    chat = _chat()
    agent = chat.built.agent
    agent.compaction_policy = CompactionPolicy(at=100, keep=20, hard=300)
    turn = [
        {"role": "user", "content": "read the big file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "z" * 900},
    ]
    messages = [{"role": "system", "content": "You help."}] + _talk(3, size=300) + turn
    context = create_context(messages=messages)
    asyncio.run(agent._compact_if_full(context))
    assert context.messages[0] == {"role": "system", "content": "You help."}
    assert context.messages[1]["content"].startswith(WORDS["summaryPrefix"])
    assert context.messages[-3:] == turn
    assert agent.last_compaction is not None and agent.last_compaction.stage == "summarized"
    json.dumps(context.messages)
