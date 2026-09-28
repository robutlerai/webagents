"""
One retry when the platform goes away before any output (2026-09-28).

A deploy or a restart drains the /llm socket with 1001 (going away); the chat
turn failed with the raw close words even when nothing had been said yet. The
proxy now sends the request once more in that case, and only in that case.
"""

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from webagents.agents.skills.core.llm.proxy import skill as proxy_mod
from webagents.agents.skills.core.llm.proxy.skill import LLMProxySkill


def _drained():
    return ConnectionClosedError(Close(1001, "server draining"), Close(1001, "server draining"), True)


def _skill(monkeypatch, plan):
    s = LLMProxySkill.__new__(LLMProxySkill)
    s.model = "auto/balanced"
    s.logger = type("L", (), {"warning": lambda *a, **k: None})()
    calls = []

    async def once(messages, model, tools, may_retry=False, **kwargs):
        calls.append(may_retry)
        step = plan[len(calls) - 1]
        for chunk in step.get("chunks", []):
            yield chunk
        if step.get("raise"):
            raise step["raise"]

    monkeypatch.setattr(s, "_stream_once", once, raising=False)
    monkeypatch.setattr(proxy_mod, "SERVER_AWAY_RETRY_DELAY_S", 0)
    return s, calls


async def _collect(s):
    return [c async for c in s.chat_completion_stream([{"role": "user", "content": "hi"}])]


@pytest.mark.asyncio
async def test_a_drain_before_any_output_is_sent_once_more(monkeypatch):
    s, calls = _skill(monkeypatch, [{"raise": _drained()}, {"chunks": [{"choices": [{"delta": {"content": "YO"}}]}]}])
    assert await _collect(s) == [{"choices": [{"delta": {"content": "YO"}}]}]
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_a_drain_after_output_is_not_retried(monkeypatch):
    s, calls = _skill(monkeypatch, [{"chunks": [{"choices": [{"delta": {"content": "Y"}}]}], "raise": _drained()}])
    with pytest.raises(ConnectionClosedError):
        await _collect(s)
    assert calls == [True]


@pytest.mark.asyncio
async def test_a_second_drain_is_the_error(monkeypatch):
    s, calls = _skill(monkeypatch, [{"raise": _drained()}, {"raise": _drained()}])
    with pytest.raises(ConnectionClosedError):
        await _collect(s)
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_other_closes_are_not_retried(monkeypatch):
    s, calls = _skill(monkeypatch, [{"raise": ConnectionClosedError(Close(1011, "internal error"), None, None)}])
    with pytest.raises(ConnectionClosedError):
        await _collect(s)
    assert calls == [True]
