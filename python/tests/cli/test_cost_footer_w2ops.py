"""
The run's cost in credits, in the chat footer next to the tokens (2026-09-26,
gap-closure plan item 2.4, lane w2-ops), against the shared fixture
`tests/fixtures/w2ops/cost.json`, which the TypeScript suite reads too
(`cost-footer-w2ops.test.ts`): the price table, how a model finds its row,
the estimate, the number's format, and what the footer, /status and the
goodbye line show.

The chat is driven with its counters set directly (a turn's usage is what
`_turn` adds; the footer reads the totals), under a throwaway HOME with the
FILE secrets backend, no key and no sign-in.
"""

from __future__ import annotations

import asyncio
import json
import re
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.agents.skills.core.llm.pricing import (
    NO_COST,
    PROVIDER_LIST_PRICES,
    add_turn_cost,
    cost_words,
    estimate_cost_credits,
    format_credits,
    price_row_for,
    reported_cost_credits,
)
from webagents.cli.repl.render import TurnRenderer, Usage, events_from_chunk
from webagents.cli.repl.session import WebAgentsSession

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "w2ops" / "cost.json").read_text())
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY")


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in KEY_VARS + ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    yield project


class TestThePriceTable:
    def test_equals_the_fixture_row_for_row(self):
        assert {k: list(v) for k, v in PROVIDER_LIST_PRICES.items()} == FIXTURE["list_prices"]

    @pytest.mark.parametrize("case", FIXTURE["lookup"], ids=lambda c: c["model"])
    def test_finds_the_row(self, case):
        assert price_row_for(case["model"]) == case["row"]

    @pytest.mark.parametrize("case", FIXTURE["estimates"], ids=lambda c: c["model"])
    def test_estimates(self, case):
        estimate = estimate_cost_credits(case["model"], case["input_tokens"], case["output_tokens"])
        if case["credits"] is None:
            assert estimate is None
        else:
            assert estimate == pytest.approx(case["credits"], abs=1e-12)


class TestTheNumber:
    @pytest.mark.parametrize("case", FIXTURE["format"]["credits"], ids=lambda c: c["text"])
    def test_writes(self, case):
        assert format_credits(case["value"]) == case["text"]

    def test_names_credits_as_credits_with_a_tilde_for_an_estimate(self):
        assert cost_words(0.0042, False) == FIXTURE["format"]["words"]["reported"].replace("{credits}", "0.0042")
        assert cost_words(0.0042, True) == FIXTURE["format"]["words"]["estimated"].replace("{credits}", "0.0042")

    def test_reads_a_platform_reported_cost_in_credits_and_nothing_in_another_currency(self):
        assert reported_cost_credits({"cost": {"input_cost": 0.001, "output_cost": 0.0032, "total_cost": 0.0042, "currency": "credits"}}) == 0.0042
        assert reported_cost_credits({"cost": {"input_cost": 1, "output_cost": 1, "total_cost": 2, "currency": "USD"}}) is None
        assert reported_cost_credits({}) is None
        assert reported_cost_credits(None) is None

    def test_the_renderer_carries_a_reported_cost_and_none_when_the_platform_sent_none(self):
        console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None)
        renderer = TurnRenderer(console)
        for event in events_from_chunk({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": {"total_cost": 0.01, "currency": "credits"}}}):
            renderer.feed(event)
        for event in events_from_chunk({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}):
            renderer.feed(event)
        assert renderer.usage == Usage(2, 2, 0.01)
        quiet = TurnRenderer(console)
        for event in events_from_chunk({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}):
            quiet.feed(event)
        assert quiet.usage == Usage(1, 1, None)


def _chat():
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=200, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    return session


class _Access:
    def __init__(self, kind: str, model: str) -> None:
        self.kind, self.model, self.provider, self.reason = kind, model, None, ""


def _chat_after(case, session) -> None:
    """The chat's counters after one turn of the case, as `_turn` sets them."""
    session.built.access = _Access("proxy" if case["reported"] is not None else "direct", case["model"])
    session.built.model_label = case["model"]
    session.input_tokens = case["input_tokens"]
    session.output_tokens = case["output_tokens"]
    session.session_tokens = case["input_tokens"] + case["output_tokens"]
    session.turns = 1
    session.messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    session.cost = add_turn_cost(NO_COST, case["model"], case["input_tokens"], case["output_tokens"], case["reported"])
    session.session_cost = add_turn_cost(NO_COST, case["model"], case["input_tokens"], case["output_tokens"], case["reported"])


def _printed(session) -> str:
    return session.console.export_text(clear=True)


class TestTheFooter:
    @pytest.mark.parametrize("case", FIXTURE["footer"], ids=lambda c: c["case"])
    def test_footer_status_and_goodbye(self, case):
        session = _chat()
        _chat_after(case, session)
        # The footer: agent, model, the tokens and cost, the folder.
        parts = session._footer_parts()
        assert ", ".join(parts[2:-1]) == case["footer"]
        # /status: the Conversation row.
        _printed(session)
        asyncio.run(session.handle_input("/status"))
        status = next(line for line in _printed(session).split("\n") if "Conversation" in line)
        assert re.sub(r"^\s*Conversation\s+", "", status).strip() == case["status"]
        # The goodbye line: replies, tokens, cost, duration.
        session._goodbye()
        goodbye = _printed(session).strip()
        assert " · ".join(goodbye.split(" · ")[1:-1]) == case["goodbye"]

    def test_a_new_conversation_starts_over_and_a_resumed_one_brings_its_cost_back(self):
        session = _chat()
        case = FIXTURE["footer"][0]
        _chat_after(case, session)
        session.save_conversation()
        session_id = session.session_id
        session.start_new_conversation(False)
        assert session.cost == NO_COST
        asyncio.run(session.cmd_resume(session_id[:8]))
        assert session.cost.known is True
        assert session.cost.estimated is True
        assert session.cost.credits == pytest.approx(0.0006, abs=1e-12)
