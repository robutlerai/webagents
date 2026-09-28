"""
The charge the platform reports, as the Python chat reads it (B4,
2026-09-28), against the shared fixture `w2ops/final_sdk_total_cost.json` the
TypeScript suite reads too. The platform never sent the charge, so the chat
showed a list-price estimate for Robutler's models; the final-billing lane
makes it send `usage.total_cost`, and the chat shows that number as it is.
"""

import asyncio
import json
from pathlib import Path

import pytest

from webagents.agents.skills.core.llm.pricing import NO_COST, add_turn_cost, cost_words, reported_cost_credits

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "w2ops" / "final_sdk_total_cost.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_the_reader(case):
    assert reported_cost_credits(case["usage"]) == case["reported"]


def test_the_footer_shows_the_charge_with_no_tilde():
    f = FIXTURE["footer"]
    cost = add_turn_cost(NO_COST, f["model"], f["usage"]["input_tokens"], f["usage"]["output_tokens"], reported_cost_credits(f["usage"]))
    assert cost.estimated is False and cost_words(cost.credits, cost.estimated) == f["words"]


class _Socket:
    def __init__(self, events):
        self.sent = []
        self._events = [json.dumps(e) for e in events]

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self._events:
            await asyncio.sleep(60)
        return self._events.pop(0)

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def test_the_proxy_skill_passes_the_charge_to_the_chat(monkeypatch):
    from webagents.agents.skills.core.llm.proxy import skill as proxy_module
    from webagents.cli.repl.render import Usage, events_from_chunk

    usage = FIXTURE["cases"][0]["usage"]
    ws = _Socket([
        {"type": "session.created", "session_id": "s"},
        {"type": "response.created", "response_id": "r"},
        {"type": "response.delta", "response_id": "r", "delta": {"type": "text", "text": "ok"}},
        {"type": "response.done", "response_id": "r", "response": {"output": [], "usage": usage}},
    ])
    monkeypatch.setattr(proxy_module.websockets.client, "connect", lambda *a, **k: ws)
    skill = proxy_module.LLMProxySkill({"model": "auto/balanced", "payment_token": "tok"})

    async def run():
        return [chunk async for chunk in skill.chat_completion_stream([{"role": "user", "content": "hi"}])]

    chunks = asyncio.run(run())
    final = [c for c in chunks if c.get("usage")][-1]
    assert final["usage"]["total_cost"] == usage["total_cost"]
    shown = [e for c in chunks for e in events_from_chunk(c) if isinstance(e, Usage)]
    assert shown[-1].cost_credits == FIXTURE["cases"][0]["reported"]
