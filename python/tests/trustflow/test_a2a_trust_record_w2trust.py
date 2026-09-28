"""
The TrustFlow record in the A2A card (plan item 2.7, 2026-09-26): an `a2a`
entry with `trust_record: true` fetches this agent's signed record from the
platform, as the agent, and serves it as the card extension both SDKs write;
the card signature covers it, and a peer verifies both (TypeScript
tests/unit/transport/a2a-trust-record-w2trust.test.ts).
"""

from __future__ import annotations

import json
from pathlib import Path

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.transport.a2a.card import verify_agent_card
from webagents.agents.skills.core.transport.a2a.skill import A2ATransportSkill
from webagents.crypto.identity import AgentSigningIdentity
from webagents.crypto.jwks import JWKSManager
from webagents.trustflow.trust_record import trust_record_from_card, verify_trust_record

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "trust" / "trustflow_record.json").read_text())
PRINCIPAL = "https://agents.example.com/agents/scout"
ANSWER = {"record": FIXTURE["records"]["valid"], "payload": FIXTURE["payload"], "jwks_url": f"{FIXTURE['issuer']}/.well-known/jwks.json"}


class StubLookup:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    async def record(self, agent):
        self.asked.append(agent)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def build(tmp_path, config, answer):
    skill = A2ATransportSkill(config)
    agent = BaseAgent(name="scout", instructions="Scout.", skills={"a2a": skill})
    agent.description = "Scout"
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_ed25519_key("scout")
    agent.signing_identity = AgentSigningIdentity(PRINCIPAL, manager)
    skill.trust_lookup = StubLookup(answer)
    return skill, agent, manager


async def test_carries_the_platform_record_as_the_extension_under_the_card_signature(tmp_path):
    skill, agent, manager = build(tmp_path, {"trust_record": True}, ANSWER)
    card = await skill.build_card(agent, None)
    assert card["capabilities"]["extensions"] == FIXTURE["extension"]["card_with_record"]["capabilities"]["extensions"]
    assert skill.trust_lookup.asked == [PRINCIPAL]
    keys = manager.get_jwks()["keys"]
    assert (await verify_agent_card(card, keys=keys)).ok
    record = trust_record_from_card(card)
    verified = await verify_trust_record(record, keys=FIXTURE["jwks"]["keys"], issuer=FIXTURE["issuer"], now=FIXTURE["now"], subject={"url": PRINCIPAL})
    assert verified.ok
    tampered = dict(card, capabilities=dict(card["capabilities"], extensions=[dict(card["capabilities"]["extensions"][0], params={"record": FIXTURE["records"]["tampered"]})]))
    assert (await verify_agent_card(tampered, keys=keys)).ok is False


async def test_holds_the_record_one_platform_call_for_many_cards(tmp_path, monkeypatch):
    # The clock is pinned to the fixture's `now` (2026-09-21 14:15Z). The skill holds
    # a record until an hour before its `exp`, and the fixture's valid record
    # expires 2026-09-28 14:13:20Z. Against the real clock this test began
    # failing at 13:13:20Z that day, with two platform calls instead of one:
    # a date trap, not a regression. The skill reads `time.time()` through the
    # module, so patching the module's function pins it for this test only.
    import time

    monkeypatch.setattr(time, "time", lambda: float(FIXTURE["now"]))
    skill, agent, _ = build(tmp_path, {"trust_record": True}, ANSWER)
    await skill.build_card(agent, None)
    await skill.build_card(agent, None)
    assert skill.trust_lookup.asked == [PRINCIPAL]


async def test_serves_the_card_without_the_record_when_the_platform_cannot_answer(tmp_path):
    skill, agent, _ = build(tmp_path, {"trust_record": True}, ConnectionError("unreachable"))
    card = await skill.build_card(agent, None)
    assert "extensions" not in card["capabilities"]
    await skill.build_card(agent, None)
    assert skill.trust_lookup.asked == [PRINCIPAL]


async def test_asks_the_platform_nothing_unless_the_entry_says_so(tmp_path):
    skill, agent, _ = build(tmp_path, {}, ANSWER)
    card = await skill.build_card(agent, None)
    assert "extensions" not in card["capabilities"]
    assert skill.trust_lookup.asked == []
    assert skill.settings["trust_record"] is False
