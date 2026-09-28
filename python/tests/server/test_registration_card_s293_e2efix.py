"""
S-293 (2026-09-26): `webagents serve` showed an agent's full instructions to
any caller, with no credential. The registration card put `agent.instructions`
into `description`, and the server's `GET /` listing, `GET /<name>` info and
the health answers carried the whole system prompt. They carry the file's
`description:` now, as the TypeScript card always did, pinned by the shared
fixture `tests/fixtures/registration/card.json`, which the TypeScript suite
reads too (`tests/unit/server/registration-card-s293-e2efix.test.ts`).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.server.core.app import WebAgentsServer
from webagents.server.core.registration import build_agent_card

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "registration" / "card.json").read_text())
AGENT = FIXTURE["agent"]
MARKER = FIXTURE["never_carries"]


def _agent() -> BaseAgent:
    agent = BaseAgent(name=AGENT["name"], instructions=AGENT["instructions"], skills={})
    agent.description = AGENT["description"]
    return agent


def test_the_card_carries_the_description_and_never_the_instructions():
    card = build_agent_card(_agent(), FIXTURE["principal"])
    for key, expected in FIXTURE["card"].items():
        assert card[key] == expected, key
    assert MARKER not in json.dumps(card)


def test_an_agent_with_no_description_gets_an_empty_one_not_its_instructions():
    agent = BaseAgent(name=AGENT["name"], instructions=AGENT["instructions"], skills={})
    card = build_agent_card(agent, FIXTURE["principal"])
    assert card["description"] == ""
    assert MARKER not in json.dumps(card)


def test_no_unauthenticated_route_carries_the_instructions(tmp_path):
    server = WebAgentsServer(
        agents=[_agent()],
        keys_dir=str(tmp_path / "keys"),
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
        enable_request_logging=False,
        heartbeat=False,
        quiet=True,
    )
    client = TestClient(server.app)
    routes = FIXTURE["python_routes"]
    field = routes["field"]

    listing = client.get(routes["listing"])
    assert listing.status_code == 200
    entry = next(a for a in listing.json()["agents"] if a["name"] == AGENT["name"])
    assert entry[field] == AGENT["description"]
    assert "instructions" not in entry

    info = client.get(routes["info"])
    assert info.status_code == 200
    assert info.json()[field] == AGENT["description"]
    assert "instructions" not in info.json()

    health = client.get(routes["health"])
    assert health.status_code == 200
    assert health.json()[field] == AGENT["description"]
    assert "instructions_preview" not in health.json()

    card = client.get(routes["card"])
    assert card.status_code == 200
    assert card.json()["description"] == AGENT["description"]

    for response in (listing, info, health, card):
        assert MARKER not in response.text
