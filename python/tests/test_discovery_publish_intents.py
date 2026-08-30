"""`DiscoverySkill.publish_intents_tool` stays ADDITIVE over a REPLACING API.

`POST /api/discovery/announce` replaces the caller's whole intent set in one
transaction. The tool it replaced (`/api/intents/create`) was additive, so
publishing in two batches would have silently left only the second batch
listed. The skill therefore re-sends the union it has published, and
`replace=True` is the explicit opt-in to the platform's raw semantics.
"""

import json

import pytest

from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeClient:
    """Records every announce payload; the last status is configurable."""

    calls: list = []
    status_code = 200

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        FakeClient.calls.append(json)
        return FakeResponse(FakeClient.status_code)


@pytest.fixture
def skill(monkeypatch):
    import httpx

    FakeClient.calls = []
    FakeClient.status_code = 200
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("WEBAGENTS_PUBLIC_URL", "https://agent.example.com/agents/mini")
    s = DiscoverySkill({"robutler_api_key": "rok_test", "robutler_api_url": "http://portal.test"})
    s.robutler_api_key = "rok_test"
    s.robutler_api_url = "http://portal.test"
    # `logger` is normally set in initialize(); these tests drive the tool
    # directly, so give it one.
    import logging

    s.logger = logging.getLogger("test.discovery")
    return s


def announced(index):
    return sorted(i["intent"] for i in FakeClient.calls[index]["intents"])


@pytest.mark.asyncio
async def test_second_call_does_not_drop_the_first_batch(skill):
    first = await skill.publish_intents_tool(intents=["translate"], description="d1")
    assert first["success"] is True

    second = await skill.publish_intents_tool(intents=["summarize"], description="d2")
    assert second["success"] is True

    assert announced(0) == ["translate"]
    # The second wire call carries BOTH — the endpoint replaces, so anything
    # left out of this payload is de-listed.
    assert announced(1) == ["summarize", "translate"]
    assert sorted(second["published_intents"]) == ["summarize", "translate"]
    assert second["newly_published"] == ["summarize"]


@pytest.mark.asyncio
async def test_replace_true_drops_the_earlier_intents(skill):
    await skill.publish_intents_tool(intents=["translate"], description="d1")
    result = await skill.publish_intents_tool(
        intents=["summarize"], description="d2", replace=True
    )
    assert announced(1) == ["summarize"]
    assert result["replaced"] is True


@pytest.mark.asyncio
async def test_empty_list_with_replace_delists(skill):
    await skill.publish_intents_tool(intents=["translate"], description="d1")
    await skill.publish_intents_tool(intents=[], description="", replace=True)
    assert FakeClient.calls[1]["intents"] == []


@pytest.mark.asyncio
async def test_a_failed_announce_does_not_poison_the_accumulator(skill):
    await skill.publish_intents_tool(intents=["translate"], description="d1")
    FakeClient.status_code = 500
    failed = await skill.publish_intents_tool(intents=["summarize"], description="d2")
    assert failed["success"] is False

    FakeClient.status_code = 200
    await skill.publish_intents_tool(intents=["classify"], description="d3")
    # "summarize" never landed, so it must not reappear in a later payload.
    assert announced(2) == ["classify", "translate"]
