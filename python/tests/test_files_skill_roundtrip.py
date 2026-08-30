"""RobutlerFilesSkill: an upload must be visible to the listing.

The defect this pins is invisible in isolation — both halves "work", they
just use different principals: `store_file_from_*` POSTs
`/api/content/upload`, which files the row under the BEARER'S SUBJECT, while
the listing GETs `/api/agents/{id}/content`, whose `{id}` must be a principal
with a content link. List under the agent id while uploading under the key's
subject and the listing is empty forever.

The stub portal below models exactly that rule (`principal -> [content]`) and
nothing else, so the test fails the moment the two halves diverge again.

It also pins the degraded-not-fatal contract for a missing key: `initialize()`
must not raise (skills initialize lazily on the agent's FIRST run, so raising
kills the whole request), while the tools that need the credential must fail
loudly.
"""

import json

import pytest

from webagents.agents.skills.robutler.storage.files.skill import (
    MISSING_KEY_MESSAGE,
    RobutlerFilesSkill,
)


def api_key_for(subject: str, agent_id: str) -> str:
    """A per-agent api key in the shape the portal mints: `sub` is the OWNER,
    `agent_id` is a claim (lib/auth/session.ts signApiKeyJwt sets the subject
    to the token owner's user id)."""
    import base64

    def seg(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg({'sub': subject, 'agent_id': agent_id})}.x"


class StubPortal:
    """`/api/content/upload` and `/api/agents/{id}/content`, with the portal's
    real principal rule and nothing else."""

    def __init__(self):
        self.by_principal: dict = {}
        self.next_id = 0

    def upload(self, subject, filename):
        self.next_id += 1
        record = {
            "id": f"content-{self.next_id}",
            "displayName": filename,
            "url": f"/api/content/content-{self.next_id}",
            "size": 3,
            "mimeType": "text/plain",
            "visibility": "private",
        }
        # saveUserContent(<bearer subject>, ...) — the link row is the
        # subject's, never the agent's.
        self.by_principal.setdefault(subject, []).append(record)
        return record

    def list(self, principal):
        return {"items": self.by_principal.get(principal, [])}


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def text(self):
        return json.dumps(self._payload)

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    """Routes the two calls the skill makes; asserts a bearer is present."""

    def __init__(self, portal):
        self.portal = portal

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, data=None, headers=None):
        assert url.endswith("/api/content/upload"), url
        subject = _subject_of(headers["Authorization"])
        filename = data._fields[0][0]["filename"]
        return FakeResponse(200, self.portal.upload(subject, filename))

    def get(self, url, headers=None):
        principal = url.rsplit("/api/agents/", 1)[1].split("/")[0]
        return FakeResponse(200, self.portal.list(principal))


def _subject_of(auth_header: str) -> str:
    import base64

    token = auth_header.split(" ", 1)[1]
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]


@pytest.fixture
def patched_aiohttp(monkeypatch):
    portal = StubPortal()
    import webagents.agents.skills.robutler.storage.files.skill as files_skill

    monkeypatch.setattr(
        files_skill.aiohttp, "ClientSession", lambda *a, **k: FakeSession(portal)
    )
    return portal


class DummyAgent:
    name = "mini"


async def call(coro):
    """`list_files` carries @pricing, which wraps the return as
    (result, usage) when a fixed price is configured. Unwrap it."""
    result = await coro
    if isinstance(result, tuple):
        result = result[0]
    return json.loads(result)


@pytest.mark.asyncio
async def test_stored_file_id_round_trips_through_the_listing(patched_aiohttp):
    key = api_key_for(subject="owner-1", agent_id="agent-9")
    skill = RobutlerFilesSkill({"api_key": key, "portal_url": "http://portal.test"})
    await skill.initialize(DummyAgent())

    stored = await call(
        skill.store_file_from_base64(filename="note.txt", base64_data="aGk=")
    )
    assert stored["success"] is True
    stored_id = stored["id"]

    listed = await call(skill.list_files())
    assert listed["success"] is True
    assert [f["id"] for f in listed["files"]] == [stored_id]


@pytest.mark.asyncio
async def test_agent_subject_credential_also_round_trips(patched_aiohttp):
    """A daemon token whose subject IS the agent must work unchanged."""
    key = api_key_for(subject="agent-9", agent_id="agent-9")
    skill = RobutlerFilesSkill({"api_key": key, "portal_url": "http://portal.test"})
    await skill.initialize(DummyAgent())

    stored = await call(
        skill.store_file_from_base64(filename="a.txt", base64_data="aGk=")
    )
    listed = await call(skill.list_files())
    assert [f["id"] for f in listed["files"]] == [stored["id"]]


@pytest.mark.asyncio
async def test_missing_key_degrades_the_skill_and_never_the_agent(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_API_KEY", raising=False)
    monkeypatch.delenv("ROBUTLER_API_KEY", raising=False)
    skill = RobutlerFilesSkill({"portal_url": "http://portal.test"})

    # Must NOT raise: skills initialize on the agent's first run.
    await skill.initialize(DummyAgent())

    result = await call(
        skill.store_file_from_base64(filename="a.txt", base64_data="aGk=")
    )
    assert result["success"] is False
    assert MISSING_KEY_MESSAGE in result["error"]

    listed = await call(skill.list_files())
    assert listed["success"] is False
    assert MISSING_KEY_MESSAGE in listed["error"]
