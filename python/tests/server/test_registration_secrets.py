"""
``register_with_platform`` persisting its bearer, and reading it back.

WHY THIS MATTERS MORE THAN A CACHE NORMALLY WOULD: the token the platform
answers with is good for seven days, carries ``agents:own``, and is minted
with no ``jti``, so nothing revokes it (platform security log S-037). An agent
that re-registers on every restart mints another one of those each time and
leaves the previous ones live until they lapse. Reading one back mints none,
so the reuse path is a security property rather than a speed-up.

The other half these pin is that the store stays OPTIONAL. Registration must
survive a missing store, a store that throws on read, and a store that throws
on write, because an agent on a box with no keystore still has to come up.
"""

import base64
import json
import time

import pytest

from webagents.server.core.registration import (
    PLATFORM_TOKEN_SECRET,
    register_with_platform,
)

PLATFORM = "https://platform.example.com"
ISSUER = "https://agent.example.com"


def token_expiring_in(seconds: int) -> str:
    """A structurally valid JWT with a chosen ``exp``. Signature is a placeholder."""

    def part(value):
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    header = part({"alg": "RS256"})
    payload = part({"exp": int(time.time()) + seconds, "scopes": ["agents:own"]})
    return f"{header}.{payload}.placeholder-signature-not-verified-locally"


STORED = token_expiring_in(6 * 24 * 3600)
MINTED = token_expiring_in(7 * 24 * 3600)


class MemoryStore:
    """An in-memory store of the shape registration accepts."""

    def __init__(self, seed=None):
        self.items = dict(seed or {})
        self.get_calls = []
        self.set_calls = []
        self.deleted = []
        self.raise_on_get = None
        self.raise_on_set = None

    def get(self, name):
        self.get_calls.append(name)
        if self.raise_on_get:
            raise self.raise_on_get
        return self.items.get(name)

    def set(self, name, value):
        self.set_calls.append(name)
        if self.raise_on_set:
            raise self.raise_on_set
        self.items[name] = value
        return "keystore"

    def delete(self, name):
        self.deleted.append(name)
        return self.items.pop(name, None) is not None


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeClient:
    """Stands in for ``httpx.AsyncClient`` as an async context manager."""

    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self._calls.append(url)
        return self._response


@pytest.fixture
def platform_call(monkeypatch, tmp_path):
    """Patch the network and the key material so nothing leaves the machine."""
    import httpx

    calls = []
    state = {
        "response": FakeResponse(
            body={
                "access_token": MINTED,
                "user_id": "u-1",
                "username": "example.com.agents.demo",
            }
        )
    }

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: FakeClient(state["response"], calls)
    )
    monkeypatch.setenv("WEBAGENTS_PUBLIC_URL", ISSUER)
    monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path / "keys"))

    from webagents.crypto.jwks import JWKSManager

    monkeypatch.setattr(
        JWKSManager,
        "mint_aoauth_token",
        lambda self, *a, **kw: "dummy-aoauth-assertion-not-a-real-token",
    )
    monkeypatch.setattr(JWKSManager, "ensure_keys", lambda self, *a, **kw: None)

    return {"calls": calls, "state": state}


async def register(secrets=None, **kwargs):
    return await register_with_platform(
        "demo", public_url=ISSUER, platform_url=PLATFORM, secrets=secrets, **kwargs
    )


@pytest.mark.asyncio
async def test_reuses_a_stored_bearer_and_calls_nothing(platform_call):
    secrets = MemoryStore({PLATFORM_TOKEN_SECRET: STORED})
    result = await register(secrets)

    assert result["ok"] is True
    assert result["status"] == 0
    assert result["reused"] is True
    assert result["access_token"] == STORED
    assert platform_call["calls"] == []
    assert secrets.set_calls == []


@pytest.mark.asyncio
async def test_persists_a_freshly_minted_bearer(platform_call):
    secrets = MemoryStore()
    result = await register(secrets)

    assert result["ok"] is True
    assert result["status"] == 200
    assert result["reused"] is False
    assert result["stored"] == "saved"
    assert result["access_token"] == MINTED
    assert secrets.items[PLATFORM_TOKEN_SECRET] == MINTED
    assert len(platform_call["calls"]) == 1


@pytest.mark.asyncio
async def test_honours_a_custom_secret_name(platform_call):
    secrets = MemoryStore()
    await register(secrets, token_name="staging_platform_token")
    assert "staging_platform_token" in secrets.items
    assert PLATFORM_TOKEN_SECRET not in secrets.items


@pytest.mark.asyncio
async def test_lapsed_bearer_is_dropped_and_replaced(platform_call):
    secrets = MemoryStore({PLATFORM_TOKEN_SECRET: token_expiring_in(-60)})
    result = await register(secrets)

    assert secrets.deleted == [PLATFORM_TOKEN_SECRET]
    assert result["reused"] is False
    assert result["access_token"] == MINTED
    assert len(platform_call["calls"]) == 1


@pytest.mark.asyncio
async def test_bearer_inside_the_expiry_skew_is_treated_as_spent(platform_call):
    # 60 seconds left, default skew 300: usable once, then dead mid-run.
    secrets = MemoryStore({PLATFORM_TOKEN_SECRET: token_expiring_in(60)})
    result = await register(secrets)
    assert result["reused"] is False
    assert len(platform_call["calls"]) == 1


@pytest.mark.asyncio
async def test_bearer_with_no_readable_exp_is_not_assumed_eternal(platform_call):
    secrets = MemoryStore({PLATFORM_TOKEN_SECRET: "not-a-jwt"})
    result = await register(secrets)
    assert result["reused"] is False
    assert secrets.deleted == [PLATFORM_TOKEN_SECRET]


@pytest.mark.asyncio
async def test_refresh_registers_again_despite_a_live_stored_bearer(platform_call):
    secrets = MemoryStore({PLATFORM_TOKEN_SECRET: STORED})
    result = await register(secrets, refresh=True)
    assert result["reused"] is False
    assert result["access_token"] == MINTED
    assert secrets.items[PLATFORM_TOKEN_SECRET] == MINTED


@pytest.mark.asyncio
async def test_registers_normally_with_no_store_at_all(platform_call):
    result = await register()
    assert result["ok"] is True
    assert result["status"] == 200
    assert result["stored"] == "not-requested"
    assert result["access_token"] == MINTED


@pytest.mark.asyncio
async def test_registers_anyway_when_the_store_cannot_be_read(platform_call):
    secrets = MemoryStore()
    secrets.raise_on_get = RuntimeError("keystore locked")

    result = await register(secrets)

    assert result["ok"] is True
    assert result["access_token"] == MINTED


@pytest.mark.asyncio
async def test_hands_back_a_working_bearer_when_the_store_cannot_be_written(
    platform_call,
):
    secrets = MemoryStore()
    secrets.raise_on_set = RuntimeError("disk full")

    result = await register(secrets)

    # Registration SUCCEEDED. Throwing away a live credential over a storage
    # failure would be the worse outcome, so it is reported instead.
    assert result["ok"] is True
    assert result["access_token"] == MINTED
    assert "disk full" in result["stored"]


@pytest.mark.asyncio
async def test_stores_nothing_when_registration_fails(platform_call):
    platform_call["state"]["response"] = FakeResponse(
        status_code=401, text="unauthorized"
    )
    secrets = MemoryStore()

    result = await register(secrets)

    assert result["ok"] is False
    assert result["status"] == 401
    assert secrets.set_calls == []
    assert secrets.items == {}


@pytest.mark.asyncio
async def test_the_real_store_satisfies_the_shape_registration_expects(
    platform_call, tmp_path
):
    """The duck type is only useful if the shipped store actually fits it."""
    from webagents.agents.skills.local.secrets import open_secret_store

    store = open_secret_store(
        namespace="reg-agent",
        secrets_dir=str(tmp_path),
        backend="file",
        quiet=True,
    )

    first = await register(store)
    assert first["reused"] is False
    assert first["stored"] == "saved"

    second = await register(store)
    assert second["reused"] is True
    assert second["access_token"] == MINTED
    # One network call across two registrations is the whole point.
    assert len(platform_call["calls"]) == 1
