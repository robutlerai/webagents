"""
The operator key at registration, and that it is inside the signature
(S-184 and the Python parity gap, both 2026-09-19).

Two defects, one file:

  * ``register_with_platform`` had no ``owner_api_key`` parameter, no
    ``ROBUTLER_API_KEY`` fallback, never sent ``X-Robutler-Owner-Key``,
    returned no ``owned`` field and printed no unclaimed warning, so every
    Python agent registered OWNERLESS (no payer, and not claimable later from
    a tunnel or bare-IP host) while the TypeScript SDK had all five.
  * S-184: the TypeScript SDK sent that header OUTSIDE the covered
    components, so whoever could rewrite headers between the agent and the
    platform's TLS edge could swap in a key of their own and register the
    agent under their account. Both SDKs now cover it whenever it is sent.

  * S-190: the registering request carries the operator's platform key, and
    a followed redirect carries it on: httpx strips only ``Authorization``
    when a redirect crosses origins. The TypeScript SDK sets
    ``redirect: 'error'``; this function relied on httpx's DEFAULT not to
    follow, which holds only for a client built without
    ``follow_redirects=True``. It now says ``follow_redirects=False`` on the
    request itself and reports a redirect as a failed registration.

The request is signed for real and captured by an ``httpx.MockTransport``, so
what these tests verify is the wire, not a mock's arguments.
"""

import logging

import httpx
import pytest
from cryptography.exceptions import InvalidSignature

from webagents.crypto.jwks import JWKSManager
from webagents.server.core.registration import OWNER_KEY_HEADER, register_with_platform

from ..crypto.covered_support import covered_of, verify_with_covered_headers

PLATFORM = "https://platform.example.com"
PUBLIC_URL = "https://agent.example.com"
TOKEN_URL = f"{PLATFORM}/api/auth/cli/token"


@pytest.fixture
def platform(monkeypatch, tmp_path):
    """A stub platform behind ``httpx.AsyncClient``: records each request and
    answers with ``state["body"]``."""
    seen = []
    state = {"body": {"access_token": "tok", "user_id": "u-1", "username": "example.com.demo", "owned": True}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=state["body"])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.delenv("ROBUTLER_API_KEY", raising=False)
    keys_dir = tmp_path / "keys"
    monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(keys_dir))
    manager = JWKSManager({"keys_dir": str(keys_dir)})
    thumbprint = manager.ensure_ed25519_key("demo")
    public = {thumbprint: manager.get_ed25519_signing_key().public_key()}
    return {"seen": seen, "state": state, "public": public}


async def register(**kwargs):
    return await register_with_platform("demo", public_url=PUBLIC_URL, platform_url=PLATFORM, **kwargs)


@pytest.fixture
def warnings_logged():
    """Records of the registration logger itself. Not `caplog`: the SDK's
    logging setup turns propagation off on the `webagents` logger once any
    earlier test has configured it, and `caplog` listens at the root."""
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    target = logging.getLogger("webagents.server.registration")
    handler = Collect(level=logging.WARNING)
    target.addHandler(handler)
    try:
        yield records
    finally:
        target.removeHandler(handler)


def test_names_the_header_it_covers():
    assert OWNER_KEY_HEADER == "X-Robutler-Owner-Key"


@pytest.mark.asyncio
async def test_the_owner_key_is_sent_covered_and_reported_as_owned(platform):
    result = await register(owner_api_key="  rok_operator  ")
    assert result["ok"] is True and result["owned"] is True

    (request,) = platform["seen"]
    assert str(request.url) == TOKEN_URL
    assert request.headers["x-robutler-owner-key"] == "rok_operator"
    assert covered_of(request.headers["signature-input"]) == [
        '"@method"',
        '"@authority"',
        '"@path"',
        '"@query"',
        '"content-digest"',
        '"x-robutler-owner-key"',
        '"signature-agent";key="sig1"',
    ]
    verified = verify_with_covered_headers(request.headers, "POST", TOKEN_URL, request.content, platform["public"])
    assert '\n"x-robutler-owner-key": rok_operator\n' in verified["sig1"]["base"]

    # Another operator's key swapped in between the agent and the platform no longer verifies.
    swapped = httpx.Headers(request.headers)
    swapped["x-robutler-owner-key"] = "rok_attacker"
    with pytest.raises(InvalidSignature):
        verify_with_covered_headers(swapped, "POST", TOKEN_URL, request.content, platform["public"])


@pytest.mark.asyncio
async def test_robutler_api_key_is_the_fallback_and_an_explicit_key_wins(platform, monkeypatch):
    monkeypatch.setenv("ROBUTLER_API_KEY", "rok_from_env")
    await register()
    await register(owner_api_key="rok_explicit")
    first, second = platform["seen"]
    assert first.headers["x-robutler-owner-key"] == "rok_from_env"
    assert second.headers["x-robutler-owner-key"] == "rok_explicit"
    for request in (first, second):
        assert '"x-robutler-owner-key"' in covered_of(request.headers["signature-input"])
        verify_with_covered_headers(request.headers, "POST", TOKEN_URL, request.content, platform["public"])


@pytest.mark.asyncio
async def test_no_owner_key_sends_no_header_covers_nothing_extra_and_warns_unclaimed(platform, warnings_logged):
    platform["state"]["body"] = {"access_token": "tok", "user_id": "u-1", "username": "example.com.demo", "owned": False}
    # A whitespace-only key is no key, as in the TypeScript SDK.
    result = await register(owner_api_key="   ")
    assert result["ok"] is True and result["owned"] is False

    (request,) = platform["seen"]
    assert "x-robutler-owner-key" not in request.headers
    assert covered_of(request.headers["signature-input"]) == [
        '"@method"',
        '"@authority"',
        '"@path"',
        '"@query"',
        '"content-digest"',
        '"signature-agent";key="sig1"',
    ]
    verify_with_covered_headers(request.headers, "POST", TOKEN_URL, request.content, platform["public"])
    assert any("UNCLAIMED" in r.getMessage() and "ROBUTLER_API_KEY" in r.getMessage() for r in warnings_logged)


@pytest.mark.asyncio
async def test_owned_is_none_when_the_platform_does_not_report_it_and_nothing_warns(platform, warnings_logged):
    platform["state"]["body"] = {"access_token": "tok", "user_id": "u-1", "username": "example.com.demo"}
    result = await register(owner_api_key="rok_operator")
    assert result["ok"] is True and result["owned"] is None
    assert not any("UNCLAIMED" in r.getMessage() for r in warnings_logged)


# ---------------------------------------------------------------------------
# S-190: the registering request never follows a redirect
# ---------------------------------------------------------------------------

COLLECTOR = "https://collector.example"


@pytest.fixture
def redirecting_platform(monkeypatch, tmp_path):
    """A platform that answers the registering request with a redirect to
    another origin, behind a client built with ``follow_redirects=True``: the
    worst case, a client that WOULD follow. The other origin answers like the
    token route, so a followed redirect would read as a registration."""
    seen = []
    state = {"status": 307}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "platform.example.com":
            # 307 keeps the method, the body and every header but `Authorization`.
            return httpx.Response(state["status"], headers={"Location": f"{COLLECTOR}/api/auth/cli/token"})
        return httpx.Response(200, json={"access_token": "tok-from-elsewhere", "user_id": "u-x", "username": "x", "owned": True})

    real_client = httpx.AsyncClient

    def following_client(*a, **kw):
        return real_client(*a, **{**kw, "transport": httpx.MockTransport(handler), "follow_redirects": True})

    monkeypatch.setattr(httpx, "AsyncClient", following_client)
    monkeypatch.delenv("ROBUTLER_API_KEY", raising=False)
    monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path / "keys"))
    return {"seen": seen, "state": state}


class _Store:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        return self.values.pop(name, None) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_s190_a_redirect_is_never_followed_even_by_a_client_built_to_follow(redirecting_platform, status):
    redirecting_platform["state"]["status"] = status
    store = _Store()
    result = await register(owner_api_key="rok_operator", secrets=store)

    # One request, to the platform. The operator key went nowhere else.
    assert [r.url.host for r in redirecting_platform["seen"]] == ["platform.example.com"]
    assert redirecting_platform["seen"][0].headers["x-robutler-owner-key"] == "rok_operator"

    # A failed registration that says why, as in the TypeScript SDK; never the other origin's answer.
    assert result["ok"] is False
    assert result["status"] == status
    assert "redirect" in result["error"]
    assert f"{COLLECTOR}/api/auth/cli/token" in result["error"]
    assert "access_token" not in result
    assert store.values == {}
