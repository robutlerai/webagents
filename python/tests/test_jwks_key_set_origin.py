"""
S-135 twin (2026-09-17): a key set is fetched from CONFIGURATION, never from the
unverified token's own `iss`.

The TypeScript SDK's `JWKSManager.verifyJwt` was found to decode a bearer, take
its `iss`, build `${iss}/.well-known/jwks.json` and fetch it before checking a
single claim. The Python SDK had the same shape in two places, both feeding
`webagents.crypto.jwks.JWKSManager.fetch_jwks` (httpx GET, redirects followed,
one cache entry per distinct URL, never evicted):

* `PaymentSkillX402._verify_payment_token` (payments_x402/skill.py), reached by
  the `before_http_call` hook on any priced endpoint that receives a JWT-shaped
  `X-PAYMENT` header, with the JWKS manager auto-created by default;
* the AOAuth `AuthSkill.validate_token` (local/auth/skill.py), reached by its
  `on_connection` hook on any `Authorization: Bearer`, whose "unknown but
  allowed issuer" branch is open to every issuer while the allow list is empty
  (the default).

The two reproduction classes are written as the post-fix expectation, so on the
pre-fix code they fail with the metadata-service URL in the recorded dial list
and a cache entry per forged issuer. The remaining classes pin what the fix
must keep working and the destination policy itself. The fake stands in for
`httpx.AsyncClient` exactly the way tests/test_local_auth_skill.py does, so
nothing leaves the process.
"""

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402
from webagents.crypto.jwks import (
    KEY_SET_PATH,
    JWKSManager,
    is_public_key_set_url,
    is_well_formed_key_set_url,
)


METADATA_ISS = "http://169.254.169.254/latest#"
PLATFORM = "https://robutler.ai"
INTERNAL = "http://portal.production.svc.cluster.local"
KID = "platform-sig-key"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def forged_token(**claims) -> str:
    """A token nobody signed: RS256 header, attacker-chosen claims, junk signature."""
    header = {"alg": "RS256", "kid": "forged", "typ": "JWT"}
    payload = {
        "sub": "anyone",
        "exp": int(time.time()) + 3600,
        "payment": {"balance": 1.0},
        **claims,
    }
    return ".".join(
        [
            _b64url(json.dumps(header).encode()),
            _b64url(json.dumps(payload).encode()),
            _b64url(b"not-a-signature"),
        ]
    )


@pytest.fixture(scope="module")
def platform_key():
    """One RSA pair for the module: the private half signs, the JWK is what the fake serves."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update({"kid": KID, "use": "sig", "alg": "RS256"})
    return private, jwk


def signed_token(private, **claims) -> str:
    now = int(time.time())
    payload = {"sub": "user-1", "iat": now, "exp": now + 3600, **claims}
    return pyjwt.encode(payload, private, algorithm="RS256", headers={"kid": KID})


class Recorder:
    """Every URL the JWKS manager asked httpx for, and the key set it is served.

    `fail` set to an exception makes every dial raise it (a dead origin);
    `keys` is what a successful dial serves, as `{"keys": [...]}`.
    """

    def __init__(self):
        self.urls = []
        self.calls = []  # (url, follow_redirects)
        self.keys = []
        self.fail = None

    def body(self) -> bytes:
        return json.dumps({"keys": list(self.keys)}).encode()


class _StreamedResponse:
    """The slice of `httpx.Response` a streamed read touches: status, headers,
    `raise_for_status()` and `aiter_bytes()`, served in 8 KiB chunks so a cap
    is exercised mid-body rather than on one oversized chunk."""

    status_code = 200
    headers: dict = {}

    def __init__(self, rec: Recorder):
        self._rec = rec

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        body = self._rec.body()
        for i in range(0, len(body), 8192):
            yield body[i : i + 8192]


@pytest.fixture
def network():
    """A fake `httpx.AsyncClient` recording every dial. It answers BOTH the
    buffered `.get` read (what `fetch_jwks` did until 2026-09-18) and the
    capped `.stream` read (S-135 residual fix), so the dial-count and
    body-size assertions below mean the same thing on either code."""
    rec = Recorder()
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.raise_for_status = MagicMock()
    response.json.side_effect = lambda: {"keys": list(rec.keys)}

    def _record(url, kwargs):
        rec.urls.append(str(url))
        rec.calls.append((str(url), kwargs.get("follow_redirects")))
        if rec.fail is not None:
            raise rec.fail

    async def _get(url, **kwargs):
        _record(url, kwargs)
        return response

    class _Stream:
        def __init__(self, url, kwargs):
            self._url, self._kwargs = url, kwargs

        async def __aenter__(self):
            _record(self._url, self._kwargs)
            return _StreamedResponse(rec)

        async def __aexit__(self, *exc):
            return False

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(side_effect=_get)
    client.stream = MagicMock(side_effect=lambda method, url, **kwargs: _Stream(url, kwargs))
    with patch("httpx.AsyncClient", return_value=client):
        yield rec


@pytest.fixture(autouse=True)
def _no_platform_env(monkeypatch):
    for name in (
        "ROBUTLER_INTERNAL_API_URL",
        "ROBUTLER_API_URL",
        "ROBUTLER_PLATFORM_ISSUER",
        "OWNER_ASSERTION_JWKS_URL",
        "X402_FACILITATOR_URL",
    ):
        monkeypatch.delenv(name, raising=False)


class TestX402Reproduction:
    """S-135 twin, payments_x402: nothing is ever fetched from the token's own iss."""

    async def test_a_unsigned_token_naming_the_metadata_service_triggers_no_request(self, network):
        skill = PaymentSkillX402(config={"webagents_api_url": PLATFORM})
        result = await skill._verify_payment_token(forged_token(iss=METADATA_ISS))
        assert result is None
        assert network.urls == []

    async def test_c_distinct_forged_iss_values_leave_no_cache_entries(self, network):
        skill = PaymentSkillX402(config={"webagents_api_url": PLATFORM})
        for i in range(1, 4):
            await skill._verify_payment_token(forged_token(iss=f"http://10.0.0.{i}"))
        assert skill._jwks_manager.get_cache_stats()["total_entries"] == 0
        assert network.urls == []


class TestX402Pinned:
    """What the fix must keep working for platform-minted payment tokens."""

    async def test_platform_token_verifies_against_the_platform_key_set_at_the_internal_url(
        self, network, platform_key
    ):
        private, jwk = platform_key
        network.keys = [jwk]
        skill = PaymentSkillX402(config={"webagents_api_url": INTERNAL, "platform_issuer": PLATFORM})
        token = signed_token(private, iss=PLATFORM, payment={"balance": 4.5})
        result = await skill._verify_payment_token(token)
        assert result == {"isValid": True, "balance": 4.5}
        # The platform's key set at the URL this agent can reach, not the public issuer.
        assert network.urls == [f"{INTERNAL}{KEY_SET_PATH}"]
        # The platform URL is operator configuration: redirects are still followed there.
        assert network.calls[0][1] is True

    async def test_env_ladder_public_issuer_internal_key_set(self, network, platform_key, monkeypatch):
        monkeypatch.setenv("ROBUTLER_API_URL", PLATFORM)
        monkeypatch.setenv("ROBUTLER_INTERNAL_API_URL", INTERNAL)
        private, jwk = platform_key
        network.keys = [jwk]
        skill = PaymentSkillX402(config={})
        token = signed_token(private, iss=PLATFORM, payment={"balance": 1.25})
        assert await skill._verify_payment_token(token) == {"isValid": True, "balance": 1.25}
        assert network.urls == [f"{INTERNAL}{KEY_SET_PATH}"]

    async def test_platform_signed_token_with_another_issuer_is_refused_without_a_request(
        self, network, platform_key
    ):
        private, jwk = platform_key
        network.keys = [jwk]
        skill = PaymentSkillX402(config={"webagents_api_url": INTERNAL, "platform_issuer": PLATFORM})
        token = signed_token(private, iss="https://idp.example", payment={"balance": 4.5})
        assert await skill._verify_payment_token(token) is None
        assert network.urls == []

    async def test_expected_audience_is_still_enforced(self, network, platform_key):
        private, jwk = platform_key
        network.keys = [jwk]
        skill = PaymentSkillX402(config={"webagents_api_url": INTERNAL, "platform_issuer": PLATFORM})
        token = signed_token(private, iss=PLATFORM, aud="agent-1", payment={"balance": 2.0})
        assert await skill._verify_payment_token(token, expected_audience=["agent-1"]) is not None
        assert await skill._verify_payment_token(token, expected_audience=["other"]) is None


class TestLocalAuthReproduction:
    """S-135 twin, AOAuth local auth: the allow-all default must not dial a private issuer."""

    async def test_validate_token_never_dials_a_private_issuer_named_by_the_token(self, network):
        from webagents.agents.skills.local.auth.skill import AuthSkill as LocalAuthSkill

        skill = LocalAuthSkill(config={"agent_id": "me", "base_url": "https://me.example"})
        result = await skill.validate_token(forged_token(iss=METADATA_ISS))
        assert result is None
        assert network.urls == []


class TestLocalAuthPinned:
    """The AOAuth branch keeps dialling an allowed issuer's own key set, filtered and redirect-free."""

    async def test_allowed_public_issuer_is_fetched_without_following_redirects(
        self, network, platform_key
    ):
        from webagents.agents.skills.local.auth.skill import AuthSkill as LocalAuthSkill

        private, jwk = platform_key
        network.keys = [jwk]
        skill = LocalAuthSkill(config={"agent_id": "me", "base_url": "https://me.example"})
        token = signed_token(private, iss="https://idp.example", sub="other-agent", scope="read")
        result = await skill.validate_token(token)
        assert result is not None
        assert result.issuer == "https://idp.example"
        assert network.urls == [f"https://idp.example{KEY_SET_PATH}"]
        assert network.calls[0][1] is False

    async def test_trusted_issuer_jwks_uri_is_operator_config_and_may_be_in_cluster_http(
        self, network, platform_key
    ):
        from webagents.agents.skills.local.auth.skill import AuthSkill as LocalAuthSkill

        private, jwk = platform_key
        network.keys = [jwk]
        skill = LocalAuthSkill(
            config={
                "agent_id": "me",
                "base_url": "https://me.example",
                "trusted_issuers": [
                    {
                        "issuer": "https://portal.example",
                        "jwks_uri": f"{INTERNAL}{KEY_SET_PATH}",
                        "type": "portal",
                    }
                ],
            }
        )
        token = signed_token(private, iss="https://portal.example", sub="alice", scope="read")
        assert await skill.validate_token(token) is not None
        assert network.urls == [f"{INTERNAL}{KEY_SET_PATH}"]
        assert network.calls[0][1] is True

    @pytest.mark.parametrize(
        "issuer",
        [
            "http://idp.example",
            "https://user:pw@idp.example",
            "https://idp.example/?x=1",
            "https://10.0.0.5",
            "https://127.0.0.1",
            "https://[fd00:ec2::254]",
            "https://[::ffff:169.254.169.254]",
        ],
    )
    async def test_allowed_but_unsafe_issuers_are_refused_without_a_request(self, network, issuer):
        from webagents.agents.skills.local.auth.skill import AuthSkill as LocalAuthSkill

        skill = LocalAuthSkill(config={"agent_id": "me", "base_url": "https://me.example"})
        assert await skill.validate_token(forged_token(iss=issuer)) is None
        assert network.urls == []


class TestKeySetUrlPolicy:
    """The destination predicates and the choke point that enforces them."""

    @pytest.mark.parametrize(
        "url",
        [
            f"http://idp.example{KEY_SET_PATH}",
            f"https://user:secret@idp.example{KEY_SET_PATH}",
            f"https://idp.example{KEY_SET_PATH}?jwks=1",
            f"https://idp.example{KEY_SET_PATH}#frag",
            f"https://169.254.169.254{KEY_SET_PATH}",
            f"http://169.254.169.254/latest{KEY_SET_PATH}",
            f"https://127.0.0.1{KEY_SET_PATH}",
            f"https://10.0.0.5{KEY_SET_PATH}",
            f"https://172.31.255.1{KEY_SET_PATH}",
            f"https://192.168.1.1{KEY_SET_PATH}",
            f"https://100.64.0.1{KEY_SET_PATH}",
            f"https://0.0.0.0{KEY_SET_PATH}",
            f"https://192.0.0.192{KEY_SET_PATH}",
            f"https://198.19.0.1{KEY_SET_PATH}",
            f"https://224.0.0.1{KEY_SET_PATH}",
            f"https://240.0.0.1{KEY_SET_PATH}",
            f"https://[::1]{KEY_SET_PATH}",
            f"https://[::]{KEY_SET_PATH}",
            f"https://[::ffff:169.254.169.254]{KEY_SET_PATH}",
            f"https://[::ffff:a00:1]{KEY_SET_PATH}",
            f"https://[fd00:ec2::254]{KEY_SET_PATH}",
            f"https://[fe80::1]{KEY_SET_PATH}",
            f"https://[64:ff9b::a9fe:a9fe]{KEY_SET_PATH}",
            f"https://[ff02::1]{KEY_SET_PATH}",
            f"ftp://idp.example{KEY_SET_PATH}",
            "robutler/.well-known/jwks.json",
            "",
        ],
    )
    def test_public_policy_refuses(self, url):
        assert is_public_key_set_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            f"https://idp.example{KEY_SET_PATH}",
            f"https://idp.example/tenant{KEY_SET_PATH}",
            f"https://93.184.216.34{KEY_SET_PATH}",
            f"https://[2606:4700::1111]{KEY_SET_PATH}",
            f"https://metadata.google.internal{KEY_SET_PATH}",  # names are not resolved here
        ],
    )
    def test_public_policy_allows(self, url):
        assert is_public_key_set_url(url) is True

    def test_well_formed_floor_keeps_the_operator_platform_shapes(self):
        assert is_well_formed_key_set_url(f"{INTERNAL}{KEY_SET_PATH}") is True
        assert is_well_formed_key_set_url(f"http://127.0.0.1:3000{KEY_SET_PATH}") is True
        assert is_well_formed_key_set_url(f"https://u:p@portal.example{KEY_SET_PATH}") is False
        assert is_well_formed_key_set_url(f"https://portal.example{KEY_SET_PATH}?x") is False
        assert is_well_formed_key_set_url(f"https://portal.example{KEY_SET_PATH}#x") is False
        assert is_well_formed_key_set_url(f"ftp://portal.example{KEY_SET_PATH}") is False
        assert is_well_formed_key_set_url("") is False

    async def test_fetch_jwks_refuses_a_hostile_shape_for_every_caller(self, network):
        manager = JWKSManager({})
        assert await manager.fetch_jwks(f"https://u:p@idp.example{KEY_SET_PATH}") == []
        assert await manager.fetch_jwks(f"https://idp.example{KEY_SET_PATH}#f", untrusted_origin=False) == []
        assert network.urls == []

    async def test_untrusted_origin_refuses_private_and_skips_redirects(self, network):
        manager = JWKSManager({})
        assert await manager.fetch_jwks(f"https://10.0.0.5{KEY_SET_PATH}", untrusted_origin=True) == []
        assert network.urls == []
        await manager.fetch_jwks(f"https://idp.example{KEY_SET_PATH}", untrusted_origin=True)
        assert network.calls == [(f"https://idp.example{KEY_SET_PATH}", False)]

    async def test_cache_is_bounded_least_recently_fetched_first(self, network):
        manager = JWKSManager({"jwks_cache_max_entries": 3})
        urls = [f"https://idp-{i}.example{KEY_SET_PATH}" for i in range(5)]
        for url in urls:
            await manager.fetch_jwks(url)
        stats = manager.get_cache_stats()
        assert stats["total_entries"] == 3
        assert set(stats["entries"]) == set(urls[2:])


class TestS135Residual:
    """S-135 residual (found 2026-09-18 in review of the W2 pass, addendum 2
    under S-135 in SECURITY_ISSUES_LOG.md): the strict policy still let a
    token-derived issuer reach loopback BY NAME (`http://localhost:<any port>`,
    `*.localhost`, and `https://localhost`), and admitted numeric IPv4
    spellings `ipaddress` cannot parse but the socket layer resolves
    (`https://127.1`, `https://2130706433`, `https://0x7f000001`); a
    successful dial buffered the whole body with no cap; and a `kid` miss on
    a fresh fetch dialled AGAIN with `force_refresh`, with nothing remembered
    about a dead origin, so one unauthenticated bearer cost two outbound
    requests every time. Written as the post-fix expectation; on the pre-fix
    code the policy rows answer True, the miss case records two dials, the
    dead-origin case four, and the oversized set is returned in full.
    """

    @pytest.mark.parametrize(
        "url",
        [
            f"http://localhost:6379{KEY_SET_PATH}",
            f"http://localhost:3000{KEY_SET_PATH}",
            f"http://agent.localhost{KEY_SET_PATH}",
            f"https://localhost{KEY_SET_PATH}",
            f"https://svc.localhost{KEY_SET_PATH}",
            f"https://127.1{KEY_SET_PATH}",
            f"https://2130706433{KEY_SET_PATH}",
            f"https://0x7f000001{KEY_SET_PATH}",
            f"https://0177.0.0.1{KEY_SET_PATH}",
            f"https://10.1{KEY_SET_PATH}",
            f"https://[0:0:0:0:0:ffff:7f00:1]{KEY_SET_PATH}",
            # Ends in a number but is not an IPv4 address: the WHATWG parser
            # (what the TypeScript SDK and the platform parse with) rejects
            # the host outright, so the Python policy refuses it too.
            f"https://1.2.3.4.5{KEY_SET_PATH}",
        ],
    )
    def test_public_policy_refuses_loopback_by_name_and_numeric_shorthand(self, url):
        assert is_public_key_set_url(url) is False

    @pytest.mark.parametrize(
        "issuer",
        [
            "http://localhost:8500/v1/anything",
            "https://localhost",
            "https://127.1",
            "https://2130706433",
        ],
    )
    async def test_validate_token_never_dials_loopback_named_by_the_token(self, network, issuer):
        from webagents.agents.skills.local.auth.skill import AuthSkill as LocalAuthSkill

        skill = LocalAuthSkill(config={"agent_id": "me", "base_url": "https://me.example"})
        assert await skill.validate_token(forged_token(iss=issuer)) is None
        assert network.urls == []

    async def test_a_kid_miss_on_a_fresh_fetch_is_final_one_dial_not_two(self, network, platform_key):
        _, jwk = platform_key
        network.keys = [jwk]
        manager = JWKSManager({})
        url = f"https://idp.example{KEY_SET_PATH}"
        assert await manager.get_public_key_from_jwks(url, "not-there", untrusted_origin=True) is None
        assert network.urls == [url]
        # Served from cache afterwards, the miss may refresh ONCE (rotation),
        # and the refetch rate limit then holds: two dials in total, not three.
        assert await manager.get_public_key_from_jwks(url, "not-there", untrusted_origin=True) is None
        assert len(network.urls) <= 2

    async def test_a_dead_origin_is_remembered_so_the_next_bearer_costs_nothing(self, network):
        network.fail = OSError("connection refused")
        manager = JWKSManager({})
        url = f"https://idp.example{KEY_SET_PATH}"
        assert await manager.get_public_key_from_jwks(url, "k1", untrusted_origin=True) is None
        assert network.urls == [url]
        assert await manager.get_public_key_from_jwks(url, "k2", untrusted_origin=True) is None
        assert network.urls == [url]

    async def test_an_oversized_key_set_is_refused_and_not_cached(self, network):
        from webagents.crypto.jwks import KEY_SET_MAX_BYTES

        # About 200 KiB of well-formed keys: past the cap, and a real key set
        # is a few hundred bytes per entry with a handful of entries.
        network.keys = [{"kty": "OKP", "crv": "Ed25519", "x": "A" * 43, "kid": f"k{i}"} for i in range(2500)]
        assert len(network.body()) > KEY_SET_MAX_BYTES
        manager = JWKSManager({})
        url = f"https://idp.example{KEY_SET_PATH}"
        assert await manager.fetch_jwks(url, untrusted_origin=True) == []
        assert manager.get_cache_stats()["total_entries"] == 0 or manager._jwks_cache[url].keys == []

    async def test_a_key_set_under_the_cap_is_still_served_whole(self, network, platform_key):
        _, jwk = platform_key
        network.keys = [jwk]
        manager = JWKSManager({})
        keys = await manager.fetch_jwks(f"https://idp.example{KEY_SET_PATH}", untrusted_origin=True)
        assert [k["kid"] for k in keys] == [KID]
