"""
The MPP buyer (machine-purchase design sections 2.1, 6.3 and 6.4; pass P9b,
2026-09-18), the Python twin of
`webagents/typescript/tests/unit/skills/payments/mpp-buyer.test.ts`, case for
case. Every network exchange here goes through an `httpx.MockTransport`
that records the signed request it was handed, so the suite reads what the
platform would read: which headers rode the retry, which of them the
signature covered, whether the body was the same bytes, and (beyond the
TypeScript suite) that the signature verifies over exactly those headers.

The signing identity is a fresh Ed25519 key; the signer's own suites pin
the bytes, this one pins the loop.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any, Callable, List, Optional, Union

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.agents.skills.robutler.payments_x402 import MppBuyer as ExportedMppBuyer
from webagents.agents.skills.robutler.payments_x402.mpp_buyer import (
    MPP_CREDENTIAL_HEADER,
    MPP_RECEIPT_HEADER,
    PAYMENT_METHODS_HINT_HEADER,
    PURCHASE_HINT_HEADER,
    TERMS_ACCEPTED_HEADER,
    TERMS_VERSION_HEADER,
    MppBuyer,
    MppBuyerPolicy,
    MppChallengeFields,
    MppTermsRequest,
    SptRequest,
    TempoTransferRequest,
    base64url_decode,
    base64url_encode,
    challenge_amount_cents,
    decode_challenge_request,
    encode_mpp_credential,
    jcs_canonicalize,
    parse_payment_receipt,
    parse_www_authenticate_payment,
    read_mpp_challenge,
    read_terms_notice,
    retry_after_seconds,
    tempo_credential_payload,
    after_credential,
    tempo_payer_did,
    seller_pins_from_discovery,
    MppSellerPins,
)
from webagents.crypto.http_signature import SigningKey

from .crypto.covered_support import covered_of, verify_with_covered_headers

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

AGENT_URL = "https://agent.example/agents/mini"
RESOURCE = "https://robutler.ai/agents/acme/v1/chat/completions"
PURCHASE_URL = "https://robutler.ai/api/mpp/credits"
TERMS = "2026-07-31"
TERMS_URL = "https://robutler.ai/doc/terms-of-service"
NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
EXPIRES = "2026-09-18T12:05:00.000Z"


def new_key() -> SigningKey:
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())


def decode(b64: str) -> dict:
    raw = base64url_decode(b64)
    assert raw is not None
    return json.loads(raw)


def format_challenge(f: MppChallengeFields) -> str:
    params = [("id", f.id), ("realm", f.realm), ("method", f.method), ("intent", f.intent)]
    if f.expires:
        params.append(("expires", f.expires))
    params.append(("request", f.request))
    for name in ("digest", "opaque", "header"):
        value = getattr(f, name)
        if value:
            params.append((name, value))
    quoted = ", ".join(f'{k}="' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' for k, v in params)
    return f"Payment {quoted}"


_counter = 0


@dataclass
class Ch:
    fields: MppChallengeFields
    header: str


def challenge(
    *,
    amount_cents: int = 500,
    method: str = "stripe",
    realm: str = "robutler.ai",
    intent: str = "charge",
    expires: Optional[str] = EXPIRES,
    header: Optional[str] = MPP_CREDENTIAL_HEADER,
    kind: str = "pack",
    network_id: str = "profile_test",
    recipient: str = "0xDEPOSIT",
) -> Ch:
    global _counter
    _counter += 1
    if method == "stripe":
        request = {"amount": str(amount_cents), "currency": "usd", "methodDetails": {"networkId": network_id, "paymentMethodTypes": ["card"]}}
    else:
        request = {
            "amount": str(amount_cents * 10_000),
            "currency": "0xTOKEN",
            "recipient": recipient,
            "externalId": "mpp_5",
            "methodDetails": {"chainId": 4217, "memo": "0x" + "ab" * 32, "supportedModes": ["pull"]},
        }
    fields = MppChallengeFields(
        id=base64url_encode(hashlib.sha256(f"challenge {_counter}".encode()).digest()),
        realm=realm,
        method=method,
        intent=intent,
        request=base64url_encode(jcs_canonicalize(request)),
        expires=expires,
        opaque=base64url_encode(jcs_canonicalize({"packId": "mpp_5", "userId": "u1", "kind": kind, "terms": TERMS})),
        header=header,
    )
    return Ch(fields=fields, header=format_challenge(fields))


def problem402(ch: Ch, *, terms_version: Optional[str] = TERMS, problem: str = "payment-required", body: Optional[dict] = None) -> httpx.Response:
    headers = {"WWW-Authenticate": ch.header, "Content-Type": "application/problem+json", "Cache-Control": "no-store"}
    if terms_version:
        headers[TERMS_VERSION_HEADER] = terms_version
        headers["Link"] = f'<{TERMS_URL}>; rel="terms-of-service"'
    doc = {
        "type": f"https://paymentauth.org/problems/{problem}",
        "title": "Payment Required",
        "status": 402,
        "accepts": [{"scheme": "token", "network": "robutler"}],
    }
    if terms_version:
        doc["terms"] = {"url": TERMS_URL, "version": terms_version, "header": TERMS_ACCEPTED_HEADER}
    doc.update(body or {})
    return httpx.Response(402, headers=headers, content=json.dumps(doc).encode())


def receipt_header(reference: str = "pi_test_1") -> str:
    return base64url_encode(jcs_canonicalize({"method": "stripe", "reference": reference, "status": "success", "timestamp": "2026-09-18T12:00:00.000Z"}))


def ok200(reference: str = "pi_test_1") -> httpx.Response:
    return httpx.Response(200, headers={"Content-Type": "application/json", MPP_RECEIPT_HEADER: receipt_header(reference)}, content=b'{"ok":true}')


def pending503(**extra: Any) -> httpx.Response:
    body = {"error": "charge_outcome_unknown", **extra, "retry": {"credentialHeader": MPP_CREDENTIAL_HEADER, "sameCredential": True}}
    return httpx.Response(503, headers={"Retry-After": "5", "Content-Type": "application/problem+json"}, content=json.dumps(body).encode())


def granted_but_unfunded402(ch: Ch) -> httpx.Response:
    """The door after a grant it could not fund from (S-151): a fresh 402 whose body carries the purchase block."""
    return problem402(ch, body={"purchase": {"granted": True, "packId": "mpp_5", "paymentIntentId": "pi_granted"}})


PAYER = "0x" + "1f" * 20
SIGNED_TX = "0x76" + "ab" * 24


@dataclass
class Seen:
    method: str
    url: str
    headers: httpx.Headers
    body: bytes
    covered: List[str]


Responder = Union[httpx.Response, Callable[[Seen], httpx.Response]]


class Queue:
    """A transport that answers from a queue and records every signed request as the platform would see it."""

    def __init__(self, *responders: Responder) -> None:
        self.responders = list(responders)
        self.seen: List[Seen] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        signature_input = request.headers.get("signature-input", "")
        record = Seen(
            method=request.method,
            url=str(request.url),
            headers=request.headers,
            body=request.content,
            covered=covered_of(signature_input) if "(" in signature_input else [],
        )
        self.seen.append(record)
        if not self.responders:
            raise AssertionError(f"transport exhausted after {len(self.seen)} requests")
        nxt = self.responders.pop(0)
        return nxt(record) if callable(nxt) else nxt

    def client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self), **kwargs)


class Card:
    def __init__(self, result: Any = "spt_test_0001") -> None:
        self.calls: List[SptRequest] = []
        self.result = result

    async def get_spt(self, request: SptRequest) -> str:
        self.calls.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@dataclass
class Harness:
    buyer: MppBuyer
    card: Card
    key: SigningKey
    purchases: list = field(default_factory=list)
    refusals: list = field(default_factory=list)
    sleeps: list = field(default_factory=list)

    def verify(self, seen: Seen) -> None:
        verify_with_covered_headers(seen.headers, seen.method, seen.url, seen.body, {self.key.thumbprint: self.key.private_key.public_key()})


def make(
    queue: Optional[Queue] = None,
    *,
    policy: Optional[dict] = None,
    card: Any = "default",
    stablecoin: Any = None,
    client: Optional[httpx.AsyncClient] = None,
    now: Optional[Callable[[], datetime]] = None,
    sleep: Optional[Callable[[float], Any]] = None,
) -> Harness:
    key = new_key()
    spy = Card() if card == "default" else card
    purchases: list = []
    refusals: list = []
    sleeps: list = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if sleep is not None:
            await sleep(seconds)

    buyer = MppBuyer(
        keys=[key],
        agent_url=AGENT_URL,
        policy=MppBuyerPolicy(**{
            "max_per_purchase_cents": 2000,
            "accept_terms": TERMS,
            "realms": ["robutler.ai"],
            # S-154: the sellers the fixtures' challenges name.
            "stripe_profile_id": "profile_test",
            "tempo_deposit_address": "0xDEPOSIT",
            **(policy or {}),
        }),
        card=spy,
        stablecoin=stablecoin,
        client=client if client is not None else (queue or Queue()).client(),
        now=now or (lambda: NOW),
        sleep=record_sleep,
        on_purchase=purchases.append,
        on_refusal=refusals.append,
    )
    return Harness(buyer=buyer, card=spy, key=key, purchases=purchases, refusals=refusals, sleeps=sleeps)


async def post(h: Harness, body: bytes = b"{}", **kwargs: Any) -> httpx.Response:
    return await h.buyer.paying_request("POST", RESOURCE, content=body, **kwargs)


# --------------------------------------------------------------------------
# The pure pieces
# --------------------------------------------------------------------------


class TestParseWwwAuthenticatePayment:
    def test_parses_quoted_and_bare_values_unescapes_once_and_keeps_optionals_only_when_present(self):
        parsed = parse_www_authenticate_payment(
            'Payment id="a\\"b", realm=robutler.ai, method="stripe", intent="charge", request="cmVx", header="Payment-Authorization"'
        )
        assert parsed is not None
        assert parsed.to_dict() == {"id": 'a"b', "realm": "robutler.ai", "method": "stripe", "intent": "charge", "request": "cmVx", "header": "Payment-Authorization"}
        assert parsed.expires is None

    def test_finds_the_payment_challenge_among_other_schemes(self):
        value = 'Bearer realm="api", Payment id="x", realm="robutler.ai", method="stripe", intent="charge", request="cmVx", Basic realm="other"'
        parsed = parse_www_authenticate_payment(value)
        assert parsed is not None and parsed.id == "x" and parsed.realm == "robutler.ai"
        trailing = 'Payment id="x", realm="robutler.ai", method="stripe", intent="charge", request="cmVx", Bearer realm="other"'
        assert parse_www_authenticate_payment(trailing).realm == "robutler.ai"

    def test_none_without_a_payment_scheme_or_a_required_parameter(self):
        assert parse_www_authenticate_payment('Bearer realm="api"') is None
        assert parse_www_authenticate_payment('Payment id="x", realm="r", method="stripe", intent="charge"') is None
        assert parse_www_authenticate_payment(None) is None
        assert parse_www_authenticate_payment("") is None

    def test_round_trips_the_platform_format(self):
        ch = challenge()
        assert parse_www_authenticate_payment(ch.header) == ch.fields

    def test_whitespace_is_javascripts_class_not_pythons(self):
        # Checked against the TypeScript regex on 2026-09-18: U+FEFF is
        # whitespace to JavaScript and not to Python, U+001C the reverse.
        # The parser spells out the JavaScript class, so both SDKs stop at
        # the same byte.
        fields = 'realm="robutler.ai", method="stripe", intent="charge", request="cmVx"'
        assert parse_www_authenticate_payment('Payment\ufeffid="x", ' + fields).id == "x"
        assert parse_www_authenticate_payment('Payment id="x",\x1c' + fields) is None
        # A trailing line break is whitespace before the end, in both.
        assert parse_www_authenticate_payment('Payment id="x", ' + fields + "\n").id == "x"


class TestJcsAndBase64url:
    def test_sorts_keys_nulls_none_and_keeps_ecmascript_number_forms(self):
        value = {"b": 1, "a": [None, "x"], "Z": True, "d": {"y": None, "x": 1e30}, "f": 5.0, "g": 1.5e-7}
        assert jcs_canonicalize(value) == '{"Z":true,"a":[null,"x"],"b":1,"d":{"x":1e+30,"y":null},"f":5,"g":1.5e-7}'
        with pytest.raises(ValueError, match="non-finite"):
            jcs_canonicalize({"a": float("nan")})

    def test_sorts_keys_by_utf16_code_units(self):
        # U+1F600 is a surrogate pair (0xD83D ...), which sorts below U+FFFF.
        assert jcs_canonicalize({"\uffff": 1, "\U0001F600": 2, "z": 3}) == '{"z":3,"\U0001F600":2,"\uffff":1}'

    def test_encodes_without_padding_and_decodes_strictly(self):
        raw = bytes([0, 255, 62, 63, 250])
        encoded = base64url_encode(raw)
        assert not any(c in encoded for c in "+/=")
        assert base64url_decode(encoded) == raw
        assert base64url_decode("abc=") is None
        assert base64url_decode("a+b") is None
        assert base64url_decode("a") is None
        assert base64url_decode(base64url_encode('{"a":1}')) == b'{"a":1}'


class TestDecodedChallenge:
    def test_reads_the_stripe_and_tempo_requests_in_whole_cents(self):
        stripe = read_mpp_challenge(challenge(amount_cents=500).header)
        assert stripe.request.amount == "500" and stripe.request.currency == "usd"
        assert stripe.request.method_details["networkId"] == "profile_test"
        assert stripe.amount_cents == 500
        assert stripe.expires_at == datetime(2026, 9, 18, 12, 5, tzinfo=timezone.utc)
        assert stripe.credential_header == MPP_CREDENTIAL_HEADER

        tempo = read_mpp_challenge(challenge(method="tempo", amount_cents=7).header)
        assert tempo.request.amount == "70000"
        assert tempo.request.recipient == "0xDEPOSIT"
        assert tempo.amount_cents == 7
        assert challenge_amount_cents("tempo", "70001") is None
        assert challenge_amount_cents("stripe", "0") is None
        assert challenge_amount_cents("stripe", "9007199254740993") is None

    def test_defaults_the_credential_header_to_authorization_and_tolerates_no_expiry(self):
        ch = read_mpp_challenge(challenge(header=None, expires=None).header)
        assert ch.credential_header == "Authorization"
        assert ch.expires_at is None

    def test_none_when_the_request_is_not_a_charge_request(self):
        ch = challenge()
        assert decode_challenge_request(base64url_encode('{"amount":"5.00"}')) is None
        assert decode_challenge_request("not-json") is None
        # ASCII digits only: JavaScript's \\d, not Python's Unicode one.
        assert decode_challenge_request(base64url_encode(jcs_canonicalize({"amount": "\u0665", "currency": "usd", "methodDetails": {}}))) is None
        bad = MppChallengeFields(**{**ch.fields.to_dict(), "request": base64url_encode("[]")})
        assert read_mpp_challenge(format_challenge(bad)) is None


class TestCredentialAndReceipt:
    def test_encodes_payment_base64url_jcs_with_the_challenge_echoed(self):
        ch = challenge()
        value = encode_mpp_credential(ch.fields, {"spt": "spt_1"})
        assert value.startswith("Payment ")
        body = decode(value[len("Payment "):])
        assert body == {"challenge": ch.fields.to_dict(), "payload": {"spt": "spt_1"}}
        assert "digest" not in body["challenge"]
        assert decode(encode_mpp_credential(ch.fields, {"spt": "spt_1"}, "0xPAYER")[8:])["source"] == "0xPAYER"

    def test_is_byte_stable_whatever_the_key_order_given(self):
        ch = challenge()
        shuffled = dict(reversed(list(ch.fields.to_dict().items())))
        assert encode_mpp_credential(shuffled, {"b": 2, "a": 1}) == encode_mpp_credential(ch.fields, {"a": 1, "b": 2})

    def test_wraps_a_tempo_transaction(self):
        assert tempo_credential_payload("0xsigned") == {"type": "transaction", "signature": "0xsigned"}

    def test_parses_a_receipt_and_refuses_anything_else(self):
        receipt = parse_payment_receipt(receipt_header("pi_9"))
        assert (receipt.method, receipt.reference, receipt.status) == ("stripe", "pi_9", "success")
        assert parse_payment_receipt(base64url_encode('{"status":"success"}')) is None
        assert parse_payment_receipt(None) is None


class TestResponseReaders:
    def test_retry_after_reads_a_delta_an_http_date_and_falls_back(self):
        assert retry_after_seconds("5", NOW) == 5
        assert retry_after_seconds(format_datetime(NOW + timedelta(seconds=12), usegmt=True), NOW) == 12
        assert retry_after_seconds("soon", NOW) == 5
        assert retry_after_seconds(None, NOW, 9) == 9

    def test_read_terms_notice_prefers_the_header_then_the_body_and_takes_the_url_from_body_or_link(self):
        from_header = problem402(challenge(), terms_version="2026-08-01")
        assert read_terms_notice(from_header, None) == MppTermsRequest(version="2026-08-01", url=TERMS_URL)
        body_only = httpx.Response(402, content=b"{}")
        assert read_terms_notice(body_only, {"terms": {"version": "2026-08-02", "url": "https://x/terms"}}) == MppTermsRequest("2026-08-02", "https://x/terms")
        assert read_terms_notice(body_only, None) is None


# --------------------------------------------------------------------------
# The buyer
# --------------------------------------------------------------------------


class TestConstruction:
    def test_refuses_a_buyer_with_no_source_no_ceiling_or_no_acceptance_policy(self):
        base = {"keys": [new_key()], "agent_url": AGENT_URL}
        card = Card()
        with pytest.raises(ValueError, match="at least one payment source"):
            MppBuyer(**base, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS))
        with pytest.raises(ValueError, match="max_per_purchase_cents"):
            MppBuyer(**base, card=card, policy=MppBuyerPolicy(max_per_purchase_cents=0, accept_terms=TERMS))
        with pytest.raises(ValueError, match="empty version"):
            MppBuyer(**base, card=card, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=" "))
        with pytest.raises(ValueError, match="never accepts the Terms on its own"):
            MppBuyer(**base, card=card, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=None))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown method"):
            MppBuyer(**base, card=card, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS, methods=["paypal"]))
        with pytest.raises(ValueError, match="prefer_purchase"):
            MppBuyer(**base, card=card, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS, prefer_purchase="bulk"))
        with pytest.raises(ValueError, match="identity"):
            MppBuyer(keys=[], agent_url=AGENT_URL, card=card, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS))

    def test_s147_refuses_a_buyer_that_names_no_host_and_defaults_the_list_to_the_platform_host(self):
        base = {"keys": [new_key()], "agent_url": AGENT_URL, "card": Card()}
        with pytest.raises(ValueError, match="policy.realms"):
            MppBuyer(**base, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS))
        with pytest.raises(ValueError, match="policy.realms"):
            MppBuyer(**base, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS, realms=[]))
        with pytest.raises(ValueError, match="not a host"):
            MppBuyer(**base, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS, realms=["robutler.ai/x"]))
        by_platform = MppBuyer(**base, platform_url="https://robutler.ai", policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS))
        assert by_platform.allows_url("https://robutler.ai/agents/acme/v1/chat/completions")
        assert not by_platform.allows_url("https://evil.example/agents/acme")
        assert not by_platform.allows_url("https://robutler.ai.evil.example/")
        with_port = MppBuyer(**base, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS, realms=["localhost:3000"]))
        assert with_port.allows_url("http://localhost:3000/api/mpp/credits")
        assert not with_port.allows_url("http://localhost:4000/api/mpp/credits")

    def test_refuses_a_loopback_agent_url_at_construction(self):
        from webagents.crypto.http_signature import SigningError

        with pytest.raises(SigningError, match="loopback"):
            MppBuyer(keys=[new_key()], agent_url="http://localhost:2224/agents/x", card=Card(), policy={"max_per_purchase_cents": 1, "accept_terms": TERMS})

    def test_pays_in_the_policy_order_only_with_a_source_default_stablecoin_first(self):
        class Wallet:
            def sign_tempo_transfer(self, request):
                return "0x1"

        def buyer(**kw):
            return MppBuyer(keys=[new_key()], agent_url=AGENT_URL, policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms=TERMS, realms=["robutler.ai"], **kw.pop("policy", {})), **kw)

        assert buyer(card=Card(), stablecoin=Wallet()).method_order() == ["tempo", "stripe"]
        assert buyer(card=Card(), stablecoin=Wallet(), policy={"methods": ["stripe", "tempo"]}).method_order() == ["stripe", "tempo"]
        assert buyer(card=Card()).method_order() == ["stripe"]

    def test_the_package_exports_the_buyer(self):
        assert ExportedMppBuyer is MppBuyer


class TestPayingFetch:
    async def test_signs_the_request_and_returns_a_200_untouched(self):
        q = Queue(ok200())
        h = make(q)
        response = await post(h, b'{"messages":[]}', headers={"content-type": "application/json"})
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert len(q.seen) == 1
        first = q.seen[0]
        assert 'tag="web-bot-auth"' in first.headers["signature-input"]
        assert first.headers["content-digest"].startswith("sha-256=:")
        for name in (PURCHASE_HINT_HEADER, TERMS_ACCEPTED_HEADER):
            assert name not in first.headers
        # 2026-09-18: without this a card-only buyer met the door's one Tempo
        # challenge whenever stablecoin was on, and refused it.
        assert first.headers[PAYMENT_METHODS_HINT_HEADER] == "stripe"
        assert '"robutler-payment-methods"' in first.covered
        h.verify(first)
        assert h.card.calls == []
        assert h.purchases == []

    async def test_buys_the_402_and_re_sends_the_same_request_once_with_credential_and_assent_covered(self):
        ch = challenge(amount_cents=500)
        q = Queue(problem402(ch), ok200("pi_abc"))
        h = make(q)
        response = await post(h, b'{"messages":[1]}', headers={"content-type": "application/json"})
        assert response.status_code == 200
        assert len(q.seen) == 2
        first, retry = q.seen

        assert h.card.calls == [
            SptRequest(
                network_id="profile_test",
                amount_cents=500,
                currency="usd",
                expires_at=datetime(2026, 9, 18, 12, 5, tzinfo=timezone.utc),
                challenge_id=ch.fields.id,
                payment_method_types=["card"],
            )
        ]
        # The same request: method, URL, body bytes and digest.
        assert retry.method == "POST"
        assert retry.url == first.url
        assert retry.body == b'{"messages":[1]}'
        assert retry.headers["content-digest"] == first.headers["content-digest"]
        # A fresh signature: a new nonce.
        assert retry.headers["signature-input"] != first.headers["signature-input"]

        credential = retry.headers[MPP_CREDENTIAL_HEADER]
        assert decode(credential[len("Payment "):]) == {"challenge": ch.fields.to_dict(), "payload": {"spt": "spt_test_0001"}}
        assert retry.headers[TERMS_ACCEPTED_HEADER] == TERMS
        assert "authorization" not in retry.headers
        assert retry.covered == [
            '"@method"', '"@authority"', '"@path"', '"@query"', '"content-digest"',
            # The card-only buyer's method hint (2026-09-18), then the credential and the assent.
            '"robutler-payment-methods"', '"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent";key="sig1"',
        ]
        assert '"payment-authorization"' not in first.covered
        # The signature verifies over exactly those headers.
        h.verify(first)
        h.verify(retry)

        assert len(h.purchases) == 1
        record = h.purchases[0]
        assert (record.url, record.challenge_id, record.method, record.amount_cents, record.currency, record.terms_version, record.status) == (
            RESOURCE, ch.fields.id, "stripe", 500, "usd", TERMS, 200,
        )
        assert record.receipt.reference == "pi_abc"
        assert h.buyer.spent_today_cents() == 500
        assert h.refusals == []

    async def test_sends_the_policy_hints_covered_on_every_request(self):
        q = Queue(problem402(challenge()), ok200())
        h = make(q, policy={"prefer_purchase": "exact", "methods": ["stripe"]})
        await post(h)
        for seen in q.seen:
            assert seen.headers[PURCHASE_HINT_HEADER] == "exact"
            assert seen.headers[PAYMENT_METHODS_HINT_HEADER] == "stripe"
            assert '"robutler-purchase"' in seen.covered
            assert '"robutler-payment-methods"' in seen.covered
            h.verify(seen)

    async def test_never_pays_a_challenge_whose_realm_is_not_the_host_asked(self):
        q = Queue(problem402(challenge(realm="agents.robutler.ai")))
        h = make(q, policy={"realms": ["robutler.ai", "agents.robutler.ai"]})
        response = await post(h)
        assert response.status_code == 402
        assert response.json()["status"] == 402
        assert h.card.calls == []
        assert len(h.refusals) == 1
        refusal = h.refusals[0]
        assert (refusal.reason, refusal.amount_cents, refusal.method, refusal.terms_version) == ("realm_mismatch", 500, "stripe", TERMS)

    async def test_s147_a_third_party_host_answering_402_with_its_own_challenge_is_refused_before_any_source_call(self):
        # The delegate case: the model chose a URL, and the host there names
        # itself as the realm, so "realm equals the host asked" holds.
        q = Queue(problem402(challenge(realm="evil.example")))
        h = make(q)
        response = await h.buyer.paying_request("POST", "https://evil.example/agents/x/chat/completions", content=b"{}")
        assert response.status_code == 402
        assert h.card.calls == []
        assert len(q.seen) == 1
        assert [r.reason for r in h.refusals] == ["realm_not_allowed"]
        # A realm on the list fetched from an off-list host is refused too.
        h2 = make(Queue(problem402(challenge(realm="robutler.ai"))))
        await h2.buyer.paying_request("POST", "https://evil.example/agents/x/chat/completions", content=b"{}")
        assert h2.card.calls == []
        assert [r.reason for r in h2.refusals] == ["realm_not_allowed"]

    @pytest.mark.parametrize(
        "make_response, reason, policy",
        [
            (lambda: problem402(challenge(amount_cents=2001)), "over_max_per_purchase", {}),
            (lambda: problem402(challenge(amount_cents=600)), "over_daily_cap", {"daily_cap_cents": 500}),
            (lambda: problem402(challenge(intent="session")), "unsupported_intent", {}),
            (lambda: problem402(challenge(expires="2026-09-18T11:59:59.000Z")), "challenge_expired", {}),
            (lambda: problem402(challenge(method="tempo")), "method_unavailable", {}),
        ],
    )
    async def test_refuses_by_policy_before_any_source_call(self, make_response, reason, policy):
        h = make(Queue(make_response()), policy=policy)
        out = await post(h)
        assert out.status_code == 402
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == [reason]

    async def test_the_daily_cap_counts_what_was_presented_in_the_last_24_hours_across_calls(self):
        q = Queue(problem402(challenge(amount_cents=400)), ok200(), problem402(challenge(amount_cents=400)))
        h = make(q, policy={"daily_cap_cents": 700})
        assert (await post(h)).status_code == 200
        assert h.buyer.spent_today_cents() == 400
        assert (await post(h)).status_code == 402
        assert [r.reason for r in h.refusals] == ["over_daily_cap"]
        assert len(h.card.calls) == 1

    async def test_a_pinned_terms_version_pays_only_that_version_and_a_callback_decides(self):
        stale = make(Queue(problem402(challenge(), terms_version="2026-08-15")))
        assert (await post(stale)).status_code == 402
        assert stale.card.calls == []
        assert (stale.refusals[0].reason, stale.refusals[0].terms_version) == ("terms_refused", "2026-08-15")

        asked: list = []

        def refuse(terms: MppTermsRequest) -> bool:
            asked.append(terms)
            return False

        refusing = make(Queue(problem402(challenge())), policy={"accept_terms": refuse})
        assert (await post(refusing)).status_code == 402
        assert asked == [MppTermsRequest(version=TERMS, url=TERMS_URL)]
        assert refusing.card.calls == []

        async def accept(terms: MppTermsRequest) -> bool:
            return True

        q = Queue(problem402(challenge()), ok200())
        accepting = make(q, policy={"accept_terms": accept})
        assert (await post(accepting)).status_code == 200
        assert q.seen[1].headers[TERMS_ACCEPTED_HEADER] == TERMS

    async def test_a_402_with_no_terms_notice_is_paid_with_no_assent(self):
        q = Queue(problem402(challenge(), terms_version=None), ok200())
        h = make(q)
        assert (await post(h)).status_code == 200
        assert TERMS_ACCEPTED_HEADER not in q.seen[1].headers
        assert '"robutler-terms-accepted"' not in q.seen[1].covered
        assert h.purchases[0].terms_version is None

    async def test_re_presents_the_same_credential_after_retry_after_and_never_asks_the_source_again(self):
        q = Queue(problem402(challenge()), pending503(), pending503(), ok200("pi_settled"))
        h = make(q)
        response = await post(h)
        assert response.status_code == 200
        assert len(q.seen) == 4
        assert len(h.card.calls) == 1
        assert len({s.headers[MPP_CREDENTIAL_HEADER] for s in q.seen[1:]}) == 1
        # Every re-presentation is freshly signed: a new nonce each time.
        assert len({s.headers["signature-input"] for s in q.seen[1:]}) == 3
        # Backoff: Retry-After 5 s, doubling, capped at max_retry_after_seconds.
        assert h.sleeps == [5, 10]
        assert h.purchases[0].receipt.reference == "pi_settled"
        assert h.buyer.spent_today_cents() == 500

    async def test_bounds_the_settlement_retries_and_keeps_the_presentation_counted(self):
        q = Queue(problem402(challenge()), pending503(), pending503())
        h = make(q, policy={"max_settlement_retries": 1})
        response = await post(h)
        assert response.status_code == 503
        assert [r.reason for r in h.refusals] == ["settlement_retries_exhausted"]
        assert h.buyer.spent_today_cents() == 500

    async def test_caps_the_retry_after_it_honours(self):
        slow = httpx.Response(503, headers={"Retry-After": "600", "Content-Type": "application/problem+json"}, content=b'{"retry":{"sameCredential":true}}')
        h = make(Queue(problem402(challenge()), slow, ok200()), policy={"max_retry_after_seconds": 7})
        await post(h)
        assert h.sleeps == [7]

    async def test_a_fresh_challenge_after_paying_is_paid_again_bounded_by_max_challenges(self):
        first, second = challenge(), challenge()
        versions: list = []

        def accept(terms: MppTermsRequest) -> bool:
            versions.append(terms.version)
            return True

        q = Queue(problem402(first), problem402(second, terms_version="2026-10-01", problem="terms-version-stale"), ok200())
        h = make(q, policy={"accept_terms": accept})
        assert (await post(h)).status_code == 200
        assert len(h.card.calls) == 2
        assert versions == [TERMS, "2026-10-01"]
        assert q.seen[2].headers[TERMS_ACCEPTED_HEADER] == "2026-10-01"
        assert decode(q.seen[2].headers[MPP_CREDENTIAL_HEADER][8:])["challenge"] == second.fields.to_dict()
        # The first presentation was refused before any charge, so only the second counts.
        assert h.buyer.spent_today_cents() == 500

        bounded = make(Queue(problem402(challenge()), problem402(challenge(), problem="payment-expired")), policy={"max_challenges": 1})
        out = await post(bounded)
        assert out.status_code == 402
        assert len(bounded.card.calls) == 1
        assert [r.reason for r in bounded.refusals] == ["challenges_exhausted"]

    @pytest.mark.parametrize(
        "make_response",
        [
            lambda: httpx.Response(402, headers={"WWW-Authenticate": 'Bearer realm="x"'}, content=b'{"status":402}'),
            lambda: httpx.Response(401, content=b"{}"),
            lambda: httpx.Response(403, content=b"{}"),
        ],
    )
    async def test_returns_a_402_with_no_readable_challenge_a_401_and_a_403_as_they_came(self, make_response):
        original = make_response()
        q = Queue(original)
        h = make(q)
        out = await post(h)
        assert out is original
        assert len(q.seen) == 1
        assert h.card.calls == []

    async def test_an_error_after_the_credential_releases_the_presentation_unless_a_purchase_block(self):
        released = make(Queue(problem402(challenge()), httpx.Response(403, content=b'{"error":"buyer_inactive"}')))
        assert (await post(released)).status_code == 403
        assert released.buyer.spent_today_cents() == 0

        kept = make(Queue(problem402(challenge()), httpx.Response(500, headers={"Content-Type": "application/json"}, content=b'{"error":"run_failed","purchase":{"granted":true}}')))
        assert (await post(kept)).status_code == 500
        assert kept.buyer.spent_today_cents() == 500
        # The purchase block means the pack was granted: it is a purchase.
        assert [p.status for p in kept.purchases] == [500]
        assert kept.buyer.pending_credentials() == []

    async def test_pays_a_tempo_challenge_with_an_unbroadcast_transaction_naming_the_payer(self):
        ch = challenge(method="tempo", amount_cents=7)
        calls: List[TempoTransferRequest] = []

        class Wallet:
            address = PAYER

            def sign_tempo_transfer(self, request: TempoTransferRequest) -> str:  # a sync source is awaited too
                calls.append(request)
                return SIGNED_TX

        q = Queue(problem402(ch), ok200("0xhash"))
        h = make(q, card=None, stablecoin=Wallet())
        assert (await post(h)).status_code == 200
        assert calls == [
            TempoTransferRequest(
                chain_id=4217,
                currency="0xTOKEN",
                recipient="0xDEPOSIT",
                amount="70000",
                memo="0x" + "ab" * 32,
                valid_before=datetime(2026, 9, 18, 12, 5, tzinfo=timezone.utc),
                challenge_id=ch.fields.id,
            )
        ]
        body = decode(q.seen[1].headers[MPP_CREDENTIAL_HEADER][8:])
        # The source is the DID the platform's parseTempoPayerDid requires, on the challenge's chain.
        assert body == {"challenge": ch.fields.to_dict(), "payload": {"type": "transaction", "signature": SIGNED_TX}, "source": f"did:pkh:eip155:4217:{PAYER}"}
        record = h.purchases[0]
        assert (record.method, record.amount_cents, record.currency) == ("tempo", 7, "0xTOKEN")
        # A stablecoin-only buyer names its one method.
        assert q.seen[0].headers[PAYMENT_METHODS_HINT_HEADER] == "tempo"

    async def test_refuses_a_payer_the_platform_would_refuse_and_a_transaction_that_is_not_0x76_before_anything_is_presented(self):
        for address in ("0xPAYER", f"did:pkh:eip155:1:{PAYER}"):
            signed: list = []

            class Wallet:
                def __init__(self, a: str) -> None:
                    self.address = a

                def sign_tempo_transfer(self, request: TempoTransferRequest) -> str:
                    signed.append(request)
                    return SIGNED_TX

            q = Queue(problem402(challenge(method="tempo", amount_cents=7)))
            h = make(q, card=None, stablecoin=Wallet(address))
            with pytest.raises(ValueError, match="stablecoin source"):
                await post(h)
            # Nothing was signed for a credential the platform would refuse.
            assert signed == []
            assert len(q.seen) == 1
            assert h.buyer.spent_today_cents() == 0

        class Garbage:
            address = PAYER

            def sign_tempo_transfer(self, request: TempoTransferRequest) -> str:
                return "0xsignedtx"

        q = Queue(problem402(challenge(method="tempo", amount_cents=7)))
        h = make(q, card=None, stablecoin=Garbage())
        with pytest.raises(ValueError, match="0x76"):
            await post(h)
        assert len(q.seen) == 1

    async def test_a_source_that_fails_propagates_and_nothing_is_retried(self):
        q = Queue(problem402(challenge()))
        h = make(q, card=Card(RuntimeError("wallet declined")))
        with pytest.raises(RuntimeError, match="wallet declined"):
            await post(h)
        assert len(q.seen) == 1
        assert h.buyer.spent_today_cents() == 0

    async def test_a_source_that_returns_no_spt_id_is_refused_before_anything_is_sent(self):
        q = Queue(problem402(challenge()))
        h = make(q, card=Card("tok_visa"))
        with pytest.raises(ValueError, match="spt_"):
            await post(h)
        assert len(q.seen) == 1

    async def test_uses_the_challenge_header_parameter_and_authorization_when_it_names_none(self):
        q = Queue(problem402(challenge(header=None)), ok200())
        h = make(q)
        await post(h)
        assert q.seen[1].headers["authorization"].startswith("Payment ")
        assert MPP_CREDENTIAL_HEADER not in q.seen[1].headers
        assert '"authorization"' in q.seen[1].covered
        h.verify(q.seen[1])

    async def test_the_clients_own_auth_never_re_signs_over_the_buyer(self):
        class Stamp(httpx.Auth):
            def auth_flow(self, request):
                request.headers["X-Stamped"] = "1"
                yield request

        q = Queue(ok200())
        h = make(client=q.client(auth=Stamp()))
        await post(h)
        assert "x-stamped" not in q.seen[0].headers

    async def test_paying_fetch_takes_an_httpx_request_and_keeps_its_body_and_timeout(self):
        q = Queue(problem402(challenge()), ok200())
        h = make(q)
        request = httpx.Request("POST", RESOURCE, json={"messages": [1]}, extensions={"timeout": httpx.Timeout(3.0).as_dict()})
        assert (await h.buyer.paying_fetch(request)).status_code == 200
        assert q.seen[0].body == q.seen[1].body == request.content
        h.verify(q.seen[1])


class TestPurchaseInBand:
    async def test_pays_a_challenge_from_the_socket_at_the_purchase_url_post_with_no_body(self):
        ch = challenge()
        q = Queue(ok200("pi_inband"))
        h = make(q)
        outcome = await h.buyer.purchase(PURCHASE_URL, ch.header, {"url": TERMS_URL, "version": TERMS})
        assert outcome.ok
        record = outcome.record
        assert (record.url, record.challenge_id, record.amount_cents, record.terms_version) == (PURCHASE_URL, ch.fields.id, 500, TERMS)
        assert record.receipt.reference == "pi_inband"
        assert len(q.seen) == 1
        sent = q.seen[0]
        assert sent.method == "POST"
        assert sent.url == PURCHASE_URL
        assert sent.body == b""
        assert "content-digest" not in sent.headers
        assert sent.headers[TERMS_ACCEPTED_HEADER] == TERMS
        assert sent.covered == [
            '"@method"', '"@authority"', '"@path"', '"@query"',
            '"robutler-payment-methods"', '"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent";key="sig1"',
        ]
        h.verify(sent)
        # An in-band purchase is a purchase: the operator's on_purchase sees it too.
        assert h.purchases == [record]
        assert h.buyer.spent_today_cents() == 500

    async def test_reports_a_refusal_a_garbage_challenge_and_an_error_status(self):
        b1 = make(Queue(), policy={"max_per_purchase_cents": 100})
        refused = await b1.buyer.purchase(PURCHASE_URL, challenge(amount_cents=500).header, {"version": TERMS})
        assert (refused.ok, refused.reason) == (False, "over_max_per_purchase")
        assert b1.card.calls == []

        b2 = make(Queue())
        garbage = await b2.buyer.purchase(PURCHASE_URL, 'Bearer realm="x"')
        assert (garbage.ok, garbage.reason) == (False, "no_challenge")

        b3 = make(Queue(httpx.Response(403, content=b'{"error":"buyer_inactive"}')))
        failed = await b3.buyer.purchase(PURCHASE_URL, challenge().header, MppTermsRequest(version=TERMS))
        assert (failed.ok, failed.reason, failed.status) == (False, "unexpected_status", 403)

    async def test_the_realm_must_be_the_purchase_host_and_a_503_there_is_re_presented(self):
        q = Queue(pending503(), ok200())
        h = make(q)
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge().header, {"version": TERMS})
        assert outcome.ok
        assert len(h.card.calls) == 1
        credentials = [s.headers[MPP_CREDENTIAL_HEADER] for s in q.seen]
        assert len(credentials) == 2 and credentials[0] == credentials[1]

        elsewhere = make(Queue())
        out = await elsewhere.buyer.purchase("https://other.example/api/mpp/credits", challenge().header)
        assert (out.ok, out.reason) == (False, "realm_not_allowed")

    async def test_s147_a_uamp_peer_naming_its_own_purchase_url_and_realm_is_refused_with_no_source_call_and_no_request(self):
        q = Queue()
        h = make(q)
        outcome = await h.buyer.purchase("https://peer.example/buy", challenge(realm="peer.example").header, {"version": TERMS})
        assert (outcome.ok, outcome.reason) == (False, "realm_not_allowed")
        assert h.card.calls == []
        assert q.seen == []

    async def test_every_in_band_request_carries_an_explicit_timeout_never_the_httpx_default(self):
        seen: list = []

        def responder(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions.get("timeout"))
            return ok200()

        h = make(client=httpx.AsyncClient(transport=httpx.MockTransport(responder)))
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge().header, {"version": TERMS})
        assert outcome.ok
        assert seen == [httpx.Timeout(60).as_dict()]


# --------------------------------------------------------------------------
# 2026-09-18 fix pass: S-150, S-151, the 503 contract, the ledger, the upgrade
# (the TypeScript suite's cases, one for one)
# --------------------------------------------------------------------------


class TestS151PaidPackAfterGrant:
    async def test_counts_it_reports_it_and_does_not_pay_the_fresh_challenge_by_default(self):
        h = make(Queue(problem402(challenge()), granted_but_unfunded402(challenge())), policy={"daily_cap_cents": 1000})
        response = await post(h)
        assert response.status_code == 402
        assert len(h.card.calls) == 1
        assert [(p.status, p.amount_cents) for p in h.purchases] == [(402, 500)]
        assert h.buyer.spent_today_cents() == 500
        assert [r.reason for r in h.refusals] == ["purchase_limit_per_call"]

    async def test_a_second_pack_is_paid_only_when_the_policy_allows_it_and_both_are_counted(self):
        h = make(Queue(problem402(challenge()), granted_but_unfunded402(challenge()), ok200("pi_second")), policy={"max_purchases_per_call": 2})
        assert (await post(h)).status_code == 200
        assert len(h.card.calls) == 2
        assert [p.status for p in h.purchases] == [402, 200]
        assert h.buyer.spent_today_cents() == 1000

        # The reviewer's repro: a cap of one pack now binds, even with a second purchase allowed.
        capped = make(Queue(problem402(challenge()), granted_but_unfunded402(challenge())), policy={"max_purchases_per_call": 2, "daily_cap_cents": 500})
        assert (await post(capped)).status_code == 402
        assert len(capped.card.calls) == 1
        assert [r.reason for r in capped.refusals] == ["over_daily_cap"]
        assert capped.buyer.spent_today_cents() == 500


class TestThe503Contract:
    def test_reads_each_answer_after_a_credential_by_the_contract_never_by_retry_after_alone(self):
        assert after_credential(200, None) == "settled"
        assert after_credential(402, {"purchase": {}}) == "settled"
        # By code, not the flag alone: only funding_failed (or a purchase block) is paid.
        assert after_credential(503, {"error": "funding_failed", "retry": {"sameCredential": False}}) == "settled"
        assert after_credential(503, {"error": "mpp_not_configured", "retry": {"sameCredential": False}}) == "refused"
        assert after_credential(503, {"error": "metered_not_ready", "retry": {"sameCredential": False}}) == "refused"
        assert after_credential(503, {"error": "sale_serve_pending", "retry": {"sameCredential": True}}) == "retry_same"
        assert after_credential(409, {"error": "purchase_already_granted", "purchase": {}, "retry": {"sameCredential": False}}) == "settled"
        assert after_credential(409, {"error": "call_already_served", "retry": {"sameCredential": False}}) == "served_elsewhere"
        assert after_credential(409, {"error": "call_in_progress", "retry": {"sameCredential": False}}) == "served_elsewhere"
        assert after_credential(409, {"error": "call_refunded", "retry": {"sameCredential": False}}) == "refunded"
        assert after_credential(429, {"error": "metered_principal_cap"}) == "refused"
        assert after_credential(503, {"retry": {"sameCredential": True}}) == "retry_same"
        assert after_credential(503, None) == "unknown"
        assert after_credential(502, None) == "unknown"
        assert after_credential(402, None) == "refused"
        assert after_credential(403, {"error": "buyer_inactive"}) == "refused"

    async def test_same_credential_false_the_purchase_is_done_and_the_call_is_sent_once_more_without_it(self):
        funding_failed = httpx.Response(
            503,
            headers={"Retry-After": "5", "Content-Type": "application/problem+json"},
            content=json.dumps({"error": "funding_failed", "purchase": {"granted": True}, "retry": {"sameCredential": False}}).encode(),
        )
        q = Queue(problem402(challenge()), funding_failed, ok200())
        h = make(q)
        assert (await post(h, b'{"n":1}')).status_code == 200
        assert len(q.seen) == 3
        assert MPP_CREDENTIAL_HEADER not in q.seen[2].headers
        assert TERMS_ACCEPTED_HEADER not in q.seen[2].headers
        assert q.seen[2].body == b'{"n":1}'
        assert h.sleeps == [5]
        assert [p.status for p in h.purchases] == [503]
        assert h.buyer.spent_today_cents() == 500
        assert h.buyer.pending_credentials() == []

    async def test_a_503_with_retry_after_but_no_retry_member_is_not_re_presented_and_the_credential_is_held(self):
        bare = httpx.Response(503, headers={"Retry-After": "5"}, content=b"{}")
        q = Queue(problem402(challenge()), bare)
        h = make(q)
        assert (await post(h)).status_code == 503
        assert len(q.seen) == 2
        assert h.sleeps == []
        assert [r.reason for r in h.refusals] == ["settlement_outcome_unknown"]
        assert h.refusals[0].pending_credential.value == q.seen[1].headers[MPP_CREDENTIAL_HEADER]
        assert len(h.buyer.pending_credentials()) == 1
        assert h.buyer.spent_today_cents() == 500


class TestS150NeverDiscarded:
    async def test_re_presents_with_backoff_until_expiry_plus_grace_then_holds_and_exposes_the_credential(self):
        clock = {"now": NOW}
        tx = "0x" + "cd" * 32

        async def advance(seconds: float) -> None:
            clock["now"] = clock["now"] + timedelta(seconds=seconds)

        class Wallet:
            address = PAYER

            def sign_tempo_transfer(self, request):
                return SIGNED_TX

        responders: list = [problem402(challenge(method="tempo", amount_cents=7))]
        responders += [pending503(error="tempo_settlement_pending", transactionHash=tx) for _ in range(40)]
        q = Queue(*responders)
        h = make(q, card=None, stablecoin=Wallet(), now=lambda: clock["now"], sleep=advance, policy={"settlement_grace_seconds": 60})
        response = await post(h)
        assert response.status_code == 503
        # Expiry 12:05 plus 60 s: every re-presentation lands before 12:06.
        assert clock["now"] <= datetime(2026, 9, 18, 12, 6, tzinfo=timezone.utc)
        assert clock["now"] >= datetime(2026, 9, 18, 12, 5, 30, tzinfo=timezone.utc)
        assert len(q.seen) > 5
        assert len({s.headers[MPP_CREDENTIAL_HEADER] for s in q.seen[1:]}) == 1
        refusal = h.refusals[0]
        assert refusal.reason == "settlement_retries_exhausted"
        held = refusal.pending_credential
        assert (held.transaction_hash, held.payment_method, held.amount_cents, held.http_method, held.url, held.header_name) == (
            tx, "tempo", 7, "POST", RESOURCE, MPP_CREDENTIAL_HEADER,
        )
        assert h.buyer.pending_credentials() == [held]
        assert h.buyer.spent_today_cents() == 7

    async def test_the_next_call_to_the_same_resource_re_presents_the_held_credential_before_paying_again(self):
        q = Queue(problem402(challenge()), pending503(), problem402(challenge()), ok200("pi_completed"))
        h = make(q, policy={"max_settlement_retries": 0})
        assert (await post(h, b'{"call":1}')).status_code == 503
        held = h.buyer.pending_credentials()
        assert len(held) == 1

        second = await post(h, b'{"call":2}')
        assert second.status_code == 200
        # One SPT for both calls: the second call carried the held credential.
        assert len(h.card.calls) == 1
        assert q.seen[3].headers[MPP_CREDENTIAL_HEADER] == held[0].value
        assert q.seen[3].body == b'{"call":2}'
        h.verify(q.seen[3])
        assert [(p.challenge_id, p.receipt.reference) for p in h.purchases] == [(held[0].id, "pi_completed")]
        assert h.buyer.pending_credentials() == []
        assert h.buyer.spent_today_cents() == 500

    async def test_a_held_credential_the_platform_now_refuses_is_cleared_and_reported_and_nothing_new_is_paid(self):
        q = Queue(problem402(challenge()), pending503(), problem402(challenge()), problem402(challenge(), problem="payment-expired"))
        h = make(q, policy={"max_settlement_retries": 0})
        await post(h)
        out = await post(h)
        assert out.status_code == 402
        assert len(h.card.calls) == 1
        assert [r.reason for r in h.refusals] == ["settlement_retries_exhausted", "pending_credential_refused"]
        assert h.refusals[1].pending_credential.id == h.refusals[0].pending_credential.id
        assert h.buyer.pending_credentials() == []
        # Still counted: money may have moved before the doubt began.
        assert h.buyer.spent_today_cents() == 500

    async def test_an_error_while_the_credential_is_out_holds_it_and_still_reaches_the_caller(self):
        def hang_up(seen: Seen) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out")

        h = make(Queue(problem402(challenge()), hang_up))
        with pytest.raises(httpx.ReadTimeout):
            await post(h)
        assert [r.reason for r in h.refusals] == ["settlement_outcome_unknown"]
        assert len(h.buyer.pending_credentials()) == 1
        assert h.buyer.spent_today_cents() == 500

    async def test_a_held_pack_credential_from_a_door_is_re_presented_at_the_in_band_purchase_url(self):
        q = Queue(problem402(challenge()), pending503(), ok200("pi_inband_completed"))
        h = make(q, policy={"max_settlement_retries": 0})
        await post(h)
        [held] = h.buyer.pending_credentials()
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge().header, {"version": TERMS})
        assert outcome.ok
        assert len(h.card.calls) == 1
        assert q.seen[2].url == PURCHASE_URL
        assert q.seen[2].headers[MPP_CREDENTIAL_HEADER] == held.value
        assert outcome.record.challenge_id == held.id

    async def test_a_held_credential_is_never_re_presented_to_a_host_off_the_allowlist(self):
        q = Queue(problem402(challenge()), pending503())
        h = make(q, policy={"max_settlement_retries": 0})
        await post(h)
        outcome = await h.buyer.purchase("https://peer.example/buy", challenge(realm="peer.example").header)
        assert (outcome.ok, outcome.reason) == (False, "realm_not_allowed")
        assert len(q.seen) == 2
        assert len(h.buyer.pending_credentials()) == 1

    async def test_the_purchase_outcome_carries_the_held_credential(self):
        q = Queue(pending503())
        h = make(q, policy={"max_settlement_retries": 0})
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge().header, {"version": TERMS})
        assert not outcome.ok
        assert outcome.reason == "settlement_retries_exhausted"
        assert outcome.pending_credential.value == q.seen[0].headers[MPP_CREDENTIAL_HEADER]

    async def test_policy_persist_carries_the_held_credential_and_the_ledger_to_a_new_process(self):
        saved: list = []

        class Store:
            def load(self):
                return None

            def save(self, state):
                saved.append(json.loads(json.dumps(state)))

        first = make(Queue(problem402(challenge()), pending503()), policy={"max_settlement_retries": 0, "persist": Store()})
        await post(first)
        last = saved[-1]
        assert len(last["pending"]) == 1
        assert [e["cents"] for e in last["ledger"]] == [500]
        # The TypeScript MppPendingCredential, key for key.
        assert sorted(last["pending"][0]) == sorted([
            "id", "realm", "httpMethod", "url", "headerName", "value", "paymentMethod", "kind", "amountCents",
            "currency", "termsVersion", "expiresAt", "deadline", "transactionHash", "heldAt", "reason",
        ])
        assert last["pending"][0]["deadline"] == "2026-09-18T12:07:00.000Z"

        class Restored:
            async def load(self):
                return last

            def save(self, state):
                return None

        q = Queue(problem402(challenge()), ok200("pi_after_restart"))
        restarted = make(q, policy={"daily_cap_cents": 600, "persist": Restored()})
        assert (await post(restarted)).status_code == 200
        assert restarted.card.calls == []
        assert q.seen[1].headers[MPP_CREDENTIAL_HEADER] == last["pending"][0]["value"]
        assert restarted.buyer.spent_today_cents() == 500
        assert restarted.buyer.pending_credentials() == []


class TestDailyCapUnderConcurrency:
    async def test_is_reserved_before_any_await_so_concurrent_calls_cannot_all_pass(self):
        import asyncio

        responders: list = [lambda seen: problem402(challenge(amount_cents=500)) for _ in range(6)]
        responders += [lambda seen: ok200() for _ in range(6)]

        class SlowCard(Card):
            async def get_spt(self, request: SptRequest) -> str:
                self.calls.append(request)
                await asyncio.sleep(0.005)
                return "spt_test_0001"

        h = make(Queue(*responders), card=SlowCard(), policy={"daily_cap_cents": 1000})
        answers = await asyncio.gather(*(post(h) for _ in range(6)))
        assert len(h.card.calls) == 2
        assert len([r for r in answers if r.status_code == 200]) == 2
        assert h.buyer.spent_today_cents() == 1000
        assert len([r for r in h.refusals if r.reason == "over_daily_cap"]) == 4


class TestUpgradeHeaders:
    async def test_signs_a_get_on_the_https_form_of_the_socket_url_with_the_hints_covered(self):
        h = make()
        headers = await h.buyer.upgrade_headers("wss://robutler.ai/agents/acme/uamp")
        assert headers[PAYMENT_METHODS_HINT_HEADER] == "stripe"
        assert headers["signature-input"].startswith(
            'sig1=("@method" "@authority" "@path" "@query" "robutler-payment-methods" "signature-agent";key="sig1")'
        )
        assert "content-digest" not in headers
        assert AGENT_URL in headers["signature-agent"]
        assert headers["signature"].startswith("sig1=:")
        verify_with_covered_headers(
            httpx.Headers(headers), "GET", "https://robutler.ai/agents/acme/uamp", b"",
            {h.key.thumbprint: h.key.private_key.public_key()},
        )

    async def test_signs_nothing_for_a_host_the_buyer_does_not_buy_from(self):
        assert await make().buyer.upgrade_headers("wss://peer.example/agents/acme/uamp") == {}


class TestTempoPayerDid:
    def test_builds_the_did_the_platform_parses_passes_a_matching_did_through_and_refuses_anything_else(self):
        assert tempo_payer_did(4217, PAYER) == f"did:pkh:eip155:4217:{PAYER}"
        assert tempo_payer_did(4217, f"did:pkh:eip155:4217:{PAYER}") == f"did:pkh:eip155:4217:{PAYER}"
        with pytest.raises(ValueError, match="chain"):
            tempo_payer_did(4217, f"did:pkh:eip155:42431:{PAYER}")
        with pytest.raises(ValueError, match="40 hex"):
            tempo_payer_did(4217, "0xPAYER")



# --------------------------------------------------------------------------
# 2026-09-18, second fix pass: the portal door's new answers to a credential
# (the TypeScript suite's cases, one for one)
# --------------------------------------------------------------------------


def problem_response(status: int, body: dict, headers: Optional[dict] = None) -> httpx.Response:
    return httpx.Response(status, headers={"Content-Type": "application/problem+json", **(headers or {})}, content=json.dumps(body).encode())


BODY_RECEIPT = {"method": "stripe", "reference": "pi_lost_200", "status": "success", "timestamp": "2026-09-18T12:00:01.000Z"}


class TestPortalDoorAnswers:
    async def test_409_purchase_already_granted_is_paid_once_and_the_call_goes_again_without_a_credential(self):
        already_granted = problem_response(409, {
            "error": "purchase_already_granted",
            "receipt": BODY_RECEIPT,
            "purchase": {"granted": False, "receipt": BODY_RECEIPT},
            "retry": {"sameCredential": False},
        })

        def hang_up(seen: Seen) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out")

        q = Queue(problem402(challenge()), hang_up, problem402(challenge()), already_granted, ok200("served_from_balance"))
        h = make(q)
        with pytest.raises(httpx.ReadTimeout):
            await post(h, b'{"call":1}')
        [held] = h.buyer.pending_credentials()

        second = await post(h, b'{"call":2}')
        assert second.status_code == 200
        assert len(h.card.calls) == 1
        assert q.seen[3].headers[MPP_CREDENTIAL_HEADER] == held.value
        assert MPP_CREDENTIAL_HEADER not in q.seen[4].headers
        assert q.seen[4].body == b'{"call":2}'
        assert h.sleeps == []
        assert [(p.challenge_id, p.status, p.receipt.reference) for p in h.purchases] == [(held.id, 409, "pi_lost_200")]
        assert h.buyer.pending_credentials() == []
        assert h.buyer.spent_today_cents() == 500

    @pytest.mark.parametrize("code", ["call_already_served", "call_in_progress"])
    async def test_409_served_elsewhere_is_terminal_paid_never_paid_again_never_re_presented(self, code):
        q = Queue(problem_response(409, {"error": code, "challengeId": "x", "receipt": BODY_RECEIPT, "retry": {"sameCredential": False}}))
        h = make(q)
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge().header, {"version": TERMS})
        assert not outcome.ok
        assert outcome.reason == code
        assert outcome.receipt.reference == "pi_lost_200"
        assert len(q.seen) == 1
        assert h.sleeps == []
        assert len(h.card.calls) == 1
        assert [p.status for p in h.purchases] == [409]
        assert h.refusals == []
        assert h.buyer.spent_today_cents() == 500
        assert h.buyer.pending_credentials() == []

    async def test_409_call_refunded_frees_the_reservation_and_drops_the_held_credential(self):
        q = Queue(
            problem402(challenge()), pending503(), problem402(challenge()),
            problem_response(409, {"error": "call_refunded", "receipt": BODY_RECEIPT, "retry": {"sameCredential": False}}),
        )
        h = make(q, policy={"max_settlement_retries": 0})
        await post(h)
        assert h.buyer.spent_today_cents() == 500
        out = await post(h)
        assert out.status_code == 409
        assert len(h.card.calls) == 1
        assert h.buyer.spent_today_cents() == 0
        assert h.buyer.pending_credentials() == []
        assert h.purchases == []

    @pytest.mark.parametrize("code", ["metered_principal_cap", "metered_instrument_cap", "metered_platform_ceiling", "buyer_daily_cap"])
    async def test_the_velocity_429s_are_refusals_freed_reported_never_retried(self, code):
        q = Queue(problem402(challenge()), problem_response(429, {"error": code}, {"Retry-After": "3600"}))
        h = make(q)
        assert (await post(h)).status_code == 429
        assert len(q.seen) == 2
        assert h.sleeps == []
        assert h.buyer.spent_today_cents() == 0
        assert h.purchases == []
        assert [r.reason for r in h.refusals] == ["platform_refused"]
        assert code in h.refusals[0].detail
        assert h.refusals[0].challenge_id

    async def test_503_sale_serve_pending_is_re_presented_with_the_same_credential(self):
        pending_serve = problem_response(
            503, {"error": "sale_serve_pending", "challengeId": "x", "retry": {"credentialHeader": MPP_CREDENTIAL_HEADER, "sameCredential": True}},
            {"Retry-After": "5"},
        )
        q = Queue(problem402(challenge()), pending_serve, ok200())
        h = make(q)
        assert (await post(h)).status_code == 200
        assert q.seen[2].headers[MPP_CREDENTIAL_HEADER] == q.seen[1].headers[MPP_CREDENTIAL_HEADER]
        assert len(h.card.calls) == 1
        assert len(h.purchases) == 1

    @pytest.mark.parametrize("code", ["mpp_not_configured", "metered_not_ready"])
    async def test_503_no_payment_codes_with_same_credential_false_are_not_a_purchase(self, code):
        q = Queue(problem402(challenge()), problem_response(503, {"error": code, "retry": {"sameCredential": False}}, {"Retry-After": "3600"}))
        h = make(q)
        assert (await post(h)).status_code == 503
        assert len(q.seen) == 2
        assert h.purchases == []
        assert h.buyer.spent_today_cents() == 0
        assert h.buyer.pending_credentials() == []
        assert [r.reason for r in h.refusals] == ["platform_refused"]



# --------------------------------------------------------------------------
# 2026-09-18, S-154 residual: the seller is pinned (the TypeScript cases)
# --------------------------------------------------------------------------


def discovery_doc(offers: list) -> httpx.Response:
    doc = {"openapi": "3.1.0", "paths": {"/api/mpp/credits": {"post": {"x-payment-info": {"offers": offers}}}}}
    return httpx.Response(200, headers={"Content-Type": "application/json"}, content=json.dumps(doc).encode())


class TestS154SellerPinned:
    async def test_a_forged_card_challenge_with_a_foreign_profile_is_refused_and_the_genuine_one_is_paid(self):
        q = Queue(problem402(challenge(network_id="profile_attacker")), problem402(challenge()), ok200())
        h = make(q)
        assert (await post(h)).status_code == 402
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]
        assert "profile_attacker" in h.refusals[0].detail
        assert h.buyer.spent_today_cents() == 0
        assert (await post(h)).status_code == 200
        assert len(h.card.calls) == 1

    async def test_an_in_band_challenge_naming_a_foreign_profile_is_refused_with_no_request(self):
        q = Queue()
        h = make(q)
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge(network_id="profile_attacker").header, {"version": TERMS})
        assert (outcome.ok, outcome.reason) == (False, "seller_not_pinned")
        assert q.seen == []
        assert h.card.calls == []

    async def test_a_forged_tempo_challenge_naming_a_foreign_recipient_is_refused_before_the_wallet_signs(self):
        signed: list = []

        class Wallet:
            address = PAYER

            def sign_tempo_transfer(self, request):
                signed.append(request)
                return SIGNED_TX

        q = Queue(
            problem402(challenge(method="tempo", amount_cents=7, recipient="0xATTACKER")),
            problem402(challenge(method="tempo", amount_cents=7, recipient="0xdeposit")),
            ok200(),
        )
        h = make(q, card=None, stablecoin=Wallet())
        assert (await post(h)).status_code == 402
        assert signed == []
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]
        # Addresses compare case-insensitively.
        assert (await post(h)).status_code == 200
        assert len(signed) == 1

    async def test_with_no_explicit_pin_the_pin_is_read_once_from_the_platform_discovery_document(self):
        q = Queue(
            problem402(challenge()),
            discovery_doc([{"intent": "charge", "method": "stripe", "amount": None, "currency": "usd", "payTo": "profile_test"}]),
            ok200(),
            problem402(challenge(network_id="profile_attacker")),
        )
        h = make(q, policy={"stripe_profile_id": None})
        assert (await post(h)).status_code == 200
        assert (q.seen[1].method, q.seen[1].url) == ("GET", "https://robutler.ai/openapi.json")
        # Unsigned: discovery is public, and nothing identifies the agent to it.
        assert "signature-input" not in q.seen[1].headers
        assert (await post(h)).status_code == 402
        assert len([s for s in q.seen if s.url.endswith("/openapi.json")]) == 1
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]
        assert len(h.card.calls) == 1

    @pytest.mark.parametrize(
        "offers",
        [
            [{"intent": "charge", "method": "stripe", "amount": None, "currency": "usd"}],
            [{"method": "stripe", "payTo": "profile_test"}, {"method": "stripe", "payTo": "profile_other"}],
        ],
    )
    async def test_fails_closed_when_discovery_names_no_seller_or_two(self, offers):
        h = make(Queue(problem402(challenge()), discovery_doc(offers)), policy={"stripe_profile_id": None})
        assert (await post(h)).status_code == 402
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]

    async def test_discovery_goes_over_https_even_for_a_plaintext_platform_and_a_failed_fetch_pays_nothing(self):
        def refused(seen: Seen) -> httpx.Response:
            raise httpx.ConnectError("connect ECONNREFUSED")

        q = Queue(problem402(challenge(realm="localhost")), refused)
        h = make(q, policy={"stripe_profile_id": None, "realms": ["localhost:3000"]})
        response = await h.buyer.paying_request("POST", "http://localhost:3000/agents/acme/v1/chat/completions", content=b"{}")
        assert response.status_code == 402
        assert q.seen[1].url == "https://localhost:3000/openapi.json"
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]
        assert seller_pins_from_discovery(
            {"paths": {"/x": {"post": {"x-payment-info": {"offers": [{"method": "tempo", "extra": {"recipient": "0xABC"}}]}}}}}
        ) == MppSellerPins(stripe_profile_id=None, tempo_deposit_address="0xabc")
        assert seller_pins_from_discovery(None) == MppSellerPins()


# --------------------------------------------------------------------------
# 2026-09-19 review, the twins of the TypeScript cases of the same day: no
# redirect is ever followed (S-180), the credential header is checked when the
# challenge is read, a discovery pin is believed for an hour, and the URL sent
# is the URL signed.
# --------------------------------------------------------------------------

from webagents.agents.skills.robutler.payments_x402.mpp_buyer import (  # noqa: E402 - grouped with its cases
    CREDENTIAL_HEADERS_REFUSED,
    MppRedirectError,
    credential_header_refusal,
)

ELSEWHERE = "https://elsewhere.example/collect"


class Hosts:
    """A transport for two hosts: the allowlisted platform answers from a
    queue, and everything any OTHER host hears is recorded. The assertion
    that matters in the S-180 cases is that `elsewhere` stays empty."""

    def __init__(self, *platform: Responder) -> None:
        self.platform = Queue(*platform)
        self.elsewhere: List[Seen] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "robutler.ai":
            return self.platform(request)
        self.elsewhere.append(Seen(request.method, str(request.url), request.headers, request.content, []))
        return httpx.Response(200, json={"ok": True})

    def client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self), **kwargs)


def redirect(status: int = 307) -> httpx.Response:
    return httpx.Response(status, headers={"Location": ELSEWHERE})


class TestS180NoRedirectIsFollowed:
    async def test_the_hazard_is_real_a_client_that_follows_carries_the_credential_header_along(self):
        # What `client.send(request)` did for an operator's client built with
        # follow_redirects=True, and why the buyer now says False itself.
        hosts = Hosts(redirect())
        async with hosts.client(follow_redirects=True) as client:
            await client.post(RESOURCE, headers={MPP_CREDENTIAL_HEADER: "Payment abc", TERMS_ACCEPTED_HEADER: TERMS}, content=b"{}")
        assert [s.headers.get(MPP_CREDENTIAL_HEADER) for s in hosts.elsewhere] == ["Payment abc"]
        assert hosts.elsewhere[0].headers.get(TERMS_ACCEPTED_HEADER) == TERMS

    @pytest.mark.parametrize("follow", [True, False])
    async def test_a_paid_retry_answered_with_a_redirect_goes_nowhere_spends_nothing_and_is_typed(self, follow):
        hosts = Hosts(problem402(challenge()), redirect(307))
        h = make(client=hosts.client(follow_redirects=follow), policy={"daily_cap_cents": 5000})
        with pytest.raises(MppRedirectError) as raised:
            await post(h)
        err = raised.value
        assert (err.code, err.status, err.location, err.followed) == ("redirect_refused", 307, ELSEWHERE, False)
        assert err.response is not None and err.response.status_code == 307

        # The credential left once, to the allowlisted host, and went nowhere else.
        assert [MPP_CREDENTIAL_HEADER in s.headers for s in hosts.platform.seen] == [False, True]
        assert hosts.elsewhere == []
        # Spends nothing, releases the reservation, holds nothing, and is not a purchase.
        assert len(h.card.calls) == 1
        assert h.buyer.spent_today_cents() == 0
        assert h.buyer.pending_credentials() == []
        assert h.purchases == []
        assert [r.reason for r in h.refusals] == ["redirect_refused"]
        assert (h.refusals[0].amount_cents, h.refusals[0].method, h.refusals[0].pending_credential) == (500, "stripe", None)

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    async def test_the_first_request_answered_with_a_redirect_is_not_followed_either(self, status):
        hosts = Hosts(redirect(status))
        h = make(client=hosts.client(follow_redirects=True))
        with pytest.raises(MppRedirectError):
            await post(h)
        assert len(hosts.platform.seen) == 1
        assert hosts.elsewhere == []
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["redirect_refused"]

    async def test_a_304_or_a_300_is_an_answer_not_a_redirect(self):
        for status in (300, 304):
            h = make(Queue(httpx.Response(status)))
            assert (await post(h)).status_code == status
            assert h.refusals == []

    async def test_an_in_band_purchase_answered_with_a_redirect_is_an_outcome_not_a_followed_request(self):
        hosts = Hosts(redirect(308))
        h = make(client=hosts.client(follow_redirects=True))
        outcome = await h.buyer.purchase(PURCHASE_URL, challenge().header, {"version": TERMS})
        assert (outcome.ok, outcome.reason, outcome.status, outcome.response, outcome.pending_credential) == (
            False, "redirect_refused", None, None, None,
        )
        assert len(hosts.platform.seen) == 1 and MPP_CREDENTIAL_HEADER in hosts.platform.seen[0].headers
        assert hosts.elsewhere == []
        assert h.buyer.spent_today_cents() == 0
        assert [r.reason for r in h.refusals] == ["redirect_refused"]

    async def test_a_redirect_answered_to_a_credential_already_in_doubt_keeps_it_held(self):
        hosts = Hosts(problem402(challenge()), pending503(), redirect(302))
        h = make(client=hosts.client())
        with pytest.raises(MppRedirectError) as raised:
            await post(h)
        # A redirect says nothing about whether it settled: still counted, still held (S-150).
        assert h.buyer.spent_today_cents() == 500
        assert len(h.buyer.pending_credentials()) == 1
        assert [r.reason for r in h.refusals] == ["redirect_refused"]
        assert h.refusals[0].pending_credential is not None
        assert raised.value.pending_credential == h.buyer.pending_credentials()[0]
        assert hosts.elsewhere == []

    async def test_the_retry_without_a_credential_refuses_a_redirect_too(self):
        funding_failed = problem_response(503, {"error": "funding_failed", "retry": {"sameCredential": False}}, {"Retry-After": "0"})
        hosts = Hosts(problem402(challenge()), funding_failed, redirect(307))
        h = make(client=hosts.client(follow_redirects=True))
        with pytest.raises(MppRedirectError):
            await post(h)
        # The purchase itself settled and stays counted; only the follow-up call was refused.
        assert len(h.purchases) == 1
        assert h.buyer.spent_today_cents() == 500
        assert hosts.elsewhere == []


class TestCredentialHeaderCheckedWhenTheChallengeIsRead:
    REFUSED = [
        "Content-Digest", "signature", "Signature-Input", "Signature-Agent",
        TERMS_ACCEPTED_HEADER, PURCHASE_HINT_HEADER, PAYMENT_METHODS_HINT_HEADER,
        "Host", "Content-Length", "Transfer-Encoding", "Connection", "Cookie",
        "bad name", "@method", "x:y",
    ]

    def test_a_field_the_signer_the_buyer_or_the_http_client_owns_makes_the_challenge_unreadable(self):
        for name in self.REFUSED:
            assert read_mpp_challenge(challenge(header=name).header) is None, name
            assert isinstance(credential_header_refusal(name), str), name
        # The platform's name, the core spec's default (named or absent), and an ordinary extension field are fine.
        assert read_mpp_challenge(challenge().header).credential_header == MPP_CREDENTIAL_HEADER
        assert read_mpp_challenge(challenge(header="authorization").header).credential_header == "authorization"
        assert read_mpp_challenge(challenge(header=None).header).credential_header == "Authorization"
        assert read_mpp_challenge(challenge(header="X-Payment-Credential").header).credential_header == "X-Payment-Credential"
        assert {"content-digest", "signature", "signature-input", "signature-agent"} <= CREDENTIAL_HEADERS_REFUSED

    async def test_it_is_refused_before_the_source_is_asked(self):
        for name in ("Content-Digest", "Signature", "Content-Length"):
            q = Queue(problem402(challenge(header=name)))
            h = make(q, policy={"daily_cap_cents": 5000})
            # It used to raise out of the signer (or out of the HTTP client) AFTER get_spt, and hold the credential against the cap.
            assert (await post(h)).status_code == 402
            assert h.card.calls == []
            assert len(q.seen) == 1
            assert h.buyer.spent_today_cents() == 0
            assert h.buyer.pending_credentials() == []

            in_band = make(Queue())
            outcome = await in_band.buyer.purchase(PURCHASE_URL, challenge(header=name).header, {"version": TERMS})
            assert (outcome.ok, outcome.reason) == (False, "no_challenge")
            assert in_band.card.calls == []

    async def test_a_paid_retry_the_signer_refuses_before_it_is_sent_releases_the_reservation(self):
        from webagents.crypto.http_signature import SigningError

        # A Terms version the operator's callback accepts but that cannot be
        # sent or covered (not ASCII). It rides the problem BODY here: httpx
        # will not even build a response header out of it.
        odd_version = "v1" + chr(0xE9)
        unsendable = problem402(challenge(), terms_version=None, body={"terms": {"url": TERMS_URL, "version": odd_version}})
        q = Queue(unsendable)
        h = make(q, policy={"accept_terms": lambda terms: True, "daily_cap_cents": 5000})
        with pytest.raises((SigningError, UnicodeEncodeError)):
            await post(h)
        assert len(h.card.calls) == 1
        assert len(q.seen) == 1
        assert h.buyer.spent_today_cents() == 0
        assert h.buyer.pending_credentials() == []


class TestDiscoveryPinIsBelievedForAnHour:
    async def test_a_rotated_deposit_address_is_picked_up_after_an_hour_without_a_restart(self):
        clock = {"now": NOW}
        signed: List[TempoTransferRequest] = []

        class Wallet:
            address = PAYER

            def sign_tempo_transfer(self, request: TempoTransferRequest) -> str:
                signed.append(request)
                return SIGNED_TX

        def tempo_offer(recipient: str) -> httpx.Response:
            return discovery_doc([{"intent": "charge", "method": "tempo", "recipient": recipient}])

        q = Queue(
            problem402(challenge(method="tempo", recipient="0xDEPOSIT", expires=None)),
            tempo_offer("0xDEPOSIT"),
            ok200(),
            # 59 minutes later: still believed, so the rotated address is refused and nothing is fetched.
            problem402(challenge(method="tempo", recipient="0xROTATED", expires=None)),
            # 61 minutes later: read again.
            problem402(challenge(method="tempo", recipient="0xROTATED", expires=None)),
            tempo_offer("0xROTATED"),
            ok200(),
        )
        h = make(q, card=None, stablecoin=Wallet(), policy={"tempo_deposit_address": None}, now=lambda: clock["now"])
        assert (await post(h)).status_code == 200

        clock["now"] = NOW + timedelta(minutes=59)
        assert (await post(h)).status_code == 402
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]
        assert len([s for s in q.seen if s.url.endswith("/openapi.json")]) == 1

        clock["now"] = NOW + timedelta(minutes=61)
        assert (await post(h)).status_code == 200
        assert len([s for s in q.seen if s.url.endswith("/openapi.json")]) == 2
        assert [r.recipient for r in signed] == ["0xDEPOSIT", "0xROTATED"]

    async def test_an_expired_pin_is_never_a_fallback(self):
        def unreachable(_seen: Seen) -> httpx.Response:
            raise httpx.ConnectError("connect ETIMEDOUT")

        two_sellers = discovery_doc([{"method": "stripe", "payTo": "profile_test"}, {"method": "stripe", "payTo": "profile_other"}])
        for reread in (unreachable, two_sellers):
            clock = {"now": NOW}
            q = Queue(problem402(challenge()), discovery_doc([{"method": "stripe", "payTo": "profile_test"}]), ok200(), problem402(challenge()), reread)
            h = make(q, policy={"stripe_profile_id": None}, now=lambda: clock["now"])
            assert (await post(h)).status_code == 200
            clock["now"] = NOW + timedelta(minutes=61)
            assert (await post(h)).status_code == 402
            assert len(h.card.calls) == 1
            assert [r.reason for r in h.refusals] == ["seller_not_pinned"]


class TestTheUrlSentIsTheUrlSigned:
    async def test_a_dot_segment_url_is_signed_and_sent_in_the_whatwg_spelling_and_verifies(self):
        # `%2e%2e` is resolved by the platform's parser and by the TypeScript
        # signer, and httpx sends it as written: signed and sent must agree.
        q = Queue(problem402(challenge()), ok200())
        h = make(q)
        response = await h.buyer.paying_request("POST", "https://robutler.ai/agents/x/%2e%2e/acme/./v1/chat/completions?a=b c", content=b"{}")
        assert response.status_code == 200
        for sent in q.seen:
            assert sent.url == "https://robutler.ai/agents/acme/v1/chat/completions?a=b%20c"
            h.verify(sent)
