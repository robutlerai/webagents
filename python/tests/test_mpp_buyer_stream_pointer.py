"""
Two 2026-09-19 changes to the MPP buyer, each pinned from the outside; the
Python twin of
`typescript/tests/unit/skills/payments/mpp-buyer-stream-pointer.test.ts`.

A STREAMED ANSWER IS NEVER READ (finding sdk-1). Every send went through
httpx's non-streaming `send`, which reads the whole body before it returns,
so a streamed completion after a paid retry reached the caller only once its
LAST chunk had been sent, buffered whole. The first half of this file is not
a mock: a real HTTP server writes one chunk, then WAITS for the test to say so
before it writes the last, and the assertion is that the caller already holds
the first chunk while the server is still waiting.

THE PURCHASE POINTER. Both LLM rails now answer a token holder whose token ran
dry with an `mpp` requirement that names `purchase_url` and carries NO
challenge (the portal's lib/payments/purchase-pointer.ts: nobody on those
rails was verified, so nothing could be minted for them). Both SDKs ignored
it. The second half pins what following it means: the buyer's own signed
request to the purchase URL, the challenge it is answered with paid under the
unchanged policy (realm allowlist S-147, seller pin S-154, the caps), and the
original call sent once more without its payment token.

THE CEILING (the same day). A pointer carries no secret, so a peer behind an
allowlisted host can write one, call after call. Only `daily_cap_cents` bounds
the total, so a buyer with none follows no pointer (`pointer_needs_daily_cap`)
and still pays a challenge on a URL its caller chose. Every test that expects
a pointer to be followed builds its buyer with `with_queue`, which sets a cap
and says so; the tests of the ceiling itself use `uncapped`.

The purchase 200 fixture is the body the portal sends from 2026-09-19: no
`credits` and no `creditsNano` (the granted amount is no longer stated).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import pytest

from webagents.agents.skills.robutler.payments_x402.mpp_buyer import (
    BUYER_BODY_MAX_BYTES,
    MPP_BUYER_REFUSAL_REASONS,
    MPP_CREDENTIAL_HEADER,
    MPP_RECEIPT_HEADER,
    TERMS_ACCEPTED_HEADER,
    TERMS_VERSION_HEADER,
    MppBuyer,
    MppBuyerPolicy,
    MppRequirementEntry,
    MppTermsRequest,
    base64url_encode,
    buyer_reads_body,
    is_json_media_type,
    jcs_canonicalize,
    mpp_requirement_of,
)

from .test_mpp_buyer import AGENT_URL, TERMS, TERMS_URL, Card, Queue, new_key

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

_counter = 0


def challenge_header(realm: str, *, cents: int = 500, network_id: str = "profile_test") -> str:
    global _counter
    _counter += 1
    expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    fields = [
        ("id", base64url_encode(hashlib.sha256(f"stream pointer challenge {_counter}".encode()).digest())),
        ("realm", realm),
        ("method", "stripe"),
        ("intent", "charge"),
        ("expires", expires),
        ("request", base64url_encode(jcs_canonicalize({"amount": str(cents), "currency": "usd", "methodDetails": {"networkId": network_id, "paymentMethodTypes": ["card"]}}))),
        ("opaque", base64url_encode(jcs_canonicalize({"packId": "mpp_5", "kind": "pack", "terms": TERMS}))),
        ("header", MPP_CREDENTIAL_HEADER),
    ]
    return "Payment " + ", ".join(f'{k}="{v}"' for k, v in fields)


def receipt_header(reference: str = "pi_test_1") -> str:
    return base64url_encode(jcs_canonicalize({"method": "stripe", "reference": reference, "status": "success", "timestamp": "2026-09-19T12:00:00.000Z"}))


class Harness:
    def __init__(self, realms: List[str], client: Optional[httpx.AsyncClient], policy: Optional[dict] = None) -> None:
        self.card = Card()
        self.purchases: list = []
        self.refusals: list = []

        async def no_sleep(_seconds: float) -> None:
            return None

        self.buyer = MppBuyer(
            keys=[new_key()],
            agent_url=AGENT_URL,
            policy=MppBuyerPolicy(**{
                "max_per_purchase_cents": 2000,
                "accept_terms": TERMS,
                "realms": realms,
                "stripe_profile_id": "profile_test",
                **(policy or {}),
            }),
            card=self.card,
            client=client,
            sleep=no_sleep,
            on_purchase=self.purchases.append,
            on_refusal=self.refusals.append,
        )


# --------------------------------------------------------------------------
# sdk-1: a streamed answer reaches the caller as it arrives
# --------------------------------------------------------------------------


class SlowServer:
    """A real HTTP/1.1 server whose paid answer writes `first`, then waits for
    `release()` before it writes `last` and ends. No Content-Length on that
    answer: it is framed as `Transfer-Encoding: chunked`."""

    def __init__(self, answer_headers: Dict[str, str], first: bytes, last: bytes) -> None:
        self.answer_headers = answer_headers
        self.first, self.last = first, last
        self.released = asyncio.Event()
        self.last_sent = False
        self.paid_requests = 0
        self.host = ""
        self._server: Optional[asyncio.AbstractServer] = None

    async def __aenter__(self) -> "SlowServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.host = f"127.0.0.1:{self._server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.released.set()
        assert self._server is not None
        self._server.close()

    @property
    def origin(self) -> str:
        return f"http://{self.host}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
                headers = {k.strip().lower(): v.strip() for k, v in (line.split(":", 1) for line in head.split("\r\n")[1:] if ":" in line)}
                length = int(headers.get("content-length", "0"))
                if length:
                    await reader.readexactly(length)
                if MPP_CREDENTIAL_HEADER.lower() not in headers:
                    body = json.dumps({"status": 402, "terms": {"url": TERMS_URL, "version": TERMS}}).encode()
                    writer.write(self._head(402, "Payment Required", {
                        "WWW-Authenticate": challenge_header(self.host),
                        "Content-Type": "application/problem+json",
                        TERMS_VERSION_HEADER: TERMS,
                        "Content-Length": str(len(body)),
                    }) + body)
                    await writer.drain()
                    continue
                self.paid_requests += 1
                writer.write(self._head(200, "OK", {**self.answer_headers, "Transfer-Encoding": "chunked"}) + self._chunk(self.first))
                await writer.drain()
                await self.released.wait()
                self.last_sent = True
                writer.write(self._chunk(self.last) + b"0\r\n\r\n")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    @staticmethod
    def _head(status: int, reason: str, headers: Dict[str, str]) -> bytes:
        return (f"HTTP/1.1 {status} {reason}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n").encode("latin-1")

    @staticmethod
    def _chunk(data: bytes) -> bytes:
        return f"{len(data):x}\r\n".encode() + data + b"\r\n"


STREAMS = [
    pytest.param({"Content-Type": "text/event-stream", MPP_RECEIPT_HEADER: receipt_header("pi_stream")}, b'data: {"n":1}\n\n', b"data: [DONE]\n\n", id="an event stream"),
    # A JSON content type is not a reason to read a 2xx either: with no declared length it is a stream somebody is waiting on.
    pytest.param({"Content-Type": "application/json", MPP_RECEIPT_HEADER: receipt_header("pi_stream")}, b'{"choices":[', b"]}", id="a chunked JSON answer"),
]


@pytest.mark.parametrize("own_client", [True, False], ids=["a client the buyer opens", "the operator's client"])
@pytest.mark.parametrize("headers,first,last", STREAMS)
async def test_the_first_chunk_reaches_the_caller_before_the_last_is_sent(headers, first, last, own_client):
    async with SlowServer(headers, first, last) as server:
        operator_client = None if own_client else httpx.AsyncClient(timeout=5.0)
        try:
            h = Harness([server.host], operator_client)
            # The defect: this did not return until the server had sent its LAST chunk.
            response = await asyncio.wait_for(
                h.buyer.paying_request("POST", f"{server.origin}/agents/acme/v1/chat/completions", content=b'{"stream":true}'),
                timeout=2.0,
            )
            assert response.status_code == 200
            assert server.paid_requests == 1
            # Handed over as it arrived: unread and open.
            assert not response.is_closed and not response.is_stream_consumed

            body = response.aiter_bytes()
            assert await asyncio.wait_for(body.__anext__(), timeout=2.0) == first
            # The whole point: the caller holds the first chunk and the server has not sent the last.
            assert server.last_sent is False

            # "Was this paid" was decided from the head: the status, and the receipt header for the record.
            assert len(h.purchases) == 1
            assert h.purchases[0].status == 200 and h.purchases[0].amount_cents == 500
            assert h.purchases[0].receipt.reference == "pi_stream"

            server.released.set()
            rest = b"".join([chunk async for chunk in body])
            assert rest == last
            assert server.last_sent is True
            await response.aclose()
        finally:
            if operator_client is not None:
                await operator_client.aclose()


async def test_an_answer_the_buyer_did_read_is_a_read_response_for_the_caller_as_it_always_was():
    async with SlowServer({"Content-Type": "text/event-stream"}, b"x", b"y") as server:
        # The policy refuses the challenge, so the 402 itself comes back: read, closed, `.json()` works.
        refusing = Harness([server.host], None, policy={"max_per_purchase_cents": 100})
        response = await refusing.buyer.paying_request("POST", f"{server.origin}/x", content=b"{}")
        assert response.status_code == 402
        assert response.is_closed
        assert response.json()["status"] == 402
        assert [r.reason for r in refusing.refusals] == ["over_max_per_purchase"]


class TestWhichBodiesTheBuyerReads:
    def test_decides_from_the_head(self):
        def head(status: int, headers: Dict[str, str]) -> httpx.Response:
            return httpx.Response(status, headers=headers)

        assert is_json_media_type("application/json; charset=utf-8")
        assert is_json_media_type("application/problem+json")
        assert not is_json_media_type("text/event-stream")
        assert not is_json_media_type("application/jsonl")
        assert not is_json_media_type(None)

        # The platform's own error answers are chunked: no declared length is needed off a 2xx.
        assert buyer_reads_body(head(402, {"Content-Type": "application/problem+json"}))
        assert buyer_reads_body(head(503, {"Content-Type": "application/json"}))
        assert not buyer_reads_body(head(503, {"Content-Type": "text/html"}))
        assert not buyer_reads_body(head(402, {"Content-Type": "application/json", "Content-Length": str(BUYER_BODY_MAX_BYTES + 1)}))

        assert not buyer_reads_body(head(200, {"Content-Type": "application/json"}))
        assert buyer_reads_body(head(200, {"Content-Type": "application/json", "Content-Length": "11"}))
        assert not buyer_reads_body(head(200, {"Content-Type": "text/event-stream", "Content-Length": "11"}))
        # The answer of a purchase URL the buyer itself called is a purchase document whatever its framing.
        assert buyer_reads_body(head(200, {"Content-Type": "application/json"}), purchase_document=True)
        assert not buyer_reads_body(head(200, {"Content-Type": "text/plain"}), purchase_document=True)

    async def test_an_error_body_past_the_cap_is_unreadable_to_the_buyer_and_whole_for_the_caller(self):
        huge = json.dumps({"retry": {"sameCredential": True}, "pad": "x" * (BUYER_BODY_MAX_BYTES * 2)}).encode()

        class Chunked(httpx.AsyncByteStream):
            """No declared length, 8 KiB at a time, and it counts what was pulled."""

            def __init__(self) -> None:
                self.pulled = 0

            async def __aiter__(self):
                for i in range(0, len(huge), 8192):
                    self.pulled += 1
                    yield huge[i : i + 8192]

        stream = Chunked()
        q = Queue(
            httpx.Response(402, headers={"WWW-Authenticate": challenge_header("robutler.ai"), "Content-Type": "application/problem+json", TERMS_VERSION_HEADER: TERMS}, content=b"{}"),
            httpx.Response(503, headers={"Content-Type": "application/problem+json"}, stream=stream),
        )
        h = Harness(["robutler.ai"], q.client())
        response = await h.buyer.paying_request("POST", "https://robutler.ai/agents/acme/v1/chat/completions", content=b"{}")
        assert response.status_code == 503
        # `sameCredential: true` was past the cap, so this is a 5xx that says nothing: held, never re-presented blind.
        assert [r.reason for r in h.refusals] == ["settlement_outcome_unknown"]
        assert len(h.buyer.pending_credentials()) == 1
        # Nothing past the cap was buffered, and nothing the buyer took is lost to the caller.
        assert stream.pulled == BUYER_BODY_MAX_BYTES // 8192 + 1
        assert not response.is_closed
        assert await response.aread() == huge


# --------------------------------------------------------------------------
# The purchase pointer
# --------------------------------------------------------------------------

RESOURCE = "https://robutler.ai/api/llm/chat/completions"
PURCHASE_URL = "https://robutler.ai/api/mpp/credits"


def pointer402(purchase_url: str = PURCHASE_URL) -> httpx.Response:
    """The HTTP twin's 402 to a token holder: the token scheme first, then the pointer; no `WWW-Authenticate: Payment`."""
    return httpx.Response(402, json={
        "error": "Budget exhausted",
        "details": "Insufficient token balance",
        "requirements": {
            "amount": "0.142857", "currency": "USD",
            "schemes": [{"scheme": "token"}, {"scheme": "mpp", "purchase_url": purchase_url}],
            "reason": "Insufficient token balance",
        },
    })


def challenge402(*, realm: str = "robutler.ai", cents: int = 500, network_id: str = "profile_test") -> httpx.Response:
    """What the purchase URL answers a signed request that carries no credential (the portal's `handlePurchase`)."""
    return httpx.Response(
        402,
        headers={
            "WWW-Authenticate": challenge_header(realm, cents=cents, network_id=network_id),
            "Content-Type": "application/problem+json",
            TERMS_VERSION_HEADER: TERMS,
            "Link": f'<{TERMS_URL}>; rel="terms-of-service"',
        },
        content=json.dumps({"type": "https://paymentauth.org/problems/payment-required", "status": 402, "terms": {"url": TERMS_URL, "version": TERMS}}).encode(),
    )


#: The purchase 200 as the portal sends it from 2026-09-19: `credits` and `creditsNano` are gone.
PURCHASE_200_BODY = {
    "ok": True,
    "packId": "mpp_5",
    "granted": True,
    "balance": {"nano": "5000000000"},
    "paymentIntentId": "pi_pointer",
    "receipt": {"status": "success", "method": "stripe", "timestamp": "2026-09-19T12:00:00.000Z", "reference": "pi_pointer"},
}


def purchased200() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"Content-Type": "application/json", "Cache-Control": "no-store", MPP_RECEIPT_HEADER: receipt_header("pi_pointer")},
        content=json.dumps(PURCHASE_200_BODY).encode(),
    )


def served200() -> httpx.Response:
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=b'data: {"ok":true}\n\n')


TOKEN_HEADERS = {"X-Payment-Token": "jwt.dry.token", "X-Chat-Id": "chat-1", "Content-Type": "application/json"}


#: What makes a buyer one that FOLLOWS pointers: a configured daily cap, and nothing else.
DAILY_CAP_CENTS = 5000


def with_queue(*responders: Any, policy: Optional[dict] = None):
    """A buyer that follows pointers (it has a daily cap), over a queue of answers."""
    q = Queue(*responders)
    return q, Harness(["robutler.ai"], q.client(), policy={"daily_cap_cents": DAILY_CAP_CENTS, **(policy or {})})


def uncapped(*responders: Any):
    """The buyer the ceiling tests are about: the same policy with NO daily cap."""
    q = Queue(*responders)
    return q, Harness(["robutler.ai"], q.client())


async def token_call(h: Harness, url: str = RESOURCE, headers: Optional[dict] = None) -> httpx.Response:
    return await h.buyer.paying_request("POST", url, content=b'{"messages":[]}', headers=headers or TOKEN_HEADERS)


class TestThePurchasePointerThroughPayingFetch:
    async def test_asks_the_purchase_url_as_itself_pays_the_challenge_and_sends_the_call_again_without_its_token(self):
        q, h = with_queue(pointer402(), challenge402(), purchased200(), served200())
        response = await token_call(h)
        assert response.status_code == 200
        assert (await response.aread()) == b'data: {"ok":true}\n\n'

        assert [f"{s.method} {s.url}" for s in q.seen] == [f"POST {RESOURCE}", f"POST {PURCHASE_URL}", f"POST {PURCHASE_URL}", f"POST {RESOURCE}"]
        # Every one of them is the buyer's own signed request.
        for seen in q.seen:
            assert 'tag="web-bot-auth"' in seen.headers["signature-input"]
        # The ask carries no credential and no body; the payment carries the credential and the assent, both covered.
        assert MPP_CREDENTIAL_HEADER not in q.seen[1].headers
        assert q.seen[1].body == b""
        assert q.seen[2].headers[MPP_CREDENTIAL_HEADER].startswith("Payment ")
        assert q.seen[2].headers[TERMS_ACCEPTED_HEADER] == TERMS
        assert {MPP_CREDENTIAL_HEADER.lower(), TERMS_ACCEPTED_HEADER.lower()} <= {c.strip('"') for c in q.seen[2].covered}
        # The caller's token never goes to the purchase URL, and the credential never goes to the resource.
        assert "x-payment-token" not in q.seen[1].headers
        assert "x-payment-token" not in q.seen[2].headers
        # The original call as it was, then the same call minus the token that ran dry: the
        # door serves the bought balance only to a signed request that names no token.
        assert q.seen[0].headers["x-payment-token"] == "jwt.dry.token"
        assert "x-payment-token" not in q.seen[3].headers
        assert q.seen[3].headers["x-chat-id"] == "chat-1"
        assert MPP_CREDENTIAL_HEADER not in q.seen[3].headers
        assert q.seen[3].body == b'{"messages":[]}'

        # One purchase, at the purchase URL, counted like any other.
        assert len(h.card.calls) == 1
        assert len(h.purchases) == 1
        record = h.purchases[0]
        assert (record.url, record.amount_cents, record.status, record.terms_version) == (PURCHASE_URL, 500, 200, TERMS)
        assert record.receipt.reference == "pi_pointer"
        assert h.buyer.spent_today_cents() == 500
        assert h.refusals == []

    async def test_a_token_in_the_query_or_in_x_payment_is_dropped_from_the_resend_too(self):
        q, h = with_queue(pointer402(), challenge402(), purchased200(), served200())
        await token_call(h, f"{RESOURCE}?payment_token=jwt&model=auto", headers={"X-PAYMENT": "x402-token"})
        assert q.seen[0].url == f"{RESOURCE}?payment_token=jwt&model=auto"
        assert q.seen[3].url == f"{RESOURCE}?model=auto"
        assert "x-payment" not in q.seen[3].headers

    async def test_s147_a_pointer_named_by_a_host_off_the_allowlist_is_not_followed_and_nothing_is_sent(self):
        q, h = with_queue(pointer402())
        response = await token_call(h, "https://delegate.example/chat/completions")
        assert response.status_code == 402
        assert response.json()["error"] == "Budget exhausted"
        assert len(q.seen) == 1
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["realm_not_allowed"]
        assert h.buyer.spent_today_cents() == 0

    async def test_s147_a_pointer_to_a_host_off_the_allowlist_is_not_followed(self):
        q, h = with_queue(pointer402("https://collector.example/api/mpp/credits"))
        assert (await token_call(h)).status_code == 402
        assert len(q.seen) == 1
        assert [r.reason for r in h.refusals] == ["realm_not_allowed"]

    async def test_s147_the_challenge_the_purchase_url_answers_with_must_name_that_host_as_its_realm(self):
        q, h = with_queue(pointer402(), challenge402(realm="other.example"))
        assert (await token_call(h)).status_code == 402
        assert len(q.seen) == 2
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["realm_not_allowed"]

    async def test_s154_a_challenge_at_the_purchase_url_that_names_another_seller_is_never_paid(self):
        q, h = with_queue(pointer402(), challenge402(network_id="profile_attacker"))
        response = await token_call(h)
        # The caller gets the 402 it was answered, not the purchase URL's.
        assert response.json()["error"] == "Budget exhausted"
        assert len(q.seen) == 2
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["seller_not_pinned"]
        assert h.buyer.spent_today_cents() == 0

    async def test_the_caps_hold(self):
        _, over = with_queue(pointer402(), challenge402(cents=2500))
        assert (await token_call(over)).status_code == 402
        assert [r.reason for r in over.refusals] == ["over_max_per_purchase"]

        _, capped = with_queue(pointer402(), challenge402(), purchased200(), served200(), pointer402(), challenge402(), policy={"daily_cap_cents": 800})
        assert (await token_call(capped)).status_code == 200
        assert (await token_call(capped)).status_code == 402
        assert [r.reason for r in capped.refusals] == ["over_daily_cap"]
        assert len(capped.card.calls) == 1
        assert capped.buyer.spent_today_cents() == 500

    async def test_max_purchases_per_call_a_second_pointer_on_the_same_call_buys_nothing_more(self):
        q, h = with_queue(pointer402(), challenge402(), purchased200(), pointer402())
        assert (await token_call(h)).status_code == 402
        assert len(q.seen) == 4
        assert len(h.card.calls) == 1
        assert len(h.purchases) == 1
        assert [r.reason for r in h.refusals] == ["purchase_limit_per_call"]

    @pytest.mark.parametrize("answer", [
        lambda: httpx.Response(402, json={"error": "Budget exhausted"}),
        lambda: httpx.Response(402, headers={"Content-Type": "text/plain"}, content=b"Payment Required"),
        lambda: httpx.Response(402, json={"requirements": {"schemes": [{"scheme": "token"}, {"scheme": "mpp"}]}}),
        lambda: httpx.Response(402, json={"requirements": {"schemes": [{"scheme": "mpp", "purchase_url": PURCHASE_URL, "challenge": "  "}]}}),
    ])
    async def test_a_402_that_names_no_pointer_is_returned_as_it_came(self, answer):
        q, h = uncapped(answer())
        assert (await token_call(h)).status_code == 402
        assert len(q.seen) == 1
        assert h.refusals == []


class TestPurchaseAt:
    SOCKET = "wss://robutler.ai/llm"

    async def test_buys_at_the_purchase_url_and_nothing_in_the_record_depends_on_a_granted_amount(self):
        q, h = with_queue(challenge402(), purchased200())
        outcome = await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET)
        assert outcome.ok
        record = outcome.record
        assert (record.url, record.method, record.amount_cents, record.currency, record.terms_version, record.status) == (
            PURCHASE_URL, "stripe", 500, "usd", TERMS, 200,
        )
        assert record.receipt.reference == "pi_pointer"
        # The body is the caller's to read, and it is the portal's: no `credits`, no `creditsNano`.
        assert sorted(outcome.response.json()) == ["balance", "granted", "ok", "packId", "paymentIntentId", "receipt"]
        assert [f"{s.method} {s.url}" for s in q.seen] == [f"POST {PURCHASE_URL}", f"POST {PURCHASE_URL}"]

    @pytest.mark.parametrize("url,source_url", [
        (PURCHASE_URL, "wss://delegate.example/agents/x/uamp"),
        ("https://collector.example/api/mpp/credits", "wss://robutler.ai/llm"),
        (PURCHASE_URL, "not a url"),
    ])
    async def test_s147_nothing_is_sent_unless_both_hosts_are_on_the_allowlist(self, url, source_url):
        q, h = with_queue()
        outcome = await h.buyer.purchase_at(url, source_url=source_url)
        assert (outcome.ok, outcome.reason, outcome.status, outcome.response) == (False, "realm_not_allowed", None, None)
        assert q.seen == []
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["realm_not_allowed"]

    async def test_max_purchases_per_call_holds_across_the_separate_purchases_of_one_call(self):
        q, h = with_queue(challenge402(), purchased200(), challenge402(), purchased200())
        call = object()
        assert (await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET, call=call)).ok
        # The same call again: refused before anything is sent.
        again = await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET, call=call)
        assert (again.ok, again.reason) == (False, "purchase_limit_per_call")
        assert len(q.seen) == 2
        # An in-band challenge for the same call is a second purchase too.
        in_band = await h.buyer.purchase(PURCHASE_URL, challenge_header("robutler.ai"), MppTermsRequest(version=TERMS), call=call)
        assert (in_band.ok, in_band.reason) == (False, "purchase_limit_per_call")
        assert len(h.card.calls) == 1
        assert (await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET, call=object())).ok
        assert len(h.card.calls) == 2

    async def test_a_purchase_url_that_answers_without_a_challenge_bought_nothing(self):
        _, h = with_queue(httpx.Response(200, json={"ok": True}))
        outcome = await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET)
        assert (outcome.ok, outcome.reason) == (False, "served_without_purchase")
        assert h.card.calls == []


class TestTheCeilingAPointerIsFollowedOnlyUnderADailyCap:
    SOCKET = "wss://robutler.ai/llm"

    @staticmethod
    def challenge_entry402() -> httpx.Response:
        """A 402 BODY that names a purchase URL and carries a challenge with it: no platform rail sends this over HTTP, and any peer can."""
        return httpx.Response(402, json={
            "error": "Budget exhausted",
            "requirements": {"schemes": [{
                "scheme": "mpp", "challenge": challenge_header("robutler.ai"), "purchase_url": PURCHASE_URL,
                "terms": {"url": TERMS_URL, "version": TERMS},
            }]},
        })

    async def test_paying_fetch_with_no_daily_cap_the_pointer_is_refused_before_anything_is_sent(self):
        q, h = uncapped(pointer402())
        response = await token_call(h)
        assert response.status_code == 402
        assert response.json()["error"] == "Budget exhausted"
        # The call itself, and nothing else: the purchase URL was never asked.
        assert [s.url for s in q.seen] == [RESOURCE]
        assert h.card.calls == []
        assert h.buyer.spent_today_cents() == 0
        [refusal] = h.refusals
        assert (refusal.reason, refusal.url, refusal.challenge_id, refusal.pending_credential) == ("pointer_needs_daily_cap", RESOURCE, None, None)
        # The operator is told which setting, and why.
        assert "policy.daily_cap_cents" in refusal.detail
        assert "Nothing was sent" in refusal.detail

    async def test_purchase_at_with_no_daily_cap_the_pointer_is_refused_with_the_typed_reason(self):
        q, h = uncapped()
        outcome = await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET)
        assert (outcome.ok, outcome.reason, outcome.status, outcome.response) == (False, "pointer_needs_daily_cap", None, None)
        assert q.seen == []
        assert h.card.calls == []
        assert [r.reason for r in h.refusals] == ["pointer_needs_daily_cap"]
        assert "pointer_needs_daily_cap" in MPP_BUYER_REFUSAL_REASONS

    @pytest.mark.parametrize("build", [uncapped, with_queue])
    async def test_an_off_list_pointer_is_refused_for_its_host_whether_or_not_a_cap_is_set(self, build):
        _, h = build()
        outcome = await h.buyer.purchase_at("https://collector.example/api/mpp/credits", source_url=self.SOCKET)
        assert (outcome.ok, outcome.reason) == (False, "realm_not_allowed")

    async def test_a_purchase_url_named_in_a_402_body_is_held_to_the_same_rule_when_the_entry_carries_a_challenge(self):
        none, without = uncapped(self.challenge_entry402())
        assert (await token_call(without)).status_code == 402
        assert [s.url for s in none.seen] == [RESOURCE]
        assert without.card.calls == []
        assert [r.reason for r in without.refusals] == ["pointer_needs_daily_cap"]

        # Under a cap it is paid as `purchase` pays one: the challenge is in hand, so the purchase URL is asked once, with the credential.
        q, capped = with_queue(self.challenge_entry402(), purchased200(), served200())
        assert (await token_call(capped)).status_code == 200
        assert [s.url for s in q.seen] == [RESOURCE, PURCHASE_URL, RESOURCE]
        assert q.seen[1].headers[MPP_CREDENTIAL_HEADER].startswith("Payment ")
        assert capped.buyer.spent_today_cents() == 500

    async def test_with_no_daily_cap_a_challenge_on_the_url_the_caller_chose_and_an_in_band_purchase_are_paid_as_before(self):
        q, h = uncapped(challenge402(), served200())
        response = await h.buyer.paying_request("POST", RESOURCE, content=b'{"messages":[]}')
        assert response.status_code == 200
        assert [s.url for s in q.seen] == [RESOURCE, RESOURCE]
        assert q.seen[1].headers[MPP_CREDENTIAL_HEADER].startswith("Payment ")
        assert len(h.card.calls) == 1
        assert h.refusals == []

        _, in_band = uncapped(purchased200())
        assert (await in_band.buyer.purchase(PURCHASE_URL, challenge_header("robutler.ai"), MppTermsRequest(version=TERMS))).ok
        assert in_band.refusals == []

    async def test_the_cap_that_admits_a_pointer_is_what_bounds_it_call_after_call(self):
        # A peer starts a new call each time, so `max_purchases_per_call` never bites. 500 cents a pack under a 1200 cent cap: two, never three.
        _, h = with_queue(challenge402(), purchased200(), challenge402(), purchased200(), challenge402(), policy={"daily_cap_cents": 1200})
        assert (await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET, call=object())).ok
        assert (await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET, call=object())).ok
        third = await h.buyer.purchase_at(PURCHASE_URL, source_url=self.SOCKET, call=object())
        assert (third.ok, third.reason) == (False, "over_daily_cap")
        assert len(h.card.calls) == 2
        assert h.buyer.spent_today_cents() == 1000


def test_mpp_requirement_of_reads_the_pointer_the_entry_with_a_challenge_and_nothing_else():
    assert mpp_requirement_of({"schemes": [{"scheme": "token"}, {"scheme": "mpp", "purchase_url": PURCHASE_URL}]}) == MppRequirementEntry(purchase_url=PURCHASE_URL)
    assert mpp_requirement_of({"schemes": [{"scheme": "mpp", "challenge": 'Payment id="c1"', "purchase_url": PURCHASE_URL, "terms": {"url": TERMS_URL, "version": TERMS}}]}) == MppRequirementEntry(
        purchase_url=PURCHASE_URL, challenge='Payment id="c1"', terms=MppTermsRequest(version=TERMS, url=TERMS_URL),
    )
    for nothing in (None, {}, {"schemes": "mpp"}, {"schemes": [{"scheme": "mpp"}]}, {"schemes": [{"scheme": "mpp", "purchase_url": " "}]}, {"schemes": [{"scheme": "mpp", "purchase_url": PURCHASE_URL, "challenge": 7}]}):
        assert mpp_requirement_of(nothing) is None
