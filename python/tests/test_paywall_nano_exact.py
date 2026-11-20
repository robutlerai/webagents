"""
`nano:mainnet` exact on a priced endpoint: the paywall with a `NanoFacilitator`
as its chain seller (a fake node, no network). The 402 names an order bound to
the request, the block that pays it is verified before the handler runs and
claimed once after it, and the claim holds across a restart. A Paywall with
any other facilitator is untouched: it is called exactly as before.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import Response

from webagents.agents.skills.robutler.payments.paywall import Paywall, PricedEndpoint
from webagents.agents.skills.robutler.payments.x402_nano import NANO_ASSET, NANO_DECIMALS, NANO_NETWORK, NanoFacilitator, address_of
from webagents.agents.skills.robutler.payments.x402_wire import X402_HEADERS, credits_to_asset_units, decode_base64_json, encode_base64_json, is_well_formed_requirement

PAY_TO = address_of(hashlib.sha256(b"webagents test seller").digest())
PAYER = address_of(hashlib.sha256(b"webagents test payer").digest())
BLOCK = "B" * 63 + "2"
SECRET = "o" * 40
ENDPOINT = PricedEndpoint(path="/quote", method="POST", pricing={"credits_per_call": 0.01}, description="A quote")
UNITS_PER_CREDIT = "0.5"
AMOUNT = credits_to_asset_units(0.01, NANO_DECIMALS, UNITS_PER_CREDIT)


class FakeNode:
    def __init__(self):
        self.blocks: Dict[str, Dict[str, Any]] = {}

    async def block_info(self, block_hash):
        return self.blocks.get(block_hash)


def nano_paywall(tmp_path, node: FakeNode) -> Paywall:
    facilitator = NanoFacilitator(PAY_TO, order_secret=SECRET, claims_dir=str(tmp_path / "claims"), node=node)
    return Paywall(chain={"pay_to": PAY_TO, "network": NANO_NETWORK, "asset": NANO_ASSET, "decimals": NANO_DECIMALS, "units_per_credit": UNITS_PER_CREDIT, "facilitator": facilitator})


def request(body: bytes, headers: Optional[Dict[str, str]] = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "https",
        "server": ("agent.example.com", 443),
        "path": "/agents/mini/quote",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in {"host": "agent.example.com", **(headers or {})}.items()],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=receive)


async def handler():
    return Response(json.dumps({"price": 42}), status_code=200, media_type="application/json")


async def never():
    raise AssertionError("the handler must not run")


async def challenge(p: Paywall, body: bytes) -> Dict[str, Any]:
    res = await p.handle(ENDPOINT, request(body), never)
    assert res.status_code == 402
    entry = decode_base64_json(res.headers[X402_HEADERS["required"]])["accepts"][0]
    assert is_well_formed_requirement(entry)
    return entry


def pay(node: FakeNode, entry: Dict[str, Any], amount: Optional[str] = None) -> None:
    node.blocks[BLOCK] = {
        "block_account": PAYER,
        "amount": amount or entry["amount"],
        "confirmed": "true",
        "subtype": "send",
        "local_timestamp": str(int(entry["extra"]["order"].split(".")[1]) - 100),
        "contents": {"type": "state", "account": PAYER, "link_as_account": PAY_TO, "representative": entry["extra"]["representative"]},
    }


def payment(entry: Dict[str, Any]) -> Dict[str, str]:
    accepted = {k: v for k, v in entry.items()}
    return {X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": accepted, "payload": {"block": BLOCK, "order": entry["extra"]["order"]}})}


async def test_the_402_offers_nano_exact_with_a_bound_order(tmp_path):
    entry = await challenge(nano_paywall(tmp_path, FakeNode()), b'{"q":1}')
    assert (entry["scheme"], entry["network"], entry["asset"], entry["payTo"]) == ("exact", "nano:mainnet", "XNO", PAY_TO)
    assert entry["amount"] == AMOUNT == str(5 * 10 ** 27)
    assert set(entry["extra"]) == {"order", "representative", "binding"} and entry["extra"]["binding"] == "xno-binding/v1/rep"


async def test_a_paid_block_runs_the_handler_settles_once_and_a_replay_gets_nothing(tmp_path):
    node = FakeNode()
    p = nano_paywall(tmp_path, node)
    entry = await challenge(p, b'{"q":1}')
    pay(node, entry)
    res = await p.handle(ENDPOINT, request(b'{"q":1}', payment(entry)), handler)
    assert res.status_code == 200 and json.loads(res.body) == {"price": 42}
    settled = decode_base64_json(res.headers[X402_HEADERS["response"]])
    assert settled["success"] is True and settled["transaction"] == BLOCK and settled["network"] == "nano:mainnet"
    # The same block again, before and after a restart: refused, the handler never runs.
    again = await p.handle(ENDPOINT, request(b'{"q":1}', payment(entry)), never)
    assert again.status_code == 402
    assert decode_base64_json(again.headers[X402_HEADERS["required"]])["error"] == "replayed"
    restarted = nano_paywall(tmp_path, node)
    after = await restarted.handle(ENDPOINT, request(b'{"q":1}', payment(entry)), never)
    assert after.status_code == 402 and decode_base64_json(after.headers[X402_HEADERS["required"]])["error"] == "replayed"


async def test_a_payment_for_one_body_does_not_buy_another(tmp_path):
    node = FakeNode()
    p = nano_paywall(tmp_path, node)
    entry = await challenge(p, b'{"q":1}')
    pay(node, entry)
    res = await p.handle(ENDPOINT, request(b'{"q":2}', payment(entry)), never)
    assert res.status_code == 402
    assert decode_base64_json(res.headers[X402_HEADERS["required"]])["error"] == "order_not_for_this_request"


async def test_a_short_block_is_refused_before_the_handler(tmp_path):
    node = FakeNode()
    p = nano_paywall(tmp_path, node)
    entry = await challenge(p, b'{"q":1}')
    pay(node, entry, amount=str(int(entry["amount"]) - 1))
    res = await p.handle(ENDPOINT, request(b'{"q":1}', payment(entry)), never)
    assert res.status_code == 402 and decode_base64_json(res.headers[X402_HEADERS["required"]])["error"] == "wrong_amount"


async def test_a_handler_error_claims_nothing(tmp_path):
    node = FakeNode()
    p = nano_paywall(tmp_path, node)
    entry = await challenge(p, b'{"q":1}')
    pay(node, entry)

    async def failing():
        return Response("{}", status_code=500, media_type="application/json")

    assert (await p.handle(ENDPOINT, request(b'{"q":1}', payment(entry)), failing)).status_code == 500
    assert (await p.handle(ENDPOINT, request(b'{"q":1}', payment(entry)), handler)).status_code == 200


class PlainFacilitator:
    """A facilitator that does not bind: called with two arguments, as before."""

    def __init__(self):
        self.calls = []

    async def verify(self, payload, requirement):
        self.calls.append(("verify", requirement.get("extra")))
        return {"isValid": True}

    async def settle(self, payload, requirement):
        self.calls.append(("settle", requirement["amount"]))
        return {"success": True, "transaction": "0xabc", "network": requirement["network"]}


async def test_a_facilitator_that_does_not_bind_is_called_as_before():
    plain = PlainFacilitator()
    p = Paywall(chain={"pay_to": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C", "network": "eip155:84532", "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e", "decimals": 6, "extra": {"name": "USDC", "version": "2"}, "facilitator": plain})
    entry = await challenge(p, b"{}")
    assert entry["extra"] == {"name": "USDC", "version": "2"}
    res = await p.handle(ENDPOINT, request(b"{}", {X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": entry, "payload": {"signature": "0x00"}})}), handler)
    assert res.status_code == 200 and plain.calls == [("verify", {"name": "USDC", "version": "2"}), ("settle", "10000")]


def test_the_payment_skill_builds_a_nano_chain_seller_from_config(tmp_path):
    from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402

    skill = PaymentSkillX402(config={
        "webagents_api_url": "https://test.example",
        "robutler_api_key": "k",
        "x402": {"credits": False, "chain": {"pay_to": PAY_TO, "network": "nano:mainnet", "units_per_credit": UNITS_PER_CREDIT, "decimals": 6, "facilitator": {"order_secret": SECRET, "claims_dir": str(tmp_path / "c"), "node": FakeNode()}}},
    })
    chain = skill.paywall.chain
    assert (chain["network"], chain["asset"], chain["decimals"], chain["schemes"], chain["extra"]) == ("nano:mainnet", "XNO", 30, ["exact"], {})
    assert isinstance(chain["facilitator"], NanoFacilitator) and chain["pay_to"] == PAY_TO


def test_without_nano_config_the_chain_seller_is_unchanged():
    from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402
    from webagents.agents.skills.robutler.payments.x402_facilitator import HttpFacilitatorClient

    skill = PaymentSkillX402(config={"webagents_api_url": "https://test.example", "robutler_api_key": "k", "x402": {"credits": False, "chain": {"pay_to": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C"}}})
    chain = skill.paywall.chain
    assert chain["network"] == "eip155:84532" and chain["decimals"] == 6 and isinstance(chain["facilitator"], HttpFacilitatorClient)
