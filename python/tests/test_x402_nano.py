"""
The `nano:mainnet` exact facilitator (`payments/x402_nano.py`): addresses and
their checksum, the request binding (order + representative), every refusal
of a block, the replay claim that survives a restart, and the raw amount the
paywall derives with 30 decimals.

`x402_nano.py` and `x402_wire.py` import only the standard library, so this
file loads them by path and runs with `python -m unittest` as well as under
pytest, with no install.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

_PAYMENTS = Path(__file__).resolve().parents[1] / "webagents" / "agents" / "skills" / "robutler" / "payments"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_x402_leaf_{name}", _PAYMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nano = _load("x402_nano")
wire = _load("x402_wire")

# Public keys only (no key pair exists behind them): the accounts are fixtures.
PAY_TO = nano.address_of(hashlib.sha256(b"webagents test seller").digest())
PAYER = nano.address_of(hashlib.sha256(b"webagents test payer").digest())
BLOCK = "A" * 63 + "1"
URL = "https://agent.example.com/agents/mini/quote"
BINDING = "POST|/agents/mini/quote||" + hashlib.sha256(b'{"q":1}').hexdigest()
OTHER_BINDING = "POST|/agents/mini/quote||" + hashlib.sha256(b'{"q":2}').hexdigest()
SECRET = "s" * 40
NOW = 1_800_000_000
# 0.01 credit at 0.5 XNO a credit = 0.005 XNO, derived, never restated.
AMOUNT = wire.credits_to_asset_units("0.01", nano.NANO_DECIMALS, "0.5")


def run(coro):
    return asyncio.run(coro)


class FakeNode:
    def __init__(self, blocks=None, fail=False):
        self.blocks = dict(blocks or {})
        self.fail = fail
        self.reads = []

    async def block_info(self, block_hash):
        self.reads.append(block_hash)
        if self.fail:
            raise nano.NodeUnavailable("down")
        return self.blocks.get(block_hash)


def requirement(amount=None, pay_to=None, **over):
    req = {"scheme": "exact", "network": nano.NANO_NETWORK, "amount": amount or AMOUNT, "asset": nano.NANO_ASSET, "payTo": pay_to or PAY_TO, "maxTimeoutSeconds": 60}
    req.update(over)
    return req


def send_block(representative, amount=None, to=None, confirmed="true", subtype="send", at=NOW + 5):
    return {
        "block_account": PAYER,
        "amount": amount or AMOUNT,
        "confirmed": confirmed,
        "subtype": subtype,
        "local_timestamp": str(at),
        "contents": {"type": "state", "account": PAYER, "link_as_account": to or PAY_TO, "representative": representative},
    }


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.claims = os.path.join(self.tmp.name, "claims")
        self.now = [NOW]
        self.node = FakeNode()
        self.fac = self.make()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, node=None):
        return nano.NanoFacilitator(PAY_TO, order_secret=SECRET, claims_dir=self.claims, node=node or self.node, clock=lambda: self.now[0])

    def offer(self, binding=BINDING, url=URL):
        extra = self.fac.offer_extra(requirement(), url, binding)
        return extra["order"], extra["representative"]

    def paid(self, binding=BINDING, **block):
        order, rep = self.offer(binding)
        self.node.blocks[BLOCK] = send_block(block.pop("representative", rep), **block)
        return {"x402Version": 2, "accepted": requirement(), "payload": {"block": BLOCK, "order": order}}


class Addresses(unittest.TestCase):
    def test_round_trip_and_checksum(self):
        key = hashlib.sha256(b"k").digest()
        address = nano.address_of(key)
        self.assertTrue(address.startswith("nano_") and len(address) == 65)
        self.assertEqual(nano.public_key_of(address), key)
        self.assertEqual(nano.normalize_address("xrb_" + address[5:]), address)
        broken = address[:-1] + ("1" if address[-1] != "1" else "3")
        with self.assertRaises(ValueError):
            nano.public_key_of(broken)
        for bad in ("", "nano_", "0xabc", None, address + "1"):
            with self.assertRaises(ValueError):
                nano.public_key_of(bad)

    def test_facilitator_refuses_a_bad_pay_to_or_a_short_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                nano.NanoFacilitator(PAY_TO[:-1] + "x", order_secret=SECRET, claims_dir=tmp, node=FakeNode())
            with self.assertRaises(ValueError):
                nano.NanoFacilitator(PAY_TO, order_secret="short", claims_dir=tmp, node=FakeNode())
            with self.assertRaises(ValueError):
                nano.NanoFacilitator(PAY_TO, order_secret=SECRET, claims_dir="", node=FakeNode())

    def test_representative_is_the_binding_standard_commitment(self):
        expected = nano.address_of(hashlib.sha256(f"xno-binding/v1/rep|ord|{PAY_TO}|{AMOUNT}".encode()).digest())
        self.assertEqual(nano.representative_for("ord", PAY_TO, AMOUNT), expected)


class Amounts(unittest.TestCase):
    def test_raw_is_derived_with_thirty_decimals(self):
        self.assertEqual(AMOUNT, str(5 * 10 ** 27))
        self.assertEqual(wire.credits_to_asset_units("1", 30, "1"), str(nano.RAW_PER_XNO))

    def test_more_than_28_digits_is_not_rounded(self):
        # 0.1234567890123456789012345678901 XNO is 123456789012345678901234567890.1 raw.
        self.assertEqual(wire.credits_to_asset_units("0.1234567890123456789012345678901", 30, "1"), "123456789012345678901234567890")
        self.assertEqual(wire.credits_to_asset_units("1", 30, "1.000000000000000000000000000001"), str(10 ** 30 + 1))

    def test_six_decimal_amounts_are_unchanged(self):
        self.assertEqual(wire.credits_to_asset_units(0.01, 6, 1), "10000")
        self.assertEqual(wire.credits_to_asset_units("2.5", 6, "0.01"), "25000")


class Verify(Base):
    def test_a_bound_confirmed_send_is_valid_and_settles_once(self):
        payload = self.paid()
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING)), {"isValid": True, "payer": PAYER})
        settled = run(self.fac.settle(payload, requirement(), url=URL, binding=BINDING))
        self.assertTrue(settled["success"])
        self.assertEqual((settled["transaction"], settled["network"], settled["amount"]), (BLOCK, "nano:mainnet", AMOUNT))
        again = run(self.fac.settle(payload, requirement(), url=URL, binding=BINDING))
        self.assertEqual((again["success"], again["errorReason"]), (False, "replayed"))
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING))["invalidReason"], "replayed")

    def test_the_claim_survives_a_restart(self):
        payload = self.paid()
        self.assertTrue(run(self.fac.settle(payload, requirement(), url=URL, binding=BINDING))["success"])
        restarted = self.make(node=FakeNode(self.node.blocks))
        self.assertEqual(run(restarted.settle(payload, requirement(), url=URL, binding=BINDING))["errorReason"], "replayed")
        self.assertEqual(run(restarted.verify(payload, requirement(), url=URL, binding=BINDING))["invalidReason"], "replayed")

    def test_the_order_is_bound_to_the_request(self):
        payload = self.paid()
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=OTHER_BINDING))["invalidReason"], "order_not_for_this_request")
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL + "x", binding=BINDING))["invalidReason"], "order_not_for_this_request")
        other_amount = str(int(AMOUNT) - 1)
        self.assertEqual(run(self.fac.verify(payload, requirement(amount=other_amount), url=URL, binding=BINDING))["invalidReason"], "order_not_for_this_request")
        forged = {**payload, "payload": {"block": BLOCK, "order": payload["payload"]["order"][:-2] + "AA"}}
        self.assertEqual(run(self.fac.verify(forged, requirement(), url=URL, binding=BINDING))["invalidReason"], "order_not_for_this_request")

    def test_a_block_paid_for_another_order_is_refused(self):
        other_order, other_rep = self.offer(OTHER_BINDING)
        payload = self.paid(representative=other_rep)
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING))["invalidReason"], "block_not_for_this_request")

    def test_each_block_refusal(self):
        cases = {
            "wrong_amount": {"amount": str(int(AMOUNT) - 1)},
            "wrong_recipient": {"to": PAYER},
            "block_unconfirmed": {"confirmed": "false"},
            "not_a_send": {"subtype": "receive"},
            "sent_after_order_expired": {"at": NOW + 600 + 61},
        }
        for reason, block in cases.items():
            with self.subTest(reason=reason):
                payload = self.paid(**block)
                self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING))["invalidReason"], reason)
                self.assertFalse(run(self.fac.settle(payload, requirement(), url=URL, binding=BINDING))["success"])
        self.assertEqual(os.listdir(self.claims), [])

    def test_more_than_the_amount_is_refused_too(self):
        payload = self.paid(amount=str(int(AMOUNT) * 10))
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING))["invalidReason"], "wrong_amount")

    def test_unknown_block_node_down_and_malformed_payloads(self):
        order, _ = self.offer()
        self.assertEqual(run(self.fac.verify({"payload": {"block": BLOCK, "order": order}}, requirement(), url=URL, binding=BINDING))["invalidReason"], "block_unknown")
        down = self.make(node=FakeNode(fail=True))
        self.assertEqual(run(down.verify({"payload": {"block": BLOCK, "order": order}}, requirement(), url=URL, binding=BINDING))["invalidReason"], "node_unavailable")
        for inner in ({}, {"block": "zz", "order": order}, {"block": BLOCK}, {"block": BLOCK, "order": "a.b"}):
            with self.subTest(inner=inner):
                self.assertFalse(run(self.fac.verify({"payload": inner}, requirement(), url=URL, binding=BINDING))["isValid"])
        self.assertEqual(self.node.reads, [BLOCK])

    def test_an_expired_order_is_refused_before_the_node_is_read(self):
        payload = self.paid()
        self.node.reads.clear()
        self.now[0] = NOW + 600 + 601
        self.assertEqual(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING))["invalidReason"], "order_expired")
        self.assertEqual(self.node.reads, [])

    def test_a_requirement_that_is_not_ours_is_refused(self):
        payload = self.paid()
        for over, reason in (({"scheme": "upto"}, "unsupported_scheme"), ({"network": "eip155:8453"}, "unsupported_scheme"), ({"asset": "USDC"}, "invalid_asset"), ({"payTo": PAYER}, "invalid_pay_to"), ({"amount": "0"}, "invalid_amount")):
            with self.subTest(reason=reason):
                self.assertEqual(run(self.fac.verify(payload, requirement(**over), url=URL, binding=BINDING))["invalidReason"], reason)

    def test_settle_verifies_the_requirement_it_is_given(self):
        payload = self.paid()
        self.assertTrue(run(self.fac.verify(payload, requirement(), url=URL, binding=BINDING))["isValid"])
        cheaper = run(self.fac.settle(payload, requirement(amount=str(int(AMOUNT) // 2)), url=URL, binding=BINDING))
        self.assertEqual((cheaper["success"], cheaper["errorReason"]), (False, "order_not_for_this_request"))
        self.assertTrue(run(self.fac.settle(payload, requirement(), url=URL, binding=BINDING))["success"])


class Config(unittest.TestCase):
    def test_from_config_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            fac = nano.nano_facilitator_from_config(PAY_TO.upper().replace("NANO_", "nano_"), {"node": FakeNode()}, env={"NANO_ORDER_SECRET": SECRET, "NANO_CLAIMS_DIR": tmp})
            self.assertEqual(fac.pay_to, PAY_TO)
            with self.assertRaises(ValueError):
                nano.nano_facilitator_from_config(PAY_TO, {}, env={"NANO_CLAIMS_DIR": tmp, "NANO_NODE_URL": "http://127.0.0.1:7076"})
            with self.assertRaises(ValueError):
                nano.nano_facilitator_from_config(PAY_TO, {}, env={"NANO_ORDER_SECRET": SECRET, "NANO_CLAIMS_DIR": tmp})


if __name__ == "__main__":
    unittest.main()
