"""
The signed UAMP upgrade, pinned across both SDKs (2026-09-18).

Nothing signed a UAMP upgrade until the fix pass of 2026-09-18, so the
platform's socket door (`signedUpgradeEligible`,
lib/payments/machine-door-socket.ts) never saw a signed agent and never sent
an SDK client the in-band `mpp` entry. `MppBuyer.upgrade_headers` now signs a
GET on the http(s) form of the socket URL, the request the door rebuilds from
the upgrade's Host header and path. The TypeScript suite
(`typescript/tests/unit/skills/payments/mpp-buyer-upgrade-vectors.test.ts`)
wrote `tests/fixtures/web_bot_auth/vectors-upgrade.json`; this suite
reproduces every case and every Signature-Agent form from the Python buyer
and compares byte for byte. The first two vector files are not touched.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.agents.skills.robutler.payments_x402.mpp_buyer import MppBuyer, MppBuyerPolicy
from webagents.crypto.http_signature import SigningKey, sign_request

from .covered_support import verify_with_covered_headers

VECTORS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "web_bot_auth" / "vectors-upgrade.json"


class _Card:
    def get_spt(self, request):
        return "spt_x"


class _Wallet:
    def sign_tempo_transfer(self, request):
        return "0x7600"


def _vectors() -> dict:
    return json.loads(VECTORS_PATH.read_text())


def _key(doc: dict) -> SigningKey:
    d = doc["key"]["privateJwk"]["d"]
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(d + "=" * (-len(d) % 4))))


def test_the_file_is_the_one_the_typescript_suite_writes():
    doc = _vectors()
    assert doc["key"]["thumbprint"] == _key(doc).thumbprint
    assert [c["name"] for c in doc["cases"]] == [
        "wss to the platform, card-only buyer",
        "ws on a local port with a query, stablecoin-only buyer",
    ]


@pytest.mark.parametrize("case_index", [0, 1])
async def test_the_python_buyer_signs_every_upgrade_case_byte_for_byte(case_index: int):
    doc = _vectors()
    case = doc["cases"][case_index]
    key = _key(doc)
    params = doc["params"]
    for vector in case["vectors"]:
        buyer = MppBuyer(
            keys=[key],
            agent_url=doc["agentUrl"],
            policy=MppBuyerPolicy(max_per_purchase_cents=1, accept_terms="2026-07-31", realms=case["realms"]),
            card=_Card() if case["source"] == "card" else None,
            stablecoin=_Wallet() if case["source"] == "stablecoin" else None,
            form=vector["form"],
            allow_http=True,
        )
        headers = await buyer.upgrade_headers(case["socketUrl"], created=params["created"], nonces=[params["nonce"]])
        assert {k.lower(): v for k, v in headers.items()} == vector["headers"], f"{case['name']} / {vector['form']}"

        # The base lines, from the signer on the http(s) request itself.
        reference = sign_request(
            [key],
            doc["agentUrl"],
            "GET",
            case["signedRequest"]["url"],
            b"",
            form=vector["form"],
            created=params["created"],
            nonces=[params["nonce"]],
            allow_http=True,
            headers=case["hints"],
            covered_headers=list(case["hints"]),
        )
        assert reference.signatures[0].base.split("\n") == vector["signatureBaseLines"]
        verify_with_covered_headers(
            httpx.Headers(vector["headers"]), "GET", case["signedRequest"]["url"], b"",
            {key.thumbprint: key.private_key.public_key()},
        )
