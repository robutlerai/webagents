"""
x402 on the wire, v2 and v1 (webagents gap-closure plan 2.6, 2026-09-26).

The pure half of the paywall (`paywall.py`), the twin of the TypeScript
`skills/payments/x402-wire.ts`: header names, the standard base64 codec, the
`PaymentRequired`, `PaymentPayload` and `SettleResponse` shapes, the v1
projection of a v2 requirement, the strict-parser well-formedness rule and
the requirement matching rule, as the spec pack pins them
(`docs/internal/architecture/webagents-gap-closure/spec-payments-identity.txt`
section 1). The fixture `tests/fixtures/payments/paywall_x402_mpp.json` holds
the names and the section 1.10 vectors both suites read.

WHAT CHANGED. The SDK used to answer a 402 in a private shape (`scheme:
'token'` on `network: 'robutler'`, `maxAmountRequired`, `payTo:` the agent)
and to verify through a `client.facilitator` that did not exist. A standard
x402 client dropped every entry as unknown; a strict parser refused the body
(`network` without a `:`). Now v2 travels in the headers (`PAYMENT-REQUIRED`,
`PAYMENT-SIGNATURE`, `PAYMENT-RESPONSE`, standard base64 WITH padding), v1 in
the 402 body and `X-PAYMENT` / `X-PAYMENT-RESPONSE`, and every entry is
well-formed for a strict parser (section 1.9).

THE CREDITS SCHEME: `robutler-credits` on `robutler:1`, asset `credits`
(named as credits, never a currency), `payTo:` the role constant `robutler`
(Robutler is the seller of record), amounts in nanocredits (9 decimals), the
payload a Robutler payment token bound to the request by the server nonce in
`extra.nonce` (`x402_credits.py`).
"""

from __future__ import annotations

import base64
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Mapping, Optional, Tuple

X402_HEADERS = {
    "required": "PAYMENT-REQUIRED",
    "signature": "PAYMENT-SIGNATURE",
    "response": "PAYMENT-RESPONSE",
    "v1_payment": "X-PAYMENT",
    "v1_response": "X-PAYMENT-RESPONSE",
    "settlement_overrides": "Settlement-Overrides",
}

#: What a browser client must be allowed to READ.
X402_CORS_EXPOSE_HEADERS: Tuple[str, ...] = (
    "PAYMENT-REQUIRED",
    "PAYMENT-RESPONSE",
    "X-PAYMENT-RESPONSE",
    "WWW-Authenticate",
    "Payment-Receipt",
)
#: What a browser client must be allowed to SEND (the official fetch wrapper sends a request header named `Access-Control-Expose-Headers`).
X402_CORS_ALLOW_HEADERS: Tuple[str, ...] = (
    "Content-Type",
    "Authorization",
    "PAYMENT-SIGNATURE",
    "X-PAYMENT",
    "Access-Control-Expose-Headers",
    "Payment-Authorization",
)

CREDITS_SCHEME = "robutler-credits"
CREDITS_NETWORK = "robutler:1"
CREDITS_ASSET = "credits"
CREDITS_PAY_TO = "robutler"
CREDITS_DECIMALS = 9
CREDITS_MAX_TIMEOUT_SECONDS = 300

#: The v1 network names the reference SDKs map from CAIP-2.
V1_NETWORK_NAMES: Dict[str, str] = {
    "eip155:84532": "base-sepolia",
    "eip155:8453": "base",
    "eip155:43113": "avalanche-fuji",
    "eip155:43114": "avalanche",
    "eip155:137": "polygon",
    "eip155:80002": "polygon-amoy",
}

_STANDARD_BASE64 = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
_AMOUNT_RE = re.compile(r"^[0-9]+$")


def _json_bytes(value: Any) -> bytes:
    """`JSON.stringify` bytes: compact separators, literal UTF-8, insertion order."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


# ── Standard base64, with padding ────────────────────────────────────────────


def encode_base64_json(value: Any) -> str:
    """`base64(JSON)` as x402 transports carry it: standard alphabet, `=` padding."""
    return base64.b64encode(_json_bytes(value)).decode("ascii")


def decode_base64_json(text: Any) -> Optional[Dict[str, Any]]:
    """Strict: the standard alphabet with padding, decoding to a JSON object; None otherwise."""
    if not isinstance(text, str) or not text or len(text) % 4 != 0 or not _STANDARD_BASE64.match(text):
        return None
    try:
        raw = base64.b64decode(text, validate=True)
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


# ── Amounts ──────────────────────────────────────────────────────────────────


def credits_to_nanocredits(credits: Any) -> str:
    """A decimal credit amount as nanocredits, the atomic unit, as a decimal string."""
    try:
        value = Decimal(str(credits))
    except Exception as error:
        raise ValueError(f"x402: not a credit amount: {credits!r}") from error
    if not value.is_finite() or value < 0:
        raise ValueError(f"x402: not a credit amount: {credits!r}")
    return str(int((value * (10 ** CREDITS_DECIMALS)).to_integral_value(rounding=ROUND_HALF_UP)))


def credits_to_asset_units(credits: Any, decimals: int, units_per_credit: Any = 1) -> str:
    """A credit amount as atomic units of a chain asset with `decimals` decimals."""
    if not isinstance(decimals, int) or isinstance(decimals, bool) or decimals < 0 or decimals > 30:
        raise ValueError(f"x402: bad decimals {decimals!r}")
    value = Decimal(str(credits)) * Decimal(str(units_per_credit)) * (10 ** decimals)
    if not value.is_finite() or value < 0:
        raise ValueError(f"x402: not a credit amount: {credits!r}")
    return str(int(value.to_integral_value(rounding=ROUND_HALF_UP)))


def is_amount_string(text: Any) -> bool:
    return isinstance(text, str) and bool(_AMOUNT_RE.match(text))


# ── Well-formedness and matching (sections 1.3 and 1.9) ──────────────────────


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and value not in (float("inf"), float("-inf"))


#: The entry's required strings, one rule each: a non-empty string, and for the network one with a `:`.
_REQUIRED_STRINGS = {
    "scheme": lambda text: len(text) > 0,
    "network": lambda text: ":" in text,
    "amount": lambda text: len(text) > 0,
    "asset": lambda text: len(text) > 0,
    "payTo": lambda text: len(text) > 0,
}


def is_well_formed_requirement(value: Any) -> bool:
    """Well-formed for a strict parser: `network` contains `:`, the strings are non-empty, the timeout is positive."""
    if not isinstance(value, dict):
        return False
    for key, rule in _REQUIRED_STRINGS.items():
        text = value.get(key)
        if not isinstance(text, str) or not rule(text):
            return False
    timeout = value.get("maxTimeoutSeconds")
    extra = value.get("extra")
    return _is_number(timeout) and timeout > 0 and (extra is None or isinstance(extra, dict))


def _deep_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def requirement_matches(offered: Mapping[str, Any], accepted: Any) -> bool:
    """Every field but `extra` deep-equal, types included; the offered `extra` contained in the accepted one."""
    if not is_well_formed_requirement(accepted):
        return False
    offered_rest = {k: v for k, v in offered.items() if k != "extra"}
    accepted_rest = {k: v for k, v in accepted.items() if k != "extra"}
    if not _deep_equal(offered_rest, accepted_rest):
        return False
    offered_extra = offered.get("extra")
    if not offered_extra:
        return True
    accepted_extra = accepted.get("extra")
    if not isinstance(accepted_extra, dict):
        return False
    return all(k in accepted_extra and _deep_equal(v, accepted_extra[k]) for k, v in offered_extra.items())


# ── The 402, v2 header and v1 body ───────────────────────────────────────────


def to_v1_requirement(requirement: Mapping[str, Any], resource: Mapping[str, Any], output_schema: Any = None) -> Dict[str, Any]:
    """The v1 projection of a v2 requirement: the resource folded in, `amount` renamed."""
    v1: Dict[str, Any] = {
        "scheme": requirement["scheme"],
        "network": V1_NETWORK_NAMES.get(requirement["network"], requirement["network"]),
        "maxAmountRequired": requirement["amount"],
        "asset": requirement["asset"],
        "payTo": requirement["payTo"],
        "resource": resource["url"],
        "description": resource.get("description") or "",
        "mimeType": resource.get("mimeType") or "application/json",
        "maxTimeoutSeconds": requirement["maxTimeoutSeconds"],
    }
    if output_schema is not None:
        v1["outputSchema"] = output_schema
    if requirement.get("extra"):
        v1["extra"] = requirement["extra"]
    return v1


def build_payment_required(
    resource: Mapping[str, Any],
    accepts: List[Mapping[str, Any]],
    error: Optional[str] = None,
    extensions: Optional[Mapping[str, Any]] = None,
    bazaar: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The 402's two documents from one set of offers: `(v2, v1)`."""
    for entry in accepts:
        if not is_well_formed_requirement(entry):
            raise ValueError(f"x402: malformed requirement for {entry.get('scheme', '?') if isinstance(entry, dict) else '?'}")
    error_text = error or f"{X402_HEADERS['signature']} header is required"
    ext: Dict[str, Any] = dict(extensions or {})
    output_schema = None
    if bazaar:
        info: Dict[str, Any] = {"input": bazaar["input"]}
        if bazaar.get("output") is not None:
            info["output"] = bazaar["output"]
            output_schema = bazaar["output"]
        entry: Dict[str, Any] = {"info": info}
        if bazaar.get("schema") is not None:
            entry["schema"] = bazaar["schema"]
        ext["bazaar"] = entry
    v2: Dict[str, Any] = {"x402Version": 2, "error": error_text, "resource": dict(resource), "accepts": [dict(a) for a in accepts]}
    if ext:
        v2["extensions"] = ext
    v1: Dict[str, Any] = {"x402Version": 1, "error": error_text, "accepts": [to_v1_requirement(a, resource, output_schema) for a in accepts]}
    return v2, v1


# ── The retry's payload ──────────────────────────────────────────────────────


def read_payment_payload(headers: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The payment a request carries: `PAYMENT-SIGNATURE` (v2) first, else `X-PAYMENT` (v1).

    None when neither is present; `{"version": 0, "error": ...}` when one is
    present and malformed (a 400, never a 402).
    """
    v2 = headers.get(X402_HEADERS["signature"])
    if v2:
        decoded = decode_base64_json(v2.strip())
        if decoded is None:
            return {"version": 0, "error": f"{X402_HEADERS['signature']} is not base64 JSON"}
        if decoded.get("x402Version") != 2:
            return {"version": 0, "error": f"{X402_HEADERS['signature']} is not x402Version 2"}
        if not isinstance(decoded.get("accepted"), dict) or not isinstance(decoded.get("payload"), dict):
            return {"version": 0, "error": f"{X402_HEADERS['signature']} lacks accepted or payload"}
        return {"version": 2, "payload": decoded, "raw": v2.strip()}
    v1 = headers.get(X402_HEADERS["v1_payment"])
    if v1:
        decoded = decode_base64_json(v1.strip())
        if decoded is None:
            return {"version": 0, "error": f"{X402_HEADERS['v1_payment']} is not base64 JSON"}
        if decoded.get("x402Version") != 1:
            return {"version": 0, "error": f"{X402_HEADERS['v1_payment']} is not x402Version 1"}
        if not isinstance(decoded.get("scheme"), str) or not isinstance(decoded.get("network"), str) or not isinstance(decoded.get("payload"), dict):
            return {"version": 0, "error": f"{X402_HEADERS['v1_payment']} lacks scheme, network or payload"}
        return {"version": 1, "payload": decoded, "raw": v1.strip()}
    return None


def v1_network_to_caip2(name: str) -> str:
    if ":" in name:
        return name
    for caip2, v1 in V1_NETWORK_NAMES.items():
        if v1 == name:
            return caip2
    return name


# ── The answer's settle response ─────────────────────────────────────────────


def settle_response_header(version: int, settle: Mapping[str, Any]) -> Tuple[str, str]:
    """v2 `PAYMENT-RESPONSE`, v1 `X-PAYMENT-RESPONSE`, the same base64 JSON."""
    return (X402_HEADERS["response"] if version == 2 else X402_HEADERS["v1_response"], encode_base64_json(dict(settle)))


def read_settlement_overrides(headers: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """A metered handler's `Settlement-Overrides: {"amount":"500"}` (or `{"credits":"0.002"}`), or None."""
    raw = headers.get(X402_HEADERS["settlement_overrides"])
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, str] = {}
    if is_amount_string(parsed.get("amount")):
        out["amount"] = parsed["amount"]
    credits = parsed.get("credits")
    if isinstance(credits, (str, int, float)) and not isinstance(credits, bool):
        out["credits"] = str(credits)
    return out or None


__all__ = [
    "X402_HEADERS",
    "X402_CORS_EXPOSE_HEADERS",
    "X402_CORS_ALLOW_HEADERS",
    "CREDITS_SCHEME",
    "CREDITS_NETWORK",
    "CREDITS_ASSET",
    "CREDITS_PAY_TO",
    "CREDITS_DECIMALS",
    "CREDITS_MAX_TIMEOUT_SECONDS",
    "V1_NETWORK_NAMES",
    "encode_base64_json",
    "decode_base64_json",
    "credits_to_nanocredits",
    "credits_to_asset_units",
    "is_amount_string",
    "is_well_formed_requirement",
    "requirement_matches",
    "to_v1_requirement",
    "build_payment_required",
    "read_payment_payload",
    "v1_network_to_caip2",
    "settle_response_header",
    "read_settlement_overrides",
]
