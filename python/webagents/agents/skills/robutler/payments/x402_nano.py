"""
A facilitator for the `nano:mainnet` exact scheme, for a self-hosted agent
whose chain seller is Nano (XNO). It plugs into `Paywall.chain["facilitator"]`
beside `HttpFacilitatorClient` and needs no third party: the payer sends the
block itself, this reads it from a Nano node the operator names, and nothing
here holds a key or a seed.

THE PAYMENT IS BOUND TO THE REQUEST. A Nano send block has no memo, so the 402
names an `order` in the chain entry's `extra`: a nonce, an expiry and an HMAC
over the resource URL, `request_binding(request)` (method, path, query, body
digest, the same S-297 binding the credits nonce covers), the amount and
`payTo`. The payer's send block must set its `representative` to the account
`SHA256("xno-binding/v1/rep|" + order + "|" + payTo + "|" + amount)`
(xno-call-binding, option C). A block paying the right amount to the right
account for another request, or for this request with other inputs, carries
another representative and is refused.

REPLAY. `settle` claims the block hash in a directory of claim files, one file
per block made with `O_CREAT | O_EXCL`, so a block is settled once across
restarts and across processes that share the directory. `verify` refuses a
block already claimed, and `settle` answers `replayed`, which the paywall
already treats as a replayed request: the handler's answer is withheld.

The payment payload is `{"block": "<64 hex>", "order": "<the 402's order>"}`.
Amounts are raw (1 XNO = 10**30 raw); the paywall derives them from credits
with `decimals = 30` and the operator's `units_per_credit` (XNO per credit).

Standard library only.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

NANO_NETWORK = "nano:mainnet"
NANO_ASSET = "XNO"
NANO_DECIMALS = 30
RAW_PER_XNO = 10 ** NANO_DECIMALS
BINDING = "xno-binding/v1/rep"

#: How long a 402's order may be paid for.
DEFAULT_ORDER_TTL_SECONDS = 600
#: A paid order may still be presented this long after its expiry (a slow confirmation).
PRESENT_GRACE_SECONDS = 600
#: A block's node time may run this far past the order's expiry.
SEND_GRACE_SECONDS = 60

_ALPHABET = "13456789abcdefghijkmnopqrstuwxyz"
_ADDRESS = re.compile(r"(?:nano|xrb)_([13][13456789abcdefghijkmnopqrstuwxyz]{59})")
_HASH = re.compile(r"[0-9A-Fa-f]{64}")
_RAW = re.compile(r"[0-9]{1,40}")


# ── Accounts ─────────────────────────────────────────────────────────────────


def _b32(value: int, chars: int) -> str:
    out = []
    for _ in range(chars):
        out.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(out))


def _unb32(text: str) -> int:
    value = 0
    for char in text:
        value = value * 32 + _ALPHABET.index(char)
    return value


def _checksum(public_key: bytes) -> bytes:
    return hashlib.blake2b(public_key, digest_size=5).digest()[::-1]


def public_key_of(address: Any) -> bytes:
    """The 32-byte public key of a `nano_` (or `xrb_`) address; ValueError when it
    is malformed or its checksum is wrong."""
    match = _ADDRESS.fullmatch(address.strip().lower()) if isinstance(address, str) else None
    if match is None:
        raise ValueError("not a nano_ address")
    body = match.group(1)
    key = _unb32(body[:52])
    if key >> 256:
        raise ValueError("not a nano_ address")
    public_key = key.to_bytes(32, "big")
    if _unb32(body[52:]).to_bytes(5, "big") != _checksum(public_key):
        raise ValueError("the nano_ address checksum is wrong")
    return public_key


def address_of(public_key: bytes) -> str:
    if len(public_key) != 32:
        raise ValueError("a public key is 32 bytes")
    return "nano_" + _b32(int.from_bytes(public_key, "big"), 52) + _b32(int.from_bytes(_checksum(public_key), "big"), 8)


def normalize_address(address: Any) -> str:
    """The checksum-validated `nano_` form of an address."""
    return address_of(public_key_of(address))


def same_account(a: Any, b: Any) -> bool:
    try:
        return public_key_of(a) == public_key_of(b)
    except ValueError:
        return False


def representative_for(order: str, pay_to: str, raw: str) -> str:
    """The representative a send block for `order` must name (xno-call-binding option C)."""
    return address_of(hashlib.sha256(f"{BINDING}|{order}|{pay_to}|{raw}".encode("utf-8")).digest())


# ── Replay ───────────────────────────────────────────────────────────────────


class ClaimStore:
    """Settled block hashes as files in a directory: one `O_EXCL` file per block,
    so a claim survives a restart and two processes cannot both win it."""

    def __init__(self, directory: str) -> None:
        if not directory:
            raise ValueError("x402 nano: a claims directory is required")
        self.directory = directory
        os.makedirs(directory, mode=0o700, exist_ok=True)

    def _path(self, block_hash: str) -> str:
        if not _HASH.fullmatch(block_hash):
            raise ValueError("a block hash is 64 hex digits")
        return os.path.join(self.directory, block_hash.lower())

    def claimed(self, block_hash: str) -> bool:
        return os.path.exists(self._path(block_hash))

    def claim(self, block_hash: str, record: Mapping[str, Any]) -> bool:
        """True when this call claimed the block, False when it was claimed before."""
        try:
            fd = os.open(self._path(block_hash), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                return False
            raise
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        return True


# ── The node ─────────────────────────────────────────────────────────────────


class NodeUnavailable(Exception):
    """The node gave no usable answer; nothing can be concluded."""


class BlockReader(Protocol):
    async def block_info(self, block_hash: str) -> Optional[Dict[str, Any]]: ...


class NodeRpc:
    """`block_info` from a Nano node's JSON-RPC (the operator's own, or one it trusts)."""

    def __init__(self, url: str, timeout: float = 15.0) -> None:
        if not url:
            raise ValueError("x402 nano: a node URL is required")
        self.url = url
        self.timeout = timeout

    def _post(self, body: Dict[str, Any]) -> Any:
        request = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise NodeUnavailable(f"block_info: {exc}") from exc

    async def block_info(self, block_hash: str) -> Optional[Dict[str, Any]]:
        answer = await asyncio.to_thread(self._post, {"action": "block_info", "json_block": "true", "hash": block_hash})
        if not isinstance(answer, dict):
            raise NodeUnavailable("block_info: not an object")
        if "error" in answer:
            if "not found" in str(answer["error"]).lower():
                return None
            raise NodeUnavailable(f"block_info: {answer['error']}")
        return answer


# ── The facilitator ──────────────────────────────────────────────────────────


def _invalid(reason: str) -> Dict[str, Any]:
    return {"isValid": False, "invalidReason": reason}


class NanoFacilitator:
    """`verify` and `settle` for `nano:mainnet` exact, bound to the request.

    `binds_request` tells the paywall to call `offer_extra` when it writes a 402
    and to pass `url` and `binding` to `verify` and `settle`.
    """

    binds_request = True

    def __init__(
        self,
        pay_to: str,
        order_secret: str,
        claims_dir: str,
        node_url: str = "",
        node: Optional[BlockReader] = None,
        order_ttl_seconds: int = DEFAULT_ORDER_TTL_SECONDS,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.pay_to = normalize_address(pay_to)
        if not isinstance(order_secret, str) or len(order_secret) < 32:
            raise ValueError("x402 nano: the order secret must be 32 or more characters")
        self._secret = order_secret.encode("utf-8")
        self.claims = ClaimStore(claims_dir)
        self.node: BlockReader = node if node is not None else NodeRpc(node_url)
        self.order_ttl_seconds = int(order_ttl_seconds)
        self._clock = clock or time.time

    # ── The 402 ──

    def _mac(self, nonce: str, expires: int, url: str, binding: str, amount: str, pay_to: str) -> str:
        message = json.dumps(["webagents-nano/v1", nonce, expires, url, binding, amount, pay_to], separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(hmac.new(self._secret, message, hashlib.sha256).digest()[:16]).decode("ascii").rstrip("=")

    def offer_extra(self, requirement: Mapping[str, Any], url: str, binding: str) -> Dict[str, Any]:
        """What the 402's chain entry adds: the order and the representative its block must name."""
        amount = str(requirement["amount"])
        nonce = secrets.token_hex(16)
        expires = int(self._clock()) + self.order_ttl_seconds
        order = f"{nonce}.{expires}.{self._mac(nonce, expires, url, binding, amount, self.pay_to)}"
        return {"order": order, "representative": representative_for(order, self.pay_to, amount), "binding": BINDING}

    # ── Verify and settle ──

    def _check_requirement(self, requirement: Mapping[str, Any]) -> Optional[str]:
        if requirement.get("scheme") != "exact" or requirement.get("network") != NANO_NETWORK:
            return "unsupported_scheme"
        if requirement.get("asset") != NANO_ASSET:
            return "invalid_asset"
        if not same_account(requirement.get("payTo"), self.pay_to):
            return "invalid_pay_to"
        amount = requirement.get("amount")
        if not isinstance(amount, str) or not _RAW.fullmatch(amount) or int(amount) <= 0:
            return "invalid_amount"
        return None

    def _check_order(self, order: Any, url: str, binding: str, amount: str) -> tuple:
        try:
            nonce, expires_text, mac = str(order).split(".")
            expires = int(expires_text)
        except ValueError:
            return None, "malformed_order"
        if not hmac.compare_digest(mac, self._mac(nonce, expires, url, binding, amount, self.pay_to)):
            return None, "order_not_for_this_request"
        if self._clock() > expires + PRESENT_GRACE_SECONDS:
            return None, "order_expired"
        return expires, None

    async def _verify(self, payload: Mapping[str, Any], requirement: Mapping[str, Any], url: str, binding: str) -> Dict[str, Any]:
        reason = self._check_requirement(requirement)
        if reason:
            return _invalid(reason)
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        block_hash, order = inner.get("block"), inner.get("order")
        if not isinstance(block_hash, str) or not _HASH.fullmatch(block_hash):
            return _invalid("malformed_payload")
        amount = requirement["amount"]
        expires, reason = self._check_order(order, url, binding, amount)
        if reason:
            return _invalid(reason)
        if self.claims.claimed(block_hash):
            return _invalid("replayed")
        try:
            info = await self.node.block_info(block_hash.upper())
        except NodeUnavailable:
            return _invalid("node_unavailable")
        if not info:
            return _invalid("block_unknown")
        contents = info.get("contents") if isinstance(info.get("contents"), dict) else {}
        if str(info.get("confirmed", "")).lower() != "true":
            return _invalid("block_unconfirmed")
        if info.get("subtype") != "send" or contents.get("type") != "state":
            return _invalid("not_a_send")
        if not same_account(contents.get("link_as_account") or info.get("link_as_account"), self.pay_to):
            return _invalid("wrong_recipient")
        sent = str(info.get("amount", ""))
        if not _RAW.fullmatch(sent) or int(sent) != int(amount):
            return _invalid("wrong_amount")
        if not same_account(contents.get("representative"), representative_for(str(order), self.pay_to, amount)):
            return _invalid("block_not_for_this_request")
        try:
            sent_at = int(info.get("local_timestamp") or 0)
        except (TypeError, ValueError):
            sent_at = 0
        if sent_at <= 0 or sent_at > expires + SEND_GRACE_SECONDS:
            return _invalid("sent_after_order_expired")
        try:
            payer = normalize_address(info.get("block_account") or contents.get("account"))
        except ValueError:
            return _invalid("no_valid_sender")
        return {"isValid": True, "payer": payer}

    async def verify(self, payload: Mapping[str, Any], requirement: Mapping[str, Any], url: str = "", binding: str = "") -> Dict[str, Any]:
        return await self._verify(payload, requirement, url, binding)

    async def settle(self, payload: Mapping[str, Any], requirement: Mapping[str, Any], url: str = "", binding: str = "") -> Dict[str, Any]:
        """Verify again on the requirement given here, then claim the block once.
        The payer already moved the value: there is nothing to submit."""
        failed = {"success": False, "transaction": "", "network": NANO_NETWORK}
        checked = await self._verify(payload, requirement, url, binding)
        if not checked.get("isValid"):
            return {**failed, "errorReason": checked.get("invalidReason") or "verification_failed"}
        block_hash = str(payload["payload"]["block"]).upper()
        if not self.claims.claim(block_hash, {"block": block_hash, "url": url, "amount": requirement["amount"], "payer": checked["payer"], "at": int(self._clock())}):
            return {**failed, "errorReason": "replayed"}
        return {"success": True, "transaction": block_hash, "network": NANO_NETWORK, "payer": checked["payer"], "amount": requirement["amount"]}


def nano_facilitator_from_config(pay_to: str, config: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, str]] = None) -> NanoFacilitator:
    """A Nano facilitator from config, the environment the fallback for each part:
    `node_url` (`NANO_NODE_URL`), `order_secret` (`NANO_ORDER_SECRET`),
    `claims_dir` (`NANO_CLAIMS_DIR`), `order_ttl_seconds`."""
    config = dict(config or {})
    env = dict(env if env is not None else os.environ)
    return NanoFacilitator(
        pay_to,
        order_secret=config.get("order_secret") or env.get("NANO_ORDER_SECRET") or "",
        claims_dir=config.get("claims_dir") or env.get("NANO_CLAIMS_DIR") or "",
        node_url=config.get("node_url") or env.get("NANO_NODE_URL") or "",
        node=config.get("node"),
        order_ttl_seconds=int(config.get("order_ttl_seconds", DEFAULT_ORDER_TTL_SECONDS)),
    )


__all__ = [
    "NANO_NETWORK",
    "NANO_ASSET",
    "NANO_DECIMALS",
    "RAW_PER_XNO",
    "ClaimStore",
    "NodeRpc",
    "NodeUnavailable",
    "NanoFacilitator",
    "nano_facilitator_from_config",
    "normalize_address",
    "representative_for",
]
