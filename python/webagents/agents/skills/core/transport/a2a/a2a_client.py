"""
The A2A v1.0 client: how an agent built with this SDK calls a peer that
speaks A2A, the Python half of `a2a/a2a-client.ts`; the shared fixture
`tests/fixtures/a2a/vectors.json` (`client`) pins the interface choice and
the result shapes both read. Plan item 1.3, 2026-09-26.

DISCOVERY IS HERMES' WALK (spec pack 3.7): `GET {base}/.well-known/agent-card.json`,
then `/.well-known/agent.json` on a 404, then the first `supportedInterfaces`
entry whose binding is `JSONRPC` and whose version is 1.0. Hermes takes the
first JSONRPC entry without checking its version; the version is checked here
because a v0.3 card (`url` plus `preferredTransport`, no `supportedInterfaces`)
names no interface a v1.0 `SendMessage` could reach, and refusing it at the
card is clearer than a -32601 or a parse error one hop later.

THE CALL is one JSON-RPC `SendMessage` with `A2A-Version: 1.0`, the text as a
`ROLE_USER` part with a `mediaType` (what Hermes sends), a bearer configured
out of band (the agent file's `peers`, `peer_token_for`; never discovered),
and ONE retry as `message/send` with the same body on -32601, the way OpenClaw
retries. Both result shapes are read, `{"task"}` and `{"message"}`, plus a bare
legacy task or message, and a task that is not yet settled is polled with
`GetTask` until it is or the deadline passes. A 3xx is refused: an A2A
endpoint is called where it is, never followed.

A card's signature is checked when asked (`verify_card`), with the key fetched
from the signature's `jku`, which `card.py` allows only at the card's own
origin.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

from .card import AGENT_CARD_WELL_KNOWN_SUFFIX, LEGACY_CARD_WELL_KNOWN_SUFFIX, VerifyCardResult, verify_agent_card
from .protocol import reply_text
from .types import A2A_VERSION, A2A_VERSION_HEADER, A2A_VERSIONS_ACCEPTED, is_settled

#: What OpenClaw and Hermes wait for a blocking send.
A2A_CLIENT_TIMEOUT_SECONDS = 120.0
A2A_POLL_INTERVAL_SECONDS = 0.5
METHOD_NOT_FOUND = -32601

#: The v0.3 spelling a v1.0 method is retried under, once, on -32601.
_DOTTED_ALIASES = {"SendMessage": "message/send", "GetTask": "tasks/get", "CancelTask": "tasks/cancel"}

__all__ = [
    "A2AClientError",
    "A2A_CLIENT_TIMEOUT_SECONDS",
    "A2A_POLL_INTERVAL_SECONDS",
    "METHOD_NOT_FOUND",
    "call_agent",
    "card_names_jku",
    "fetch_agent_card",
    "get_task",
    "message_text",
    "pick_interface",
    "read_send_result",
    "reply_of_result",
    "rpc",
    "send_message",
    "text_message",
]


class A2AClientError(Exception):
    """What a call could not do: `code` is the JSON-RPC error code when the
    peer answered one, `status` the HTTP status when that is what went wrong."""

    def __init__(self, message: str, *, code: Optional[int] = None, status: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.data = data


def _client(client: Any, timeout: float) -> Tuple[Any, bool]:
    """`(client, owned)`: the given httpx client, or a new one this call closes."""
    if client is not None:
        return client, False
    import httpx

    return httpx.AsyncClient(timeout=timeout, follow_redirects=False), True


async def _request(client: Any, method: str, url: str, *, token: Optional[str], headers: Optional[Dict[str, str]], json_body: Any = None) -> Any:
    """One request with the client's headers and bearer; a redirect is refused."""
    merged: Dict[str, str] = {"accept": "application/json", **(headers or {})}
    if token:
        merged["authorization"] = f"Bearer {token}"
    response = await client.request(method, url, headers=merged, json=json_body)
    if 300 <= response.status_code < 400:
        raise A2AClientError(f"{url} answered a redirect ({response.status_code}); an A2A endpoint is called where it is", status=response.status_code)
    return response


async def fetch_agent_card(
    base_url: str,
    *,
    token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
    client: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """`(card, card_url)`: the v1.0 path, then the legacy path on a 404."""
    base = base_url.rstrip("/")
    http, owned = _client(client, timeout)
    try:
        for suffix in (AGENT_CARD_WELL_KNOWN_SUFFIX, LEGACY_CARD_WELL_KNOWN_SUFFIX):
            url = f"{base}{suffix}"
            response = await _request(http, "GET", url, token=token, headers=headers)
            if response.status_code == 404:
                continue
            if response.status_code >= 400:
                raise A2AClientError(f"{url} answered {response.status_code}", status=response.status_code)
            try:
                card = response.json()
            except ValueError:
                raise A2AClientError(f"{url} is not JSON") from None
            if not isinstance(card, dict):
                raise A2AClientError(f"{url} is not a JSON object")
            return card, url
        raise A2AClientError(
            f"no agent card at {base}: {AGENT_CARD_WELL_KNOWN_SUFFIX} and {LEGACY_CARD_WELL_KNOWN_SUFFIX} both answered 404",
            status=404,
        )
    finally:
        if owned:
            await http.aclose()


def pick_interface(card: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The first JSON-RPC interface at protocol version 1.0 (`{"url", "tenant"?}`); None when the card names none."""
    for entry in card.get("supportedInterfaces") or []:
        if not isinstance(entry, dict) or entry.get("protocolBinding") != "JSONRPC":
            continue
        version = entry.get("protocolVersion")
        if not isinstance(version, str) or version.strip() not in A2A_VERSIONS_ACCEPTED:
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            continue
        picked = {"url": url}
        tenant = entry.get("tenant")
        if isinstance(tenant, str) and tenant:
            picked["tenant"] = tenant
        return picked
    return None


def card_names_jku(card: Dict[str, Any]) -> bool:
    """Whether any of the card's signatures names a `jku` in its protected
    header: what `verify_card="jku"` verifies against."""
    import base64

    for entry in card.get("signatures") or []:
        protected = entry.get("protected") if isinstance(entry, dict) else None
        if not isinstance(protected, str):
            continue
        try:
            header = json.loads(base64.urlsafe_b64decode(protected + "=" * ((4 - len(protected) % 4) % 4)))
        except Exception:  # noqa: BLE001 - not a readable header: nothing named
            continue
        if isinstance(header, dict) and isinstance(header.get("jku"), str) and header["jku"]:
            return True
    return False


def text_message(text: str, *, context_id: Optional[str] = None, message_id: Optional[str] = None) -> Dict[str, Any]:
    """A `ROLE_USER` text message, the part with a `mediaType` as Hermes sends it."""
    message: Dict[str, Any] = {
        "messageId": message_id or str(uuid.uuid4()),
        "role": "ROLE_USER",
        "parts": [{"text": text, "mediaType": "text/plain"}],
    }
    if context_id:
        message["contextId"] = context_id
    return message


def message_text(message: Dict[str, Any]) -> str:
    """The text of a message result."""
    return "".join(p["text"] for p in message.get("parts") or [] if isinstance(p, dict) and isinstance(p.get("text"), str))


def read_send_result(result: Any) -> Dict[str, Any]:
    """A send's result as either SDK answers it (`{"task"}` or `{"message"}`),
    or as a legacy server does (the bare task or message)."""
    if not isinstance(result, dict):
        raise A2AClientError("the peer answered no result object")
    if isinstance(result.get("task"), dict):
        return {"task": result["task"]}
    if isinstance(result.get("message"), dict):
        return {"message": result["message"]}
    if isinstance(result.get("id"), str) and isinstance(result.get("status"), dict):
        task = dict(result)
        task.setdefault("artifacts", [])
        task.setdefault("history", [])
        return {"task": task}
    if isinstance(result.get("parts"), list):
        return {"message": result}
    raise A2AClientError("the peer answered neither a task nor a message")


def reply_of_result(result: Dict[str, Any]) -> str:
    """The reply text of a send result: artifacts first, then the status message; a message's text."""
    return reply_text(result["task"]) if "task" in result else message_text(result["message"])


async def rpc(
    rpc_url: str,
    method: str,
    params: Dict[str, Any],
    *,
    token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = A2A_CLIENT_TIMEOUT_SECONDS,
    client: Any = None,
) -> Any:
    """One JSON-RPC call with `A2A-Version: 1.0`; on -32601 the v0.3 alias is
    tried once with the same params. A JSON-RPC error is raised with its code."""
    http, owned = _client(client, timeout)

    async def call(name: str) -> Any:
        response = await _request(
            http,
            "POST",
            rpc_url,
            token=token,
            headers={"content-type": "application/json", A2A_VERSION_HEADER: A2A_VERSION, **(headers or {})},
            json_body={"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": name, "params": params},
        )
        try:
            body = response.json()
        except ValueError:
            raise A2AClientError(f"{rpc_url} answered {name} with {response.status_code} and no JSON", status=response.status_code) from None
        if not isinstance(body, dict):
            raise A2AClientError(f"{rpc_url} answered {name} with something other than a JSON-RPC response", status=response.status_code)
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), int) else None
            raise A2AClientError(error.get("message") if isinstance(error.get("message"), str) else f"{name} failed", code=code, status=response.status_code, data=error.get("data"))
        if response.status_code >= 400:
            raise A2AClientError(f"{rpc_url} answered {response.status_code} to {name}", status=response.status_code, data=body)
        return body.get("result")

    try:
        try:
            return await call(method)
        except A2AClientError as error:
            alias = _DOTTED_ALIASES.get(method)
            if alias and error.code == METHOD_NOT_FOUND:
                return await call(alias)
            raise
    finally:
        if owned:
            await http.aclose()


async def send_message(
    rpc_url: str,
    message: Dict[str, Any],
    *,
    configuration: Optional[Dict[str, Any]] = None,
    tenant: Optional[str] = None,
    **options: Any,
) -> Dict[str, Any]:
    """`SendMessage` to a JSON-RPC interface: `{"task": ...}` or `{"message": ...}`."""
    params: Dict[str, Any] = {"message": message}
    if configuration:
        params["configuration"] = configuration
    if tenant:
        params["tenant"] = tenant
    return read_send_result(await rpc(rpc_url, "SendMessage", params, **options))


async def get_task(rpc_url: str, task_id: str, *, tenant: Optional[str] = None, **options: Any) -> Dict[str, Any]:
    """`GetTask` on a JSON-RPC interface."""
    params: Dict[str, Any] = {"id": task_id}
    if tenant:
        params["tenant"] = tenant
    result = read_send_result(await rpc(rpc_url, "GetTask", params, **options))
    if "task" not in result:
        raise A2AClientError(f"GetTask {task_id} answered a message, not a task")
    return result["task"]


async def call_agent(
    base_url: str,
    message_or_text: Union[str, Dict[str, Any]],
    *,
    token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    context_id: Optional[str] = None,
    return_immediately: bool = False,
    poll_interval: float = A2A_POLL_INTERVAL_SECONDS,
    timeout: float = A2A_CLIENT_TIMEOUT_SECONDS,
    verify_card: Union[bool, str] = False,
    verify: Optional[Dict[str, Any]] = None,
    client: Any = None,
    card: Optional[Tuple[Dict[str, Any], str]] = None,
) -> Dict[str, Any]:
    """Call a peer: its card, the interface, one send, and the task polled to
    a settled state. Answers `card`, `card_url`, `rpc_url`, `verified` (when
    asked), `task` or `message`, and `reply`.

    `verify_card=True` verifies every card, so an unsigned one is refused;
    `"jku"` verifies a card whose signature names a `jku` and lets an unsigned
    card through as unverified (`delegate` asks for this: OpenClaw's card is
    unsigned). `card` is `(card, card_url)` already fetched from `base_url` (a
    probe), so discovery is not repeated."""
    http, owned = _client(client, timeout)
    try:
        if card is not None:
            card, card_url = card
        else:
            card, card_url = await fetch_agent_card(base_url, token=token, headers=headers, timeout=timeout, client=http)
        verified: Optional[VerifyCardResult] = None
        if verify_card is True or (verify_card == "jku" and card_names_jku(card)):
            verified = await verify_agent_card(card, card_url=card_url, **(verify or {}))
            if not verified.ok:
                raise A2AClientError(f"the card at {card_url} does not verify: {verified.reason or 'no signature verified'}")
        interface = pick_interface(card)
        if interface is None:
            raise A2AClientError(f"{card_url} names no A2A 1.0 JSON-RPC interface")
        message = text_message(message_or_text, context_id=context_id) if isinstance(message_or_text, str) else message_or_text
        deadline = time.monotonic() + timeout
        options = {"token": token, "headers": headers, "timeout": timeout, "client": http}
        result = await send_message(
            interface["url"],
            message,
            configuration={"returnImmediately": True} if return_immediately else None,
            tenant=interface.get("tenant"),
            **options,
        )
        out: Dict[str, Any] = {"card": card, "card_url": card_url, "rpc_url": interface["url"], "verified": verified}
        if "message" in result:
            out["message"] = result["message"]
            out["reply"] = message_text(result["message"])
            return out
        task = result["task"]
        while not is_settled(task["status"]["state"]) and time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            task = await get_task(interface["url"], task["id"], tenant=interface.get("tenant"), **options)
        out["task"] = task
        out["reply"] = reply_text(task)
        return out
    finally:
        if owned:
            await http.aclose()
