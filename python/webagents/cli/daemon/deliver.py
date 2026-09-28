"""
Where a scheduled run's reply goes (plan item 1.7, 2026-09-26): the `deliver`
target of a `cron:` schedule (`cli/loader/schedules.py`), one deliverer per
kind, the same words and bytes as `typescript/src/daemon/deliver.ts`, held by
the shared fixture (`tests/fixtures/daemon/cron.json`: `details`, `webhook`,
`chat`).

`file` appends an entry to a path inside the agent's folder. The loader
already refused a path that names its way out (`..`, an absolute path); this
checks the REAL path too, because a link inside the folder can point outside
it, and appending through one would write where the file never said.

`webhook` POSTs the run as JSON, signed the way the REST tool signs an
outbound request (Web Bot Auth, `crypto/http_signature.py`) when the agent
holds a signing identity (`agent.signing_identity`, what `WebAgentsServer`
attaches to a served agent); an agent without one posts unsigned and the run's
record says so, as the REST tool reports its own unsigned requests. A hop that
cannot be reached, or that answers 408, 429 or 5xx, is tried again up to
`retries` times with doubling backoff (1, 2, 4 ... capped at 30 s); any other
4xx is final, since asking again does not change the answer. Every try is
signed afresh: a signature carries a nonce and a 60 s window, and a replayed
one is refused by any verifier that keeps nonces.

`chat` records the run into the owner's chat with the agent on Robutler,
through the route the chat's `robutler` session backend uses
(`cli/robutler_sessions.py`, `POST /api/agents/{id}/conversations`): the
schedule's prompt as the owner's words, the reply as the agent's, a
heartbeat's report alone. The session id is uuid5 of the agent and schedule
names in the URL namespace (`chat_session_id`), so every run of one schedule
lands in the same chat and nothing has to be stored for it. It needs the
person signed in and the agent published, as the chat does; without either
the run fails and its record says which.

`channel` is the channel relay's, refused by the loader until it exists; a
kind with no deliverer is reported as such in the run's record rather than
silently dropped, so a schedule never looks delivered when nothing happened.

The seams on `DeliveryContext` (`sleep`, `chat`) exist for the tests, which
deliver to a local server and a mocked platform client.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..loader.schedules import DeliverTarget

#: `(outcome, detail)`: `delivered`, `nothing` (a heartbeat with nothing to report) or `failed`.
Outcome = Tuple[str, str]

#: One appended entry (the fixture's `file_entry`).
FILE_ENTRY = "## {schedule}, {ran_at}\n\n{content}\n\n"
#: Seconds waited before each retry of a webhook (the fixture's `webhook.backoff_seconds`).
WEBHOOK_BACKOFF_SECONDS = (1, 2, 4, 8, 16, 30, 30, 30, 30, 30)
#: Answers worth a retry beside 5xx (the fixture's `webhook.retry_statuses`).
WEBHOOK_RETRY_STATUSES = frozenset({408, 429})
#: The uuid5 namespace a chat session id is derived in (RFC 4122's URL namespace, the fixture's `chat.namespace`).
CHAT_SESSION_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
#: Characters one recorded message keeps (`cli/robutler_sessions.py` RECORD_CHARS, the platform's limit).
_CHAT_RECORD_CHARS = 100_000


@dataclass
class RunResult:
    """What one scheduled turn produced."""

    agent: str
    schedule: str
    #: `cron` or `every`.
    kind: str
    #: The message the turn was given; None for a heartbeat.
    prompt: Optional[str]
    content: str
    #: When the slot fired, ISO 8601 UTC to the second.
    ran_at: str


@dataclass
class DeliveryContext:
    """Where the agent lives and, when a deliverer signs or posts as it, the agent itself."""

    agent_dir: Path
    agent: Any = None
    #: The webhook deliverer's wait between tries; `asyncio.sleep` by default.
    sleep: Optional[Callable[[float], Awaitable[None]]] = None
    #: The chat deliverer's platform client (`ChatClient`); the person's sign-in and the folder's link by default.
    chat: Any = None


Deliverer = Callable[[DeliverTarget, RunResult, DeliveryContext], Awaitable[Outcome]]


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


def tries(n: int) -> str:
    """`1 try` or `N tries` (the fixture's `{tries}`)."""
    return "1 try" if n == 1 else f"{n} tries"


# -- file -----------------------------------------------------------------------------------


async def deliver_file(target: DeliverTarget, result: RunResult, ctx: DeliveryContext) -> Outcome:
    """Append the entry to `target.path` under the agent's folder, never outside it."""
    root = Path(ctx.agent_dir).resolve()
    wanted = root / str(target.path)
    wanted.parent.mkdir(parents=True, exist_ok=True)
    dest = wanted.parent.resolve() / wanted.name
    # The file itself may be a link, and a DANGLING one is invisible to
    # `exists()`: appending through it would create the target outside.
    if dest.is_symlink():
        dest = (dest.parent / os.readlink(dest)).resolve()
    if dest != root and root not in dest.parents:
        return "failed", f"file {target.path}: outside the agent's folder"
    with dest.open("a", encoding="utf-8") as out:
        out.write(FILE_ENTRY.format(schedule=result.schedule, ran_at=result.ran_at, content=result.content))
    return "delivered", f"file {target.path}"


# -- webhook --------------------------------------------------------------------------------


def webhook_body(result: RunResult) -> str:
    """The JSON a webhook receives (the fixture's `webhook.body`): these keys, this order."""
    return json.dumps(
        {
            "agent": result.agent,
            "schedule": result.schedule,
            "kind": result.kind,
            "prompt": result.prompt,
            "content": result.content,
            "ran_at": result.ran_at,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _signing_identity_of(agent: Any) -> Any:
    """The agent's signing identity, where the server leaves it (the REST tool reads the same)."""
    identity = getattr(agent, "signing_identity", None)
    if identity is None or not isinstance(getattr(identity, "issuer", None), str) or not callable(getattr(identity, "held_keys", None)):
        return None
    return identity


def _signed_headers(agent: Any, url: str, body: bytes) -> Tuple[Dict[str, str], str, str]:
    """`(headers, url on the wire, words for the record)` for one try: `signed as ...` or `unsigned: ...`."""
    identity = _signing_identity_of(agent)
    if identity is None:
        return {}, url, "unsigned: this agent holds no signing key"
    try:
        from webagents.crypto.http_signature import sign_request

        signed = sign_request(identity.held_keys(), identity.issuer, "POST", url, body)
    except Exception as error:  # noqa: BLE001 - a key that cannot sign is the record's business, not a crash
        return {}, url, f"unsigned: this agent's key could not sign: {_describe(error)}"
    wire_url = signed.target.url if signed.target is not None else url
    return dict(signed.headers), wire_url, f"signed as {identity.issuer}"


async def deliver_webhook(target: DeliverTarget, result: RunResult, ctx: DeliveryContext) -> Outcome:
    """POST the run to `target.url` (file comment, `webhook`)."""
    import httpx

    sleep = ctx.sleep or asyncio.sleep
    url = str(target.url)
    body = webhook_body(result).encode("utf-8")
    attempts = int(target.retries or 0) + 1
    timeout = float(target.timeout or 15)
    last_status: Optional[int] = None
    for attempt in range(1, attempts + 1):
        headers, wire_url, signing = _signed_headers(ctx.agent, url, body)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(wire_url, content=body, headers={"Content-Type": "application/json", **headers})
            last_status = response.status_code
            if 200 <= response.status_code < 300:
                return "delivered", f"webhook {url} ({signing})"
            if not (response.status_code in WEBHOOK_RETRY_STATUSES or response.status_code >= 500):
                return "failed", f"webhook {url}: answered {response.status_code} after {tries(attempt)}"
        except httpx.HTTPError:
            last_status = None
        if attempt < attempts:
            await sleep(WEBHOOK_BACKOFF_SECONDS[min(attempt - 1, len(WEBHOOK_BACKOFF_SECONDS) - 1)])
    reason = "could not be reached" if last_status is None else f"answered {last_status}"
    return "failed", f"webhook {url}: {reason} after {tries(attempts)}"


# -- chat -----------------------------------------------------------------------------------


def chat_session_id(agent: str, schedule: str) -> str:
    """uuid5 of `webagents:cron:<agent>/<schedule>` in the URL namespace (the fixture's `chat.sessions`)."""
    return str(uuid.uuid5(CHAT_SESSION_NAMESPACE, f"webagents:cron:{agent}/{schedule}"))


def chat_turn(result: RunResult) -> Dict[str, Any]:
    """The turn the run records (the fixture's `chat.turn` and `chat.heartbeat_turn`)."""
    messages: List[Dict[str, str]] = []
    if result.prompt is not None:
        messages.append({"role": "user", "content": result.prompt[:_CHAT_RECORD_CHARS]})
    messages.append({"role": "assistant", "content": result.content[:_CHAT_RECORD_CHARS]})
    return {"sessionId": chat_session_id(result.agent, result.schedule), "messages": messages}


class ChatClient:
    """The platform client the chat deliverer posts through (`chat` on the
    context replaces it in tests): `target` answers a `ConversationsTarget`, or
    `signed_out` / `not_published`; `record` records a turn and answers the
    chat's id."""

    async def target(self, agent_dir: Path, agent_name: str) -> Any:
        from ..config_store import resolve_platform_url
        from ..credentials import get_token
        from ..robutler_sessions import ConversationsTarget, linked_platform_agent

        token = get_token()
        if not token:
            return "signed_out"
        agent_id = linked_platform_agent(Path(agent_dir), agent_name)
        if not agent_id:
            return "not_published"
        return ConversationsTarget(base=resolve_platform_url()[0], token=token, agent_id=agent_id)

    async def record(self, target: Any, turn: Dict[str, Any], command: Callable[[str], str]) -> str:
        from ..robutler_sessions import record_platform_turn

        return await record_platform_turn(target, turn, command)


async def deliver_chat(target: DeliverTarget, result: RunResult, ctx: DeliveryContext) -> Outcome:
    """Record the run into the owner's chat with the agent (file comment, `chat`)."""
    from ..config_store import cli_command

    client = ctx.chat if ctx.chat is not None else ChatClient()
    where = await client.target(Path(ctx.agent_dir), result.agent)
    if where == "signed_out":
        return "failed", f"chat owner: not signed in: run `{cli_command('login')}`"
    if where == "not_published":
        return "failed", f"chat owner: this agent is not on Robutler: run `{cli_command('publish')}`"
    try:
        chat_id = await client.record(where, chat_turn(result), cli_command)
    except Exception as error:  # noqa: BLE001 - the platform's refusal is the record's business
        return "failed", f"chat owner: {_describe(error)}"
    return "delivered", f"chat owner ({chat_id})"


# -- dispatch -------------------------------------------------------------------------------

#: The deliverer for each `deliver` kind.
DELIVERERS: Dict[str, Deliverer] = {"file": deliver_file, "webhook": deliver_webhook, "chat": deliver_chat}


async def deliver(target: DeliverTarget, result: RunResult, ctx: DeliveryContext) -> Outcome:
    """Deliver `result` where `target` says; a kind nothing serves is a failed run, said."""
    deliverer = DELIVERERS.get(target.kind)
    if deliverer is None:
        return "failed", f"{target.kind} delivery is not available in this build"
    return await deliverer(target, result, ctx)
