"""
The chat's conversations on Robutler (2026-09-25): the ``robutler`` backend of
the session skill, as the chat uses it.

ON ROBUTLER A CONVERSATION IS A CHAT. So a person's conversation with their
own agent in this terminal is recorded into their chat with that agent on the
platform (``/api/agents/{id}/conversations``, the portal's
``lib/messaging/recorded-conversations.ts``): their words as them, the agent's
replies as the agent, without waking the agent there or notifying anyone. It
shows in their chat list, and ``/resume`` on another machine, or after a chat
on the web, continues it.

WHAT IT NEEDS: the person signed in (``webagents login``, the token the request
carries; the platform answers only the agent's owner) and the agent on
Robutler (``webagents publish``, whose folder link names its platform id).
Without either, conversations stay on this machine and the chat says why,
once. The TypeScript chat's twin is ``typescript/src/cli/robutler-sessions.ts``;
``tests/fixtures/sessions/sessions.json`` holds the words both say.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Messages one record call carries, and characters one message keeps: the platform's own limits.
RECORD_BATCH = 50
RECORD_CHARS = 100_000


@dataclass
class PlatformConversation:
    """One of the owner's conversations with the agent, as the platform lists it."""

    chat_id: str
    #: The session id this chat was recorded from; None for a chat started on the web.
    session_id: Optional[str]
    updated_at: str
    message_count: int
    preview: str


@dataclass
class ConversationsTarget:
    """Where and as whom: the platform, the person's token, the agent's platform id."""

    base: str
    token: str
    agent_id: str


def unavailable_reason(why: str, command: Callable[[str], str]) -> str:
    """The sentence the chat shows, once, when the ``robutler`` backend cannot work here."""
    if why == "signed_out":
        return f"Conversations stay on this machine: sign in with `{command('login')}` to keep them on Robutler too."
    return f"Conversations stay on this machine: publish this agent with `{command('publish')}` to keep them on Robutler too."


class PlatformConversationsError(Exception):
    """A refusal or failure from the platform, with the sentence to show."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def failure_sentence(status: int, command: Callable[[str], str]) -> str:
    """The sentence for a platform answer that is not a success (both SDKs say the same)."""
    if status == 401:
        return f"your sign-in has expired: run `{command('login')}`"
    if status == 404:
        return "Robutler does not know this agent as yours"
    if status == 0:
        return "Robutler could not be reached"
    return f"Robutler answered {status}"


def linked_platform_agent(folder: Path, agent_name: str) -> Optional[str]:
    """The folder's link to its platform agent (``webagents publish`` / ``link``),
    when it names THIS agent: ``link.agentId`` and ``link.agentName`` in the
    folder's own ``.webagents/config.json``."""
    try:
        data = json.loads((Path(folder) / ".webagents" / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    agent_id = data.get("link.agentId")
    name = data.get("link.agentName")
    if not isinstance(agent_id, str) or not agent_id:
        return None
    # `alice.helper` is the agent `helper`; a link naming another agent in
    # this folder is not this one's.
    if isinstance(name, str) and name and name != agent_name and not name.endswith(f".{agent_name}"):
        return None
    return agent_id


async def _call(
    target: ConversationsTarget,
    route: str,
    command: Callable[[str], str],
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import httpx

    url = f"{target.base}/api/agents/{target.agent_id}/conversations{route}"
    headers = {"Authorization": f"Bearer {target.token}"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if body is None:
                response = await client.get(url, headers=headers)
            else:
                response = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError:
        raise PlatformConversationsError(failure_sentence(0, command), 0) from None
    if response.status_code >= 400:
        raise PlatformConversationsError(failure_sentence(response.status_code, command), response.status_code)
    try:
        data = response.json()
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


async def list_platform_conversations(
    target: ConversationsTarget, command: Callable[[str], str]
) -> List[PlatformConversation]:
    """The owner's conversations with the agent, newest first."""
    body = await _call(target, "?limit=20", command)
    out: List[PlatformConversation] = []
    for c in body.get("conversations") or []:
        if not isinstance(c, dict) or not isinstance(c.get("chatId"), str):
            continue
        session = c.get("sessionId")
        out.append(
            PlatformConversation(
                chat_id=c["chatId"],
                session_id=session if isinstance(session, str) else None,
                updated_at=str(c.get("updatedAt") or ""),
                message_count=int(c.get("messageCount") or 0),
                preview=str(c.get("preview") or ""),
            )
        )
    return out


async def read_platform_conversation(
    target: ConversationsTarget, chat_id: str, command: Callable[[str], str]
) -> List[Dict[str, str]]:
    """One conversation's words, oldest first."""
    body = await _call(target, f"/{chat_id}", command)
    return [
        {"role": m["role"], "content": m["content"]}
        for m in body.get("messages") or []
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]


async def record_platform_turn(
    target: ConversationsTarget, turn: Dict[str, Any], command: Callable[[str], str]
) -> str:
    """Record messages into the owner's chat with the agent; answers the chat's id."""
    body = await _call(target, "", command, body=turn)
    chat_id = body.get("chatId")
    if not isinstance(chat_id, str):
        raise PlatformConversationsError(failure_sentence(500, command), 500)
    return chat_id


def words_since(messages: Sequence[Any], start: int) -> List[Dict[str, str]]:
    """What a turn adds to the record: the person's and the agent's words since
    ``start``, text only (the platform takes nothing else), in order, each cut to
    ``RECORD_CHARS``."""
    out: List[Dict[str, str]] = []
    for m in list(messages)[start:]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            out.append({"role": m["role"], "content": content[:RECORD_CHARS]})
    return out


async def record_words(
    target: ConversationsTarget,
    session_id: str,
    chat_id: Optional[str],
    words: List[Dict[str, str]],
    command: Callable[[str], str],
) -> Optional[str]:
    """Record ``words`` into the conversation, ``RECORD_BATCH`` at a time, the
    first batch finding or making the chat; answers the chat's id."""
    chat = chat_id
    for i in range(0, len(words), RECORD_BATCH):
        turn: Dict[str, Any] = {"sessionId": session_id, "messages": words[i : i + RECORD_BATCH]}
        if chat:
            turn["chatId"] = chat
        chat = await record_platform_turn(target, turn, command)
    return chat


@dataclass
class ConversationEntry:
    """One row of ``/resume``: a conversation on this machine, on Robutler, or both."""

    updated_at: str
    local_count: int
    platform_count: int
    preview: str
    #: Only on Robutler: started on the web, or on another machine.
    only_on_robutler: bool
    #: Its id on this machine, when it is here.
    id: Optional[str] = None
    #: Its platform chat, when it is on Robutler.
    chat_id: Optional[str] = None
    #: The session it was recorded from, for a conversation only on Robutler.
    session_id: Optional[str] = None


def _when(iso: str) -> float:
    try:
        then = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return then.timestamp()


def merge_conversations(local: Sequence[Any], platform: Sequence[PlatformConversation]) -> List[ConversationEntry]:
    """This machine's conversations and Robutler's, as one list, newest first:
    a conversation recorded from here is one row (matched by its chat, or by
    the session it was recorded from), the rest are rows of their own."""
    used: set = set()
    rows: List[ConversationEntry] = []
    for entry in local:
        match = next(
            (
                c
                for c in platform
                if c.chat_id not in used and (c.chat_id == entry.chat_id or (c.session_id is not None and c.session_id == entry.id))
            ),
            None,
        )
        if match is not None:
            used.add(match.chat_id)
        rows.append(
            ConversationEntry(
                id=entry.id,
                chat_id=match.chat_id if match is not None else entry.chat_id,
                updated_at=match.updated_at if match is not None and _when(match.updated_at) > _when(entry.updated_at) else entry.updated_at,
                local_count=entry.message_count,
                platform_count=match.message_count if match is not None else 0,
                preview=entry.preview or (match.preview if match is not None else ""),
                only_on_robutler=False,
            )
        )
    for c in platform:
        if c.chat_id in used:
            continue
        rows.append(
            ConversationEntry(
                chat_id=c.chat_id,
                session_id=c.session_id,
                updated_at=c.updated_at,
                local_count=0,
                platform_count=c.message_count,
                preview=c.preview,
                only_on_robutler=True,
            )
        )
    rows.sort(key=lambda r: _when(r.updated_at), reverse=True)
    return rows
