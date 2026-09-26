"""
The conversations the chat keeps (2026-09-24), in the same files the
TypeScript chat writes (`typescript/src/cli/sessions.ts`), so either CLI can
pick up the other's.

WHERE: ``~/.webagents[-profile]/sessions/<folder>/<agent>/<id>.json``, with a
``.latest`` pointer beside them. Under the profile's own directory, never in
the project: a chat started in any folder must not leave files there. The chat
used to write ``.webagents/sessions`` next to the agent file, which for the
built-in agent was inside the installed package, one pile shared by every
folder. ``<folder>`` is the folder's absolute path with every character other
than a letter, a digit, ``.``, ``_`` or ``-`` turned into ``-``
(``/Users/me/x`` -> ``-Users-me-x``), the same in both SDKs.

WHAT: ``session_id``, ``agent_name``, ``created_at``, ``updated_at``,
``messages`` (OpenAI shape), ``metadata``, ``input_tokens``, ``output_tokens``,
the same in both SDKs. Owner-only files, written whole and then renamed into
place, so a crash never leaves half a conversation: they hold the conversation.

WHOSE (2026-09-25). The person at the terminal is the agent's owner, and their
conversations are the folder's own directory. A SERVED agent's session skill
(``agents/skills/local/session/skill.py``) keeps every other verified caller's
under ``callers/<hash of the caller>/`` beside them, one namespace per caller,
so no caller's conversations are another's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def slug_for(text: str) -> str:
    """A path or a name as one directory name."""
    return _UNSAFE.sub("-", text)


def sessions_dir(folder: Path, agent_name: str, profile: Optional[str] = None) -> Path:
    """Where this folder's conversations with ``agent_name`` are kept."""
    from .config_store import global_dir, profile_name

    return global_dir(profile_name(profile)) / "sessions" / slug_for(str(Path(folder).resolve())) / slug_for(agent_name)


def caller_key(principal: str) -> str:
    """One caller's directory name: the first 32 hex digits of the SHA-256 of their principal (``user:...``)."""
    return hashlib.sha256(principal.encode("utf-8")).hexdigest()[:32]


def caller_sessions_dir(folder: Path, agent_name: str, principal: str, profile: Optional[str] = None) -> Path:
    """Where a served agent keeps one verified caller's conversations (module docstring, WHOSE)."""
    return sessions_dir(folder, agent_name, profile) / "callers" / caller_key(principal)


def new_session_id() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_session(directory: Path, session: Dict[str, Any]) -> None:
    """Write the session and point ``.latest`` at it."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = dict(session)
    now = _now()
    data["updated_at"] = now
    data["created_at"] = data.get("created_at") or now
    path = directory / f"{slug_for(str(data['session_id']))}.json"
    _write_private(path, json.dumps(data, indent=2) + "\n")
    _write_private(directory / ".latest", str(data["session_id"]))


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, owner-only, whole or not at all."""
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.replace(temp, path)


def mark_recorded(directory: Path, session_id: str, chat_id: str, count: int) -> None:
    """Note on a saved conversation that its first ``count`` messages are
    recorded into the platform chat ``chat_id``, leaving everything else as it
    was (its ``updated_at`` and ``.latest`` included). For a turn recorded
    after the person already moved on to another conversation."""
    session = load_session(directory, session_id)
    if session is None:
        return
    session["metadata"] = {**session["metadata"], "robutler_chat_id": chat_id, "robutler_recorded": count}
    _write_private(directory / f"{slug_for(session_id)}.json", json.dumps(session, indent=2) + "\n")


def load_session(directory: Path, session_id: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((directory / f"{slug_for(session_id)}.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return None
    return {
        "session_id": data.get("session_id", session_id),
        "agent_name": data.get("agent_name", ""),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "messages": data["messages"],
        "metadata": data.get("metadata") or {},
        "input_tokens": data.get("input_tokens") or 0,
        "output_tokens": data.get("output_tokens") or 0,
    }


def session_preview(messages: List[Dict[str, Any]]) -> str:
    """The first thing the person said, as one line."""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") == "user" and isinstance(content, str) and content.strip():
            return " ".join(content.split())
    return ""


@dataclass
class SessionSummary:
    id: str
    updated_at: str
    message_count: int
    preview: str
    #: The platform chat it is recorded into, when it is (``robutler_sessions.py``).
    chat_id: Optional[str] = None


def list_sessions(directory: Path) -> List[SessionSummary]:
    """This folder's conversations with the agent, newest first."""
    try:
        names = [p.name for p in directory.iterdir() if p.suffix == ".json"]
    except OSError:
        return []
    out: List[SessionSummary] = []
    for name in names:
        session = load_session(directory, name[: -len(".json")])
        if not session or not any(m.get("role") == "user" for m in session["messages"] if isinstance(m, dict)):
            continue
        chat_id = session["metadata"].get("robutler_chat_id")
        out.append(
            SessionSummary(
                id=session["session_id"],
                updated_at=session["updated_at"],
                message_count=len(session["messages"]),
                preview=session_preview(session["messages"]),
                chat_id=chat_id if isinstance(chat_id, str) and chat_id else None,
            )
        )
    out.sort(key=lambda s: s.updated_at, reverse=True)
    return out


def when_label(iso: str, now: Optional[datetime] = None) -> str:
    """"just now", "5 min ago", "3 h ago", "2 days ago", or the date."""
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    seconds = max(0.0, (current - then).total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 7 * 86400:
        days = int(seconds // 86400)
        return "yesterday" if days == 1 else f"{days} days ago"
    return iso[:10]
