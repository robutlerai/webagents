"""
The session skill (2026-09-25): an agent keeps its conversations.

ONE MEANING IN BOTH SDKS. In an agent file::

    skills:
      - session                        # conversations kept on this machine
      - session: {backend: robutler}   # ... and yours on Robutler too, as chats

* IN THE CHAT (and ``-p``) the chat keeps the conversation itself, on this
  machine (``cli/sessions.py``), and with ``backend: robutler`` also as your
  chat with the agent on Robutler (``cli/robutler_sessions.py``). This skill is
  never loaded there (``build_agent`` leaves it out): it would be a second
  writer.
* SERVED (``serve``, the daemon), this skill keeps each VERIFIED caller's
  conversation, when the request names it with ``metadata.session_id``: the
  owner's where the chat keeps theirs, so ``/resume`` finds them; anyone
  else's under ``callers/<hash>/``, one namespace per caller. Anonymous callers
  are not kept, and a session id is only ever looked up inside its caller's
  namespace, so naming someone else's id finds nothing of theirs. A request
  carries the whole conversation, as an OpenAI-style client sends it, and what
  is kept is that conversation (the person's and the agent's words) and the
  reply (``conversation_to_keep``). ``backend: robutler`` changes nothing here:
  conversations that come through Robutler are chats there already, and a
  served agent does not write chats in anyone's name.

WHAT THIS REPLACED. ``SessionManagerSkill`` recorded every caller's turns into
ONE shared session (``.latest``) in the project folder, with default file
permissions, and its ``/sessions`` HTTP routes listed, read, created and
deleted them for anyone who could reach the port, no credential needed (S-249).
The TypeScript skill of the same name was a key-value scratchpad keyed on a
chat id the caller chose (S-262). Both are this now:
``typescript/src/skills/session/skill.ts``, pinned with this one by
``tests/fixtures/sessions/sessions.json``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...base import Skill
from webagents.agents.tools.decorators import hook

BACKENDS = ("local", "robutler")

_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_PRINCIPAL = re.compile(r"^(user|agent|key):.")


def session_backend_of(value: Any) -> str:
    """The backend an entry names; raises the sentence a person can act on."""
    if value is None or value == "local":
        return "local"
    if value == "robutler":
        return "robutler"
    import json

    raise ValueError(f'session: backend must be "local" or "robutler", not {json.dumps(value)}.')


def request_session_id(metadata: Any) -> Optional[str]:
    """The session a request names (``metadata.session_id``), when it is one that can name a file."""
    value = metadata.get("session_id") if isinstance(metadata, dict) else None
    if isinstance(value, str) and _SESSION_ID.match(value) and value not in (".", ".."):
        return value
    return None


def conversation_owner(auth: Any) -> Optional[str]:
    """Whose conversation a caller's is: ``owner`` for the agent's owner, else the
    first identity something verified (``user:``, ``agent:``, ``key:``), else
    None: an anonymous caller's is not kept."""
    from webagents.agents.skills.local.access.skill import _tier, _user_principals

    if auth is None or getattr(auth, "authenticated", True) is False:
        return None
    if _tier(auth) == "owner":
        return "owner"
    listed = getattr(auth, "principals", None)
    if isinstance(listed, (list, tuple)):
        verified = [p for p in listed if isinstance(p, str) and _PRINCIPAL.match(p)]
    else:
        verified = _user_principals(auth)
    return verified[0] if verified else None


def conversation_to_keep(request: Sequence[Any], reply: Any) -> List[Dict[str, str]]:
    """What is kept of a turn: the request's own words (the person's and the
    agent's, text only) and the reply."""
    words: List[Dict[str, str]] = []
    for m in request:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            words.append({"role": m["role"], "content": content})
    if isinstance(reply, str) and reply.strip():
        words.append({"role": "assistant", "content": reply})
    return words


#: Where the request's own messages wait, from `on_connection` to the end of the turn.
_REQUEST_KEY = "_session_request_messages"


class SessionSkill(Skill):
    """Keeps each verified caller's conversation, when a served request names it (module docstring)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config, scope="all")
        cfg = config or {}
        self.backend = session_backend_of(cfg.get("backend"))
        # `agent_builder.load_skills` passes the agent file's folder as
        # `agent_path`, and the agent's name.
        folder = cfg.get("agent_path") or cfg.get("agent_dir")
        self.agent_dir: Optional[Path] = Path(folder) if folder else None
        self.agent_name: Optional[str] = cfg.get("agent_name")

    @hook("on_connection", priority=95)
    async def remember_request(self, context: Any) -> Any:
        """The conversation as the request brought it, before the turn adds to it."""
        messages = getattr(context, "messages", None) or []
        try:
            context.set(_REQUEST_KEY, [dict(m) for m in messages if isinstance(m, dict)])
        except Exception:  # noqa: BLE001 - a context that cannot hold it keeps nothing
            pass
        return context

    @hook("finalize_connection", priority=90)
    async def keep_conversation(self, context: Any) -> Any:
        """After a served turn: keep the caller's conversation, when the request names it (module docstring)."""
        session_id = request_session_id(getattr(context, "metadata", None))
        if not session_id:
            return context
        whose = conversation_owner(getattr(context, "auth", None))
        if not whose:
            return context
        try:
            request = context.get(_REQUEST_KEY) or []
        except Exception:  # noqa: BLE001
            request = []
        final = getattr(context, "messages", None) or []
        reply = next(
            (
                m.get("content")
                for m in reversed(final)
                if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip()
            ),
            None,
        )
        messages = conversation_to_keep(request, reply)
        if not messages:
            return context

        from webagents.cli.sessions import caller_sessions_dir, load_session, save_session, sessions_dir

        folder = self.agent_dir or Path.cwd()
        agent = getattr(self, "agent", None)
        name = self.agent_name or getattr(agent, "name", None) or "agent"
        directory = sessions_dir(folder, name) if whose == "owner" else caller_sessions_dir(folder, name, whose)
        kept = load_session(directory, session_id)
        try:
            save_session(
                directory,
                {
                    "session_id": session_id,
                    "agent_name": name,
                    "created_at": (kept or {}).get("created_at", ""),
                    "updated_at": "",
                    "messages": messages,
                    "metadata": {**((kept or {}).get("metadata") or {}), "sdk": "python"},
                    "input_tokens": (kept or {}).get("input_tokens", 0),
                    "output_tokens": (kept or {}).get("output_tokens", 0),
                },
            )
        except OSError:
            pass  # A conversation that cannot be kept is still answered.
        return context
