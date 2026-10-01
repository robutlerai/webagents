"""
The memory skill (gap-closure plan item 2.1, 2026-09-26): one skill, named
``memory`` in an agent file, the same in both SDKs::

    skills:
      - memory                                  # notes kept on this machine
      - memory: {local: true, portal: true}     # ... and on Robutler, synced
      - memory: {portal: true, local: false}    # on Robutler alone

WHAT IT GIVES THE MODEL: ``memory_search``, ``memory_read``, ``memory_write``,
``memory_forget`` and ``memory_list`` (``MEMORY_TOOL_DEFINITIONS``); an index of
its notes in the system prompt, one line each (key and description), frozen
for the session (``memory_notes.py``), from which ``memory_read`` gives a note
in full; and the summary of a compacted conversation kept as an episode
(``on_compaction``; compaction itself is the agent's,
``agents/core/context_compaction.py``, since 2026-09-29).

SCOPED BY CALLER, BY CONSTRUCTION (``memory_namespace.py``). Every entry lives
in a namespace derived from the VERIFIED caller of the turn: ``owner``, or
``caller:<principal>``. A tool call names no namespace it may not use: a
caller reads its own and ``shared``, writes its own; the owner reads and
writes everything. One caller's ``memory_search`` therefore never returns
another's entries, which ``tests/agents/skills/test_memory_isolation_w2mem.py``
pins, as ``typescript/tests/unit/skills/memory/memory-isolation-w2mem.test.ts``
does for the TypeScript twin (``typescript/src/skills/memory/skill.ts``).

TWO TIERS. ``local`` is Markdown files with a full-text index
(``local_memory_store.py``); ``portal`` is the platform's store with semantic
search (``portal_memory_store.py``). With both, reads are served locally,
every local change is pushed to the platform, and what changed on the
platform is pulled for the caller's namespaces at the start of a turn
(throttled), merged by entry id, last writer wins. The pull cursor is the
platform's opaque string, moved only by a pull (S-307, 2026-09-27: a push
used to answer a cursor past everything, and the agent never pulled again).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from ...base import Skill
from webagents.agents.tools.decorators import hook, prompt

from .local_memory_store import LocalMemoryStore, MemoryEntry
from webagents.agents.core.context_compaction import COMPACTION_RUN
from .memory_namespace import (
    OWNER_NAMESPACE,
    SHARED_NAMESPACE,
    is_valid_key,
    key_refusal,
    namespace_of,
    readable_namespaces,
    target_namespace,
)
from .memory_notes import DEFAULT_NOTES_BUDGET, NOTES_TITLES, render_notes
from .portal_memory_store import MemoryPortalError, PortalMemoryStore

logger = logging.getLogger("webagents.skills.memory")


#: The five tools, word for word the TypeScript skill's (``definitions.ts``),
#: pinned by ``tests/fixtures/memory_tool/definition.json``.
MEMORY_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search your memory: the notes you saved and the summaries of earlier conversations. Results come only "
                "from the memory of the caller you are talking to, the notes shared with every caller, and, when the "
                "caller is your owner, every namespace. Each result has key, namespace, description, content and updated_at."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in a few words."},
                    "namespace": {
                        "type": "string",
                        "description": "Owner only: search one namespace (owner, shared, or a caller namespace as memory_list names it).",
                    },
                    "limit": {"type": "integer", "description": "Most results to return (default 10, at most 50)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": (
                "Read one note in full, by the key your memory index gives it. Answers with key, namespace, "
                "description, content and updated_at."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The note's name."},
                    "namespace": {
                        "type": "string",
                        "description": "Owner only: owner, shared, or a caller namespace as memory_list names it.",
                    },
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Save a note to remember across conversations. key names the note (a short slug such as preferences or "
                "project-status: letters, digits, dots, dashes and underscores); writing an existing key replaces it. "
                "Give it a one-line description: that line is what your memory index shows in every conversation. The "
                "note is filed in the memory of the caller you are talking to. Only your owner may file under owner "
                "(the default in the owner's conversations) or shared (read by every caller)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The note's name, a slug such as preferences.",
                        "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
                    },
                    "content": {"type": "string", "description": "The note, in Markdown."},
                    "description": {
                        "type": "string",
                        "description": "One line saying what the note is for, shown in your memory index. Without it, the note's first line is shown.",
                    },
                    "namespace": {"type": "string", "enum": ["owner", "shared"], "description": "Owner only: owner (the default) or shared."},
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": (
                "Delete a note by key from the memory of the caller you are talking to. Your owner may name the "
                "namespace to delete from. Answers with how many notes were removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The note's name."},
                    "namespace": {
                        "type": "string",
                        "description": "Owner only: owner, shared, or a caller namespace as memory_list names it.",
                    },
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": (
                "List the notes in memory, most recently updated first: key, namespace, description and updated_at. "
                "Callers see their own notes and the shared ones; your owner sees every namespace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "Only keys starting with this."},
                    "namespace": {"type": "string", "description": "Owner only: list one namespace."},
                    "limit": {"type": "integer", "description": "Most notes to return (default 50, at most 200)."},
                },
                "required": [],
            },
        },
    },
]

CONFIG_KEYS = {"local", "portal", "notes_budget", "compaction"}
#: Keys the loader and the tests add to the entry; never the agent file's.
RUNTIME_KEYS = {
    "agent_name",
    "agent_path",
    "agent_dir",
    "agent_id",
    "api_key",
    "robutler_api_url",
    "webagents_api_url",
    "summarizer",
    "transport",
    "plain_index",
    "now",
}

NO_CALLER = "memory: nothing is remembered for a caller nothing verified; only shared notes can be read."
NOT_YOURS = "memory: only the owner may name that namespace."
PULL_INTERVAL_S = 60.0
# Pages one pull takes per namespace, at most: the platform's key limit (10,000) in its pages of 500.
MAX_PULL_PAGES = 20
MAX_FROZEN = 500
#: A run that writes a compaction summary (the agent's, `context_compaction.py`):
#: the notes and the pull stay out of it.
COMPACTION_FLAG = COMPACTION_RUN


def parse_memory_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``- memory: {...}`` entry, checked; every sentence is the fixture's."""
    config = dict(raw or {})
    for key in config:
        if key not in CONFIG_KEYS and key not in RUNTIME_KEYS:
            raise ValueError(f'memory: unknown key "{key}". It takes local, portal, notes_budget and compaction.')

    def flag(name: str, fallback: bool) -> bool:
        value = config.get(name)
        if value is None:
            return fallback
        if not isinstance(value, bool):
            raise ValueError(f"memory: {name} must be true or false.")
        return value

    local = flag("local", True)
    portal = flag("portal", False)
    if not local and not portal:
        raise ValueError("memory: at least one of local and portal must be true.")
    notes_budget = DEFAULT_NOTES_BUDGET
    if config.get("notes_budget") is not None:
        value = config["notes_budget"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("memory: notes_budget must be a positive number of characters.")
        notes_budget = int(value)
    # `compaction` here is the setting from before compaction was the agent's
    # (2026-09-29): still read, so a file that has it loads, and its
    # `threshold` becomes the agent's `compaction.at` when the agent file has
    # no `compaction:` block of its own (`agent_builder.py`).
    compaction: Dict[str, int] = {}
    if config.get("compaction") is not None:
        block = config["compaction"]
        if not isinstance(block, dict):
            raise ValueError("memory: compaction must be a mapping of threshold and keep.")
        if block.get("threshold") is not None:
            value = block["threshold"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("memory: compaction.threshold must be a positive number of tokens.")
            compaction["threshold"] = int(value)
        if block.get("keep") is not None:
            value = block["keep"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("memory: compaction.keep must be a number of messages.")
            compaction["keep"] = int(value)
    return {"local": local, "portal": portal, "notes_budget": notes_budget, "compaction": compaction}


def _stamp(now: Optional[Callable[[], datetime]]) -> str:
    moment = (now() if now else datetime.now(timezone.utc)).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H-%M-%S-") + f"{moment.microsecond // 1000:03d}Z"


class MemorySkill(Skill):
    """Notes and episodes, scoped by verified caller, kept locally and on the platform (module docstring)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {}, scope="all")
        cfg = dict(config or {})
        parsed = parse_memory_config(cfg)
        self.tiers = {"local": parsed["local"], "portal": parsed["portal"]}
        self.notes_budget: int = parsed["notes_budget"]
        self.compaction: Dict[str, int] = parsed["compaction"]
        folder = cfg.get("agent_dir") or cfg.get("agent_path")
        self.agent_dir: Path = Path(folder) if folder else Path.cwd()
        self.agent_name: Optional[str] = cfg.get("agent_name")
        self.agent_id: Optional[str] = cfg.get("agent_id")
        self._api_key: Optional[str] = cfg.get("api_key")
        self._portal_url_config = {k: cfg.get(k) for k in ("robutler_api_url", "webagents_api_url")}
        self._transport = cfg.get("transport")
        self._plain_index: bool = bool(cfg.get("plain_index", False))
        self._now: Optional[Callable[[], datetime]] = cfg.get("now")
        self._local: Optional[LocalMemoryStore] = None
        self._portal: Optional[PortalMemoryStore] = None
        self._ready = False
        self._frozen: Dict[str, str] = {}
        self._pulled_at: Dict[str, float] = {}
        #: Compactions this process ran, for the tests and ``doctor``.
        self.compactions = 0

    # -- setup ------------------------------------------------------------

    @property
    def store_key(self) -> str:
        """What entry ids are computed over: the platform id, else the name."""
        return self.agent_id or self.agent_name or getattr(self.agent, "name", None) or "agent"

    @property
    def memory_root(self) -> Path:
        return self.agent_dir / ".webagents" / "memory"

    async def initialize(self, agent: Any) -> None:
        await super().initialize(agent)
        for definition in MEMORY_TOOL_DEFINITIONS:
            self.register_tool(self._tool_for(definition), scope="all")
        await self.ensure_ready()

    def _tool_for(self, definition: Dict[str, Any]) -> Callable[..., Awaitable[Any]]:
        name = definition["function"]["name"]
        method = getattr(self, name)

        async def handler(**kwargs: Any) -> Any:
            return await method(**kwargs)

        handler.__name__ = name
        handler.__doc__ = definition["function"]["description"]
        handler._webagents_tool_definition = copy.deepcopy(definition)  # type: ignore[attr-defined]
        return handler

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        self._ready = True
        if self.tiers["local"]:
            self._local = LocalMemoryStore(self.memory_root, self.store_key, plain_index=self._plain_index, now=self._now)
            self._local.open()
        if self.tiers["portal"]:
            from webagents.agents.skills.robutler.platform_url import resolve_platform_url

            self._portal = PortalMemoryStore(
                resolve_platform_url({k: v for k, v in self._portal_url_config.items() if v}),
                token=self._resolve_token,
                agent_id=self.agent_id,
                transport=self._transport,
            )
            if self._local is not None:
                # Everything this machine knows about, pulled once at start;
                # the caller's namespaces are pulled again per turn.
                try:
                    await self.pull(self._local.namespaces())
                    await self.push()
                except MemoryPortalError as exc:
                    logger.warning("memory: first sync skipped: %s", exc)

    def _resolve_token(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        try:
            from webagents.utils.agent_credential import resolve_agent_credential

            found = resolve_agent_credential(self.agent_name or getattr(self.agent, "name", None), cwd=self.agent_dir)
            return found[0] if found else None
        except Exception:  # noqa: BLE001 - no credential is answered as none
            return None

    async def cleanup(self) -> None:
        if self._local is not None:
            self._local.close()

    # -- the caller ---------------------------------------------------------

    def _caller_namespace(self, context: Any = None) -> Optional[str]:
        if context is None:
            context = self.get_context()
        return namespace_of(getattr(context, "auth", None) if context is not None else None)

    # -- reads through the tiers ----------------------------------------------

    async def _list_entries(self, namespaces: Optional[Sequence[str]], **options: Any) -> List[MemoryEntry]:
        if self._local is not None:
            return self._local.list(namespaces, **options)
        assert self._portal is not None
        return await self._portal.list(namespaces, **options)

    async def _search_entries(self, query: str, namespaces: Optional[Sequence[str]], limit: int) -> List[MemoryEntry]:
        out: List[MemoryEntry] = []
        seen = set()

        def add(entries: Sequence[MemoryEntry]) -> None:
            for e in entries:
                if e.id in seen or (namespaces is not None and e.namespace not in namespaces):
                    continue
                seen.add(e.id)
                out.append(e)

        if self._local is not None:
            add(self._local.search(query, namespaces, limit))
        if self._portal is not None and len(out) < limit:
            try:
                add(await self._portal.search(query, namespaces, limit))
            except MemoryPortalError:
                if self._local is None:
                    raise
                # The platform's semantic leg is extra when the local index answered.
        return out[:limit]

    # -- tools ----------------------------------------------------------------------

    async def memory_search(self, query: Any = None, namespace: Any = None, limit: Any = None, **_: Any) -> Any:
        await self.ensure_ready()
        text = query.strip() if isinstance(query, str) else ""
        if not text:
            return {"error": "memory: query is required."}
        caller = self._caller_namespace()
        namespaces = readable_namespaces(caller)
        if isinstance(namespace, str) and namespace.strip():
            one = target_namespace(caller, namespace, "read")
            if one is None:
                return {"error": NOT_YOURS}
            namespaces = [one]
        count = min(max(1, int(limit or 10)), 50)
        try:
            entries = await self._search_entries(text, namespaces, count)
        except MemoryPortalError as exc:
            return {"error": str(exc)}
        return {
            "entries": [
                {"key": e.key, "namespace": e.namespace, "description": e.description, "content": e.content, "updated_at": e.updated_at}
                for e in entries
            ]
        }

    async def memory_read(self, key: Any = None, namespace: Any = None, **_: Any) -> Any:
        """One note in full (2026-09-29): what the memory index in the prompt names."""
        await self.ensure_ready()
        if not is_valid_key(key):
            return {"error": key_refusal(key)}
        caller = self._caller_namespace()
        if caller is None:
            return {"error": NO_CALLER}
        if isinstance(namespace, str) and namespace.strip():
            one = target_namespace(caller, namespace, "read")
            if one is None:
                return {"error": NOT_YOURS}
            namespaces = [one]
        else:
            namespaces = readable_namespaces(caller) or [OWNER_NAMESPACE, SHARED_NAMESPACE]
        try:
            for each in namespaces:
                entry = await self._get_entry(each, key)
                if entry is not None:
                    return {
                        "key": entry.key,
                        "namespace": entry.namespace,
                        "description": entry.description,
                        "content": entry.content,
                        "updated_at": entry.updated_at,
                    }
        except MemoryPortalError as exc:
            return {"error": str(exc)}
        return {"error": f"memory: no note called {key}."}

    async def memory_write(
        self, key: Any = None, content: Any = None, description: Any = None, namespace: Any = None, **_: Any
    ) -> Any:
        await self.ensure_ready()
        if not is_valid_key(key):
            return {"error": key_refusal(key)}
        if not isinstance(content, str):
            return {"error": "memory: content is required."}
        caller = self._caller_namespace()
        if caller is None:
            return {"error": NO_CALLER}
        # A non-owner's note goes into its own namespace whatever it asked for
        # (module docstring): the parameter is honoured for the owner alone.
        if caller == OWNER_NAMESPACE:
            target = target_namespace(caller, namespace, "write") or OWNER_NAMESPACE
            if isinstance(namespace, str) and namespace.strip() and target != namespace.strip():
                return {"error": "memory: namespace must be owner or shared."}
            if isinstance(namespace, str) and namespace.strip() and target not in (OWNER_NAMESPACE, SHARED_NAMESPACE):
                return {"error": "memory: namespace must be owner or shared."}
        else:
            target = caller
        try:
            entry = await self.write(target, key, content, "tool", description if isinstance(description, str) else "")
        except (MemoryPortalError, ValueError) as exc:
            return {"error": str(exc)}
        return {"ok": True, "id": entry.id, "key": entry.key, "namespace": entry.namespace, "updated_at": entry.updated_at}

    async def memory_forget(self, key: Any = None, namespace: Any = None, **_: Any) -> Any:
        await self.ensure_ready()
        if not is_valid_key(key):
            return {"error": key_refusal(key)}
        caller = self._caller_namespace()
        if caller is None:
            return {"error": NO_CALLER}
        target = target_namespace(caller, namespace, "write")
        if target is None:
            return {"error": NOT_YOURS}
        try:
            forgotten = await self.forget(target, key)
        except MemoryPortalError as exc:
            return {"error": str(exc)}
        return {"ok": True, "forgotten": 1 if forgotten else 0}

    async def memory_list(self, prefix: Any = None, namespace: Any = None, limit: Any = None, **_: Any) -> Any:
        await self.ensure_ready()
        caller = self._caller_namespace()
        namespaces = readable_namespaces(caller)
        if isinstance(namespace, str) and namespace.strip():
            one = target_namespace(caller, namespace, "read")
            if one is None:
                return {"error": NOT_YOURS}
            namespaces = [one]
        count = min(max(1, int(limit or 50)), 200)
        try:
            entries = await self._list_entries(namespaces, prefix=prefix if isinstance(prefix, str) else None, limit=count)
        except MemoryPortalError as exc:
            return {"error": str(exc)}
        return {"entries": [{"key": e.key, "namespace": e.namespace, "description": e.description, "updated_at": e.updated_at} for e in entries]}

    # -- writes through the tiers --------------------------------------------------

    async def _get_entry(self, namespace: str, key: str) -> Optional[MemoryEntry]:
        if self._local is not None:
            return self._local.get(namespace, key)
        assert self._portal is not None
        return await self._portal.get(namespace, key)

    async def write(self, namespace: str, key: str, content: str, source: str, description: str = "") -> MemoryEntry:
        """Write in the tiers that are on: the local file first, then the platform (pushed from the log)."""
        await self.ensure_ready()
        if self._local is not None:
            entry = self._local.put(namespace, key, content, source, description=description)
            if self._portal is not None:
                try:
                    await self.push()
                except MemoryPortalError as exc:
                    logger.warning("memory: push skipped: %s", exc)
            return entry
        assert self._portal is not None
        return await self._portal.put(namespace, key, content, source, description=" ".join(description.split())[:200])

    async def forget(self, namespace: str, key: str) -> bool:
        await self.ensure_ready()
        if self._local is not None:
            removed = self._local.forget(namespace, key)
            if self._portal is not None:
                try:
                    await self.push()
                except MemoryPortalError as exc:
                    logger.warning("memory: push skipped: %s", exc)
            return removed
        assert self._portal is not None
        return await self._portal.forget(namespace, key)

    # -- the owner's view (the chat's /memory, interactive-mode spec 3.7) ----------

    async def owner_summary(self) -> Dict[str, Any]:
        """What the agent remembers, for the person who owns it: where the
        notes are kept, how many are theirs, shared, or a caller's, and their
        own newest ten with each note's first line. The TypeScript skill
        answers the same shape (`skill.ts` `ownerSummary`)."""
        await self.ensure_ready()
        entries = await self._list_entries(None, limit=100_000)
        callers = set()
        owner = shared = caller_notes = 0
        for e in entries:
            if e.namespace == OWNER_NAMESPACE:
                owner += 1
            elif e.namespace == SHARED_NAMESPACE:
                shared += 1
            else:
                caller_notes += 1
                callers.add(e.namespace)
        recent = [
            {
                "key": e.key,
                # The note's description when it has one, as the index shows it (2026-09-29).
                "first_line": e.description or next((l.strip() for l in e.content.split("\n") if l.strip()), ""),
                "updated_at": e.updated_at,
            }
            for e in entries
            if e.namespace == OWNER_NAMESPACE
        ][:10]
        return {
            "local": self._local is not None,
            "portal": self._portal is not None,
            # Whether the Robutler tier has the agent's own key to reach it
            # with (B5, 2026-09-28): `/memory` said "and on Robutler" for a
            # tier that had none, so nothing ever reached Robutler.
            "portal_key": self._portal is not None and bool(self._resolve_token()),
            "owner": owner,
            "shared": shared,
            "callers": len(callers),
            "caller_notes": caller_notes,
            "recent": recent,
        }

    async def has_own_note(self, key: str) -> bool:
        """Whether the owner has a note called `key`."""
        await self.ensure_ready()
        if self._local is not None:
            return self._local.get(OWNER_NAMESPACE, key) is not None
        assert self._portal is not None
        return any(e.key == key for e in await self._portal.list([OWNER_NAMESPACE], prefix=key, limit=1000))

    async def forget_own(self, key: str) -> bool:
        """Remove one of the owner's notes (`/memory forget <key>`); False when there is none."""
        return await self.forget(OWNER_NAMESPACE, key)

    # -- sync -----------------------------------------------------------------------

    @property
    def _sync_state_file(self) -> Path:
        return self.memory_root / "sync-state.json"

    def _read_sync_state(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self._sync_state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"pushedSeq": 0, "cursors": {}}
        if not isinstance(raw, dict):
            return {"pushedSeq": 0, "cursors": {}}
        cursors = raw.get("cursors") if isinstance(raw.get("cursors"), dict) else {}
        return {
            "pushedSeq": raw.get("pushedSeq") if isinstance(raw.get("pushedSeq"), int) and not isinstance(raw.get("pushedSeq"), bool) else 0,
            # Only the platform's strings are kept. A number is the retired
            # sequence cursor (S-307: 9007199254740991 after the first push),
            # which would stop every pull; it is dropped, so that namespace's
            # next pull starts over, which the merge makes harmless.
            "cursors": {ns: c for ns, c in cursors.items() if isinstance(c, str) and c},
        }

    def _write_sync_state(self, state: Dict[str, Any]) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._sync_state_file.write_text(json.dumps(state) + "\n", encoding="utf-8")
        try:
            self._sync_state_file.chmod(0o600)
        except OSError:
            pass

    async def push(self) -> int:
        """Push every local change the platform has not seen, one request per namespace."""
        if self._local is None or self._portal is None:
            return 0
        state = self._read_sync_state()
        lines = self._local.log_since(state["pushedSeq"])
        if not lines:
            return 0
        by_namespace: Dict[str, List[Dict[str, Any]]] = {}
        for line in lines:
            by_namespace.setdefault(line["namespace"], []).append(line)
        pushed = 0
        for namespace, group in by_namespace.items():
            # No cursor comes back and none is kept: our own lines return on
            # the next pull with the times we sent, and the merge keeps what is here.
            await self._portal.push(namespace, group)
            pushed += len(group)
        state["pushedSeq"] = max([state["pushedSeq"]] + [l["seq"] for l in lines])
        self._write_sync_state(state)
        return pushed

    async def pull(self, namespaces: Sequence[str]) -> int:
        """Pull what changed on the platform in ``namespaces``, page by page, and apply what is newer here."""
        if self._local is None or self._portal is None:
            return 0
        state = self._read_sync_state()
        applied = 0
        for namespace in namespaces:
            since: Optional[str] = state["cursors"].get(namespace)
            for _page in range(MAX_PULL_PAGES):
                lines, cursor, more = await self._portal.pull(namespace, since)
                applied += self._local.apply(lines)
                since = cursor
                if not more:
                    break
            # The platform's answer replaces what was kept, None included: a
            # cursor it did not recognise is not kept for the next pull either.
            if since:
                state["cursors"][namespace] = since
            else:
                state["cursors"].pop(namespace, None)
            self._pulled_at[namespace] = time.monotonic()
            self._write_sync_state(state)
        return applied

    @hook("on_connection", priority=60)
    async def pull_for_caller(self, context: Any) -> Any:
        """At the start of a turn with both tiers: the caller's namespaces, at most once a minute each."""
        if self._local is None or self._portal is None:
            return context
        if context is not None and context.get(COMPACTION_FLAG):
            return context
        await self.ensure_ready()
        wanted = readable_namespaces(self._caller_namespace(context)) or [OWNER_NAMESPACE, SHARED_NAMESPACE]
        due = [ns for ns in wanted if time.monotonic() - self._pulled_at.get(ns, -PULL_INTERVAL_S * 2) > PULL_INTERVAL_S]
        if due:
            try:
                await self.pull(due)
            except MemoryPortalError as exc:
                logger.warning("memory: pull skipped: %s", exc)
        return context

    # -- the frozen notes -------------------------------------------------------------

    async def notes_sections_for(self, caller: Optional[str]) -> List[Dict[str, Any]]:
        """The sections a caller may see, in the order the prompt shows them."""

        async def section(title: str, namespace: str) -> Dict[str, Any]:
            entries = await self._list_entries([namespace], exclude_sources=("compaction",), limit=200)
            return {"title": title, "entries": [{"key": e.key, "description": e.description, "content": e.content} for e in entries]}

        if caller == OWNER_NAMESPACE:
            return [await section(NOTES_TITLES["owner"], OWNER_NAMESPACE), await section(NOTES_TITLES["shared"], SHARED_NAMESPACE)]
        sections = [await section(NOTES_TITLES["shared"], SHARED_NAMESPACE)]
        if caller:
            sections.append(await section(NOTES_TITLES["caller"], caller))
        return sections

    @prompt(priority=40)
    async def frozen_notes(self, context: Any = None) -> str:
        """The notes, rendered once per session (``metadata.session_id``, else
        once per caller for the life of this process) and never again
        mid-session, so the provider's prompt cache holds (plan principle 6)."""
        if context is not None and context.get(COMPACTION_FLAG):
            return ""
        await self.ensure_ready()
        caller = self._caller_namespace(context)
        from webagents.agents.skills.local.session.skill import request_session_id

        key = f"{caller or ''}|{request_session_id(getattr(context, 'metadata', None)) or ''}"
        if key in self._frozen:
            return self._frozen[key]
        try:
            text = render_notes(await self.notes_sections_for(caller), self.notes_budget)
        except MemoryPortalError:
            text = ""
        if len(self._frozen) >= MAX_FROZEN:
            self._frozen.pop(next(iter(self._frozen)))
        self._frozen[key] = text
        return text

    def unfreeze_notes(self) -> None:
        """For tests and the chat's ``/memory`` view: forget what was frozen."""
        self._frozen.clear()

    # -- compaction ---------------------------------------------------------------------
    #
    # Compaction is the agent's (2026-09-29, `agents/core/context_compaction.py`):
    # the conversation's owner compacts, once, and tells the skills. The hook
    # that did it here compacted the run's copy while the chat kept the whole
    # history, so every later turn paid for a new summary and saved another
    # episode.

    async def on_compaction(self, outcome: Any, context: Any) -> None:
        """A conversation was compacted: its summary is kept as an episode in the
        caller's memory, once, so it is still searchable later. Between turns
        in the chat there is no run, and the caller is the owner."""
        summary = getattr(outcome, "summary", None)
        if not summary:
            return
        caller = self._caller_namespace(context) if context is not None else OWNER_NAMESPACE
        if not caller:
            return
        await self.ensure_ready()
        try:
            await self.write(caller, f"episode-{_stamp(self._now)}", summary, "compaction")
        except (MemoryPortalError, ValueError) as exc:
            logger.warning("memory: episode not kept: %s", exc)
            return
        self.compactions += 1
