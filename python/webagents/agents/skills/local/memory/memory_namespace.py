"""
Whose memory a turn reads and writes (gap-closure plan item 2.1, 2026-09-26).

THE NAMESPACE COMES FROM THE VERIFIED CALLER, never from a parameter. The rule
is the session skill's ``conversation_owner`` (``local/session/skill.py``), so a
caller's memory and a caller's conversations are keyed on the same identity:
the agent's owner gets ``owner``; anyone else gets ``caller:<principal>`` for
the first identity something verified (``user:``, ``agent:``, ``key:``, and
``channel:`` for a channel sender); a caller nothing verified has no namespace
and reads only ``shared``. An admin is a caller like any other: admin is a
tier for tools, not a claim on the owner's notes.

``shared`` is written by the owner and read by everyone. A non-owner writes
only into its own namespace, whatever it asks for, so nothing a stranger says
can land in the owner's notes (plan principle 7).

The local tier keeps each namespace in a folder: ``owner``, ``shared``, and
``callers/<sha256(principal)[:32]>`` (the session skill's ``caller_key``,
pinned by both fixtures). The TypeScript twin is
``typescript/src/skills/memory/namespace.ts``; both run
``tests/fixtures/memory_tool/definition.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, List, Optional

OWNER_NAMESPACE = "owner"
SHARED_NAMESPACE = "shared"
CALLER_PREFIX = "caller:"

#: A namespace as a tool or the portal may name it. No whitespace and NO COMMA
#: in a caller principal (S-298, 2026-09-26): namespace lists used to travel to
#: the portal comma-joined, and a Web Bot Auth principal built from a
#: ``jwks_uri`` keeps a comma in its path, so a caller keyed at
#: ``https://evil.example/x,owner`` was read as two namespaces, the owner's
#: among them. Lists are repeated parameters now (``portal_memory_store.py``),
#: and the same grammar holds here, in the TypeScript twin and on the portal:
#: a principal that is not a namespace gets none (``namespace_of``).
NAMESPACE_RE = re.compile(r"^(owner|shared|caller:[^\s,]{1,300})$")

#: A key as ``memory_write`` accepts it: one path segment, no leading dot.
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_VERIFIED_PRINCIPAL = re.compile(r"^(user|agent|key|channel):.")


def key_refusal(key: Any) -> str:
    """The refusal for a key that is not a slug (the fixture's sentence)."""
    return (
        "memory: key must be a slug of letters, digits, dots, dashes and underscores (up to 128), not "
        f"{json.dumps(str(key))}."
    )


def is_valid_key(key: Any) -> bool:
    return isinstance(key, str) and bool(KEY_RE.match(key))


def namespace_of(auth: Any) -> Optional[str]:
    """The namespace of this turn's caller: ``owner``, ``caller:<principal>``,
    or None for a caller nothing verified (module docstring)."""
    from webagents.agents.skills.local.access.skill import _tier, _user_principals

    if auth is None or getattr(auth, "authenticated", True) is False:
        return None
    if _tier(auth) == "owner":
        return OWNER_NAMESPACE
    listed = getattr(auth, "principals", None)
    if isinstance(listed, (list, tuple)):
        verified = [p for p in listed if isinstance(p, str) and _VERIFIED_PRINCIPAL.match(p)]
    else:
        verified = _user_principals(auth)
    if not verified:
        return None
    # A principal that is not a namespace (whitespace, a comma) has none: such
    # a caller reads ``shared`` alone, as a caller nothing verified does (S-298).
    namespace = f"{CALLER_PREFIX}{verified[0]}"
    return namespace if NAMESPACE_RE.match(namespace) else None


def readable_namespaces(namespace: Optional[str]) -> Optional[List[str]]:
    """What a caller may read: everything for the owner (None: no filter), its
    own plus ``shared`` for a verified caller, ``shared`` alone for nobody."""
    if namespace == OWNER_NAMESPACE:
        return None
    return [namespace, SHARED_NAMESPACE] if namespace else [SHARED_NAMESPACE]


def writable_namespaces(namespace: Optional[str]) -> List[str]:
    """What a caller may write: ``owner`` and ``shared`` for the owner, its own for a caller, none for nobody."""
    if namespace == OWNER_NAMESPACE:
        return [OWNER_NAMESPACE, SHARED_NAMESPACE]
    return [namespace] if namespace else []


def target_namespace(caller_namespace: Optional[str], requested: Any, mode: str) -> Optional[str]:
    """The namespace a call acts on: the caller's own, or the one it names when
    it may (the owner naming any; ``shared`` for reads by anyone). None when
    the caller has none and named none it may use."""
    named = requested.strip() if isinstance(requested, str) and requested.strip() else None
    if not named:
        return caller_namespace
    if not NAMESPACE_RE.match(named):
        return None
    if caller_namespace == OWNER_NAMESPACE:
        return named
    allowed = writable_namespaces(caller_namespace) if mode == "write" else (readable_namespaces(caller_namespace) or [])
    return named if named in allowed else None


def caller_key(principal: str) -> str:
    """The session skill's caller key: the first 32 hex characters of the principal's SHA-256."""
    return hashlib.sha256(principal.encode("utf-8")).hexdigest()[:32]


def local_dir_of(namespace: str) -> str:
    """Where the local tier keeps a namespace, relative to the memory root."""
    if namespace in (OWNER_NAMESPACE, SHARED_NAMESPACE):
        return namespace
    if namespace.startswith(CALLER_PREFIX):
        return f"callers/{caller_key(namespace[len(CALLER_PREFIX):])}"
    raise ValueError(f"memory: not a namespace: {json.dumps(namespace)}")


def entry_id_for(store: str, namespace: str, key: str) -> str:
    """An entry's id in every tier: uuid5 in the URL namespace over the store
    (the agent's platform id, or its name when it has none), the namespace and
    the key, joined by newlines. The same key in the same namespace is the same
    entry everywhere, so a sync merge by id is a merge by key."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{store}\n{namespace}\n{key}"))
