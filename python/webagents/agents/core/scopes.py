"""
Who may use a tool, prompt, command or endpoint declared with a scope.

ONE RULE, BOTH SDKs (ADR-0045 section 5, 2026-09-25). Until then each site
carried its own copy of `{"admin": 3, "owner": 2, "all": 1}`, and an unknown
scope took the `all` level, so a tool declared `scope="group:friends"` (or a
typo) was shown to every caller. The TypeScript agent required membership for
tools but showed an unknown PROMPT scope to everyone, and did not let an admin
through an `owner` scope while Python did. One function now decides, with the
same table of cases run by both SDKs
(`tests/fixtures/scopes/scope_allows.json`):

  - `all`, `None` or an empty list: anyone.
  - `user`: a verified caller (`user`, `owner` or `admin`).
  - `owner`: the owner, or an admin. `admin`: an admin only.
  - `group:<name>`: a member of that group, or the owner, or an admin.
  - anything else: only a caller holding exactly that scope. Unknown scopes
    fail closed.
  - a list: any one of its entries.

WHERE A CALLER'S SCOPES COME FROM, and where they must not. `caller_scopes`
reads the auth object's tier (`scope`) and the groups the access check put
there (`groups`), never a token's own `scopes` list: the local AOAuth context
carries whatever `scope` claim an issuer the agent trusts chose to mint, and
reading it here would let that issuer name itself `owner`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

GROUP_PREFIX = "group:"

Required = Union[None, str, Iterable[str]]


def caller_scopes(auth: Any) -> frozenset:
    """The scopes of the caller an auth object describes; empty for anonymous."""
    if auth is None:
        return frozenset()
    found = set()
    scope = getattr(auth, "scope", None)
    value = getattr(scope, "value", scope)
    if isinstance(value, str) and value:
        found.add(value)
    groups = getattr(auth, "groups", None)
    if isinstance(groups, (list, tuple, set, frozenset)):
        found.update(f"{GROUP_PREFIX}{name}" for name in groups if isinstance(name, str) and name)
    return frozenset(found)


def _one_allows(required: str, caller: frozenset) -> bool:
    if required == "all":
        return True
    if required == "admin":
        return "admin" in caller
    if required == "owner":
        return "owner" in caller or "admin" in caller
    if required == "user":
        return bool(caller & {"user", "owner", "admin"})
    if required.startswith(GROUP_PREFIX):
        return required in caller or "owner" in caller or "admin" in caller
    return required in caller


def scope_allows(required: Required, caller: Optional[Iterable[str]]) -> bool:
    """Whether a caller holding `caller` scopes may use something declared `required`."""
    held = frozenset(caller or ())
    if required is None:
        return True
    if isinstance(required, str):
        return _one_allows(required, held)
    entries = list(required)
    if not entries:
        return True
    return any(isinstance(entry, str) and _one_allows(entry, held) for entry in entries)
