"""
The signing identity a served agent holds, as a skill reads it.

`WebAgentsServer` loads (or creates) each static agent's Ed25519 key and composes
the agent URL its key set is published under; since 2026-09-25 it also hands the
pair to the agent as `agent.signing_identity`, the way the TypeScript `serve()`
sets `agent.identity` (ADR-0045). A skill that signs for the agent (the REST
tool) reads it there instead of composing a URL and loading a key of its own,
which could name a key set nobody serves.
"""

from __future__ import annotations

from typing import Any, List


class AgentSigningIdentity:
    """`issuer` is the agent URL signatures name; `held_keys()` the keys to sign with."""

    def __init__(self, issuer: str, manager: Any):
        self.issuer = issuer
        self._manager = manager

    def held_keys(self) -> List[Any]:
        return self._manager.held_ed25519_keys()
