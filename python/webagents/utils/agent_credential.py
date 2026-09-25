"""
The agent's own platform credential, found rather than configured (2026-09-24).

WHY. One credential had four names, and a developer had to copy it by hand:
`webagents deploy` stored the agent's key in the keystore, and then the docs
told them to read it back out and export it as `WEBAGENTS_AGENT_TOKEN` (the
bridge, the heartbeat), while most platform skills read `WEBAGENTS_API_KEY`,
payments read a `robutler_api_key` config key with no environment fallback at
all, and the TypeScript payments skill read `ROBUTLER_API_KEY`, the name
registration uses for the OWNER's key. The CLI already knows everything needed
to find it: the directory is linked to its platform agent (`link.agentName` in
`./.webagents/config.json`) and the key is stored under that name.

THE ORDER, and why each step is where it is:
  1. what the code passes explicitly;
  2. `WEBAGENTS_AGENT_TOKEN`, then the older `WEBAGENTS_API_KEY`: the
     environment is how containers and CI receive secrets, so it stays the
     override there;
  3. the key `deploy` (or the TypeScript `publish`) stored, when this
     directory is linked to THIS agent. The link names one agent; a second
     agent file in the same directory must not authenticate as the first, so
     the linked name has to match.
Nothing here signs. An agent the platform can reach at a public https URL
needs no credential at all: the bridge and the platform clients sign with the
agent's own key (identity) when there is no token, and that stays the
preferred path where it is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

#: The one environment name for the agent's own credential.
AGENT_TOKEN_ENV = "WEBAGENTS_AGENT_TOKEN"
#: Older names for the same credential, still read.
LEGACY_AGENT_TOKEN_ENV = ("WEBAGENTS_API_KEY",)


def agent_key_name(platform_name: str) -> str:
    """The keystore name `deploy`/`publish` store an agent's key under."""
    return f"AGENT_KEY_{platform_name.upper().replace('.', '_').replace('-', '_')}"


def linked_agent_name(cwd: Optional[Path] = None) -> Optional[str]:
    """`link.agentName` from `./.webagents/config.json`, the project file only."""
    path = Path(cwd or Path.cwd()) / ".webagents" / "config.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    name = data.get("link.agentName") if isinstance(data, dict) else None
    return str(name) if name else None


def link_matches(linked: str, agent_name: Optional[str]) -> bool:
    """Whether the directory's link names THIS agent (`alice.helper` matches `helper`)."""
    if not agent_name:
        return False
    return linked == agent_name or linked.endswith("." + agent_name)


def _stored_key(key_name: str) -> Optional[str]:
    try:
        from ..agents.skills.local.secrets.store import open_secret_store
        from ..cli.config_store import global_dir, profile_name, scoped_namespace

        profile = profile_name()
        store = open_secret_store(
            namespace=scoped_namespace("providers", profile),
            secrets_dir=str(global_dir(profile) / "secrets"),
            quiet=True,
        )
        return store.get(key_name)
    except Exception:
        # A keystore that cannot be read is the same as no stored key here:
        # the caller reports the missing credential with its own fix.
        return None


def resolve_agent_credential(
    agent_name: Optional[str],
    explicit: Optional[str] = None,
    cwd: Optional[Path] = None,
    include_legacy: bool = True,
) -> Optional[Tuple[str, str]]:
    """`(credential, source)` for the agent, or None. See the module docstring for the order.

    `include_legacy=False` is for callers that need an AGENT-BOUND key (the
    bridge, the heartbeat): `WEBAGENTS_API_KEY` has long been set to an
    owner's key for the platform skills, and the bridge refuses those, so
    reading it there would stop an agent that signs today from starting.
    """
    if explicit:
        return explicit, "explicit"
    env_names = (AGENT_TOKEN_ENV, *LEGACY_AGENT_TOKEN_ENV) if include_legacy else (AGENT_TOKEN_ENV,)
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value, f"env:{name}"
    linked = linked_agent_name(cwd)
    if linked and link_matches(linked, agent_name):
        key_name = agent_key_name(linked)
        value = _stored_key(key_name)
        if value:
            return value, f"keystore:{key_name}"
    return None
