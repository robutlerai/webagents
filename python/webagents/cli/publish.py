"""
Sending the agent to the platform: the chat's `/publish` (2026-09-24), the
same steps and words as the TypeScript `publish` (`typescript/src/cli/publish.ts`).

Creates the agent (`POST /api/agents`) and links the folder to it, or updates
the linked one (`PATCH /api/agents/{id}`), with the link keys (`link.agentId`,
`link.agentName` in the folder's `.webagents/config.json`) both CLIs use.

Two rules the old deploy broke:

 - THE LINK COMES FROM THE PROJECT ONLY. It was read through every config
   layer, global included, so one `config set link.agentId X` (global by
   default) made every unlinked folder on the machine update agent X.
 - AN UPDATE NEVER SENDS `name`. The portal renames an agent whose display name
   differs from the one sent, and a platform username is minted once: after a
   rename in the portal, the next deploy moved `me.shipper` to `me.shipper-2`
   for good and retired the old handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from .config_store import cli_command

LINK_ID_KEY = "link.agentId"
LINK_NAME_KEY = "link.agentName"


@dataclass
class PublishIO:
    #: What happened, first: "Published me.helper (id)." or "Updated me.helper."
    ok: Callable[[str], None]
    print: Callable[[str], None]
    error: Callable[[str], None]
    #: Asked before creating a new agent; False when nobody can answer.
    confirm: Callable[[str], Awaitable[bool]]


@dataclass
class PublishResult:
    ok: bool
    username: Optional[str] = None
    agent_id: Optional[str] = None
    created: bool = False


def project_link(project_root: Path) -> Dict[str, str]:
    """`link.agentId` / `link.agentName` from the folder's own config, never another layer."""
    from .config_store import ConfigStore

    layers = dict(ConfigStore(cwd=project_root).layers())
    project = layers.get("project") or {}
    out: Dict[str, str] = {}
    for key, name in ((LINK_ID_KEY, "agent_id"), (LINK_NAME_KEY, "agent_name")):
        value = project.get(key)
        if isinstance(value, str) and value:
            out[name] = value
    return out


def platform_payload(merged: Any) -> Dict[str, Any]:
    """A merged AGENT.md as `POST /api/agents` takes it.

    `skills:` is the one shape that does not carry across: the file writes a
    LIST (`- memory`, `- mcp: {...}`) and the portal's schema is
    `z.record(z.unknown())`, a mapping. A bare name becomes an empty config so
    the two forms land the same way.
    """
    skills: Dict[str, Any] = {}
    for entry in merged.metadata.skills or []:
        if isinstance(entry, str):
            skills[entry] = {}
        elif isinstance(entry, dict) and entry:
            name = next(iter(entry))
            skills[str(name)] = entry[name]

    payload: Dict[str, Any] = {"name": merged.metadata.name}
    # Only when the file has one, as the TypeScript payload (`agent-project.ts`).
    if merged.metadata.description:
        payload["description"] = merged.metadata.description
    payload["instructions"] = merged.instructions
    if merged.metadata.model:
        payload["model"] = merged.metadata.model
    if merged.metadata.intents:
        payload["intents"] = list(merged.metadata.intents)
    if skills:
        payload["skills"] = skills
    return payload


def _stored_where(backend: str) -> str:
    return "your keychain" if backend == "keystore" else "an owner-only file"


async def publish_agent(agent_path: Path, io: PublishIO, yes: bool = False, dry_run: bool = False) -> PublishResult:
    """`agent_path` is the agent file, or the folder holding it."""
    import httpx
    from urllib.parse import quote

    from .agent_files import default_agent_file
    from .config_store import ConfigStore, platform_url
    from .credentials import get_token
    from .loader import AgentFormatError
    from .loader.hierarchy import load_agent

    portal = platform_url().rstrip("/")
    agent_file = agent_path if agent_path.is_file() else default_agent_file(agent_path)
    if agent_file is None:
        io.error(f"No AGENT.md or agent.json at {agent_path.resolve()}. Create one with `webagents init`.")
        return PublishResult(ok=False)
    try:
        merged = load_agent(agent_file)
    except AgentFormatError as error:
        io.error(str(error))
        return PublishResult(ok=False)
    payload = platform_payload(merged)

    project_root = agent_file.parent.resolve()
    link = project_link(project_root)
    agent_id = link.get("agent_id")

    body = {key: value for key, value in payload.items() if value is not None}
    if agent_id:
        body.pop("name", None)
    method = "PATCH" if agent_id else "POST"
    url = f"{portal}/api/agents/{quote(agent_id, safe='')}" if agent_id else f"{portal}/api/agents"

    # Before the sign-in check: seeing what would be sent needs no account.
    if dry_run:
        import json

        io.print(f"Would {method} {url}:")
        io.print(json.dumps(body, indent=2))
        return PublishResult(ok=True, created=not agent_id)

    token = get_token()
    if not token:
        io.error(f"Not signed in. Run `{cli_command('login')}` (or /login in the chat) first.")
        return PublishResult(ok=False)

    if not agent_id:
        io.print("This folder is not linked to an agent on Robutler.")
        io.print(
            f"Publishing creates a new agent named {payload.get('name') or 'agent'}. "
            "Its username is set once and cannot be changed."
        )
        if not yes and not await io.confirm("Create it?"):
            io.error("Not created.")
            return PublishResult(ok=False)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(
                method,
                url,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as error:
        io.error(f"Could not reach {portal}: {error}")
        return PublishResult(ok=False)

    if response.status_code >= 400:
        detail = response.text.strip()
        io.error(f"Publish failed: {response.status_code}{' ' + detail if detail else ''}")
        if response.status_code == 404 and agent_id:
            io.error("The linked agent is gone. Remove `link.agentId` from .webagents/config.json to create a new one.")
        return PublishResult(ok=False)
    try:
        data = response.json()
    except ValueError:
        data = {}
    agent = data.get("agent") or {}

    if agent_id:
        username = agent.get("username") or data.get("username") or link.get("agent_name") or payload.get("name") or "agent"
        io.ok(f"Updated {username}.")
        return PublishResult(ok=True, username=username, agent_id=agent_id, created=False)

    # THE KEY IS NEVER PRINTED (S-223). `POST /api/agents` answers
    # `{agent, rawApiKey}` and returns that key once, so it goes to the store
    # under the name both CLIs use.
    username = agent.get("username") or payload.get("name") or "agent"
    new_id = str(agent["id"]) if agent.get("id") else None
    io.ok(f"Published {username}{f' ({new_id})' if new_id else ''}.")
    if new_id:
        store = ConfigStore(cwd=project_root)
        store.set(LINK_ID_KEY, new_id, scope="project")
        store.set(LINK_NAME_KEY, username, scope="project")
        io.print("This folder is now linked to it; the next publish updates it.")
    raw_key = data.get("rawApiKey")
    if raw_key:
        key_name = "AGENT_KEY_" + username.upper().replace(".", "_").replace("-", "_")
        try:
            from .commands.secrets import _store

            backend = _store(quiet=True).set(key_name, raw_key)
            io.print(f"Stored its API key as {key_name} ({_stored_where(backend)}). The platform returns it once.")
            io.print(f"Read it with `webagents secrets get {key_name} --show`.")
        except Exception as error:  # noqa: BLE001 - losing it silently is the one outcome to avoid
            io.error(f"Could not store the agent's API key: {error}")
            io.error("It was not printed and cannot be recovered; regenerate it in Settings.")
    return PublishResult(ok=True, username=username, agent_id=new_id, created=True)
