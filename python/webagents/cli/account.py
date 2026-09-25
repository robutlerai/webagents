"""
Who you are on Robutler, and which agent this folder publishes to
(2026-09-24): `whoami`, `link` and `unlink`, in the words the TypeScript CLI
uses (`typescript/src/cli/account.ts`), so the two CLIs answer alike.

`link` binds the folder to an agent you already have, so `publish` updates it
rather than creating a second one: a platform username is minted once, and a
duplicate cannot be taken back. The binding is `link.agentId` and
`link.agentName` in the folder's own `.webagents/config.json`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config_store import cli_command

LINK_ID_KEY = "link.agentId"
LINK_NAME_KEY = "link.agentName"


def _host(url: str) -> str:
    return re.sub(r"^https?://", "", url)


@dataclass
class WhoAmI:
    ok: bool
    message: str
    code: str = ""
    fix: str = ""
    username: str = ""
    platform: str = ""


def who_am_i() -> WhoAmI:
    """The signed-in account, asked of the platform (`GET /api/users/me`)."""
    import httpx

    from .config_store import platform_url
    from .credentials import get_token

    portal = platform_url().rstrip("/")
    host = _host(portal)
    token = get_token()
    if not token:
        return WhoAmI(False, f"Not signed in to {host}.", "not_signed_in", f"Run `{cli_command('login')}`.")
    try:
        response = httpx.get(f"{portal}/api/users/me", headers={"Authorization": f"Bearer {token}"}, timeout=8)
    except httpx.HTTPError as error:
        return WhoAmI(
            False,
            f"Could not reach {host}: {error}",
            "unreachable",
            "Check the network, or `webagents config get platform.url`.",
        )
    if response.status_code == 401:
        return WhoAmI(False, f"Your sign-in on {host} has expired.", "expired", f"Run `{cli_command('login')}`.")
    if response.status_code >= 400:
        return WhoAmI(False, f"{host} answered {response.status_code}.", "http_error")
    try:
        data = response.json()
    except ValueError:
        data = {}
    user = data.get("user") if isinstance(data.get("user"), dict) else data
    username = str(user.get("username") or "")
    return WhoAmI(True, f"Signed in as @{username} on {host}.", username=username, platform=portal)


def project_link(folder: Path) -> Dict[str, str]:
    from .publish import project_link as _project_link

    return _project_link(folder)


def show_link(folder: Path, say: Callable[[str], None]) -> bool:
    """`link --show`: what this folder publishes to."""
    link = project_link(folder)
    if not link.get("agent_id"):
        say("This folder is not linked to an agent on Robutler.")
        say(f"`{cli_command('link <name>')}` links it to one of yours; `{cli_command('publish')}` creates one.")
        return True
    say(f"Linked to {link.get('agent_name') or link['agent_id']} ({link['agent_id']}).")
    return True


def link_folder(folder: Path, name: Optional[str], say: Callable[[str], None], error: Callable[[str], None]) -> bool:
    """`link [name]`: bind this folder to the agent of yours called `name` (the
    local agent file's name when omitted). Matched on the username first
    (`me.helper`, or its `helper` part), then on the display name."""
    import httpx

    from .config_store import ConfigStore, platform_url
    from .credentials import get_token

    token = get_token()
    portal = platform_url().rstrip("/")
    if not token:
        error(f"Not signed in. Run `{cli_command('login')}` first.")
        return False
    wanted = (name or "").strip()
    if not wanted:
        from .loader import AgentFormatError
        from .loader.hierarchy import find_default_agent

        try:
            merged = find_default_agent(folder)
        except AgentFormatError as problem:
            error(str(problem))
            return False
        if merged is None:
            error(f"No agent file here, so no name to look for. Pass one: `{cli_command('link <name>')}`.")
            return False
        wanted = merged.metadata.name or merged.name

    try:
        response = httpx.get(f"{portal}/api/agents", headers={"Authorization": f"Bearer {token}"}, timeout=15)
    except httpx.HTTPError as problem:
        error(f"Could not list your agents: {problem}")
        return False
    if response.status_code >= 400:
        error(f"Could not list your agents: {response.status_code}.")
        error(f"Check `{cli_command('whoami')}`, then `{cli_command('login')}`.")
        return False
    agents: List[Dict[str, Any]] = (response.json() or {}).get("agents") or []

    def by_username(agent: Dict[str, Any]) -> bool:
        username = str(agent.get("username") or "")
        return username == wanted or username.endswith(f".{wanted}")

    match = next((a for a in agents if by_username(a)), None) or next(
        (a for a in agents if (a.get("displayName") or a.get("name")) == wanted), None
    )
    if not match or not match.get("id"):
        error(f"None of your agents is called {wanted}.")
        if agents:
            error("You have:")
            for agent in agents[:10]:
                error(f"  {agent.get('username') or agent.get('displayName') or agent.get('id')}")
        error(f"`{cli_command('publish')}` creates it.")
        return False
    username = str(match.get("username") or wanted)
    store = ConfigStore(cwd=folder)
    store.set(LINK_ID_KEY, str(match["id"]), scope="project")
    store.set(LINK_NAME_KEY, username, scope="project")
    say(f"Linked this folder to {username}.")
    return True


def unlink_folder(folder: Path, say: Callable[[str], None]) -> None:
    """`unlink`: forget the binding; the agent itself is untouched."""
    from .config_store import ConfigStore

    link = project_link(folder)
    if not link.get("agent_id"):
        say("This folder is not linked.")
        return
    store = ConfigStore(cwd=folder)
    store.unset(LINK_ID_KEY, scope="project")
    store.unset(LINK_NAME_KEY, scope="project")
    say(f"Unlinked from {link.get('agent_name') or link['agent_id']}.")


def validate_key(portal: str, key: str) -> Dict[str, Any]:
    """Exchange an API key for a platform token, with the TypeScript CLI's
    reasons (`typescript/src/cli/auth-check.ts`): `{"ok": True, "token", "username"}`
    or `{"ok": False, "reason"}`. "Could not ask" and "the answer was no" are
    kept apart, and the portal's own response text is not repeated."""
    import httpx

    url = f"{portal.rstrip('/')}/api/auth/cli/token"
    try:
        response = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, json={}, timeout=15)
    except httpx.TimeoutException:
        return {"ok": False, "reason": f"no response from {portal} within 15s"}
    except httpx.HTTPError as error:
        return {"ok": False, "reason": f"could not reach {portal} ({error})"}
    if response.status_code in (401, 403):
        return {"ok": False, "reason": "the portal rejected that key"}
    if response.status_code >= 400:
        return {"ok": False, "reason": f"the portal answered {response.status_code}"}
    try:
        body = response.json()
    except ValueError:
        body = {}
    return {"ok": True, "token": body.get("access_token"), "username": body.get("username"), "user_id": body.get("user_id")}


def login_command(url: Optional[str], token: Optional[str]) -> int:
    """`webagents login`: the browser at a terminal, else a key (`--token`, or
    pasted). The TypeScript `login`'s steps and words (`typescript/src/cli/index.ts`)."""
    import asyncio
    import sys
    from datetime import datetime, timedelta

    from .config_store import PLATFORM_URL_ENV_VAR, ConfigStore, resolve_platform_url
    from .credentials import backend_status
    from .state.local import get_state

    explicit = url.strip().rstrip("/") if url else None
    resolved, resolved_from = resolve_platform_url()
    portal = explicit or resolved
    username = ""

    if not token and sys.stdin.isatty():
        from .platform.auth import login as browser_login

        try:
            result = asyncio.run(browser_login(say=print, base=portal))
        except Exception as error:  # noqa: BLE001 - one line and exit 1
            print(f"Could not sign in: {error}", file=sys.stderr)
            return 1
        username = str(result.get("username") or "")
    else:
        if not token:
            # `/settings/api-keys` is not a page in the portal; keys are issued
            # on the Developer tab of Settings.
            print(f"\nCreate an API key at: {portal}/settings?tab=developer\n")
            print("Paste your API key: ", end="", flush=True)
            token = sys.stdin.readline()
        token = (token or "").strip()
        if not token:
            print("No key entered.", file=sys.stderr)
            return 1
        checked = validate_key(portal, token)
        if not checked["ok"]:
            print(f"Could not authenticate: {checked['reason']}", file=sys.stderr)
            return 1
        username = str(checked.get("username") or "")
        # The EXCHANGED token, scoped and expiring in 7 days, not the pasted key.
        get_state().set_credentials(
            access_token=checked.get("token") or token,
            auth_type="jwt",
            username=username,
            user_id=checked.get("user_id") or "",
            expires_at=(datetime.utcnow() + timedelta(days=7)).isoformat(),
            authenticated_at=datetime.utcnow().isoformat(),
        )

    # The token and the portal stay together: `--url` means "sign in to THAT
    # portal", so later commands have to look there too.
    if explicit and explicit != resolved:
        if resolved_from == PLATFORM_URL_ENV_VAR:
            print(
                f"Note: {PLATFORM_URL_ENV_VAR}={resolved} is set and later commands will use it, "
                f"not {explicit}. Unset it, or set it to {explicit}."
            )
        else:
            written = ConfigStore().set("platform.url", explicit)
            print(f"platform.url set to {explicit} ({written}).")

    print(f"Authenticated{f' as @{username}' if username else ''} on {portal}.")
    keystore = backend_status().get("backend") == "keystore"
    print("Token stored in your OS keystore." if keystore else "No OS keystore here; token stored in an owner-only file (0600).")
    return 0


def logout_command() -> None:
    """`webagents logout`: the stored token goes, from the keystore and the file."""
    from .config_store import platform_url
    from .state.local import get_state

    get_state().clear_credentials()
    print(f"Signed out of {_host(platform_url().rstrip('/'))}.")
