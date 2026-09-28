"""
`webagents serve [path]` (2026-09-24): one agent over HTTP, the TypeScript
CLI's command (`typescript/src/cli/serve-action.ts`): the agent at `path` (an
AGENT.md, or the folder holding one), `-p/--port` (3000) and `--host`.

LOOPBACK UNLESS IT VERIFIES ITS CALLERS (S-226, S-327; the TypeScript rule
in `server/origin-policy.ts`): with no `--host`, it listens on 127.0.0.1
unless the agent has an AuthSkill, and says so. A public URL
(`WEBAGENTS_PUBLIC_URL`) names where the agent is reached and no longer opens
it to every interface by itself (2026-09-28): it did so even for a loopback
URL, so anyone on the network could run the agent's model with any bearer
string. A tunnel or proxy on this machine forwards to the loopback port.

THE MODEL NEVER RUNS ON THE SIGN-IN (S-327, 2026-09-28). The turns `serve`
runs are other callers', so the decision is `for_callers`
(`model_access.choose_model_access`): the agent's provider key, or Robutler's
models on the agent's own platform credential, each call paid by the
caller's payment token. It ran them on the signed-in owner's credits. With
neither, `serve` refuses to start with the sentence naming both, rather than
serving an agent that fails every request; `mcp serve`, whose tools need no
model, says it and starts.

ONE AGENT AT THE ROOT. The server mounts every agent under its name
(`/helper/chat/completions`); the TypeScript `serve` answers at
`/chat/completions`, `/a2a` and `/.well-known/agent-card.json`, and the docs
give those URLs for both. `RootMount` (`server/core/root_mount.py`, 2026-09-26)
maps the root paths onto the one agent's before anything else sees the
request, the credential floor included, and the named paths keep working; the
server is told the same agent is its `root_agent`, so its cards name ABSOLUTE
root URLs (`http://localhost:<port>/a2a`, a `jku` at the origin key set), as
the TypeScript cards do. With no `WEBAGENTS_PUBLIC_URL` the base is
`http://localhost:<port>`, the TypeScript `serve()` fallback: it serves, and
cannot sign a platform request, which the startup line says.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

from webagents.server.core.root_mount import ROOT_PATHS, RootMount

#: The old names, kept for anything that imported them from here.
_AtRoot = RootMount
__all__ = ["ROOT_PATHS", "RootMount", "_AtRoot", "load_served_agent", "bind_host", "loopback_bind_line", "serve_command"]


def load_served_agent(path: str, *, require_model: bool = False, for_callers: bool = True):
    """The agent at `path`, built as `serve` builds it and not yet started:
    the file's skills and model, the stored keys, the folder, the access block
    (2026-09-26, shared with `mcp serve`, so the agent an MCP client reaches is
    by construction the one `serve` would put behind HTTP). A skill the file
    names that does not exist, or a malformed block, prints the file's own
    sentence and exits 1, as the TypeScript `serve` answers.

    The model is decided for other callers (S-327): never the sign-in. With
    no model, `require_model` (`serve`) prints the sentence and exits 1;
    otherwise (`mcp serve`) the sentence is printed and the agent returned.
    `for_callers=False` is the owner's own turns (ACP, and `mcp serve` over
    stdio: the editor or MCP client the owner started here), where the
    sign-in may pay, as in the chat and as the ACP `login` method promises."""
    from .agent_builder import build_agent
    from .agent_files import default_agent_file
    from .loader import AgentFormatError

    target = Path(path)
    agent_file = target if target.is_file() else default_agent_file(target)
    # Absolute, as the TypeScript `serve` names it in every message.
    agent_file = agent_file.resolve() if agent_file is not None else None
    folder = agent_file.parent if agent_file else (target if target.is_dir() else Path.cwd())
    if agent_file is None:
        print(f"No AGENT.md or agent.json at {target.resolve()}, serving default agent")
    try:
        built = asyncio.run(
            build_agent(
                agent_file,
                working_dir=folder,
                bare=agent_file is None,
                initialize=False,
                strict_skills=True,
                for_callers=for_callers,
            )
        )
    except AgentFormatError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    if built.model_problem:
        print(f"[webagents] {built.name}: {built.model_problem}", file=sys.stderr)
        if require_model:
            raise SystemExit(1)
    return built


def loopback_bind_line(name: str, public_url: Optional[str]) -> str:
    """Why `serve` listens on loopback (the TypeScript `loopbackBindLine`,
    `server/origin-policy.ts`; fixture `cli/final_sdk_serve_model.json`)."""
    if public_url:
        return (
            f"[webagents] {name}: listening on 127.0.0.1 only, because it has no AuthSkill to verify its "
            "callers; WEBAGENTS_PUBLIC_URL alone does not open it to other machines. A tunnel or proxy on "
            "this machine can forward to it, or pass `hostname` (`--host 0.0.0.0` on the CLI) to accept "
            "other machines."
        )
    return (
        f"[webagents] {name}: listening on 127.0.0.1 only, because it has no AuthSkill to verify its "
        "callers. Pass `hostname` (`--host 0.0.0.0` on the CLI) to accept other machines."
    )


def bind_host(built, host: Optional[str]) -> str:
    """Where to listen: `host`; else every interface for an agent that
    verifies its callers (an AuthSkill), else loopback, and said (S-226,
    S-327, the TypeScript rule in `server/origin-policy.ts`). A public URL
    alone no longer widens the bind: it named every interface for an agent
    that could not tell one caller from another."""
    from webagents.server.core.origin_policy import agent_verifies_credentials

    public_url = os.environ.get("WEBAGENTS_PUBLIC_URL")
    bind = host or ("0.0.0.0" if agent_verifies_credentials(built.agent) else "127.0.0.1")
    if not host and bind == "127.0.0.1":
        _say(loopback_bind_line(built.name, public_url))
    return bind


def serve_command(path: str, port: int, host: Optional[str]) -> None:
    import uvicorn

    from webagents import __version__
    from webagents.agents.skills.local.secrets.keychain_ux import forbid_dialogs
    from webagents.server.core.app import create_server
    from webagents.server.core.origin_policy import agent_verifies_credentials

    # Before the agent is built, since building it reads its keys: `serve`
    # never waits on a macOS keychain dialog (keychain-ux, 2026-09-27).
    forbid_dialogs("serve")
    built = load_served_agent(path, require_model=True)
    public_url = os.environ.get("WEBAGENTS_PUBLIC_URL")
    bind = bind_host(built, host)
    # THE SAME STARTUP REPORT AS THE TYPESCRIPT `serve` (`server/node.ts`),
    # line for line and in its order (2026-09-25): whether the agent can
    # register, the key it was given, whether anything verifies its callers,
    # and why it does not beat. This server said none of it at the terminal.
    from webagents.server.core.registration import resolve_agent_token, resolve_portal_api_url

    token = resolve_agent_token(built.agent)
    portal = resolve_portal_api_url()
    local_only = not public_url and not token and not portal
    if local_only:
        _say(
            f"[webagents] {built.name}: serving locally, not registered with the platform. "
            "Registering needs WEBAGENTS_PUBLIC_URL, WEBAGENTS_AGENT_TOKEN and ROBUTLER_API_URL."
        )
    elif not public_url:
        _say(
            f"[webagents] {built.name}: no public URL (WEBAGENTS_PUBLIC_URL), serving as http://localhost:{port}. "
            "This identity cannot sign a platform request from a loopback address; set "
            "WEBAGENTS_PUBLIC_URL to the https address the agent is reachable at before registering."
        )
    # The base URL the cards name (module docstring): the configured public
    # URL, else the loopback address this serves at, as the TypeScript
    # `serve()` falls back. Never relative: a peer posts to the interface URL
    # as written and cannot resolve a relative one.
    server = create_server(
        title=built.name,
        description=built.description or "",
        version=__version__,
        agents=[built.agent],
        public_url=public_url or f"http://localhost:{port}",
        root_agent=built.name,
        quiet=True,
    )
    has_access_block = any(type(skill).__name__ == "AccessSkill" for skill in (built.agent.skills or {}).values())
    if not agent_verifies_credentials(built.agent) and not has_access_block:
        _say(
            f"[webagents] {built.name} has no AuthSkill: /chat/completions requires an "
            "Authorization header but cannot verify it. Add AuthSkill to validate api keys, "
            "owner assertions and platform service tokens.",
            stream=sys.stderr,
        )
        # And what that means for memory (2026-09-26, the e2e run): every
        # served caller is a caller nothing verified, who reads shared notes
        # and writes nothing. Said once here rather than one refused tool
        # call at a time (`startup_lines.py`, the fixture's `memory_without_auth`).
        if "memory" in (built.agent.skills or {}):
            from .startup_lines import memory_without_auth_line

            _say(memory_without_auth_line(built.name), stream=sys.stderr)
    if not local_only and (not token or not portal):
        missing = [name for name, absent in (
            ("WEBAGENTS_AGENT_TOKEN", not token),
            ("ROBUTLER_API_URL (or ROBUTLER_INTERNAL_API_URL)", not portal),
        ) if absent]
        _say(
            f"[webagents] no heartbeat for {built.name}: {' and '.join(missing)} "
            f"{'are' if len(missing) > 1 else 'is'} not set. "
            "The platform will show this agent as unknown until it beats."
        )
    # A busy port is one sentence and exit 1 BEFORE the address line
    # (`listen.py`, 2026-09-26): this said "on http://..." and then uvicorn
    # failed to bind.
    from .listen import bind_or_refuse

    bind_or_refuse(bind, port)
    _say(f"[webagents] {built.name} on http://{bind}:{port}")
    from webagents.server.core.request_log import RequestLog

    uvicorn.run(RequestLog(RootMount(server.app, built.name)), host=bind, port=port, log_level="warning")


def _say(line: str, stream: Any = None) -> None:
    print(line, file=stream or sys.stdout, flush=True)
