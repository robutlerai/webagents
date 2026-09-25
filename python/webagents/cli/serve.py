"""
`webagents serve [path]` (2026-09-24): one agent over HTTP, the TypeScript
CLI's command (`typescript/src/cli/serve-action.ts`): the agent at `path` (an
AGENT.md, or the folder holding one), `-p/--port` (3000) and `--host`.

LOOPBACK UNLESS IT IS MEANT TO BE REACHED (S-226, the TypeScript rule in
`server/origin-policy.ts`): with no `--host`, it listens on 127.0.0.1 unless
the agent has a public URL (`WEBAGENTS_PUBLIC_URL`) or verifies its callers
(an AuthSkill), and says so. The model is the chat's decision
(`agent_builder.build_agent`): its key, or Robutler when signed in. With
neither it still starts, as the TypeScript `serve` does, and says so first
rather than as a failure on the first request.

ONE AGENT AT THE ROOT. The server mounts every agent under its name
(`/helper/chat/completions`); the TypeScript `serve` answers at
`/chat/completions`, and the docs give that URL for both. `_AtRoot` maps the
root paths onto the one agent's before anything else sees the request, the
credential floor included, and the named paths keep working.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


#: The one agent's routes that `serve` also answers at the root.
ROOT_PATHS = ("/chat/completions",)


class _AtRoot:
    """ASGI wrapper: `/chat/completions` is `/<agent>/chat/completions`."""

    def __init__(self, app: Any, agent_name: str) -> None:
        self.app = app
        self.prefix = f"/{agent_name}"

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket") and scope.get("path") in ROOT_PATHS:
            path = self.prefix + scope["path"]
            scope = dict(scope, path=path, raw_path=path.encode())
        await self.app(scope, receive, send)


def serve_command(path: str, port: int, host: Optional[str]) -> None:
    import uvicorn

    from webagents import __version__
    from webagents.server.core.app import create_server
    from webagents.server.core.origin_policy import agent_verifies_credentials

    from .agent_builder import build_agent
    from .agent_files import default_agent_file

    target = Path(path)
    agent_file = target if target.is_file() else default_agent_file(target)
    # Absolute, as the TypeScript `serve` names it in every message.
    agent_file = agent_file.resolve() if agent_file is not None else None
    folder = agent_file.parent if agent_file else (target if target.is_dir() else Path.cwd())
    if agent_file is None:
        print(f"No AGENT.md or agent.json at {target.resolve()}, serving default agent")
    from .loader import AgentFormatError

    try:
        # A skill the file names that does not exist stops `serve`, as in TypeScript.
        built = asyncio.run(
            build_agent(agent_file, working_dir=folder, bare=agent_file is None, initialize=False, strict_skills=True)
        )
    except AgentFormatError as error:
        # The file's own sentence and exit 1, as the TypeScript `serve` answers.
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    if built.model_problem:
        print(f"[webagents] {built.name}: {built.model_problem}", file=sys.stderr)

    public_url = os.environ.get("WEBAGENTS_PUBLIC_URL")
    bind = host or ("0.0.0.0" if public_url or agent_verifies_credentials(built.agent) else "127.0.0.1")
    if not host and bind == "127.0.0.1":
        _say(
            f"[webagents] {built.name}: listening on 127.0.0.1 only, because it has no public URL and no "
            "AuthSkill. Pass `hostname` (`--host 0.0.0.0` on the CLI) to accept other machines."
        )
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
    server = create_server(title=built.name, description=built.description or "", version=__version__, agents=[built.agent], quiet=True)
    has_access_block = any(type(skill).__name__ == "AccessSkill" for skill in (built.agent.skills or {}).values())
    if not agent_verifies_credentials(built.agent) and not has_access_block:
        _say(
            f"[webagents] {built.name} has no AuthSkill: /chat/completions requires an "
            "Authorization header but cannot verify it. Add AuthSkill to validate api keys, "
            "owner assertions and platform service tokens.",
            stream=sys.stderr,
        )
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
    _say(f"[webagents] {built.name} on http://{bind}:{port}")
    from webagents.server.core.request_log import RequestLog

    uvicorn.run(RequestLog(_AtRoot(server.app, built.name)), host=bind, port=port, log_level="warning")


def _say(line: str, stream: Any = None) -> None:
    print(line, file=stream or sys.stdout, flush=True)
