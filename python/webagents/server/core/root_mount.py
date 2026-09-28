"""
One agent at the root (2026-09-26, the new-developer e2e run).

`WebAgentsServer` mounts every agent under its name (`/helper/chat/completions`,
`/helper/a2a`, `/helper/.well-known/agent-card.json`). The TypeScript `serve()`
answers its one agent at `basePath`, which is the root for `webagents serve`,
and the docs give the root URLs for both SDKs. Until 2026-09-26 the Python
`serve` mapped only `/chat/completions`: `POST /a2a` at the root was 405, the
v1.0 card was reachable only under `/<name>/` and carried RELATIVE interface
URLs and a relative `jku`, so a TypeScript peer could not call a served Python
agent at all.

`RootMount` is the ASGI wrapper that answers the one agent's routes at the
root: it rewrites a root path in `ROOT_PATHS`, or under a prefix in
`ROOT_PREFIXES`, to the agent's own path before the application sees the
request, the credential floor included, and the named paths keep working. The
server names the same agent as its `root_agent`, so that agent's principal is
the base URL itself and its cards name root URLs (`app.py`,
`_create_registration_endpoints`). `cli/serve.py` and the cross-SDK A2A test's
helper (`tests/interop/serve_a2a_echo.py`) both use it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: The one agent's routes that are also answered at the root, exactly.
ROOT_PATHS: Tuple[str, ...] = (
    "/chat/completions",
    "/a2a",
    "/.well-known/agent-card.json",
    "/.well-known/agent.json",
)

#: Root path prefixes that belong to the agent too: the A2A HTTP+JSON binding
#: (`/a2a/message:send`, `/a2a/tasks/{id}`, ...).
ROOT_PREFIXES: Tuple[str, ...] = ("/a2a/",)


def is_root_path(path: str) -> bool:
    """Whether a request path is one of the agent's, spelled at the root."""
    return path in ROOT_PATHS or any(path.startswith(prefix) for prefix in ROOT_PREFIXES)


class RootMount:
    """ASGI wrapper: the root spelling of the agent's routes is `/<agent>` + path."""

    def __init__(self, app: Any, agent_name: str) -> None:
        self.app = app
        self.prefix = f"/{agent_name}"

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket") and is_root_path(scope.get("path") or ""):
            path = self.prefix + scope["path"]
            scope = dict(scope, path=path, raw_path=path.encode())
        await self.app(scope, receive, send)
