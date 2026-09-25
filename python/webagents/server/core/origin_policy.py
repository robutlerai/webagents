"""
Who a served agent answers from another origin (2026-09-24, S-224).

THE HOLE THIS CLOSES. Every server `create_server` built, the local daemon
`webagents connect` starts included, installed Starlette's `CORSMiddleware`
with `allow_origins=["*"]`, `allow_credentials=True` and every header allowed.
The credential floor in front of the model routes only checks that a
credential is PRESENT (verifying it is `AuthSkill`'s job), so any web page the
developer visited could send `Authorization: Bearer <anything>` to the daemon on
127.0.0.1 and read the answers. Measured: an unrelated origin's preflight came
back approved, with that origin echoed. Binding loopback (S-218) does not help
against this; the browser is ON the loopback. And CORS never covered the
WebSocket routes at all.

THE RULE, the same one the TypeScript SDK applies (`origin-policy.ts`):
  - a server whose agents all verify what they are sent (each has an
    `AuthSkill`) keeps the permissive policy, for hosted agents called from
    browsers;
  - any other server answers loopback origins only, so a local web UI on
    another port still works and a page from anywhere else does not;
  - an explicit `cors_origins` always wins: `"*"` or `True` for any origin, a
    list for exactly those, `False` for none.
WebSocket handshakes are held to the same rule by `WebSocketOriginGuard`,
because a browser opens a socket to any origin and only reports where the page
came from in `Origin`. A handshake with no `Origin` did not come from a page
(the CLI, the platform, a script) and is left to the credential floor.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Union

#: A page served from this machine: http(s)://localhost, 127.0.0.1 or [::1], any port.
LOOPBACK_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_LOOPBACK = re.compile(LOOPBACK_ORIGIN_REGEX, re.IGNORECASE)

#: `None` for the rule above; `True`/`"*"` any origin; a list for those; `False` none.
CorsSetting = Union[None, bool, str, Sequence[str]]


def is_loopback_origin(origin: str) -> bool:
    return bool(_LOOPBACK.match(origin))


def agent_verifies_credentials(agent: Any) -> bool:
    """Whether the agent verifies credentials rather than only requiring one.

    Keyed on the class name so that checking does not import the platform
    skills (and the `robutler` stack behind them) into every server.
    """
    skills = getattr(agent, "skills", None) or {}
    values: Iterable[Any] = skills.values() if isinstance(skills, dict) else skills
    return any(type(skill).__name__ == "AuthSkill" for skill in values)


def _permissive(setting: CorsSetting) -> bool:
    return setting is True or setting == "*"


def cors_middleware_kwargs(setting: CorsSetting, verifies_credentials: bool) -> Optional[Dict[str, Any]]:
    """Keyword arguments for Starlette's `CORSMiddleware`, or None for no CORS at all."""
    common = {"allow_credentials": True, "allow_methods": ["*"], "allow_headers": ["*"]}
    if setting is False:
        return None
    if _permissive(setting) or (setting is None and verifies_credentials):
        return {"allow_origins": ["*"], **common}
    if setting is not None and not isinstance(setting, (str, bool)):
        return {"allow_origins": list(setting), **common}
    return {"allow_origin_regex": LOOPBACK_ORIGIN_REGEX, **common}


def origin_allowed(setting: CorsSetting, verifies_credentials: bool, origin: Optional[str]) -> bool:
    """The same decision for a WebSocket handshake's `Origin` header."""
    if not origin:
        return True
    if setting is False:
        return False
    if _permissive(setting) or (setting is None and verifies_credentials):
        return True
    if setting is not None and not isinstance(setting, (str, bool)):
        return origin in list(setting)
    return is_loopback_origin(origin)


class WebSocketOriginGuard:
    """Refuse a WebSocket handshake from a page the origin rule does not allow.

    Pure ASGI, so it sees the handshake before any route does. Closing before
    `websocket.accept` makes the ASGI server answer the upgrade with HTTP 403.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], allowed: Callable[[Optional[str]], bool]):
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope.get("type") == "websocket":
            origin: Optional[str] = None
            headers: List = scope.get("headers") or []
            for key, value in headers:
                if key == b"origin":
                    origin = value.decode("latin-1")
                    break
            if not self.allowed(origin):
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)
