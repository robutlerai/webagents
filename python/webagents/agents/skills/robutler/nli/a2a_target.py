"""
Where a `delegate` target goes (2026-09-27, the a2a-delegate lane of the
gap-closure build): the platform path the skill always had, or a peer that
speaks A2A v1.0 (`agents/skills/core/transport/a2a/a2a_client.py`). The
TypeScript twin is `skills/nli/a2a-target.ts`; the shared fixture
`tests/fixtures/a2a/delegate_routing.json` pins every case and every sentence
in both SDKs.

THREE ROUTES, DECIDED IN THIS ORDER:
  * `peer`: the target is a URL the agent file's `a2a.peers` configures, or
    lies under one (the longest configured URL that is the target or a path
    prefix of it, the rule `peer_token_for` applies). The owner wrote that
    URL, so its scheme is not judged here: a peer on loopback is a peer. The
    hop carries the peer's configured bearer, when there is one.
  * `probe`: an https URL that is not the platform's own origin. Its card is
    fetched once and, when it names an A2A 1.0 JSON-RPC interface, the hop
    goes over A2A with no bearer; a URL that serves no such card stays on the
    platform path, exactly as before. Plain http off loopback is never probed,
    and loopback is a peer only when configured: a model must not be able to
    point the agent at a local port by naming it.
  * `platform`: everything else. `@name` and a bare name are agents on the
    platform; so is a URL at the platform's origin, unless the owner pinned
    that very URL as a peer, which is an explicit ask for A2A.

WHAT A PEER HOP NEVER CARRIES: the run's payment token, the caller's auth
token, an owner assertion, the agent's platform key (`never_sent_headers` in
the fixture). A peer outside the platform is paid by nobody, and the result
says so with `A2A_UNPAID_NOTE`: no budget is derived for the hop and there is
no receipt.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

#: How long a probe waits for a card (the fixture's `probe.timeout_seconds`).
A2A_PROBE_TIMEOUT_SECONDS = 10.0
#: Appended to a peer hop's reply (the fixture's `unpaid_note`).
A2A_UNPAID_NOTE = (
    "[delegate over A2A: {url} is a peer outside the platform, paid by nobody: "
    "no budget applies to this hop and there is no receipt]"
)
#: A peer hop that could not complete (the fixture's `a2a_failed`).
A2A_FAILED = "delegation to {url} over A2A failed: {reason}"
#: Attachments name platform content a peer cannot read (the fixture's `attachments_refused`).
A2A_ATTACHMENTS_REFUSED = (
    "delegation to {url} refused: attachments cannot be forwarded to a peer outside the platform; "
    "put the content in the message or send it another way"
)
#: What an empty peer reply reads as (the fixture's `result.empty_reply`).
A2A_EMPTY_REPLY = "(no response)"

_DEFAULT_PORTS = {"http": 80, "https": 443}

__all__ = [
    "A2A_ATTACHMENTS_REFUSED",
    "A2A_EMPTY_REPLY",
    "A2A_FAILED",
    "A2A_PROBE_TIMEOUT_SECONDS",
    "A2A_UNPAID_NOTE",
    "classify_delegate_target",
    "is_loopback_url",
    "peer_entry_for",
]


def _parse_http_url(value: str) -> Optional[Tuple[str, str]]:
    """`(scheme, origin)` for an absolute http(s) URL with a host, else None."""
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    origin = f"{parts.scheme}://{host}"
    if port is not None and port != _DEFAULT_PORTS[parts.scheme]:
        origin += f":{port}"
    return parts.scheme, origin


def peer_entry_for(url: str, peers: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The peer entry `url` falls under: `{"peer", "token"}` for the longest
    configured URL that is `url` or a path prefix of it, else None."""
    target = (url or "").rstrip("/")
    best: Optional[Tuple[int, str, Optional[str]]] = None
    for peer, entry in (peers or {}).items():
        if not isinstance(peer, str):
            continue
        prefix = peer.strip().rstrip("/")
        if not prefix:
            continue
        if target != prefix and not target.startswith(prefix + "/"):
            continue
        if best is not None and len(prefix) <= best[0]:
            continue
        token = entry.get("token") if isinstance(entry, dict) else None
        best = (len(prefix), prefix, token if isinstance(token, str) and token else None)
    if best is None:
        return None
    return {"peer": best[1], "token": best[2]}


def is_loopback_url(value: str) -> bool:
    """Whether a URL names this machine: `localhost`, `*.localhost`, 127/8 or `::1`."""
    try:
        host = (urlsplit(value).hostname or "").lower()
    except ValueError:
        return False
    return host == "localhost" or host.endswith(".localhost") or host.startswith("127.") or host == "::1"


def classify_delegate_target(target: str, peers: Optional[Dict[str, Any]], platform_base: str) -> Dict[str, Any]:
    """The route for `target` (module docstring): `{"route": "platform"}`,
    `{"route": "peer", "url", "token"}` or `{"route": "probe", "url"}`.
    `platform_base` is where `@name` lives; its origin keeps the platform path."""
    trimmed = (target or "").strip()
    if not trimmed or trimmed.startswith("@") or "://" not in trimmed:
        return {"route": "platform"}
    parsed = _parse_http_url(trimmed)
    if parsed is None:
        return {"route": "platform"}
    scheme, origin = parsed
    url = trimmed.rstrip("/")
    peer = peer_entry_for(url, peers)
    if peer is not None:
        return {"route": "peer", "url": url, "token": peer["token"]}
    if scheme != "https":
        return {"route": "platform"}
    platform = _parse_http_url(platform_base or "")
    if platform is not None and platform[1] == origin:
        return {"route": "platform"}
    return {"route": "probe", "url": url}
