"""
The well-known HTTP Message Signatures directory: the key set a
`legacy-string` signer's `Signature-Agent` resolves to (2026-09-19). The
Python twin of the TypeScript SDK's `src/server/key-directory.ts`.

WHY IT EXISTS. `sign_request(..., form="legacy-string")` sends a bare ORIGIN as
`Signature-Agent`, the form an edge that reads nothing else accepts. A verifier
resolves a bare origin to
`{origin}/.well-known/http-message-signatures-directory` and REQUIRES the
answer's media type to be
`application/http-message-signatures-directory+json`
(draft-ietf-webbotauth-httpsig-protocol-00 sections 5.5.1 and 8.1; the
platform's lib/auth/web-bot-auth/discovery.ts refuses any other type as
`key_set_invalid`). Neither SDK server served that path, so selecting the form
against an agent hosted by one failed discovery on every request and counted
against the platform's failed-signature window: the option was a trap. The
signer keeps the form; the servers now serve what it names.

WHAT IT CARRIES. The Ed25519 public keys of EVERY static agent the server
hosts, because a bare origin cannot say which agent signed. Each entry is what
the agent's own key set publishes, `{kty, crv, x, kid, use}`, where `kid` IS
the RFC 7638 thumbprint, which is what the directory rule demands (a `kid` at
the well-known directory that is not the thumbprint is a refusal). The RSA
entry the agent's key set also carries for its RS256 consumers is NOT listed:
the platform counts entries against a cap of 16 (`KEY_SET_MAX_KEYS`) before it
filters them, so a server whose agents hold more than `DIRECTORY_MAX_KEYS`
keys between them publishes a directory the platform refuses. It is served as
it is, with a warning, rather than truncated: dropping a key would fail one
agent silently.

WHAT THE FORM MEANS, which serving this does not change: the principal of a
`legacy-string` signature is the ORIGIN, so every agent that selects the form
on one origin is ONE principal to the platform, and registering it also needs
a card at `{origin}/.well-known/agent.json`, which this server does not serve
(an origin card cannot self-name for an agent mounted under a path). Agents
mounted under a path should keep the default `dictionary-typed` form, which
names their own key set.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from fastapi import Response

#: P section 8.1: the path appended to a bare origin.
DIRECTORY_WELL_KNOWN_PATH = "/.well-known/http-message-signatures-directory"
#: P section 5.5.1: the media type the directory MUST be served with.
DIRECTORY_MEDIA_TYPE = "application/http-message-signatures-directory+json"
#: The platform's `KEY_SET_MAX_KEYS`: a published set with more entries is refused whole.
DIRECTORY_MAX_KEYS = 16

_log = logging.getLogger(__name__)


def directory_keys(managers: Iterable[Optional[Any]]) -> List[Dict[str, str]]:
    """The union of the hosted agents' published Ed25519 keys, one entry per
    thumbprint, in hosting order. `managers` are `JWKSManager`s (or None for an
    agent whose key could not be loaded); one that holds no Ed25519 key
    contributes nothing."""
    seen: set = set()
    keys: List[Dict[str, str]] = []
    for manager in managers:
        if manager is None:
            continue
        try:
            held = manager.held_ed25519_keys()
        except RuntimeError:
            continue
        for key in held:
            jwk = key.public_jwk()
            kid = jwk.get("kid")
            if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519" or not kid or kid in seen:
                continue
            seen.add(kid)
            keys.append({"kty": jwk["kty"], "crv": jwk["crv"], "x": jwk["x"], "kid": kid, "use": "sig"})
    return keys


def key_directory_response(managers: Iterable[Optional[Any]]) -> Response:
    """The directory answer: 200 with the directory media type and the same
    `Cache-Control: max-age` the TypeScript server sends (the platform's
    refresh clamp reads it), or 404 when no hosted agent holds a signing key."""
    keys = directory_keys(managers)
    if not keys:
        return Response(
            content=json.dumps({"error": "No signing identity is hosted here"}),
            status_code=404,
            media_type="application/json",
        )
    if len(keys) > DIRECTORY_MAX_KEYS:
        _log.warning(
            "the signatures directory lists %d keys and the platform refuses a key set above %d: a legacy-string "
            "signature from this origin will not verify. Sign with the default dictionary-typed form instead.",
            len(keys),
            DIRECTORY_MAX_KEYS,
        )
    return Response(
        content=json.dumps({"keys": keys}),
        status_code=200,
        media_type=DIRECTORY_MEDIA_TYPE,
        headers={"Cache-Control": "public, max-age=3600"},
    )


__all__ = [
    "DIRECTORY_MAX_KEYS",
    "DIRECTORY_MEDIA_TYPE",
    "DIRECTORY_WELL_KNOWN_PATH",
    "directory_keys",
    "key_directory_response",
]
