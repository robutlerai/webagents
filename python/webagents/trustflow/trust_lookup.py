"""
The TrustFlow lookup: an agent's score and topic summary, from the platform
(webagents gap-closure plan item 2.5, 2026-09-26). The TypeScript twin is
`typescript/src/trustflow/trust-lookup.ts`; the contract both read is
`tests/fixtures/trust/trust_tool_definition.json`.

TRUSTFLOW IS A PLATFORM SERVICE. The score is computed by Robutler from
interactions on the platform, so an SDK agent cannot compute it and does not
try: it asks `GET /api/trust/lookup`, AUTHENTICATED AS ITSELF, and holds the
answer for a minute. The platform decides who is asking from the credential and
what they may see; this class only carries the credential and the subject.

THE CREDENTIAL is the discovery skill's rule (`robutler/discovery/skill.py`,
module docstring): an identity that can sign signs the request (Web Bot Auth,
`WebBotAuth` as the httpx auth) and no bearer rides beside it; otherwise a
platform key is the bearer; otherwise the call is refused up front with the
sentence naming the fix, never sent to get a 401.

FAILURE IS AN ERROR, NEVER A ZERO. A lookup that cannot be made (no
credential, unreachable, refused, unreadable) raises `TrustLookupError`, so a
caller gating on trust (the access skill) fails CLOSED and a tool says why; a
zero would read as "this agent is not trusted", which the platform never said.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union
from urllib.parse import urlencode

from webagents.agents.skills.robutler.platform_url import resolve_platform_url

TRUST_LOOKUP_PATH = "/api/trust/lookup"
TRUST_RECORD_PATH = "/api/trust/record"
#: How long an answer is held; the platform marks it `max-age=60` too.
TRUST_CACHE_TTL_MS = 60_000
#: How long a signed record is held: a record is valid for a week.
TRUST_RECORD_CACHE_TTL_MS = 3_600_000
TRUST_LOOKUP_TIMEOUT_MS = 8_000

#: The sentence for an agent with neither credential, the TypeScript one word
#: for word (fixture `no_credential`).
NO_TRUST_CREDENTIAL = (
    "No credential for the platform: this agent has no signing identity and no platform key, "
    "so it cannot ask Robutler for a TrustFlow score. "
    "Publish it with `webagents publish` (the chat and `serve` then use the key it stores for this folder), "
    "serve it at a public https URL (WEBAGENTS_PUBLIC_URL) so its requests are signed, "
    "or set WEBAGENTS_AGENT_TOKEN to the agent's key."
)


@dataclass
class PlatformCredential:
    """How one platform call is authenticated (module docstring). Exactly one of the three is set."""

    #: A `WebBotAuth`: the request is signed with the agent's identity.
    auth: Optional[Any] = None
    #: A platform key, presented as `Authorization: Bearer`.
    bearer: Optional[str] = None
    #: Neither: the sentence naming the fix.
    refusal: Optional[str] = None

    @property
    def kind(self) -> str:
        if self.auth is not None:
            return "signature"
        if self.bearer:
            return "bearer"
        return "none"


def platform_credential_for(identity: Any = None, api_key: Optional[str] = None) -> PlatformCredential:
    """The credential for an agent: its identity (`AgentSigningIdentity`, or
    anything with `issuer` and `held_keys()`) when that can sign for a URL the
    platform can fetch keys from, else its key as a bearer, else the refusal
    (module docstring)."""
    key = (api_key or "").strip() or None
    issuer = getattr(identity, "issuer", None)
    held = getattr(identity, "held_keys", None)
    if isinstance(issuer, str) and issuer and callable(held):
        from webagents.crypto.http_signature import SigningError, WebBotAuth

        try:
            return PlatformCredential(auth=WebBotAuth(held(), issuer))
        except (SigningError, RuntimeError) as error:
            if not key:
                return PlatformCredential(refusal=f"this agent's identity cannot sign: {error}")
    if key:
        return PlatformCredential(bearer=key)
    return PlatformCredential(refusal=NO_TRUST_CREDENTIAL)


def trust_failure_message(code: str, agent: str, status: Optional[int] = None, reason: Optional[str] = None) -> str:
    """The sentence a tool says for each failure, the same in both SDKs (fixture `messages`)."""
    if code == "no_credential":
        return reason or NO_TRUST_CREDENTIAL
    if code == "unreachable":
        return f"The platform could not be reached, so there is no TrustFlow score for {agent}."
    if code == "not_found":
        return f"The platform knows no agent {agent} that you may look up."
    if code == "refused":
        return f"The platform refused the lookup for {agent} ({status if status is not None else 'no status'})."
    return f"The platform's answer for {agent} could not be read."


class TrustLookupError(Exception):
    """A lookup with no answer: `code` is one of no_credential, unreachable, refused, not_found, unreadable."""

    def __init__(self, code: str, agent: str, status: Optional[int] = None, reason: Optional[str] = None):
        super().__init__(trust_failure_message(code, agent, status, reason))
        self.code = code
        self.agent = agent
        self.status = status


CredentialSource = Union[PlatformCredential, Callable[[], Union[PlatformCredential, Awaitable[PlatformCredential]]]]


@dataclass
class _Held:
    value: Dict[str, Any]
    expires: float


@dataclass
class TrustLookup:
    """The lookup client: `lookup` and `record`, each held for its TTL per subject and topic."""

    #: The platform; None means the skills' resolution (`platform_url.py`).
    platform_url: Optional[str] = None
    #: The credential, or how to get it per call (an identity is attached after construction).
    credential: Optional[CredentialSource] = None
    ttl_ms: int = TRUST_CACHE_TTL_MS
    record_ttl_ms: int = TRUST_RECORD_CACHE_TTL_MS
    timeout_ms: int = TRUST_LOOKUP_TIMEOUT_MS
    #: The clock in ms, for tests.
    now: Callable[[], float] = field(default_factory=lambda: (lambda: time.time() * 1000))
    _held: Dict[Tuple[str, str], _Held] = field(default_factory=dict, repr=False)
    _held_records: Dict[str, _Held] = field(default_factory=dict, repr=False)

    def resolved_platform_url(self) -> str:
        if not self.platform_url:
            self.platform_url = resolve_platform_url()
        return self.platform_url.rstrip("/")

    async def resolve_credential(self) -> PlatformCredential:
        source = self.credential
        if callable(source):
            found = source()
            if hasattr(found, "__await__"):
                found = await found  # type: ignore[misc]
            return found  # type: ignore[return-value]
        return source or PlatformCredential(refusal=NO_TRUST_CREDENTIAL)

    def clear_cache(self) -> None:
        self._held.clear()
        self._held_records.clear()

    async def lookup(self, agent: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """The agent's trust summary, on `topic` when given. `agent` is its URL,
        `@username` or platform id, as the platform reads it. Raises
        `TrustLookupError` when there is no answer (module docstring)."""
        subject = agent.strip()
        wanted_topic = (topic or "").strip()
        key = (subject, wanted_topic)
        hit = self._held.get(key)
        if hit is not None and hit.expires > self.now():
            return hit.value
        params = [("agent", subject)] + ([("topic", wanted_topic)] if wanted_topic else [])
        body = await self._get_json(f"{self.resolved_platform_url()}{TRUST_LOOKUP_PATH}?{urlencode(params)}", subject)
        score = body.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not isinstance(body.get("subject"), dict):
            raise TrustLookupError("unreadable", subject)
        self._held[key] = _Held(body, self.now() + self.ttl_ms)
        return body

    async def record(self, agent: str) -> Dict[str, Any]:
        """The agent's signed record (`trust_record.py` verifies it). Held for an hour."""
        subject = agent.strip()
        hit = self._held_records.get(subject)
        if hit is not None and hit.expires > self.now():
            return hit.value
        body = await self._get_json(f"{self.resolved_platform_url()}{TRUST_RECORD_PATH}?{urlencode([('agent', subject)])}", subject)
        if not isinstance(body.get("record"), str) or not isinstance(body.get("payload"), dict):
            raise TrustLookupError("unreadable", subject)
        self._held_records[subject] = _Held(body, self.now() + self.record_ttl_ms)
        return body

    async def _get_json(self, url: str, subject: str) -> Dict[str, Any]:
        import httpx

        credential = await self.resolve_credential()
        if credential.kind == "none":
            raise TrustLookupError("no_credential", subject, reason=credential.refusal)
        headers = {"accept": "application/json"}
        if credential.kind == "bearer":
            headers["Authorization"] = f"Bearer {credential.bearer}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_ms / 1000) as client:
                response = await client.get(
                    url,
                    headers=headers,
                    auth=credential.auth if credential.auth is not None else httpx.USE_CLIENT_DEFAULT,
                )
        except httpx.HTTPError:
            raise TrustLookupError("unreachable", subject) from None
        if response.status_code == 404:
            raise TrustLookupError("not_found", subject, 404)
        if not response.is_success:
            raise TrustLookupError("refused", subject, response.status_code)
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        raise TrustLookupError("unreadable", subject)
