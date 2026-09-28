"""
The trust skill: the `trust` tool, and this agent's own signed TrustFlow
record (webagents gap-closure plan items 2.5 and 2.7, 2026-09-26). An agent
file names it as `- trust` (`cli/agent_builder.py`); the TypeScript twin is
`typescript/src/skills/trust/trust-skill.ts`, and the tool both offer is
pinned by `tests/fixtures/trust/trust_tool_definition.json`.

TRUSTFLOW IS A PLATFORM SERVICE (the tool's description says so to the
model): the score comes from `GET /api/trust/lookup`, asked as this agent,
through `webagents.trustflow.trust_lookup`, and held for a minute. The
credential is the discovery skill's rule: the agent's identity signs when it
can, else the agent's platform key (`utils/agent_credential.py`, then the
environment names the discovery skill reads) is the bearer, else the tool
answers with the sentence naming the fix.

Text in an answer that other people wrote (a username, a topic label) is the
platform's own data about a platform record, not free text an agent chose, so
it is not fenced the way search results are.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.platform_url import resolve_platform_url
from webagents.agents.tools.decorators import tool
from webagents.trustflow.trust_lookup import (
    PlatformCredential,
    TrustLookup,
    TrustLookupError,
    platform_credential_for,
)
from webagents.trustflow.trust_record import VerifyTrustRecordResult, verify_trust_record

#: The tool's description, the TypeScript one word for word (fixture `definition`).
TRUST_DESCRIPTION = (
    "Look up an agent's TrustFlow score on the Robutler platform. TrustFlow is a platform service: "
    "Robutler computes the score from verified interactions on the platform, so it cannot be computed "
    "locally and this tool asks the platform. Use it before delegating to an agent you do not know, or "
    "to check whether an agent is trusted on a subject.\n\n"
    "`agent` is the agent's URL, @username or platform id. `topic` (optional) scores the agent on that "
    'subject, for example "billing".\n\n'
    "Returns: subject (id, username, url), score (0 to 1, the agent's overall TrustFlow), topic (the "
    "subject asked for and the agent's score on it, null when the platform could not score it), tier, "
    "trust_level (none, silver, gold, platinum, flagged or suspended), topics (the subjects the agent's "
    "earned reputation is strongest in, with a score each), computed_at and methodology. Answers are held "
    "for a minute."
)

#: The tool's parameters, the TypeScript ones (fixture `definition`).
TRUST_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "description": "The agent: its URL, @username or platform id"},
        "topic": {"type": "string", "description": 'A subject to score the agent on, for example "billing"'},
    },
    "required": ["agent"],
}

#: The definition the model sees, the TypeScript one (fixture `definition`):
#: pinned on the tool the way the discovery skill pins `search`, because the
#: decorator derives a schema from the signature and cannot carry the
#: parameter descriptions.
TRUST_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {"name": "trust", "description": TRUST_DESCRIPTION, "parameters": TRUST_PARAMETERS},
}


class TrustSkill(Skill):
    """The `trust` tool, this agent's own record, and record verification."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {}, scope="all")
        self.config = config or {}
        self.robutler_api_url = resolve_platform_url(self.config)
        self.robutler_api_key: Optional[str] = self.config.get("robutler_api_key") or self.config.get("api_key")
        #: An identity to sign with; usually the one the server attaches to the agent (`signing_identity`).
        self.identity: Any = self.config.get("identity")
        self._lookup: Optional[TrustLookup] = None

    async def initialize(self, agent: Any) -> None:
        await super().initialize(agent)
        if not self.robutler_api_key:
            self.robutler_api_key = self._resolve_key(getattr(agent, "name", None))

    @staticmethod
    def _resolve_key(agent_name: Optional[str]) -> Optional[str]:
        """The agent's own key: `utils/agent_credential` (WEBAGENTS_AGENT_TOKEN,
        the keystore), then the names the discovery skill reads."""
        try:
            from webagents.utils.agent_credential import resolve_agent_credential

            found = resolve_agent_credential(agent_name)
            if found:
                return found[0]
        except Exception:  # noqa: BLE001 - a keystore that cannot be read is no key
            pass
        return os.getenv("WEBAGENTS_API_KEY") or os.getenv("SERVICE_TOKEN") or None

    def credential(self) -> PlatformCredential:
        """Which credential the next platform call carries, decided per call
        (an identity is attached after construction)."""
        identity = self.identity if self.identity is not None else getattr(self.agent, "signing_identity", None)
        return platform_credential_for(identity, self.robutler_api_key)

    @property
    def lookup(self) -> TrustLookup:
        """The lookup client, built once."""
        if self._lookup is None:
            self._lookup = TrustLookup(
                platform_url=self.robutler_api_url,
                credential=self.credential,
                timeout_ms=int(self.config.get("timeout") or 8000),
                ttl_ms=int(self.config.get("ttl") or 60_000),
            )
        return self._lookup

    @tool(name="trust", description=TRUST_DESCRIPTION, scope="all")
    async def trust(self, agent: str, topic: str = None) -> Dict[str, Any]:
        """Look up an agent's TrustFlow score on the Robutler platform."""
        subject = str(agent or "").strip()
        if not subject:
            return {"error": "agent is required: the agent URL, @username or platform id."}
        wanted = topic.strip() if isinstance(topic, str) and topic.strip() else None
        try:
            return await self.lookup.lookup(subject, wanted)
        except TrustLookupError as error:
            return {"error": str(error)}
        except Exception as error:  # noqa: BLE001 - the model gets a sentence, never a traceback
            return {"error": f"The lookup failed: {error}"}

    trust._webagents_tool_definition = TRUST_DEFINITION

    async def record(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """This agent's own signed record (its identity's URL), or another agent's when `agent` is given."""
        identity = self.identity if self.identity is not None else getattr(self.agent, "signing_identity", None)
        subject = agent or getattr(identity, "issuer", None)
        if not subject:
            raise TrustLookupError(
                "no_credential", "(this agent)", reason="This agent has no identity, so it has no record of its own to fetch; name the agent."
            )
        return await self.lookup.record(subject)

    async def verify(self, record: str, **options: Any) -> VerifyTrustRecordResult:
        """Verify a record from any agent against the platform's key set (`trustflow.trust_record`)."""
        return await verify_trust_record(record, **options)

    def get_dependencies(self) -> List[str]:
        return []
