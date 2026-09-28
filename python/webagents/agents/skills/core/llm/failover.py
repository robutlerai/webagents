"""
Model failover (2026-09-26, gap-closure plan item 2.8): the agent's model,
then the `fallback_models:` of its file, in order, when a call fails with a
provider error.

One LLM skill standing in for a chain of them. Each turn asks the first
member; when it answers a 5xx, a 429, a 408 or never answers (no connection,
no host, a timeout), the next member is asked the same question, and the
transcript gets a note saying so BEFORE the next reply
(`{failed} did not answer ({reason}); trying {next}`). Any other failure (a
refused key, an unknown model, a bad request, no credits) is reported as it
is: those are not outages, and asking another model would hide the thing to
fix. A member that had already started answering is never retried, because
the reply would repeat.

The note travels as a chunk with no choices and a `webagents_note` key, which
the agent loop passes on and the chat renders as a warning line; served
agents see the chunk. The TypeScript SDK's `FailoverLLMSkill`
(`llm/failover/skill.ts`) does the same with a `progress` event, and the
words, the retryable set and the cases are pinned by
`tests/fixtures/w2ops/models.json` (`failover`).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple

from webagents.agents.skills.base import Handoff, Skill
from webagents.utils.errors import describe_exception
from webagents.utils.logging import get_logger

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

FAILOVER_SKILL_NAME = "failover"
#: The HTTP statuses that mean the provider, not the request, failed.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_NETWORK_WORDS = re.compile(r"connection|timed out|timeout|unreachable|refused|name or service|nodename|ECONNREFUSED|ENOTFOUND", re.I)


def failover_note(failed: str, reason: str, next_model: str) -> str:
    """The note the transcript gets when the chain moves on (fixture `failover.note`)."""
    return f"{failed} did not answer ({reason}); trying {next_model}"


def provider_failure(error: BaseException, detail: bool = False) -> Optional[str]:
    """How the note names a provider's failure (`HTTP 503`, `could not reach
    <origin>`), or None for a failure that is the request's own: a refused
    key, an unknown model, a bad request, no credits.

    THE ORIGIN IS NAMED ONLY WITH `detail` (S-228): the note reaches whoever
    called the agent, and a served agent's upstream address has no business
    in a reply. The chain passes `detail` for the owner at the terminal, as
    the TypeScript skill sees the origin only when the CLI switched the
    request error detail on; a caller of a served agent reads
    `could not reach the provider`.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return f"HTTP {status}" if status in RETRYABLE_STATUSES else None
    described = describe_exception(error)
    unreachable = re.search(r"could not reach (https?://[^\s/:]+(?::\d+)?|[^\s:]+)", described, re.I)
    if unreachable:
        return f"could not reach {unreachable.group(1)}" if detail else "could not reach the provider"
    name = type(error).__name__
    if "Connection" in name or "Timeout" in name or _NETWORK_WORDS.search(described):
        host = re.search(r"(https?://[^\s/]+)", described, re.I)
        return f"could not reach {host.group(1)}" if host and detail else "could not reach the provider"
    return None


def _owner_at_the_terminal() -> bool:
    """Whether the current run's caller is the local owner (`run_as_local_owner`,
    the chat and `webagents -p`), who may read the upstream origin."""
    try:
        from webagents.server.context.context_vars import get_context

        auth = getattr(get_context(), "auth", None)
        return getattr(auth, "provider", None) == "local" and getattr(auth, "scope", None) == "owner"
    except Exception:  # noqa: BLE001 - no context is not the owner
        return False


class FailoverLLMSkill(Skill):
    """The chain as one skill: `chain` is `(provider/model, skill)` pairs, primary first."""

    is_failover = True

    def __init__(self, chain: Sequence[Tuple[str, Skill]]):
        super().__init__({}, scope="all")
        if not chain:
            raise ValueError("A failover chain needs at least one model")
        self.chain: List[Tuple[str, Skill]] = list(chain)
        #: The model that answered the last call (`provider/model`), for the chat's cost line.
        self.answered_model: Optional[str] = None
        #: The notes said during the last call, in order.
        self.notes: List[str] = []
        self.logger = get_logger("skill.llm.failover", "init")

    @property
    def models(self) -> List[str]:
        return [model for model, _ in self.chain]

    @property
    def model(self) -> str:
        """The model a call starts with, as the loop labels its span; the answering one is `answered_model`."""
        return self.chain[0][0]

    @property
    def provider_id(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else "unknown"

    async def initialize(self, agent: "BaseAgent") -> None:
        self.agent = agent
        self.logger = get_logger("skill.llm.failover", agent.name)
        # Registered FIRST and at priority 1, below every member's own (10), so
        # this is the agent's handoff: `register_handoff` keeps the lowest
        # priority active and sorts the list, so the turn's reset finds it too.
        agent.register_handoff(
            Handoff(
                target=FAILOVER_SKILL_NAME,
                description=f"Failover over {', '.join(self.models)}",
                scope="all",
                metadata={"function": self.chat_completion_stream, "priority": 1, "is_generator": True},
            ),
            source=FAILOVER_SKILL_NAME,
        )
        for model, skill in self.chain:
            await skill.initialize(agent)
            # A member's own decorated tools, hooks and prompts (the Google
            # skill's `analyze_image`), registered as they would be for a
            # skill the agent holds directly; the agent walks only its own
            # skills for decorators.
            register = getattr(agent, "_auto_register_skill_decorators", None)
            if callable(register):
                register(skill, f"{FAILOVER_SKILL_NAME}:{model}")

    async def cleanup(self) -> None:
        for _model, skill in self.chain:
            close = getattr(skill, "cleanup", None)
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:  # noqa: BLE001 - a member that cannot close must not stop the others
                pass

    async def chat_completion_stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        self.notes = []
        self.answered_model = None
        last = len(self.chain) - 1
        detail = _owner_at_the_terminal()
        for index, (model, skill) in enumerate(self.chain):
            started = False
            try:
                async for chunk in skill.chat_completion_stream(messages, tools=tools, **kwargs):
                    started = True
                    yield chunk
            except Exception as error:  # noqa: BLE001 - classified below
                reason = None if (started or index == last) else provider_failure(error, detail)
                if reason is None:
                    raise
                next_model = self.chain[index + 1][0]
                note = failover_note(model, reason, next_model)
                self.notes.append(note)
                self.logger.warning(note)
                yield {"object": "chat.completion.chunk", "choices": [], "webagents_note": note}
                continue
            self.answered_model = model
            return
