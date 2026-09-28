"""
Ollama (2026-09-26, gap-closure plan item 2.8): local models through Ollama's
OpenAI-compatible endpoint.

`ollama/<model>` in an agent file, or `skills: [ollama]`, reaches
`OLLAMA_BASE_URL` (`http://localhost:11434/v1` unless set) with the OpenAI
chat-completions wire shape, which Ollama serves as it is. No key: the client
sends a placeholder (`ollama`), which Ollama ignores, because the OpenAI
client refuses to start with none. A `base_url` in the entry's config still
wins over the variable, as for every model skill (`cli/agent_builder.py`).

The TypeScript SDK's `OllamaSkill` (`llm/ollama/skill.ts`) does the same; the
provider row, the variable and the default address are pinned by the shared
fixture `tests/fixtures/w2ops/models.json`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from webagents.agents.skills.base import Handoff
from webagents.utils.logging import get_logger

from ..openai.skill import OpenAISkill

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

OLLAMA_BASE_URL_VAR = "OLLAMA_BASE_URL"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "llama3.2"
#: What the client sends as the key; Ollama does not read it.
OLLAMA_PLACEHOLDER_KEY = "ollama"


def ollama_base_url(env: Optional[Dict[str, str]] = None) -> str:
    """Where Ollama answers: the variable when set, else the default address."""
    value = (os.environ if env is None else env).get(OLLAMA_BASE_URL_VAR)
    return value.strip() if value and value.strip() else OLLAMA_DEFAULT_BASE_URL


class OllamaSkill(OpenAISkill):
    """The OpenAI skill pointed at Ollama, answering as the `ollama` provider."""

    provider_id = "ollama"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = dict(config or {})
        config.setdefault("model", OLLAMA_DEFAULT_MODEL)
        if not config.get("base_url"):
            config["base_url"] = ollama_base_url()
        if not config.get("api_key"):
            config["api_key"] = OLLAMA_PLACEHOLDER_KEY
        super().__init__(config)
        self.logger = get_logger("skill.llm.ollama", "init")

    async def initialize(self, agent: "BaseAgent") -> None:
        self.agent = agent
        self.logger = get_logger("skill.llm.ollama", agent.name)
        self._get_client()
        # Its own handoff target and source, so an agent can hold an OpenAI
        # skill and this one (a failover chain) without the two colliding.
        agent.register_handoff(
            Handoff(
                target=f"ollama_{self.model.replace('.', '_').replace(':', '_')}",
                description=f"Ollama completion handler using {self.model}",
                scope="all",
                metadata={"function": self.chat_completion_stream, "priority": 10, "is_generator": True},
            ),
            source="ollama",
        )
        self.logger.info(f"Registered Ollama as handoff with model: {self.model} at {self.base_url}")
