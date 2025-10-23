# LLM Skills

Provides large language model (LLM) capabilities to agents.

Robutler supports multiple providers through dedicated skills (e.g., OpenAI, Anthropic) and via LiteLLM proxying. In most cases you can specify `model="openai/gpt-4o"` and the correct provider skill is created for you.

## Features
- Text generation, completion, and chat using supported LLM backends
- Integration with agent tool and skill system

## Example: Add LLM Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.core.llm.openai.skill import OpenAISkill

agent = BaseAgent(
    name="llm-agent",
    model="openai/gpt-4o",
    skills={
        "llm": OpenAISkill({"model": "gpt-4o-mini"})
    }
)
```

## Example: Use LLM Tool in a Skill
```python
from webagents.agents.skills import Skill, tool

class SummarizeSkill(Skill):
    def __init__(self):
        super().__init__()
        self.llm = self.agent.skills["llm"]

    @tool
    async def summarize(self, text: str) -> str:
        """Summarize a block of text using the LLM"""
        return await self.llm.generate(prompt=f"Summarize: {text}")
```

Implementation: provider-specific skills, e.g., `robutler/agents/skills/core/llm/openai/skill.py`.