# Memory Skills

Adds memory and context retention to agents.

Robutler offers multiple memory options (short-term, vector) as individual skills so you can choose the right persistence strategy. Memory skills integrate with the unified context to store/retrieve data safely in async environments.

## Features
- Store, retrieve, and manage conversational or task memory
- Integrates with agent context and skills

## Example: Add Memory Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.core.memory.short_term.skill import ShortTermMemorySkill

agent = BaseAgent(
    name="memory-agent",
    model="openai/gpt-4o",
    skills={
        "memory": ShortTermMemorySkill({"max_messages": 50})
    }
)
```

## Example: Use Memory in a Skill
```python
from webagents.agents.skills import Skill, tool

class RememberSkill(Skill):
    def __init__(self):
        super().__init__()
        self.memory = self.agent.skills["memory"]

    @tool
    async def remember(self, key: str, value: str) -> str:
        """Store a value in memory"""
        await self.memory.set(key, value)
        return f"Remembered {key} = {value}"

    @tool
    async def recall(self, key: str) -> str:
        """Retrieve a value from memory"""
        value = await self.memory.get(key)
        return value or "Not found"
```

Implementation: e.g., `robutler/agents/skills/core/memory/short_term/skill.py`.