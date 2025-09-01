# Zapier Skill

Connect agents to Zapier workflows.

## Features
- Trigger and manage Zaps
- Pass data between Robutler and Zapier

## Example: Add Zapier Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.ecosystem.zapier import ZapierSkill

agent = BaseAgent(
    name="zapier-agent",
    model="openai/gpt-4o",
    skills={
        "zapier": ZapierSkill({})
    }
)
```

## Example: Use Zapier Tool in a Skill
```python
from webagents.agents.skills import Skill, tool

class ZapOpsSkill(Skill):
    def __init__(self):
        super().__init__()
        self.zapier = self.agent.skills["zapier"]

    @tool
    async def trigger_zap(self, zap_id: str, data: dict) -> str:
        """Trigger a Zapier workflow"""
        return await self.zapier.trigger(zap_id, data)
```

**Implementation:** See `robutler/agents/skills/ecosystem/zapier/skill.py`. 