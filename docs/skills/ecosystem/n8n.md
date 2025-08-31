# n8n Skill

Status: Under construction – implementation is in progress. The API below is illustrative and may change.

## Features
- Trigger n8n workflows
- Exchange data with n8n

## Planned usage: Add n8n Skill to an Agent
```python
from webagents.agents import BaseAgent
# from webagents.agents.skills.ecosystem.n8n import N8nSkill  # coming soon

agent = BaseAgent(
    name="n8n-agent",
    model="openai/gpt-4o",
    skills={
        # "n8n": N8nSkill({})  # coming soon
    }
)
```

## Planned usage: Use n8n Tool in a Skill
```python
from webagents.agents.skills import Skill, tool  # example pattern

class N8nOpsSkill(Skill):  # illustrative
    @tool
    async def trigger_workflow(self, workflow_id: str, data: dict) -> str:
        """Trigger an n8n workflow"""
        # return await self.agent.skills["n8n"].trigger(workflow_id, data)
        raise NotImplementedError("n8n skill integration coming soon")
```

**Implementation:** Under construction in `webagents/agents/skills/ecosystem/n8n/`.