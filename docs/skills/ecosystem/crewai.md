# Crewai Skill

Status: Under construction – implementation is in progress. The API below is illustrative and may change.

## Features
- Register and discover Crewai agents
- Interoperate with Crewai workflows

## Planned usage: Add Crewai Skill to an Agent
```python
from webagents.agents import BaseAgent
# from webagents.agents.skills.ecosystem.crewai import CrewaiSkill  # coming soon

agent = BaseAgent(
    name="crewai-agent",
    model="openai/gpt-4o",
    skills={
        # "crewai": CrewaiSkill({})  # coming soon
    }
)
```

## Planned usage: Use Crewai Tool in a Skill
```python
from webagents.agents.skills import Skill, tool  # example pattern

class CrewaiOpsSkill(Skill):  # illustrative
    @tool
    async def list_agents(self) -> str:
        """List available Crewai agents"""
        # return await self.agent.skills["crewai"].list_agents()
        raise NotImplementedError("crewai skill integration coming soon")
```

**Implementation:** Under construction in `webagents/agents/skills/ecosystem/crewai/`.