# Database Skill

Status: Under construction – implementation is in progress. The API below is illustrative and may change.

## Features
- Connect to databases
- Run SQL queries
- Fetch and update records

## Planned usage: Add Database Skill to an Agent
```python
from webagents.agents import BaseAgent
# from webagents.agents.skills.ecosystem.database import DatabaseSkill  # coming soon

agent = BaseAgent(
    name="db-agent",
    model="openai/gpt-4o",
    skills={
        # "database": DatabaseSkill({})  # coming soon
    }
)
```

## Planned usage: Use Database Tool in a Skill
```python
from webagents.agents.skills import Skill, tool  # example pattern

class QuerySkill(Skill):  # illustrative
    @tool
    async def run_query(self, sql: str) -> str:
        """Run a SQL query"""
        # return await self.agent.skills["database"].query(sql)
        raise NotImplementedError("database skill integration coming soon")
```

**Implementation:** Under construction in `webagents/agents/skills/ecosystem/database/`.