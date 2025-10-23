# NLI Skill (Natural Language Interface)

Natural Language Interface skill for agent-to-agent communication.

NLI lets agents collaborate over HTTP using natural language. It adds resilient request/response primitives and optional budgeting controls (authorization caps) so one agent can safely call another.

## Features
- HTTP-based communication with other Robutler agents
- Authorization limits and cost tracking
- Communication history and success rate tracking
- Automatic timeout and retry handling
- Agent endpoint discovery and management

## Configuration
- `timeout`, `max_retries`
- `default_authorization`, `max_authorization` (optional budgeting)
- `portal_base_url` (optional for resolving agents)

## Example: Add NLI Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.nli import NLISkill

agent = BaseAgent(
    name="nli-agent",
    model="openai/gpt-4o",
    skills={
        "nli": NLISkill({
            "timeout": 20.0,
            "max_retries": 3
        })
    }
)
```

## Example: Use NLI Tool in a Skill
```python
from webagents.agents.skills import Skill, tool

class CollaborateSkill(Skill):
    def __init__(self):
        super().__init__()
        self.nli = self.agent.skills["nli"]

    @tool
    async def ask_agent(self, agent_url: str, message: str) -> str:
        """Send a message to another agent and get the response"""
        return await self.nli.nli_tool(agent_url=agent_url, message=message)
```

Implementation: `robutler/agents/skills/robutler/nli/skill.py`.