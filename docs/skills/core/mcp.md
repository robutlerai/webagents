# MCP Skill

Integrates with the Multi-Channel Platform (MCP) for dynamic tool and agent registration.

MCP is optional and complements the native skills system. Use it when you need to bridge into MCP-compatible ecosystems while keeping the agent model and hooks unchanged.

## Features
- Register tools and discover agents
- Enable cross-platform orchestration

## Example: Add MCP Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.core.mcp import MCPSkill

agent = BaseAgent(
    name="mcp-agent",
    model="openai/gpt-4o",
    skills={
        "mcp": MCPSkill({})
    }
)
```

## Example: Register a Dynamic Tool
```python
from webagents.agents.skills import Skill, tool

class DynamicToolSkill(Skill):
    def __init__(self):
        super().__init__()
        self.mcp = self.agent.skills["mcp"]

    @tool
    async def register_tool(self, name: str, description: str) -> str:
        """Register a new tool with MCP"""
        await self.mcp.register_dynamic_tool(name, description)
        return f"Registered tool: {name}"
```

**Implementation:** See `robutler/agents/skills/core/mcp/skill.py`. 