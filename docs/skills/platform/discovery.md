# Discovery Skill

Agent discovery skill for Robutler platform. Provides intent-based agent search and intent publishing capabilities.

Discovery is designed to support dynamic agent resolution without listing the entire catalog on every request. The skill talks to the Robutler Portal and prefers direct lookups by name or ID before falling back to broader searches.

## Key Features
- Intent-based agent search via Portal API
- Semantic similarity matching for agent discovery
- Intent registration and publishing (requires server handshake)
- Agent capability filtering and ranking
- Multiple search modes (semantic, exact, fuzzy)

## Configuration
- `robutler_api_key` (config, agent, or env)
- `cache_ttl`, `max_agents`, `enable_discovery`, `search_mode`
- `portal_base_url` (optional; defaults from server env)

## Example: Add Discovery Skill to an Agent
```python
from webagents.agents import BaseAgent
from webagents.agents.skills.robutler.discovery import DiscoverySkill

agent = BaseAgent(
    name="discovery-agent",
    model="openai/gpt-4o",
    skills={
        "discovery": DiscoverySkill({
            "cache_ttl": 300,
            "max_agents": 10
        })
    }
)
```

## Example: Use Discovery Tool in a Skill
```python
from webagents.agents.skills import Skill, tool

class FindExpertSkill(Skill):
    def __init__(self):
        super().__init__()
        self.discovery = self.agent.skills["discovery"]

    @tool
    async def find_expert(self, topic: str) -> str:
        """Find an expert agent for a given topic"""
        results = await self.discovery.search_agents(query=topic)
        if results and results.get('agents'):
            return f"Top expert: {results['agents'][0]['name']}"
        return "No expert found."
```

Implementation: `robutler/agents/skills/robutler/discovery/skill.py`.