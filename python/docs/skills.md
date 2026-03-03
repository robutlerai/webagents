# Skills System

Skills are modular capabilities that can be added to agents.

## Built-in Skills

| Skill | Description |
|-------|-------------|
| `NLISkill` | Call other agents via natural language |
| `DiscoverySkill` | Search for agents and intents |
| `CompletionsSkill` | OpenAI-compatible completions |
| `UampSkill` | UAMP event-based communication |
| `A2ASkill` | Google A2A task management |

## Using Skills

```python
from webagents import WebAgent
from webagents.skills import NLISkill, DiscoverySkill

agent = WebAgent(
    name="orchestrator",
    skills=[
        NLISkill(),
        DiscoverySkill(),
    ],
)

@agent.on_message
async def handle(message: str) -> str:
    # Find an agent that can help
    results = await agent.skills.discovery.search(query=message)
    if results:
        # Call the best match
        response = await agent.skills.nli.call(
            agent=results[0]["url"],
            message=message,
        )
        return response
    return "No suitable agent found"
```

## Trust Configuration

Agents can configure trust scopes to control agent-to-agent communication:

```python
agent = WebAgent(
    name="alice.secure-bot",
    config={
        "accept_from": ["family", "#verified"],
        "talk_to": ["everyone"],
    },
    skills=[NLISkill()],
)
```

See [Trust Zones](trust.md) for presets, patterns, and trust labels.

## Custom Skills

```python
from webagents.skills import Skill

class MySkill(Skill):
    name = "my-skill"
    
    async def execute(self, **kwargs):
        return {"result": "custom logic here"}
```
