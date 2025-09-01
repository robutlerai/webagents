# Creating Custom Skills

This guide shows how to build a minimal, production-ready skill that is consistent with the SDK, Quickstart, and platform conventions.

## What a Skill Provides

- `@tool` functions: executable capabilities
- `@prompt` producers: guide LLM behavior
- `@hook` handlers: react to lifecycle events (e.g., `on_message`)
- `@handoff` rules: route to other agents when needed
- Optional `@http` endpoints: custom REST handlers mounted under the agent
- Declared dependencies: ensure other skills are present (e.g., memory)

## Minimal Skill

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool
from webagents.agents.skills.decorators import hook, handoff

class NotesSkill(Skill):
    def __init__(self, config=None):
        super().__init__(
            config=config,
            scope="all",              # all | owner | admin
            dependencies=["memory"],  # requires memory for storage
        )

    @tool
    def add_note(self, text: str) -> dict:
        """Add a note to the user’s short-term memory."""
        # In a real implementation, call memory skill here
        return {"status": "saved", "text": text}
    
    @hook("on_message")
    async def normalize_message(self, context):
        # Lightweight preprocessing for downstream tools or model
        return context
    
    @handoff("notes-auditor")
    def route_to_auditor(self, text: str) -> bool:
        return "audit" in text.lower()
```

## Adding HTTP Endpoints (Optional)

```python
from webagents.agents.tools.decorators import http

@http("/notes", method="post", scope="owner")
async def create_note(payload: dict) -> dict:
    return {"received": payload, "status": "ok"}
```

- Endpoints are mounted under your agent path when served
- `scope` can restrict access to `owner` or `admin`

## Use Your Skill in an Agent

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.memory import ShortTermMemorySkill

agent = BaseAgent(
    name="notes",
    instructions="You help users capture and recall short notes.",
    model="openai/gpt-4o-mini",
    skills={
        "memory": ShortTermMemorySkill(),
        "notes": NotesSkill(),
    }
)
```

## Serve Your Agent

```python
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

## Best Practices

- Keep one clear responsibility per skill
- Validate inputs in tools and HTTP handlers
- Use `scope` appropriately (`all`, `owner`, `admin`)
- Prefer async for I/O and external API calls
- Leverage dependencies for cross-skill collaboration

## Learn More

- Skills Framework: [skills/overview.md](../skills/overview.md)
- Platform Skills: [platform/auth.md](../skills/platform/auth.md), [platform/discovery.md](../skills/platform/discovery.md), [platform/nli.md](../skills/platform/nli.md), [platform/payments.md](../skills/platform/payments.md)
- Agent Overview: [agent/overview.md](../agent/overview.md)
- Quickstart: [quickstart.md](../quickstart.md)
