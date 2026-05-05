---
title: Creating Custom Skills
description: Build a minimal, production-ready skill — tools, prompts, hooks, handoffs, HTTP endpoints, and dependencies.
---

# Creating Custom Skills

This guide shows how to build a minimal, production-ready skill that is consistent with the SDK, Quickstart, and platform conventions.

## What a Skill Provides

- `@tool` functions — executable capabilities.
- `@prompt` producers — guide LLM behaviour.
- `@hook` handlers — react to lifecycle events (e.g., `on_message`).
- `@handoff` declarations — route to other agents when needed.
- Optional `@http` / `@websocket` endpoints — custom REST / WS handlers mounted under the agent.
- Declared dependencies — ensure other skills are present (e.g., memory).

## Minimal Skill

```typescript tab="TypeScript"
import { Skill, tool, hook, handoff } from 'webagents';
import type { Context, HookData, ClientEvent } from 'webagents';

class NotesSkill extends Skill {
  readonly name = 'notes';
  readonly dependencies = ['memory'];

  @tool({ description: "Add a note to the user's short-term memory" })
  async addNote(params: { text: string }): Promise<{ status: string; text: string }> {
    // In a real implementation, call the memory skill here.
    return { status: 'saved', text: params.text };
  }

  @hook({ lifecycle: 'on_message' })
  async normalizeMessage(data: HookData, ctx: Context) {
    return data;
  }

  @handoff({
    name: 'notes-auditor',
    description: 'Route audit requests to the auditor',
  })
  async *routeToAuditor(events: ClientEvent[]) {
    yield { type: 'response.delta', delta: 'auditor handling' } as const;
  }
}
```

```python tab="Python"
from webagents import Skill, tool, hook, handoff

class NotesSkill(Skill):
    def __init__(self, config=None):
        super().__init__(
            config=config,
            scope="all",              # all | owner | admin
            dependencies=["memory"],  # requires memory for storage
        )

    @tool
    def add_note(self, text: str) -> dict:
        """Add a note to the user's short-term memory."""
        return {"status": "saved", "text": text}

    @hook("on_message")
    async def normalize_message(self, context):
        return context

    @handoff("notes-auditor")
    def route_to_auditor(self, text: str) -> bool:
        return "audit" in text.lower()
```

## Adding HTTP Endpoints (Optional)

```typescript tab="TypeScript"
import { Skill, http } from 'webagents';

class NotesSkill extends Skill {
  readonly name = 'notes';

  @http({ path: '/notes', method: 'POST', scopes: ['owner'] })
  async createNote(req: Request): Promise<Response> {
    const payload = await req.json();
    return Response.json({ received: payload, status: 'ok' });
  }
}
```

```python tab="Python"
from webagents import http

@http("/notes", method="post", scope="owner")
async def create_note(payload: dict) -> dict:
    return {"received": payload, "status": "ok"}
```

- Endpoints are mounted under your agent path when served.
- `scope` / `scopes` can restrict access to `owner` or `admin`.

## Use Your Skill in an Agent

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { SessionSkill } from 'webagents/skills/session';

const agent = new BaseAgent({
  name: 'notes',
  instructions: 'You help users capture and recall short notes.',
  model: 'openai/gpt-4o-mini',
  skills: [new SessionSkill(), new NotesSkill()],
});
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.memory import ShortTermMemorySkill

agent = BaseAgent(
    name="notes",
    instructions="You help users capture and recall short notes.",
    model="openai/gpt-4o-mini",
    skills={
        "memory": ShortTermMemorySkill(),
        "notes": NotesSkill(),
    },
)
```

## Serve Your Agent

```typescript tab="TypeScript"
import { serve } from 'webagents';

await serve(agent, { port: 8000 });
```

```python tab="Python"
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

## Best Practices

- Keep one clear responsibility per skill.
- Validate inputs in tools and HTTP handlers.
- Use `scope` / `scopes` appropriately (`all`, `owner`, `admin`).
- Prefer async for I/O and external API calls.
- Leverage dependencies for cross-skill collaboration.

## Learn More

- [Skills overview](./overview.md)
- [Platform skills](./platform/auth.md), [Discovery](./platform/discovery.md), [NLI](./platform/nli.md), [Payments](./platform/payments.md)
- [Agent overview](../agent/overview.md)
- [Quickstart](../quickstart.md)
