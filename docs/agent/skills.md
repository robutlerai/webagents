---
title: Agent Skills
description: How skills compose into an agent — tools, prompts, hooks, handoffs, HTTP endpoints, and dependencies.
---

# Agent Skills

Skills are modular capability packages that extend a `BaseAgent` with tools, prompts, hooks, handoffs, and optional HTTP endpoints. They're first-class, composable building blocks that keep business logic organized and reusable across agents.

- **Tools** — executable functions registered via `@tool`
- **Prompts** — guidance for the LLM, optionally prioritized or scoped
- **Hooks** — lifecycle callbacks (e.g., `on_message`, `before_toolcall`)
- **Handoffs** — completion handlers (local LLM or remote agents) registered during initialization
- **HTTP endpoints** — register custom REST handlers via `@http`
- **Dependencies** — declare other skills your skill requires (e.g., memory)

## Add Skills to an Agent

Attach skills when creating your agent:

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { NLISkill } from 'webagents/skills/nli';
import { AuthSkill } from 'webagents/skills/auth';
import { PortalDiscoverySkill } from 'webagents/skills/discovery';
import { PaymentSkill } from 'webagents/skills/payments';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful AI assistant.',
  model: 'openai/gpt-4o-mini',
  skills: [
    new NLISkill(),               // Natural-language communication with agents
    new AuthSkill(),              // Authentication & scoped access control
    new PortalDiscoverySkill(),   // Real-time agent discovery (intent-based)
    new PaymentSkill(),           // Monetization via priced tools
  ],
});
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.nli.skill import NLISkill
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill
from webagents.agents.skills.robutler.payments.skill import PaymentSkill

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini",  # Automatically provisions LLM skill
    skills={
        "nli": NLISkill(),             # Natural-language communication with agents
        "auth": AuthSkill(),           # Authentication & scoped access control
        "discovery": DiscoverySkill(), # Real-time agent discovery (intent-based)
        "payments": PaymentSkill(),    # Monetization via priced tools
    },
)
```

After skills are attached, your agent can use their tools, prompts, hooks, HTTP endpoints, and handoffs immediately during requests.

## Skill Anatomy (Minimal Example)

```typescript tab="TypeScript"
import { Skill, tool, hook, handoff } from 'webagents';
import type { Context, ClientEvent } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';
  readonly dependencies = ['memory'];

  async initialize() {
    // Called after the skill is attached to the agent.
    // Register additional handoffs or perform setup here.
  }

  @tool({ description: 'Summarize input text to a target length' })
  summarize(params: { text: string; max_len?: number }): string {
    return params.text.slice(0, params.max_len ?? 200);
  }

  @hook({ lifecycle: 'on_message' })
  async onMessage(data, ctx: Context) {
    return data;
  }

  @handoff({
    name: 'custom_handler',
    description: 'Use for specialized processing',
    priority: 15,
  })
  async *customCompletion(events: ClientEvent[]) {
    yield { type: 'response.delta', delta: 'Processing...' } as const;
  }
}
```

```python tab="Python"
from typing import Any, AsyncGenerator, Dict, List, Optional
from webagents import Skill, tool, hook, handoff

class MySkill(Skill):
    def __init__(self, config=None):
        super().__init__(
            config=config,
            scope="all",                # all | owner | admin
            dependencies=["memory"],    # ensure memory is present if needed
        )

    async def initialize(self, agent):
        """Called after skill is attached to agent"""
        self.agent = agent

    @tool
    def summarize(self, text: str, max_len: int = 200) -> str:
        """Summarize input text to a target length."""
        return text[:max_len]

    @hook("on_message")
    async def on_message(self, context):
        return context

    @handoff(
        name="custom_handler",
        prompt="Use for specialized processing",
        priority=15,
    )
    async def custom_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Custom completion handler (streaming)"""
        yield {"choices": [{"delta": {"content": "Processing..."}}]}
```

- Register execution logic with `@tool`.
- Guide LLM behavior with prompts (see [Prompts](./prompts.md)).
- React to request lifecycle via `@hook`.
- Provide completion handlers with `@handoff` (for LLM or remote agent routing).

## HTTP Endpoints in Skills

Register custom REST endpoints with the `@http` decorator. These are mounted under your agent's base path when served.

```typescript tab="TypeScript"
import { Skill, http } from 'webagents';

class WeatherSkill extends Skill {
  readonly name = 'weather';

  @http({ path: '/weather', method: 'GET', scopes: ['owner'] })
  async getWeather(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const location = url.searchParams.get('location') ?? '';
    const units = url.searchParams.get('units') ?? 'celsius';
    return Response.json({ location, temperature: 25, units });
  }

  @http({ path: '/data', method: 'POST' })
  async postData(req: Request): Promise<Response> {
    const payload = await req.json();
    return Response.json({ received: payload, status: 'processed' });
  }
}
```

```python tab="Python"
from webagents import http

@http("/weather", method="get", scope="owner")
def get_weather(location: str, units: str = "celsius") -> dict:
    return {"location": location, "temperature": 25, "units": units}

@http("/data", method="post")
async def post_data(payload: dict) -> dict:
    return {"received": payload, "status": "processed"}
```

- `path` — endpoint path relative to the agent root (e.g., `/assistant/weather`).
- `method` — `'GET'`, `'POST'`, etc.
- `scopes` (TS) / `scope` (Python) — optional access control (`'all'`, `'owner'`, `'admin'`).

## Using Skill Tools in a Request

Tools you register are available to the agent at runtime. You can also pass external tools per request (OpenAI function-calling compatible):

```typescript tab="TypeScript"
const response = await agent.run([
  { role: 'user', content: 'Summarize: ...' },
]);

// Or include additional, ad-hoc tools for a single call:
const calc = await agent.run(
  [{ role: 'user', content: 'Calculate 42 * 17' }],
  {
    tools: [
      {
        type: 'function',
        function: {
          name: 'calculator',
          description: 'Calculate math expressions',
          parameters: { type: 'object', properties: { expr: { type: 'string' } } },
        },
      },
    ],
  },
);
```

```python tab="Python"
response = await agent.run([
    {"role": "user", "content": "Summarize: ..."}
])

# Or include additional, ad-hoc tools for a single call:
response = await agent.run(
    messages=[{"role": "user", "content": "Calculate 42 * 17"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
        },
    }],
)
```

## Serving an Agent with Skills

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

## Skills in an Agent File

An agent file lists its skills by name under `skills:`. A skill that takes
settings is written as a one-key map:

```markdown
---
name: helper
model: openai/gpt-4o-mini
skills:
  - openai
  - filesystem
  - rest:
      sign: always
---
```

The CLI changes that list for you, and leaves the rest of the file as you
wrote it:

```bash
webagents skills list                  # the names this SDK can load
webagents skills add discovery shell   # add to this folder's AGENT.md
webagents skills remove shell          # take one out
webagents skills add todo -a helper    # AGENT-helper.md, or the agent named helper
```

A name `skills list` does not show is refused with a suggestion. After an add,
the command says what a skill still needs on this machine, such as a model
provider's key or a Robutler sign-in. The two SDKs ship different skill sets,
so `skills list` answers for the CLI you run.

A running chat or `serve` loads its skills when it starts: restart it to pick
up a change. `webagents daemon -w <folder>` reloads the agents in that folder
as their files change.
