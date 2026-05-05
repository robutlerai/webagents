---
title: Agent Overview
description: How BaseAgent works — the skill-based architecture behind every WebAgent.
---

# Agent Overview

`BaseAgent` is the core class for creating AI agents in WebAgents. It uses a flexible, skill-based architecture so you can add exactly the capabilities you need. Agents speak the OpenAI Chat Completions dialect natively, so existing clients work out of the box. The [skill system](../skills/overview.md) adds platform features like [authentication](../skills/platform/auth.md), [payments](../skills/platform/payments.md), [discovery](../skills/platform/discovery.md), and multi-agent collaboration.

- Build an agent with a few lines of code
- Add capabilities via skills (tools, hooks, prompts, handoffs)
- Serve OpenAI-compatible endpoints with a single function call

## Creating Agents

### Basic Agent

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'my-assistant',
  instructions: 'You are a helpful assistant',
  model: 'openai/gpt-4o',
});
```

```python tab="Python"
from webagents import BaseAgent

agent = BaseAgent(
    name="my-assistant",
    instructions="You are a helpful assistant",
    model="openai/gpt-4o",
)
```

**New to WebAgents?** Check out the [Quickstart](../quickstart.md) for a complete walkthrough.

### Agent with Skills

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { PortalDiscoverySkill } from 'webagents/skills/discovery';
import { SessionSkill } from 'webagents/skills/session';

const agent = new BaseAgent({
  name: 'advanced-assistant',
  instructions: 'You are an advanced assistant with memory',
  model: 'openai/gpt-4o',
  skills: [
    new SessionSkill({ maxMessages: 50 }),
    new PortalDiscoverySkill(),
  ],
});
```

```python tab="Python"
from webagents import BaseAgent
from webagents.agents.skills.core.memory.skill import ShortTermMemorySkill
from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill

agent = BaseAgent(
    name="advanced-assistant",
    instructions="You are an advanced assistant with memory",
    model="openai/gpt-4o",
    skills={
        "memory": ShortTermMemorySkill({"max_messages": 50}),
        "discovery": DiscoverySkill(),
    },
)
```

> Explore available skills in the [Skills Overview](../skills/overview.md) or learn to [create custom skills](../skills/custom.md).

## Smart Model Parameter

The `model` parameter accepts a provider-prefixed string. The correct LLM skill is provisioned automatically.

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { OpenAILLMSkill } from 'webagents/skills/llm';

new BaseAgent({ model: 'openai/gpt-4o' });        // OpenAI GPT-4o
new BaseAgent({ model: 'anthropic/claude-3-5' }); // Anthropic Claude
new BaseAgent({ model: 'xai/grok-2' });           // xAI Grok
new BaseAgent({ model: 'google/gemini-1.5-pro' });

new BaseAgent({
  skills: [
    new OpenAILLMSkill({
      apiKey: process.env.OPENAI_API_KEY,
      defaultModel: 'gpt-4o',
      temperature: 0.7,
    }),
  ],
});
```

```python tab="Python"
from webagents import BaseAgent
from webagents.agents.skills.core.llm.openai.skill import OpenAISkill

BaseAgent(model="openai/gpt-4o")        # OpenAI GPT-4o
BaseAgent(model="anthropic/claude-3")   # Anthropic Claude
BaseAgent(model="litellm/gpt-4")        # Via LiteLLM proxy
BaseAgent(model="xai/grok-beta")        # xAI Grok

BaseAgent(model=OpenAISkill({
    "api_key": "sk-...",
    "temperature": 0.7,
}))
```

See [LLM Skills](../skills/core/llm.md) for more configuration options.

## Running Agents

### Basic Conversation

```typescript tab="TypeScript"
const response = await agent.run([
  { role: 'user', content: 'Hello!' },
]);
console.log(response.content);
```

```python tab="Python"
response = await agent.run([
    {"role": "user", "content": "Hello!"}
])
print(response.choices[0].message.content)
```

### Streaming Response

```typescript tab="TypeScript"
for await (const chunk of agent.runStreaming([
  { role: 'user', content: 'Tell me a story' },
])) {
  process.stdout.write(chunk.delta ?? '');
}
```

```python tab="Python"
async for chunk in agent.run_streaming([
    {"role": "user", "content": "Tell me a story"}
]):
    print(chunk.choices[0].delta.content, end="")
```

### With Tools

Attach additional tools per request using the OpenAI function-calling format:

```typescript tab="TypeScript"
const response = await agent.run(
  [{ role: 'user', content: 'Calculate 42 * 17' }],
  {
    tools: [
      {
        type: 'function',
        function: {
          name: 'calculator',
          description: 'Calculate math expressions',
          parameters: { /* JSON schema */ },
        },
      },
    ],
  },
);
```

```python tab="Python"
response = await agent.run(
    messages=[{"role": "user", "content": "Calculate 42 * 17"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {...},
        },
    }],
)
```

Learn more about [creating tools](./tools.md) and the [OpenAI function calling format](https://platform.openai.com/docs/guides/function-calling).

## Agent Capabilities

### Skills

Skills provide modular capabilities:

- **[LLM Skills](../skills/core/llm.md)** — Language model providers (OpenAI, Anthropic, Google, xAI, …)
- **[Memory / Storage Skills](../skills/core/memory.md)** — Conversation persistence and context management
- **[Platform Skills](../skills/platform/auth.md)** — Robutler platform integration (auth, payments, discovery)
- **[Ecosystem Skills](../skills/ecosystem/index.md)** — Third-party integrations (OpenAI workflows, database, n8n)

### Tools

Tools are executable functions that extend agent capabilities:

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @tool({ description: 'Tool description' })
  async myFunction(params: { value: string }): Promise<string> {
    return `Result: ${params.value}`;
  }
}
```

```python tab="Python"
from webagents import Skill, tool

class MySkill(Skill):
    @tool
    def my_function(self, param: str) -> str:
        """Tool description"""
        return f"Result: {param}"
```

See [Tools](./tools.md) for examples and best practices.

### Hooks

Lifecycle hooks enable event-driven behavior during request processing:

```typescript tab="TypeScript"
import { Skill, hook } from 'webagents';
import type { HookData, Context } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @hook({ lifecycle: 'on_message' })
  async processMessage(data: HookData, ctx: Context) {
    return data;
  }
}
```

```python tab="Python"
from webagents import Skill, hook

class MySkill(Skill):
    @hook("on_message")
    async def process_message(self, context):
        """Process each message"""
        return context
```

Learn about [hooks](./hooks.md) and the [agent lifecycle](./lifecycle.md).

### Handoffs

Handoffs enable agents to delegate completions to specialized handlers or remote agents:

```typescript tab="TypeScript"
import { Skill, handoff } from 'webagents';
import type { ClientEvent } from 'webagents';

class SpecializedSkill extends Skill {
  readonly name = 'specialized';

  @handoff({
    name: 'math_expert',
    description: 'Use for advanced mathematical problems',
    priority: 15,
  })
  async *mathCompletion(events: ClientEvent[]) {
    for await (const chunk of this.specializedMathLLM(events)) {
      yield chunk;
    }
  }
}
```

```python tab="Python"
from webagents import Skill, handoff

class SpecializedSkill(Skill):
    @handoff(
        name="math_expert",
        prompt="Use for advanced mathematical problems",
        priority=15,
    )
    async def math_completion(self, messages, tools=None, **kwargs):
        """Handle math-focused completions"""
        async for chunk in self.specialized_math_llm(messages):
            yield chunk
```

Explore [handoff patterns](./handoffs.md), [agent discovery](../skills/platform/discovery.md), and [remote agent communication](../skills/platform/nli.md).

## Context Management

> Agents maintain a unified context object throughout execution. Skills read and write to this structure — `contextvars` in Python, an explicit `Context` parameter in TypeScript — and both are async-safe.

```typescript tab="TypeScript"
import { tool } from 'webagents';
import type { Context } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @tool({ description: 'Inspect context' })
  async whoami(_: Record<string, never>, ctx: Context) {
    return {
      userId: ctx.auth?.userId,
      streaming: ctx.metadata.stream === true,
    };
  }
}
```

```python tab="Python"
context = self.get_context()
user_id = context.peer_user_id
messages = context.messages
streaming = context.stream
```

## Agent Registration

Register agents with the server to make them available via HTTP endpoints:

```typescript tab="TypeScript"
import { serve } from 'webagents';

await serve(agent, { port: 8000 });

// Or compose multiple agents:
import { WebAgentsServer } from 'webagents';
const server = new WebAgentsServer({ agents: [agent1, agent2] });
await server.listen({ port: 8000 });
```

```python tab="Python"
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
# server = create_server(agents=[agent1, agent2])

uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Learn about [server deployment](../server/index.md), [dynamic agents](../server/dynamic-agents.md), and [server architecture](../server/architecture.md).

## Best Practices

1. **Start Simple** — Begin with a basic agent, add skills as you go.
2. **Use Dependencies** — Some skills auto-require others (e.g. [payments](../skills/platform/payments.md) depends on [auth](../skills/platform/auth.md)).
3. **Scope Appropriately** — Use tool scopes (`scope`/`scopes`) for access control.
4. **Test Thoroughly** — Treat skills as units; test hooks and tools independently.
5. **Monitor Performance** — Track usage and latency. Payments will use `context.usage`.

## Next Steps

- **[Quickstart](../quickstart.md)** — Build your first agent in 5 minutes
- **[Skills](../skills/overview.md)** — Explore available skills and create custom ones
- **[Agent Lifecycle](./lifecycle.md)** — Understand the complete request processing flow
- **[Server Deployment](../server/index.md)** — Deploy your agents to production
- **[Contributing](../developers/contributing.md)** — Contribute to the WebAgents ecosystem
