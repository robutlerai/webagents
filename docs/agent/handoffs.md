---
title: Agent Handoffs
description: Unified completion handlers — local LLM skills, remote agent delegation, streaming adaptation, and dynamic invocation.
---

# Agent Handoffs

The handoff system provides a unified interface for both local LLM completions and remote agent handoffs, with automatic streaming support and priority-based handler selection.

Handoffs enable seamless completion handling that supports:

- **Local LLM completions** (OpenAI, Anthropic, Google, xAI, Fireworks, …)
- **Remote agent handoffs** — delegate to specialized agents with full streaming support
- **Automatic streaming / non-streaming adaptation**
- **Priority-based handler selection**
- **Dynamic prompt injection**

## Overview

The handoff system is decorator-based. Mark a skill method with `@handoff` and the agent registers it as a completion handler.

```typescript tab="TypeScript"
import { Skill, handoff } from 'webagents';
import type { ClientEvent } from 'webagents';

class CustomLLMSkill extends Skill {
  readonly name = 'custom-llm';

  @handoff({
    name: 'custom_llm',
    description: 'Custom LLM using a specialized model',
    priority: 10,
    subscribes: ['input.text'],
    produces: ['response.delta'],
  })
  async *chatCompletionStream(events: ClientEvent[]) {
    for await (const chunk of this.myStreamingLlmApi(events)) {
      yield chunk;
    }
  }

  private async *myStreamingLlmApi(_: ClientEvent[]) {
    yield { type: 'response.delta', delta: 'hello' } as const;
  }
}
```

```python tab="Python"
from typing import Any, AsyncGenerator, Dict, List, Optional
from webagents import Skill
from webagents.agents.skills.base import Handoff

class CustomLLMSkill(Skill):
    """Custom LLM completion handler"""

    async def initialize(self, agent):
        # Register the streaming function for best compatibility
        agent.register_handoff(
            Handoff(
                target="custom_llm",
                description="Custom LLM using specialized model",
                scope="all",
                metadata={
                    "function": self.chat_completion_stream,
                    "priority": 10,
                    "is_generator": True,
                },
            ),
            source="custom_llm",
        )

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle LLM completion (streaming)"""
        async for chunk in self.my_streaming_llm_api(messages, tools):
            yield chunk
```

## Core Concepts

### Handoff Configuration

`@handoff(...)` accepts:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Handler identifier (required). |
| `description` / `prompt` | `string` | Description / dynamic prompt fragment for the LLM. |
| `priority` | `number` | Lower runs first (default `0` in TS, varies in Python). |
| `scope` / `scopes` | `string \| string[]` | Required caller scopes. |
| `subscribes` | `(string \| RegExp)[]` | Event types this handler consumes (TS). |
| `produces` | `string[]` | Event types this handler emits (TS). |

### Priority System

Handoffs are selected based on priority (lower = higher priority):

- **Priority 10** — Local LLM handlers (default)
- **Priority 20** — Remote agent handlers
- **Priority 50+** — Custom / specialized handlers

The **first registered handoff** (lowest priority) becomes the **default completion handler**.

### Streaming vs Non-Streaming

The system automatically adapts handlers:

- **Async generators** = streaming native
- **Regular async functions** = non-streaming native
- **Automatic adaptation** in both directions

## Dynamic Handoff Invocation

Skills can let the LLM explicitly route to a specific handoff during a conversation, enabling dynamic switching between handlers.

### Using `request_handoff`

```typescript tab="TypeScript"
import { Skill, tool, handoff } from 'webagents';

class SpecialistSkill extends Skill {
  readonly name = 'specialist';

  @handoff({ name: 'specialist', description: 'Specialized handler', priority: 15 })
  async *specialistHandler(events) {
    for await (const chunk of this.processWithSpecialist(events)) {
      yield chunk;
    }
  }

  @tool({ description: 'Switch to specialist for advanced queries' })
  async useSpecialist(): Promise<string> {
    return this.requestHandoff('specialist');
  }

  private async *processWithSpecialist(_: unknown) {
    yield { type: 'response.delta', delta: 'specialist response' } as const;
  }
}
```

```python tab="Python"
from webagents import Skill, tool, handoff

class SpecialistSkill(Skill):
    @handoff(name="specialist", prompt="Specialized handler", priority=15)
    async def specialist_handler(self, messages, tools, **kwargs):
        async for chunk in process_with_specialist(messages):
            yield chunk

    @tool(description="Switch to specialist for advanced queries")
    async def use_specialist(self) -> str:
        return self.request_handoff("specialist")
```

When the LLM calls `use_specialist()`, the framework:

1. Detects the handoff request marker.
2. Finds the registered `specialist` handoff.
3. Executes it with the current conversation.
4. Streams the response directly to the user.

This works with both local and remote handoffs, enabling the LLM to route requests to the most appropriate handler at runtime.

### Handoff Chaining and Default Reset

When a dynamic handoff is invoked:

1. The handoff executes with the current conversation context.
2. Streaming is continuous — the response streams directly to the user.
3. The agent resets to the default handoff after the turn completes.
4. The next user message uses the default handoff again (unless another dynamic handoff is requested).

```text
Turn 1: User: "Use specialist"
  → LLM calls use_specialist() → specialist handoff executes → response streams
  → After turn ends, active_handoff resets to default (e.g., openai)

Turn 2: User: "What about this?"
  → Uses default handoff (openai) again
```

**Handoff chaining** is also supported — a handoff can request another handoff during execution, allowing multi-stage processing within a single turn.

## Using the `@handoff` Decorator

### Basic Handoff with Prompt

```typescript tab="TypeScript"
import { Skill, handoff } from 'webagents';

class SpecializedSkill extends Skill {
  readonly name = 'specialized';

  @handoff({
    name: 'specialist',
    description: 'Use this handler for complex symbolic math',
    priority: 15,
  })
  async specializedCompletion(events) {
    return await this.processWithSpecialist(events);
  }

  private async processWithSpecialist(_: unknown) {
    return { content: 'symbolic math result' };
  }
}
```

```python tab="Python"
from webagents import handoff

class SpecializedSkill(Skill):
    @handoff(
        name="specialist",
        prompt="Use this handler for complex mathematical computations requiring symbolic processing",
        priority=15,
        provides="symbolic_math",  # Capability for discovery
    )
    async def specialized_completion(
        self,
        messages,
        tools=None,
        context=None,  # Auto-injected if present in signature
        **kwargs,
    ):
        """Handle specialized completions"""
        result = await self.process_with_specialist(messages)
        return result
```

### Streaming Handoff

For streaming responses, use an async generator:

```typescript tab="TypeScript"
class StreamingSkill extends Skill {
  readonly name = 'streaming';

  @handoff({
    name: 'streaming_llm',
    description: 'Streaming LLM handler for real-time responses',
    priority: 10,
  })
  async *streamingCompletion(events) {
    for await (const chunk of this.myStreamingApi(events)) {
      yield chunk;
    }
  }

  private async *myStreamingApi(_: unknown) {
    yield { type: 'response.delta', delta: 'streaming chunk' } as const;
  }
}
```

```python tab="Python"
class StreamingSkill(Skill):
    @handoff(
        name="streaming_llm",
        prompt="Streaming LLM handler for real-time responses",
        priority=10,
    )
    async def streaming_completion(
        self, messages, tools=None, **kwargs,
    ):
        """Stream LLM responses"""
        async for chunk in self.my_streaming_api(messages, tools):
            yield chunk
```

### Context Injection

In Python, the decorator automatically injects `context` if it's in your function signature. In TypeScript, the runtime always passes a `Context` argument to handoffs.

```typescript tab="TypeScript"
@handoff({ name: 'context_aware', priority: 10 })
async contextAware(events, ctx) {
  const userId = ctx.auth?.userId;
  return await this.process(events, userId);
}
```

```python tab="Python"
@handoff(name="context_aware", priority=10)
async def completion_with_context(
    self, messages, context=None, **kwargs,
):
    """Use context for billing, auth, etc."""
    user_id = context.auth.user_id if context else None
    return await self.process(messages, user_id=user_id)
```

## Built-in Handoff Skills

### Native LLM Skills (Default)

Native LLM skills automatically register as handoff handlers during initialization. Available skills:

- TypeScript: `OpenAILLMSkill`, `AnthropicLLMSkill`, `GoogleLLMSkill`, `XAILLMSkill`, `FireworksLLMSkill`, `WebLLMSkill`, `TransformersLLMSkill`, `ProxyLLMSkill`.
- Python: `OpenAISkill`, `AnthropicSkill`, `GoogleAISkill`, `XAISkill`, `FireworksAISkill`.

```typescript tab="TypeScript"
import { OpenAILLMSkill } from 'webagents/skills/llm';

const agent = new BaseAgent({
  name: 'assistant',
  skills: [new OpenAILLMSkill({ defaultModel: 'gpt-4o' })],
});
```

```python tab="Python"
from webagents.agents.skills.core.llm.openai import OpenAISkill

skills = {"openai": OpenAISkill(model="gpt-4o")}
# OpenAISkill.initialize() automatically calls agent.register_handoff(...)
# with the streaming function and priority=10.
```

## Remote Agent Handoffs

The remote-handoff skill enables seamless delegation to remote agents via NLI with full streaming. This is essential for multi-agent systems where you delegate tasks to specialists.

### Basic Setup

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { NLISkill } from 'webagents/skills/nli';
import { DynamicRoutingSkill } from 'webagents/skills/routing';

const agent = new BaseAgent({
  name: 'coordinator',
  instructions: 'Coordinate with specialist agents',
  skills: [new NLISkill(), new DynamicRoutingSkill()],
});
```

```python tab="Python"
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill
from webagents.agents.skills.robutler.nli import NLISkill

skills = {
    "nli": NLISkill(),
    "agent_handoff": AgentHandoffSkill(),
}

agent = BaseAgent(
    name="coordinator",
    instructions="Coordinate with specialist agents",
    skills=skills,
)
```

> The TypeScript SDK uses `DynamicRoutingSkill` to discover and delegate to remote agents. A dedicated `AgentHandoffSkill` is on the roadmap — see the [parity matrix](../internal/python-typescript-parity.md).

### Default Agent Configuration

Configure a default agent URL that will be used automatically:

```typescript tab="TypeScript"
import { DynamicRoutingSkill } from 'webagents/skills/routing';

const skills = [
  new DynamicRoutingSkill({
    agentUrl: 'https://robutler.ai/agents/specialist',
  }),
];
```

```python tab="Python"
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill

skills["agent_handoff"] = AgentHandoffSkill({
    "agent_url": "https://robutler.ai/agents/specialist",
})
```

### Calling Remote Agents

Hand off to specific agents using their full URL (which includes the agent ID):

```typescript tab="TypeScript"
const handoff = agent.skills.find((s) => s.name === 'dynamic-routing') as DynamicRoutingSkill;

for await (const chunk of handoff.remoteAgentHandoff({
  agentUrl: 'https://robutler.ai/agents/96f6d0ab-71d4-4035-a71d-94d1c2b72da3',
  messages,
  tools,
})) {
  yield chunk;
}
```

```python tab="Python"
async for chunk in agent.skills["agent_handoff"].remote_agent_handoff(
    agent_url="https://robutler.ai/agents/96f6d0ab-71d4-4035-a71d-94d1c2b72da3",
    messages=messages,
    tools=tools,
):
    yield chunk
```

### Dynamic Agent Discovery

You can programmatically discover and call agents:

```typescript tab="TypeScript"
import { Skill, tool } from 'webagents';

class CoordinatorSkill extends Skill {
  readonly name = 'coordinator';

  @tool({ description: 'Delegate complex music tasks to the music specialist' })
  async delegateToMusicAgent(params: { task: string }): Promise<string> {
    const musicAgentUrl = 'https://robutler.ai/agents/96f6d0ab-71d4-4035-a71d-94d1c2b72da3';
    return this.requestHandoff('dynamic-routing', { agentUrl: musicAgentUrl });
  }
}
```

```python tab="Python"
from webagents import Skill, tool

class CoordinatorSkill(Skill):
    @tool(description="Delegate complex music tasks to the music specialist")
    async def delegate_to_music_agent(self, task: str) -> str:
        """Hand off music-related tasks to r-music agent"""
        music_agent_url = "https://robutler.ai/agents/96f6d0ab-71d4-4035-a71d-94d1c2b72da3"
        return self.request_handoff("agent_handoff", agent_url=music_agent_url)
```

### How It Works

1. **Automatic registration** — the remote-handoff skill registers itself with `priority=20` during initialization.
2. **NLI communication** — uses the NLI skill's stream API for SSE streaming from remote agents.
3. **OpenAI compatibility** — returns OpenAI-compatible streaming chunks.
4. **Tool support** — remote agents can use their own tools and skills.
5. **Payment integration** — supports payment-token authorization for paid agents.

> Agent URLs must include the full agent ID: `https://robutler.ai/agents/{agent-id}`. You can find agent IDs in the portal or via the agents API.

> Remote handoffs always stream responses using SSE (Server-Sent Events), providing real-time feedback even for long-running operations.

## Manual Handoff Registration

You can also register handoffs manually without decorators (Python):

```typescript tab="TypeScript"
// In TypeScript, use @handoff on a class method. Manual registration without
// decorators is not exposed; the metadata-based decorator is the supported path.
```

```python tab="Python"
from webagents.agents.skills.base import Handoff

class MySkill(Skill):
    async def initialize(self, agent):
        agent.register_handoff(
            Handoff(
                target="my_handler",
                description="My custom completion handler",
                scope="owner",
                metadata={
                    "function": self.my_completion_handler,
                    "priority": 15,
                    "is_generator": False,  # Non-streaming
                },
            ),
            source="my_skill",
        )

    async def my_completion_handler(self, messages, tools=None, **kwargs):
        return await self.process(messages)
```

## Dynamic Prompt Integration

Handoff descriptions automatically integrate with the agent's system prompt:

```typescript tab="TypeScript"
@handoff({
  name: 'math_expert',
  description: 'Use this handler for advanced mathematical problems requiring symbolic computation, calculus, or theorem proving',
  priority: 15,
})
async mathCompletion(events) {
  return await this.mathEngine.solve(events);
}
```

```python tab="Python"
@handoff(
    name="math_expert",
    prompt="Use this handler for advanced mathematical problems requiring symbolic computation, calculus, or theorem proving",
    priority=15,
)
async def math_completion(self, messages, **kwargs):
    return await self.math_engine.solve(messages)
```

The `description` (TS) / `prompt` (Python) parameter serves dual purposes:

1. **Description** — explains when this handoff should be used.
2. **Dynamic prompt** — added to the agent's system prompt automatically.

## Best Practices

1. **Priority selection**
   - Reserve 1–10 for critical / high-priority handlers.
   - Use 10–20 for standard local / remote handlers.
   - Use 20+ for specialized / conditional handlers.
2. **Streaming support**
   - Use async generators for streaming-native handlers.
   - Don't mix streaming and non-streaming behaviour in the same function.
3. **Context usage**
   - Use the injected `Context` for auth, billing, and user preferences.
   - Don't mutate context across requests; treat it as request-scoped state.
4. **Error handling**
   - Always handle errors in custom handoffs and provide fallback responses.
   - Log failures for debugging.
5. **Prompt clarity**
   - Make handoff descriptions specific and actionable.
   - Include examples of suitable queries.

## Quick Start Example

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { OpenAILLMSkill } from 'webagents/skills/llm';
import { NLISkill } from 'webagents/skills/nli';
import { DynamicRoutingSkill } from 'webagents/skills/routing';

const agent = new BaseAgent({
  name: 'coordinator',
  instructions: 'Coordinate tasks and hand off to specialists when needed',
  skills: [
    new OpenAILLMSkill({ defaultModel: 'gpt-4o' }),
    new NLISkill(),
    new DynamicRoutingSkill(),
  ],
});
```

```python tab="Python"
from webagents.agents import BaseAgent
from webagents.agents.skills.core.llm.openai import OpenAISkill
from webagents.agents.skills.robutler.nli import NLISkill
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill

agent = BaseAgent(
    name="coordinator",
    instructions="Coordinate tasks and hand off to specialists when needed",
    skills={
        "openai": OpenAISkill(model="gpt-4o"),
        "nli": NLISkill(),
        "agent_handoff": AgentHandoffSkill(),
    },
)
```
