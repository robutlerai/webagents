---
title: TypeScript SDK Reference
description: Core classes, decorators, and server functions exported by the WebAgents TypeScript SDK.
---

# TypeScript SDK Reference

Install the SDK:

```bash
npm install webagents
```

---

## BaseAgent

The core agent class. Processes UAMP events, manages skills, and exposes run methods.

```typescript
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  instructions: 'You are a helpful assistant.',
  model: 'gpt-4o',
  skills: [new MySkill()],
});
```

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Agent display name |
| `instructions` | `string` | System prompt |
| `model` | `string` | LLM model identifier |
| `skills` | `Skill[]` | Array of skill instances |
| `scopes` | `string[]` | Required auth scopes |
| `capabilities` | `Capabilities` | UAMP capabilities (modalities, audio formats) |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(messages: Message[], options?: RunOptions) => Promise<RunResponse>` | Single-turn execution |
| `runStreaming` | `(messages: Message[], options?: RunOptions) => AsyncGenerator<StreamChunk>` | Streaming execution |
| `getCapabilities` | `() => Capabilities` | Get agent capabilities |
| `addSkill` | `(skill: ISkill) => void` | Add a skill at runtime |
| `removeSkill` | `(skillName: string) => void` | Remove a skill at runtime |
| `executeTool` | `(name: string, params: Record<string, unknown>) => Promise<unknown>` | Execute a tool by name |
| `overrideTool` | `(name: string) => void` | Mark a tool as client-executed |

---

## Skill

Abstract base class for skills. Skills bundle tools, hooks, handoffs, and endpoints.

```typescript
import { Skill, tool, hook } from 'webagents';

class WeatherSkill extends Skill {
  @tool({ description: 'Get weather for a city' })
  async getWeather(city: string): Promise<string> {
    return `Weather in ${city}: sunny`;
  }

  @hook({ lifecycle: 'before_run' })
  async onBeforeRun(data: HookData): Promise<HookResult> {
    return {};
  }
}
```

---

## Decorators

### @tool

Register a method as an LLM-callable tool.

```typescript
@tool({ name?: string, description?: string, schema?: ZodSchema })
```

### @hook

Register a lifecycle hook.

```typescript
@hook({ lifecycle: 'before_run' | 'after_run' | 'before_tool' | 'after_tool' })
```

### @handoff

Register a handoff handler for multi-agent delegation.

```typescript
@handoff({ name?: string, description?: string, subscribes?: string[], produces?: string[] })
```

### @observe

Register a non-consuming event observer.

```typescript
@observe({ subscribes: string[] })
```

### @http

Register an HTTP endpoint on the agent server.

```typescript
@http({ path: string, method?: 'get' | 'post' | 'put' | 'delete' })
```

### @websocket

Register a WebSocket endpoint.

```typescript
@websocket({ path: string })
```

---

## Server

### createAgentApp

Creates a Hono HTTP application for an agent.

```typescript
import { createAgentApp } from 'webagents';

const { app, handleUpgrade } = createAgentApp(agent, { cors: true });
```

### serve

Starts an HTTP server for the agent (Node.js). Combines `createAgentApp` with listening.

```typescript
import { serve } from 'webagents';

await serve(agent, { port: 8080 });
```

### createFetchHandler

Creates a standard `fetch` handler for serverless/edge deployments.

```typescript
import { createFetchHandler } from 'webagents';

const handler = createFetchHandler(agent);
export default { fetch: handler };
```

---

## Context

The `Context` object is available inside tools, hooks, and handoffs. It carries session state, auth info, and payment info.

```typescript
interface Context {
  session: SessionState;
  auth: AuthInfo;
  payment: PaymentInfo;
  metadata: Record<string, unknown>;
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  hasScope(scope: string): boolean;
}
```

---

## MessageRouter

Routes UAMP messages through the agent pipeline.

```typescript
import { MessageRouter } from 'webagents';

const router = new MessageRouter();
router.route('response.delta', 'myHandler');
await router.send(event, context);
```

### Sinks

Output adapters for different transports:

| Class | Description |
|-------|-------------|
| `WebSocketSink` | WebSocket transport |
| `SSESink` | Server-Sent Events |
| `WebStreamSink` | Web Streams API |
| `CallbackSink` | Custom callback function |
| `BufferSink` | In-memory buffer |

---

## Daemon

Background process manager for agent lifecycle.

```typescript
import { WebAgentsDaemon } from 'webagents';

const daemon = new WebAgentsDaemon({ port: 8080, enableCron: true });
daemon.registerAgent(agent);
await daemon.start();
```

---

## UAMP Types

Key types for the Universal Agentic Message Protocol:

| Type | Description |
|------|-------------|
| `Capabilities` | Agent capabilities declaration |
| `Modality` | Supported modalities (text, image, audio) |
| `Message` | UAMP message envelope |
| `ToolDefinition` | Tool schema for LLM |
| `ContentItem` | Multimodal content item |
| `UsageStats` | Token usage statistics |

---

## Crypto

JWT and JWKS utilities for agent authentication.

```typescript
import { JWKSManager } from 'webagents';

const jwks = new JWKSManager({ jwksCacheTtl: 3600 });
const payload = await jwks.verifyJwt(token);
```
