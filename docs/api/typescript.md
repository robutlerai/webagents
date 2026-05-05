---
title: TypeScript SDK Reference
description: Core classes, decorators, and server functions exported by the WebAgents TypeScript SDK.
---

# TypeScript SDK Reference

> Looking for the Python SDK? See the [Python SDK Reference](./python.md). For a side-by-side cross-language overview, see the [Quickstart](../quickstart.md).

Install the SDK:

```bash
npm install webagents
```

The package targets Node.js ≥ 20 and modern browsers. ESM only.

---

## BaseAgent

The core agent class. Processes UAMP events, manages skills, and exposes `run` / `runStreaming` interfaces.

```typescript
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  instructions: 'You are a helpful assistant.',
  model: 'openai/gpt-4o',
  skills: [new MySkill()],
});
```

### Constructor

`new BaseAgent(config: AgentConfig)` — see [`AgentConfig`](../../typescript/src/core/types.ts).

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Agent display name (default `'agent'`). |
| `description` | `string` | Optional human-readable description. |
| `instructions` | `string` | System prompt prepended to every run. |
| `model` | `string` | Default LLM model (e.g. `'openai/gpt-4o'`). |
| `skills` | `ISkill[]` | Skill instances to load. |
| `capabilities` | `Partial<Capabilities>` | UAMP capabilities (modalities, audio formats, streaming). |
| `maxToolIterations` | `number` | Max agentic loop iterations (default `50`). |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(messages: Message[], options?: RunOptions) => Promise<RunResponse>` | Single-turn execution. |
| `runStreaming` | `(messages: Message[], options?: RunOptions) => AsyncGenerator<StreamChunk>` | Streaming execution. |
| `getCapabilities` | `() => Capabilities` | Get aggregated agent capabilities. |
| `addSkill` | `(skill: ISkill) => void` | Add a skill at runtime. |
| `removeSkill` | `(name: string) => void` | Remove a skill at runtime. |
| `executeTool` | `(name: string, params: Record<string, unknown>) => Promise<unknown>` | Execute a tool by name. |
| `overrideTool` | `(name: string) => void` | Mark a tool as client-executed. |

---

## Skill

Abstract base for skills. Skills bundle tools, hooks, prompts, handoffs, observers, and HTTP/WebSocket endpoints.

```typescript
import { Skill, tool, hook } from 'webagents';
import type { Context, HookData, HookResult } from 'webagents';

class WeatherSkill extends Skill {
  readonly name = 'weather';

  @tool({ description: 'Get weather for a city' })
  async getWeather(params: { city: string }, ctx: Context): Promise<string> {
    return `Weather in ${params.city}: sunny`;
  }

  @hook({ lifecycle: 'before_run' })
  async onBeforeRun(data: HookData, ctx: Context): Promise<HookResult> {
    return {};
  }
}
```

### Lifecycle

| Method | Description |
|--------|-------------|
| `initialize(agent: IAgent)` | Called when the skill is registered with an agent. |
| `setAgent(agent: IAgent)` | Set the parent agent reference (auto-called by `addSkill`). |
| `tools` / `hooks` / `handoffs` / `prompts` / `httpEndpoints` / `wsEndpoints` | Read-only registries populated by decorators. |

---

## Decorators

All decorators live in `webagents/core` and are re-exported from the package root.

### `@tool`

Register a method as an LLM-callable tool.

```typescript
@tool({
  name?: string,
  description?: string,
  parameters?: JSONSchema,
  provides?: string,
  scopes?: string[],
  enabled?: boolean,
})
```

### `@hook`

Register a lifecycle hook.

```typescript
@hook({
  lifecycle: 'before_run' | 'after_run' | 'before_toolcall' | 'after_toolcall' | 'on_error' | 'on_message' | 'on_chunk' | 'on_request' | 'on_connection',
  priority?: number,
  enabled?: boolean,
})
```

### `@prompt`

Register a dynamic system-prompt contributor. Return value is appended to the system message.

```typescript
@prompt({ priority?: number, scope?: string | string[] })
```

### `@handoff`

Declare a UAMP handoff handler. Used by the router for capability-based routing.

```typescript
@handoff({
  name: string,
  description?: string,
  priority?: number,
  scopes?: string[],
  subscribes?: (string | RegExp)[],
  produces?: string[],
})
```

Defaults: `subscribes: ['input.text']`, `produces: ['response.delta']`.

### `@observe`

Register a non-consuming event observer.

```typescript
@observe({ name: string, subscribes: (string | RegExp)[] })
```

### `@http`

Register a Hono-style HTTP endpoint.

```typescript
@http({
  path: string,
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH',
  scopes?: string[],
  content_type?: string,
  auth?: 'public' | 'session' | 'agent',
  enabled?: boolean,
})
```

### `@websocket`

Register a WebSocket endpoint.

```typescript
@websocket({
  path: string,
  scopes?: string[],
  protocols?: string[],
  enabled?: boolean,
})
```

### `@pricing`

Attach pricing metadata to a tool. The payments skill's `before_toolcall` hook reads this and pre-authorizes the charge.

```typescript
@pricing({
  creditsPerCall?: number,
  lock?: number | ((params) => number),
  settle?: (result, params) => number,
  reason?: string,
  onSuccess?: (result, ctx) => void | Promise<void>,
  onFail?: (error, ctx) => void | Promise<void>,
})
```

> `@command` and `@widget` are Python-only today. See the [parity matrix](../internal/python-typescript-parity.md) for status.

---

## Server

### `createAgentApp(agent, config?)`

Create a Hono HTTP app + WebSocket upgrade handler for an agent.

```typescript
import { createAgentApp } from 'webagents';

const { app, handleUpgrade } = createAgentApp(agent, { cors: true });
```

Returns `{ app: Hono, handleUpgrade(req, socket, head) }`.

### `serve(agent, config?)`

Start a Node.js HTTP server (Hono via `@hono/node-server`, or Bun if detected). Wires UAMP, Completions, A2A, ACP, Realtime endpoints.

```typescript
import { serve } from 'webagents';

await serve(agent, {
  port: 8080,
  hostname: '0.0.0.0',
  cors: true,
  logging: true,
});
```

### `createFetchHandler(agent, options?)`

Standard `fetch` handler for serverless / edge deployments (Cloudflare Workers, Vercel Edge, Deno Deploy).

```typescript
import { createFetchHandler } from 'webagents';

export default {
  fetch: createFetchHandler(agent),
};
```

### `WebAgentsServer`

Multi-agent server with rate limiting, extension loading, and dynamic agents.

```typescript
import { WebAgentsServer } from 'webagents';

const server = new WebAgentsServer({
  agents: [agent1, agent2],
  rateLimit: { perMinute: 60 },
});

await server.listen({ port: 8080 });
```

---

## Context

Available inside tools, hooks, prompts, and handoffs.

```typescript
interface Context {
  session: SessionState;
  auth: AuthInfo;
  payment: PaymentInfo;
  client_capabilities?: Capabilities;
  agent_capabilities?: Capabilities;
  metadata: Record<string, unknown>;
  signal?: AbortSignal;
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  hasScope(scope: string): boolean;
}
```

---

## Router

`MessageRouter` routes UAMP events through the skill graph using capability-based subscriptions.

```typescript
import { MessageRouter } from 'webagents';

const router = new MessageRouter();
router.addHandler({ subscribes: ['input.text'], produces: ['response.delta'], handler });
await router.dispatch(event, context);
```

### Sinks

Output adapters for different transports:

| Class | Description |
|-------|-------------|
| `WebSocketSink` | WebSocket transport. |
| `SSESink` | Server-Sent Events. |
| `WebStreamSink` | Web Streams API. |
| `CallbackSink` | Custom callback function. |
| `BufferSink` | In-memory buffer (testing). |

---

## Daemon

Background process manager — the runtime behind the `webagents` and `robutler` CLIs.

```typescript
import { WebAgentsDaemon } from 'webagents/daemon';

const daemon = new WebAgentsDaemon({ port: 8080, enableCron: true });
daemon.registerAgent(agent);
await daemon.start();
```

---

## UAMP Types

Key exports from `webagents/uamp`:

| Type | Description |
|------|-------------|
| `Capabilities` | Agent capabilities declaration. |
| `Modality` | `'text' | 'audio' | 'image' | 'video' | 'file'`. |
| `AudioFormat` | Audio codec hint (`'pcm16'`, `'opus'`, …). |
| `Message` | UAMP message envelope. |
| `ContentItem` | Multimodal content item (text/audio/image/video/file). |
| `ToolDefinition` | Tool schema for LLM. |
| `ToolCall` | Tool invocation record. |
| `UsageStats` | Token usage statistics. |
| `ClientEvent` / `ServerEvent` | Event envelopes for the realtime/UAMP transports. |

---

## Crypto

JWKS / JWT utilities for AOAuth.

```typescript
import { JWKSManager } from 'webagents';

const jwks = new JWKSManager({ jwksCacheTtl: 3600 });
const payload = await jwks.verifyJwt(token);
```

---

## Skill Imports

Per-skill subpath exports keep tree-shaking honest:

```typescript
import { AuthSkill } from 'webagents/skills/auth';
import { PaymentSkill } from 'webagents/skills/payments';
import { PortalDiscoverySkill } from 'webagents/skills/discovery';
import { NLISkill } from 'webagents/skills/nli';
import { OpenAPISkill } from 'webagents/skills/openapi';
import { MCPSkill } from 'webagents/skills/mcp';
import { SlackSkill } from 'webagents/skills/messaging/slack';
```

The complete subpath catalogue is declared under `exports` in [`webagents/typescript/package.json`](../../typescript/package.json).
