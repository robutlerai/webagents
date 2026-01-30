# API Reference

Complete API reference for the webagents TypeScript SDK.

## Core Classes

### BaseAgent

The main agent class that orchestrates skills and processes messages.

```typescript
import { BaseAgent } from 'webagents';

const agent = new BaseAgent(config?: AgentConfig);
```

#### AgentConfig

```typescript
interface AgentConfig {
  name?: string;              // Agent name (default: 'agent')
  description?: string;       // Agent description
  instructions?: string;      // System instructions
  model?: string;            // Default model
  skills?: ISkill[];         // Skills to load
  capabilities?: Partial<Capabilities>;
}
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Agent name |
| `description` | `string \| undefined` | Agent description |

#### Methods

##### run(messages, options?)

Run the agent with messages.

```typescript
async run(
  messages: Message[],
  options?: RunOptions
): Promise<RunResponse>

// Types
interface Message {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content?: string;
  content_items?: ContentItem[];
  tool_call_id?: string;
}

interface RunOptions {
  instructions?: string;
  tools?: Tool[];
}

interface RunResponse {
  content: string;
  content_items: ContentItem[];
  usage?: UsageStats;
}
```

**Example:**
```typescript
const response = await agent.run([
  { role: 'user', content: 'Hello!' }
]);
console.log(response.content);
```

##### runStreaming(messages, options?)

Run with streaming response.

```typescript
async *runStreaming(
  messages: Message[],
  options?: RunOptions
): AsyncGenerator<StreamChunk>

type StreamChunk = 
  | { type: 'delta'; delta: string }
  | { type: 'tool_call'; tool_call: { id: string; name: string; arguments: string } }
  | { type: 'done'; response: RunResponse }
  | { type: 'error'; error: Error };
```

**Example:**
```typescript
for await (const chunk of agent.runStreaming([
  { role: 'user', content: 'Tell me a story' }
])) {
  if (chunk.type === 'delta') {
    process.stdout.write(chunk.delta);
  }
}
```

##### processUAMP(events)

Process raw UAMP events.

```typescript
async *processUAMP(
  events: ClientEvent[]
): AsyncGenerator<ServerEvent>
```

##### addSkill(skill)

Add a skill to the agent.

```typescript
addSkill(skill: ISkill): void
```

##### removeSkill(name)

Remove a skill by name.

```typescript
removeSkill(skillName: string): void
```

##### executeTool(name, params)

Execute a tool directly.

```typescript
async executeTool(
  name: string,
  params: Record<string, unknown>
): Promise<unknown>
```

##### getCapabilities()

Get agent capabilities.

```typescript
getCapabilities(): Capabilities
```

##### getToolDefinitions()

Get all tool definitions.

```typescript
getToolDefinitions(): ToolDefinition[]
```

##### initialize()

Initialize all skills.

```typescript
async initialize(): Promise<void>
```

##### cleanup()

Clean up all skills.

```typescript
async cleanup(): Promise<void>
```

##### getHttpHandler(path, method)

Get HTTP handler for a path.

```typescript
getHttpHandler(
  path: string,
  method: string
): HttpEndpoint | undefined
```

##### getWebSocketHandler(path)

Get WebSocket handler for a path.

```typescript
getWebSocketHandler(path: string): WebSocketEndpoint | undefined
```

---

### Skill

Base class for creating skills.

```typescript
import { Skill } from 'webagents';

class MySkill extends Skill {
  // ...
}
```

#### SkillConfig

```typescript
interface SkillConfig {
  name?: string;      // Skill name (default: class name)
  enabled?: boolean;  // Whether skill is enabled (default: true)
}
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Skill name |
| `enabled` | `boolean` | Whether skill is enabled |
| `tools` | `Tool[]` | Registered tools (filtered by enabled) |
| `hooks` | `Hook[]` | Registered hooks |
| `handoffs` | `Handoff[]` | Registered handoffs |
| `httpEndpoints` | `HttpEndpoint[]` | HTTP endpoints |
| `wsEndpoints` | `WebSocketEndpoint[]` | WebSocket endpoints |

#### Methods

##### initialize()

Initialize the skill. Override in subclasses.

```typescript
async initialize(): Promise<void>
```

##### cleanup()

Clean up resources. Override in subclasses.

```typescript
async cleanup(): Promise<void>
```

##### setToolEnabled(name, enabled)

Enable or disable a tool.

```typescript
setToolEnabled(name: string, enabled: boolean): void
```

##### getTool(name)

Get a tool by name.

```typescript
getTool(name: string): Tool | undefined
```

##### getHandoff(name)

Get a handoff by name.

```typescript
getHandoff(name: string): Handoff | undefined
```

---

### Context

Request context with session, auth, and payment info.

```typescript
import { createContext, ContextImpl } from 'webagents/core';

const ctx = createContext(options?: Partial<ContextOptions>);
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session` | `SessionState` | Session state |
| `auth` | `AuthInfo` | Authentication info |
| `payment` | `PaymentInfo` | Payment info |
| `metadata` | `Record<string, unknown>` | Custom metadata |
| `client_capabilities` | `Capabilities \| undefined` | Client caps |

#### Methods

##### get(key)

Get session data.

```typescript
get<T>(key: string): T | undefined
```

##### set(key, value)

Set session data.

```typescript
set(key: string, value: unknown): void
```

##### delete(key)

Delete session data.

```typescript
delete(key: string): void
```

##### hasScope(scope)

Check if context has a scope.

```typescript
hasScope(scope: string): boolean
```

##### hasScopes(scopes)

Check if context has all scopes.

```typescript
hasScopes(scopes: string[]): boolean
```

##### with(updates)

Create copy with updates.

```typescript
with(updates: Partial<ContextOptions>): Context
```

---

## Decorators

### @tool

Register a method as a tool.

```typescript
import { tool } from 'webagents';

@tool(config: ToolConfig)
async methodName(params: P, ctx: Context): Promise<R>

interface ToolConfig {
  name?: string;           // Tool name (default: method name)
  description?: string;    // Tool description
  parameters?: JSONSchema; // Parameter schema
  provides?: string;       // Capability this provides
  scopes?: string[];       // Required scopes
  enabled?: boolean;       // Whether enabled (default: true)
}
```

**Example:**
```typescript
class MySkill extends Skill {
  @tool({
    description: 'Search the web',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string' }
      },
      required: ['query']
    }
  })
  async search(params: { query: string }, ctx: Context) {
    // ...
  }
}
```

### @hook

Register a lifecycle hook.

```typescript
import { hook } from 'webagents';

@hook(config: HookConfig)
async methodName(data: HookData, ctx: Context): Promise<HookResult | void>

interface HookConfig {
  lifecycle: HookLifecycle;
  priority?: number;      // Lower = earlier (default: 50)
  enabled?: boolean;
}

type HookLifecycle = 
  | 'before_run'
  | 'after_run'
  | 'before_tool'
  | 'after_tool'
  | 'before_handoff'
  | 'after_handoff'
  | 'on_error';

interface HookData {
  messages?: Message[];
  response?: string;
  tool_name?: string;
  tool_params?: Record<string, unknown>;
  tool_result?: unknown;
  handoff_target?: string;
  error?: Error;
}

interface HookResult {
  skip_remaining?: boolean;
  abort?: boolean;
  abort_reason?: string;
  tool_params?: Record<string, unknown>;
  tool_result?: unknown;
}
```

### @handoff

Register an LLM handoff.

```typescript
import { handoff } from 'webagents';

@handoff(config: HandoffConfig)
async *methodName(
  events: ClientEvent[],
  ctx: Context
): AsyncGenerator<ServerEvent>

interface HandoffConfig {
  name: string;
  description?: string;
  priority?: number;      // Higher = preferred (default: 0)
  scopes?: string[];
  enabled?: boolean;
}
```

### @http

Register an HTTP endpoint.

```typescript
import { http } from 'webagents';

@http(config: HttpConfig)
async methodName(req: Request, ctx: Context): Promise<Response>

interface HttpConfig {
  path: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';  // default: 'GET'
  scopes?: string[];
  content_type?: string;
  enabled?: boolean;
}
```

### @websocket

Register a WebSocket endpoint.

```typescript
import { websocket } from 'webagents';

@websocket(config: WebSocketConfig)
methodName(ws: WebSocket, ctx: Context): void

interface WebSocketConfig {
  path: string;
  scopes?: string[];
  protocols?: string[];
  enabled?: boolean;
}
```

---

## UAMP Types

### Events

```typescript
import type {
  ClientEvent,
  ServerEvent,
  SessionCreateEvent,
  InputTextEvent,
  ResponseCreateEvent,
  ResponseDeltaEvent,
  ResponseDoneEvent,
  ResponseErrorEvent,
  ToolCallEvent,
  ToolResultEvent,
  ProgressEvent,
  ThinkingEvent,
} from 'webagents/uamp';
```

### Event Factories

```typescript
import {
  generateEventId,
  createBaseEvent,
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createToolResultEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
  createProgressEvent,
  parseEvent,
  serializeEvent,
  isClientEvent,
  isServerEvent,
} from 'webagents/uamp';
```

### Types

```typescript
import type {
  Modality,
  AudioFormat,
  SessionConfig,
  ContentItem,
  Capabilities,
  UsageStats,
  Message,
  ToolDefinition,
  JSONSchema,
} from 'webagents/uamp';

type Modality = 'text' | 'audio' | 'image' | 'video' | 'file';

type AudioFormat = 'pcm16' | 'g711_ulaw' | 'g711_alaw';

interface UsageStats {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  audio_input_tokens?: number;
  audio_output_tokens?: number;
  reasoning_tokens?: number;
  cost?: number;
}
```

---

## Server

### createAgentApp

Create a Hono app for an agent.

```typescript
import { createAgentApp } from 'webagents/server';

const app = createAgentApp(agent: IAgent, options?: AppOptions);

interface AppOptions {
  basePath?: string;
}
```

### serve

Start the server.

```typescript
import { serve } from 'webagents/server';

serve(app: Hono, options?: ServeOptions);

interface ServeOptions {
  port?: number;      // default: 3000
  hostname?: string;  // default: '0.0.0.0'
}
```

### createFetchHandler

Create a universal fetch handler.

```typescript
import { createFetchHandler } from 'webagents/server';

const handler = createFetchHandler(
  agent: IAgent,
  options?: HandlerOptions
);

// Returns: (request: Request) => Promise<Response>

interface HandlerOptions {
  basePath?: string;
  corsOrigin?: string;
}
```

---

## Daemon

### AgentRegistry

Manage registered agents.

```typescript
import { AgentRegistry } from 'webagents/daemon';

const registry = new AgentRegistry();

// Local agent
registry.registerLocal(agent: IAgent);

// Remote agent
registry.registerRemote(
  name: string,
  url: string,
  capabilities: Capabilities,
  source?: 'api' | 'websocket'
);

// Query
registry.get(name: string): RegisteredAgent | undefined;
registry.getAll(): RegisteredAgent[];
registry.findByCapability(cap: string): RegisteredAgent[];

// Health
registry.startHealthChecks(intervalMs?: number);
registry.stopHealthChecks();
```

### CronScheduler

Schedule agent tasks.

```typescript
import { CronScheduler } from 'webagents/daemon';

const scheduler = new CronScheduler();

// Add job
scheduler.addJob({
  id: 'daily-report',
  cron: '0 9 * * *',  // 9 AM daily
  agentName: 'report-agent',
  task: 'generate_report',
  params: { format: 'pdf' },
  enabled: true,
});

// Control
scheduler.start();
scheduler.stop();
scheduler.enableJob(id: string);
scheduler.disableJob(id: string);
scheduler.removeJob(id: string);

// Query
scheduler.getJobs(): ScheduledJob[];
scheduler.getJob(id: string): ScheduledJob | undefined;

// Events
scheduler.on('job:execute', (job) => {
  console.log('Running:', job.id);
});
```

---

## LLM Skills

### WebLLMSkill

```typescript
import { WebLLMSkill } from 'webagents/skills/llm/webllm';

interface WebLLMConfig {
  model: string;
  temperature?: number;
  maxTokens?: number;
}
```

### TransformersSkill

```typescript
import { TransformersSkill } from 'webagents/skills/llm/transformers';

interface TransformersConfig {
  model: string;
  temperature?: number;
  maxTokens?: number;
}
```

### OpenAISkill

```typescript
import { OpenAISkill } from 'webagents/skills/llm/openai';

interface OpenAIConfig {
  apiKey: string;
  model?: string;       // default: 'gpt-4o'
  baseUrl?: string;
  temperature?: number;
  maxTokens?: number;
}
```

### AnthropicSkill

```typescript
import { AnthropicSkill } from 'webagents/skills/llm/anthropic';

interface AnthropicConfig {
  apiKey: string;
  model?: string;       // default: 'claude-3-5-sonnet-20241022'
  maxTokens?: number;
}
```

### GoogleSkill

```typescript
import { GoogleSkill } from 'webagents/skills/llm/google';

interface GoogleConfig {
  apiKey: string;
  model?: string;       // default: 'gemini-1.5-pro'
}
```

### XAISkill

```typescript
import { XAISkill } from 'webagents/skills/llm/xai';

interface XAIConfig {
  apiKey: string;
  model?: string;       // default: 'grok-beta'
}
```

---

## Transport Skills

### CompletionsTransportSkill

```typescript
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';

interface CompletionsConfig {
  basePath?: string;    // default: '/v1'
}

const skill = new CompletionsTransportSkill(config);
skill.setAgent(agent);  // Required
```

### PortalTransportSkill

```typescript
import { PortalTransportSkill } from 'webagents/skills/transport/portal';

interface PortalConfig {
  portalUrl?: string;
}

const skill = new PortalTransportSkill(config);
await skill.exposeAgent(agent);
const response = await skill.callRemoteAgent(name, messages);
```

---

## Browser Skills

All browser skills extend `Skill` and provide tools via the `@tool` decorator.

```typescript
import { WakeLockSkill } from 'webagents/skills/browser/wakelock';
import { NotificationsSkill } from 'webagents/skills/browser/notifications';
import { GeolocationSkill } from 'webagents/skills/browser/geolocation';
import { StorageSkill } from 'webagents/skills/browser/storage';
import { CameraSkill } from 'webagents/skills/browser/camera';
import { MicrophoneSkill } from 'webagents/skills/browser/microphone';
```

See [Skills Guide](./skills.md) for tool details.
