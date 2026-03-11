# TypeScript API Reference

## Core Classes

### BaseAgent

The primary agent class. Composes skills to create a capable agent.

```typescript
import { BaseAgent, OpenAISkill } from 'webagents';

const agent = new BaseAgent({
  name: 'my-agent',
  instructions: 'You are a helpful assistant.',
  model: 'gpt-4o',
  skills: [new OpenAISkill({ model: 'gpt-4o' })],
  maxToolIterations: 15,
});

await agent.initialize();
const response = await agent.run([{ role: 'user', content: 'Hello!' }]);
```

### Skill

Base class for all skills. Use decorators to define tools, hooks, handoffs, and endpoints.

```typescript
import { Skill, tool, hook } from 'webagents';

class MySkill extends Skill {
  @tool({ name: 'greet', description: 'Greet a user' })
  async greet(params: { name: string }) {
    return `Hello, ${params.name}!`;
  }

  @hook({ lifecycle: 'before_run' })
  async logRun(data, context) {
    console.log('Agent run starting');
  }
}
```

## Skills Reference

### LLM Skills

| Skill | Provider | Import |
|-------|----------|--------|
| `OpenAISkill` | OpenAI / compatible | `webagents` |
| `AnthropicSkill` | Anthropic | `webagents` |
| `GoogleSkill` | Google Gemini | `webagents` |
| `XAISkill` | xAI Grok | `webagents` |
| `WebLLMSkill` | In-browser (WebGPU) | `webagents` |
| `TransformersSkill` | In-browser (WASM) | `webagents` |

### Transport Skills

| Skill | Protocol | Import |
|-------|----------|--------|
| `CompletionsTransportSkill` | OpenAI Completions | `webagents` |
| `PortalTransportSkill` | Portal WebSocket | `webagents` |
| `UAMPTransportSkill` | UAMP WebSocket | `webagents` |
| `RealtimeTransportSkill` | Realtime Audio WS | `webagents` |
| `A2ATransportSkill` | Agent-to-Agent (Google) | `webagents` |
| `ACPTransportSkill` | Agent Commerce Protocol | `webagents` |

### Platform Skills

| Skill | Capability | Import |
|-------|-----------|--------|
| `NLISkill` | Agent-to-agent NLI | `webagents` |
| `PortalDiscoverySkill` | Intent-based discovery | `webagents` |
| `DynamicRoutingSkill` | Cross-agent routing | `webagents` |
| `AuthSkill` | JWT/JWKS auth | `webagents` |
| `PaymentSkill` | x402 payments | `webagents` |

### Storage Skills

| Skill | Capability | Import |
|-------|-----------|--------|
| `RobutlerKVSkill` | Key-value storage | `webagents` |
| `RobutlerJSONSkill` | JSON document storage | `webagents` |
| `RobutlerFilesSkill` | File storage | `webagents` |

### Local Skills

| Skill | Capability | Import |
|-------|-----------|--------|
| `FilesystemSkill` | Sandboxed file ops | `webagents` |
| `ShellSkill` | Sandboxed shell | `webagents` |
| `MCPSkill` | MCP client | `webagents` |
| `SessionSkill` | Conversational state | `webagents` |
| `CheckpointSkill` | File snapshots | `webagents` |
| `TodoSkill` | Task management | `webagents` |
| `RAGSkill` | Vector search | `webagents` |
| `SandboxSkill` | Docker execution | `webagents` |
| `PluginSkill` | Dynamic skill loading | `webagents` |

### Social Skills

| Skill | Capability | Import |
|-------|-----------|--------|
| `ChatsSkill` | Chat management | `webagents` |
| `NotificationsSkill` | Push notifications | `webagents` |
| `PublishSkill` | Feed publishing | `webagents` |
| `PortalConnectSkill` | Portal registration | `webagents` |
| `PortalWSSkill` | Portal WebSocket | `webagents` |

## Server

### WebAgentsServer (Multi-Agent)

```typescript
import { WebAgentsServer, BaseAgent, OpenAISkill } from 'webagents';

const server = new WebAgentsServer({
  port: 3000,
  metricsPath: '/metrics',
  rateLimit: { maxRequests: 100, windowMs: 60000 },
});

const agent = new BaseAgent({ name: 'assistant', skills: [new OpenAISkill()] });
await server.addAgent('assistant', agent);
await server.start();
```

### Single Agent Server

```typescript
import { serve, BaseAgent, OpenAISkill } from 'webagents';

const agent = new BaseAgent({ name: 'agent', skills: [new OpenAISkill()] });
await serve(agent, { port: 3000 });
```

## Daemon

```typescript
import { WebAgentsDaemon } from 'webagents';

const daemon = new WebAgentsDaemon({
  port: 8080,
  watchDir: './agents',
  cron: true,
});
await daemon.start();
```

## CLI

```bash
webagents chat                    # Interactive chat
webagents serve                   # Serve agent
webagents daemon                  # Start daemon
webagents login                   # Portal auth
webagents discover "search web"   # Find agents
webagents init my-agent           # New project
webagents publish                 # Publish to portal
webagents skills list             # List skills
```
