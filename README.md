# webagents

TypeScript SDK for building AI agents with UAMP (Universal Agentic Message Protocol) support. Designed for in-browser AI agents using WebLLM, with support for cloud providers.

[![npm version](https://badge.fury.io/js/webagents.svg)](https://www.npmjs.com/package/webagents)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **In-Browser LLM**: Run LLMs locally using [WebLLM](https://webllm.mlc.ai) (WebGPU) or [Transformers.js](https://huggingface.co/docs/transformers.js)
- **UAMP Protocol**: Universal Agentic Message Protocol for standardized agent communication
- **Transport Skills**: OpenAI Completions API, Portal (native UAMP), and more
- **Cloud Providers**: OpenAI, Anthropic Claude, Google Gemini, xAI Grok
- **Browser APIs**: WakeLock, Notifications, Geolocation, Storage, Camera, Microphone
- **Extensible**: Build custom skills with decorators (`@tool`, `@hook`, `@handoff`)
- **CLI**: Interactive chat and agent management

## Installation

```bash
npm install webagents
```

### Peer Dependencies (Optional)

Install the providers you need:

```bash
# In-browser LLM (choose one or both)
npm install @mlc-ai/web-llm           # WebGPU optimized
npm install @huggingface/transformers # WebGPU + WASM fallback

# Cloud LLM providers
npm install openai                    # OpenAI GPT
npm install @anthropic-ai/sdk         # Anthropic Claude
npm install @google/generative-ai     # Google Gemini
# xAI uses OpenAI SDK with custom base URL
```

## Quick Start

### In-Browser Agent (WebLLM)

```typescript
import { BaseAgent } from 'webagents';
import { WebLLMSkill } from 'webagents/skills/llm/webllm';

const agent = new BaseAgent({
  name: 'browser-assistant',
  skills: [
    new WebLLMSkill({ 
      model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC' 
    })
  ]
});

// Initialize (downloads model on first run)
await agent.initialize();

// Run a conversation
const response = await agent.run([
  { role: 'user', content: 'What is WebGPU?' }
]);

console.log(response.content);
```

### Cloud Provider Agent (OpenAI)

```typescript
import { BaseAgent } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm/openai';

const agent = new BaseAgent({
  name: 'cloud-assistant',
  skills: [
    new OpenAISkill({ 
      apiKey: process.env.OPENAI_API_KEY,
      model: 'gpt-4o'
    })
  ]
});

const response = await agent.run([
  { role: 'user', content: 'Hello!' }
]);
```

### Streaming Responses

```typescript
for await (const chunk of agent.runStreaming([
  { role: 'user', content: 'Write a poem about AI' }
])) {
  if (chunk.type === 'delta') {
    process.stdout.write(chunk.delta);
  }
}
```

### Custom Tools

```typescript
import { Skill, tool } from 'webagents';

class WeatherSkill extends Skill {
  @tool({ 
    provides: 'weather',
    description: 'Get current weather for a city',
    parameters: {
      type: 'object',
      properties: {
        city: { type: 'string', description: 'City name' }
      },
      required: ['city']
    }
  })
  async getWeather(params: { city: string }) {
    const response = await fetch(
      `https://api.weather.example/v1?city=${params.city}`
    );
    return response.json();
  }
}

const agent = new BaseAgent({
  skills: [new WeatherSkill(), new OpenAISkill({ ... })]
});
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    External Clients                      │
│         (Chat UI, CLI, Other Agents, etc.)              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Transport Layer (Skills)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Completions │  │   Portal    │  │    A2A      │     │
│  │   (OpenAI)  │  │   (UAMP)    │  │  (Google)   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
└─────────┼────────────────┼────────────────┼─────────────┘
          │    adapters    │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────┐
│                    UAMP Protocol                         │
│         (Universal Agentic Message Protocol)            │
│   Events: session, input, response, tool, progress      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                     BaseAgent                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    Tools    │  │    Hooks    │  │  Handoffs   │     │
│  │  Registry   │  │   Chain     │  │  (LLMs)     │     │
│  └─────────────┘  └─────────────┘  └──────┬──────┘     │
└──────────────────────────────────────────┼──────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────┐
│                   LLM Skills                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ WebLLM  │ │ Trans-  │ │ OpenAI  │ │ Claude  │ ...   │
│  │ (local) │ │ formers │ │ (cloud) │ │ (cloud) │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Available Skills

### LLM Skills (Handoffs)

| Skill | Provider | Environment | Notes |
|-------|----------|-------------|-------|
| `WebLLMSkill` | MLC AI | Browser | WebGPU, local inference |
| `TransformersSkill` | Hugging Face | Browser/Node | WASM fallback |
| `OpenAISkill` | OpenAI | Any | GPT-4, GPT-4o |
| `AnthropicSkill` | Anthropic | Any | Claude 3.5 Sonnet |
| `GoogleSkill` | Google | Any | Gemini Pro |
| `XAISkill` | xAI | Any | Grok |

### Transport Skills

| Skill | Protocol | Purpose |
|-------|----------|---------|
| `CompletionsTransportSkill` | OpenAI API | Expose agent as OpenAI-compatible API |
| `PortalTransportSkill` | UAMP over WS | Connect to portal/daemon |

### Browser Skills

| Skill | APIs | Purpose |
|-------|------|---------|
| `WakeLockSkill` | Screen Wake Lock | Prevent screen sleep |
| `NotificationsSkill` | Web Notifications | Show notifications |
| `GeolocationSkill` | Geolocation | Get device location |
| `StorageSkill` | localStorage, IndexedDB | Persist data |
| `CameraSkill` | getUserMedia | Capture video/images |
| `MicrophoneSkill` | getUserMedia | Capture audio |

## CLI Usage

```bash
# Interactive chat
npx webagents chat

# With specific model
npx webagents chat --model gpt-4o

# Non-interactive mode
echo "Hello" | npx webagents chat

# List available models
npx webagents models

# Show agent info
npx webagents info
```

### Slash Commands

In interactive mode:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model <name>` | Switch model |
| `/history` | Show conversation history |
| `/clear` | Clear conversation |
| `/save <file>` | Save conversation |
| `/load <file>` | Load conversation |
| `/tools` | List available tools |
| `/exit` | Exit chat |

## HTTP Server

```typescript
import { BaseAgent } from 'webagents';
import { createAgentApp, serve } from 'webagents/server';

const agent = new BaseAgent({ ... });
const app = createAgentApp(agent);

serve(app, { port: 3000 });
// Agent available at http://localhost:3000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Agent info and capabilities |
| `/uamp` | POST | Process UAMP events |
| `/uamp/stream` | POST | Process UAMP events (SSE) |

## Documentation

- [Getting Started](./docs/getting-started.md)
- [Skills Guide](./docs/skills.md)
- [UAMP Protocol](./docs/uamp.md)
- [API Reference](./docs/api.md)
- [CLI Documentation](./docs/cli.md)
- [Examples](./docs/examples.md)

## External Resources

- [UAMP Protocol Specification](https://uamp.dev)
- [WebLLM Documentation](https://webllm.mlc.ai)
- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT
