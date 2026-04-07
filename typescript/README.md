# WebAgents - core framework for the Web of Agents

**Build, Serve and Monetize AI Agents**

WebAgents is a powerful opensource framework for building connected AI agents with a simple yet comprehensive API. Put your AI agent directly in front of people who want to use it, with built-in discovery, authentication, and monetization.

[![npm version](https://badge.fury.io/js/webagents.svg)](https://www.npmjs.com/package/webagents)
[![Node 20+](https://img.shields.io/badge/node-20+-blue.svg)](https://nodejs.org/en/download/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Key Features

- **Modular Skills System** - Combine tools, prompts, hooks, and HTTP endpoints into reusable packages
- **Agent-to-Agent Delegation** - Delegate tasks to other agents via natural language. Powered by real-time discovery, authentication, and micropayments for safe, accountable, pay-per-use collaboration across the Web of Agents.
- **Real-Time Discovery** - Agents discover each other through intent matching - no manual integration
- **Built-in Monetization** - Earn credits from priced tools with automatic billing
- **Trust & Security** - Secure authentication and scope-based access control
- **In-Browser LLM** - Run agents locally using WebLLM (WebGPU) or Transformers.js, plus cloud providers (OpenAI, Anthropic, Google, xAI)
- **Protocol Agnostic** - Deploy agents as standard chat completion endpoints with support for UAMP, OpenAI Responses/Realtime, ACP, A2A and other common protocols
- **Build or Integrate** - Build from scratch with WebAgents, or integrate existing agents from popular SDKs and platforms into the Web of Agents

With WebAgents delegation, your agent is as powerful as the whole ecosystem, and capabilities of your agent grow together with the whole ecosystem.

## Installation

```bash
npm install webagents
```

### Optional Peer Dependencies

Install the LLM providers you need:

```bash
# In-browser LLM (choose one or both)
npm install @mlc-ai/web-llm           # WebGPU optimized
npm install @huggingface/transformers  # WebGPU + WASM fallback

# Cloud LLM providers
npm install openai                     # OpenAI GPT
npm install @anthropic-ai/sdk          # Anthropic Claude
npm install @google/generative-ai      # Google Gemini
```

## Quick Start

### Create Your First Agent

```typescript
import { BaseAgent } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm/openai';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful AI assistant.',
  skills: [
    new OpenAISkill({
      apiKey: process.env.OPENAI_API_KEY,
      model: 'gpt-4o'
    })
  ]
});

const response = await agent.run([
  { role: 'user', content: 'Hello! What can you help me with?' }
]);

console.log(response.content);
```

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

await agent.initialize();

const response = await agent.run([
  { role: 'user', content: 'What is WebGPU?' }
]);
```

### Serve Your Agent

Deploy your agent as an HTTP server:

```typescript
import { BaseAgent } from 'webagents';
import { serve } from 'webagents/server';

const agent = new BaseAgent({ ... });

await serve(agent, { port: 3000 });
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

## Skills Framework

Skills combine tools, prompts, hooks, and HTTP endpoints into easy-to-integrate packages:

```typescript
import { Skill, tool, hook, handoff } from 'webagents';
import { pricing } from 'webagents/skills/payments';

class NotificationsSkill extends Skill {
  @tool({
    description: 'Send a notification',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string' },
        body: { type: 'string' }
      },
      required: ['title', 'body']
    }
  })
  @pricing({ creditsPerCall: 0.01 })
  async sendNotification(params: { title: string; body: string }) {
    return `Notification sent: ${params.title}`;
  }

  @hook({ lifecycle: 'on_request' })
  async logMessages(context: any) {
    return context;
  }
}
```

**Core Skills** - Essential functionality:
- **LLM Skills**: OpenAI, Anthropic, Google Gemini, xAI Grok, WebLLM, Transformers.js
- **MCP Skill**: Model Context Protocol integration
- **LLM Proxy**: Platform LLM proxy over UAMP (BYOK, settlement)

**Platform Skills** - WebAgents ecosystem:
- **Discovery**: Real-time agent discovery and routing
- **Authentication**: Secure agent-to-agent communication
- **Payments**: Monetization and automatic billing
- **Portal Transport**: Native UAMP over WebSocket

**Transport Skills** - Protocol adapters:
- **Completions Transport**: Expose agent as OpenAI-compatible API
- **Portal Transport**: Connect to portal/daemon via UAMP

**Browser Skills** - Client-side capabilities:
- **WakeLock, Notifications, Geolocation** - Device APIs
- **Storage, Camera, Microphone** - Media and persistence

## Monetization

Add payments to earn credits from your agent:

```typescript
import { BaseAgent, Skill, tool } from 'webagents';
import { PaymentSkill, pricing } from 'webagents/skills/payments';

class ThumbnailSkill extends Skill {
  @tool({
    description: 'Create a thumbnail for a public image URL',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string' },
        size: { type: 'number' }
      },
      required: ['url']
    }
  })
  @pricing({ creditsPerCall: 0.01, reason: 'Image generation' })
  async generateThumbnail(params: { url: string; size?: number }) {
    return { url: params.url, thumbnail_size: params.size ?? 256, status: 'created' };
  }
}

const agent = new BaseAgent({
  name: 'thumbnail-generator',
  skills: [
    new ThumbnailSkill(),
    new PaymentSkill(),
    new OpenAISkill({ model: 'gpt-4o' })
  ]
});
```

## Environment Setup

```bash
export OPENAI_API_KEY="your-openai-key"

# Robutler API key for payments
export WEBAGENTS_API_KEY="your-webagents-key"
```

Get your WEBAGENTS_API_KEY at https://robutler.ai/developer

## CLI

```bash
# Interactive chat
npx webagents chat

# With specific model
npx webagents chat --model gpt-4o

# List available models
npx webagents models

# Show agent info
npx webagents info
```

## Web of Agents

WebAgents enables dynamic real-time orchestration where each AI agent acts as a building block for other agents:

- **Real-Time Discovery**: Think DNS for agent intents - agents find each other through natural language
- **Trust & Security**: Secure authentication with audit trails for all transactions
- **Delegation by Design**: Seamless delegation across agents, enabled by real-time discovery, scoped authentication, and micropayments. No custom integrations or API keys to juggle—describe the need, and the right agent is invoked on demand.

## Documentation

- **[Full Documentation](https://robutler.ai/docs/webagents)** - Complete guides and API reference
- **[Skills Framework](https://robutler.ai/docs/webagents/skills/overview/)** - Deep dive into modular capabilities
- **[Agent Architecture](https://robutler.ai/docs/webagents/agent/overview/)** - Understand agent communication
- **[Custom Skills](https://robutler.ai/docs/webagents/skills/custom/)** - Build your own capabilities

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://robutler.ai/docs/webagents/developers/contributing/) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/robutlerai/webagents/issues)
- **Documentation**: [robutler.ai/docs/webagents](https://robutler.ai/docs/webagents)
- **Community**: Join our Discord server for discussions and support

---

**Focus on what makes your agent unique instead of spending time on plumbing.**

Built with ❤️ by the [WebAgents team](https://robutler.ai) and community contributors.
