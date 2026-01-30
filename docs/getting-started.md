# Getting Started

This guide will help you get up and running with webagents.

## Installation

```bash
npm install webagents
```

## Choose Your LLM Provider

webagents supports multiple LLM providers. Install the ones you need:

### In-Browser (Local Inference)

For running LLMs directly in the browser without API calls:

```bash
# WebLLM - Uses WebGPU for fast local inference
npm install @mlc-ai/web-llm

# Transformers.js - Hugging Face models, WebGPU + WASM fallback
npm install @huggingface/transformers
```

### Cloud Providers

```bash
# OpenAI (GPT-4, GPT-4o)
npm install openai

# Anthropic (Claude)
npm install @anthropic-ai/sdk

# Google (Gemini)
npm install @google/generative-ai

# xAI (Grok) - uses OpenAI SDK
npm install openai
```

## Your First Agent

### Browser Agent with WebLLM

WebLLM runs LLMs locally using WebGPU, providing privacy and offline capabilities.

```typescript
import { BaseAgent } from 'webagents';
import { WebLLMSkill } from 'webagents/skills/llm/webllm';

// Create the agent
const agent = new BaseAgent({
  name: 'my-assistant',
  instructions: 'You are a helpful assistant.',
  skills: [
    new WebLLMSkill({ 
      model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC'
    })
  ]
});

// Initialize (downloads model on first run, ~4GB)
console.log('Loading model...');
await agent.initialize();
console.log('Model loaded!');

// Have a conversation
const response = await agent.run([
  { role: 'user', content: 'Hello! What can you help me with?' }
]);

console.log(response.content);
```

### Cloud Agent with OpenAI

```typescript
import { BaseAgent } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm/openai';

const agent = new BaseAgent({
  name: 'cloud-assistant',
  instructions: 'You are a helpful assistant.',
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

console.log(response.content);
```

## Streaming Responses

For real-time output, use streaming:

```typescript
for await (const chunk of agent.runStreaming([
  { role: 'user', content: 'Tell me a story' }
])) {
  switch (chunk.type) {
    case 'delta':
      // Incremental text
      process.stdout.write(chunk.delta);
      break;
      
    case 'tool_call':
      // Agent is calling a tool
      console.log(`\nCalling tool: ${chunk.tool_call.name}`);
      break;
      
    case 'done':
      // Response complete
      console.log('\n\nDone!');
      console.log('Tokens used:', chunk.response.usage?.total_tokens);
      break;
      
    case 'error':
      console.error('Error:', chunk.error);
      break;
  }
}
```

## Adding Tools

Tools let your agent perform actions. Use the `@tool` decorator:

```typescript
import { Skill, tool } from 'webagents';
import type { Context } from 'webagents';

class CalculatorSkill extends Skill {
  @tool({
    description: 'Add two numbers together',
    parameters: {
      type: 'object',
      properties: {
        a: { type: 'number', description: 'First number' },
        b: { type: 'number', description: 'Second number' }
      },
      required: ['a', 'b']
    }
  })
  async add(params: { a: number; b: number }, ctx: Context) {
    return params.a + params.b;
  }

  @tool({
    description: 'Multiply two numbers',
    parameters: {
      type: 'object',
      properties: {
        a: { type: 'number' },
        b: { type: 'number' }
      },
      required: ['a', 'b']
    }
  })
  async multiply(params: { a: number; b: number }, ctx: Context) {
    return params.a * params.b;
  }
}

// Add the skill to your agent
const agent = new BaseAgent({
  skills: [
    new CalculatorSkill(),
    new OpenAISkill({ apiKey: '...' })
  ]
});

// The agent can now use these tools
const response = await agent.run([
  { role: 'user', content: 'What is 42 * 17?' }
]);
```

## Using Browser APIs

webagents provides skills for common browser APIs:

```typescript
import { BaseAgent } from 'webagents';
import { WakeLockSkill } from 'webagents/skills/browser/wakelock';
import { NotificationsSkill } from 'webagents/skills/browser/notifications';
import { GeolocationSkill } from 'webagents/skills/browser/geolocation';

const agent = new BaseAgent({
  skills: [
    new WakeLockSkill(),      // Keep screen awake
    new NotificationsSkill(), // Show notifications
    new GeolocationSkill(),   // Get location
    new WebLLMSkill({ ... })  // LLM for processing
  ]
});

// The agent can now:
// - Keep the screen awake during long operations
// - Show notifications to the user
// - Get the user's location (with permission)
```

## Running as a Server

Expose your agent as an HTTP server:

```typescript
import { BaseAgent } from 'webagents';
import { createAgentApp, serve } from 'webagents/server';
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';

const agent = new BaseAgent({
  name: 'api-agent',
  skills: [
    new CompletionsTransportSkill(), // Adds /v1/chat/completions
    new OpenAISkill({ ... })
  ]
});

// Set up the completions transport
const completionsSkill = agent.skills.find(
  s => s.name === 'completions'
) as CompletionsTransportSkill;
completionsSkill.setAgent(agent);

// Create and start server
const app = createAgentApp(agent);
serve(app, { port: 3000 });

console.log('Agent server running at http://localhost:3000');
```

Now you can use the agent with any OpenAI-compatible client:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Using the CLI

The CLI provides an interactive chat interface:

```bash
# Start interactive chat
npx webagents chat

# Use a specific provider
OPENAI_API_KEY=sk-... npx webagents chat --model gpt-4o

# List available models
npx webagents models
```

### CLI Slash Commands

While in the chat:

- `/help` - Show all commands
- `/model gpt-4o` - Switch models
- `/history` - View conversation
- `/clear` - Start fresh
- `/save chat.json` - Save conversation
- `/exit` - Quit

## Next Steps

- [Skills Guide](./skills.md) - Learn about built-in skills and creating custom ones
- [UAMP Protocol](./uamp.md) - Understand the underlying protocol
- [API Reference](./api.md) - Detailed API documentation
- [Examples](./examples.md) - More code examples
