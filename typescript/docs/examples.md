# Examples

Practical code examples for common use cases.

## Basic Examples

### Simple Chat Agent

```typescript
import { BaseAgent } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm/openai';

const agent = new BaseAgent({
  name: 'chat-assistant',
  instructions: 'You are a helpful assistant.',
  skills: [
    new OpenAISkill({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o',
    }),
  ],
});

// Simple request/response
const response = await agent.run([
  { role: 'user', content: 'What is the meaning of life?' },
]);

console.log(response.content);
```

### Multi-turn Conversation

```typescript
const messages = [
  { role: 'system' as const, content: 'You are a math tutor.' },
  { role: 'user' as const, content: 'What is calculus?' },
];

// First turn
const response1 = await agent.run(messages);
console.log('Assistant:', response1.content);

// Add to conversation
messages.push(
  { role: 'assistant' as const, content: response1.content },
  { role: 'user' as const, content: 'Can you give me an example?' }
);

// Second turn
const response2 = await agent.run(messages);
console.log('Assistant:', response2.content);
```

### Streaming Response

```typescript
import { BaseAgent } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm/openai';

const agent = new BaseAgent({
  skills: [new OpenAISkill({ apiKey: '...' })],
});

// Stream the response
for await (const chunk of agent.runStreaming([
  { role: 'user', content: 'Write a short poem about coding.' },
])) {
  switch (chunk.type) {
    case 'delta':
      process.stdout.write(chunk.delta);
      break;
    case 'done':
      console.log('\n\nTokens used:', chunk.response.usage?.total_tokens);
      break;
    case 'error':
      console.error('Error:', chunk.error.message);
      break;
  }
}
```

## In-Browser Examples

### WebLLM Agent (Browser)

```typescript
import { BaseAgent } from 'webagents';
import { WebLLMSkill } from 'webagents/skills/llm/webllm';

// Create agent with local LLM
const agent = new BaseAgent({
  name: 'local-assistant',
  skills: [
    new WebLLMSkill({
      model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC',
    }),
  ],
});

// Show loading progress
const webllm = agent.skills[0] as WebLLMSkill;
webllm.onProgress = (progress) => {
  document.getElementById('status')!.textContent = 
    `Loading: ${(progress * 100).toFixed(1)}%`;
};

// Initialize (downloads model)
await agent.initialize();
document.getElementById('status')!.textContent = 'Ready!';

// Chat function
async function chat(userMessage: string) {
  const output = document.getElementById('output')!;
  output.textContent = '';

  for await (const chunk of agent.runStreaming([
    { role: 'user', content: userMessage },
  ])) {
    if (chunk.type === 'delta') {
      output.textContent += chunk.delta;
    }
  }
}
```

### Browser APIs Example

```typescript
import { BaseAgent } from 'webagents';
import { WebLLMSkill } from 'webagents/skills/llm/webllm';
import { WakeLockSkill } from 'webagents/skills/browser/wakelock';
import { NotificationsSkill } from 'webagents/skills/browser/notifications';
import { GeolocationSkill } from 'webagents/skills/browser/geolocation';

const agent = new BaseAgent({
  name: 'browser-assistant',
  instructions: `You are a helpful assistant with access to browser APIs.
    You can keep the screen awake, show notifications, and get location.`,
  skills: [
    new WakeLockSkill(),
    new NotificationsSkill(),
    new GeolocationSkill(),
    new WebLLMSkill({ model: '...' }),
  ],
});

await agent.initialize();

// The agent can now use browser APIs
const response = await agent.run([
  { role: 'user', content: 'Keep the screen awake while I read.' },
]);
// Agent will call acquire_wakelock tool

const response2 = await agent.run([
  { role: 'user', content: 'Where am I located?' },
]);
// Agent will call get_location tool
```

## Custom Skills

### Weather Skill

```typescript
import { Skill, tool } from 'webagents';
import type { Context } from 'webagents';

interface WeatherResponse {
  city: string;
  temperature: number;
  conditions: string;
  humidity: number;
}

class WeatherSkill extends Skill {
  private apiKey: string;

  constructor(apiKey: string) {
    super({ name: 'weather' });
    this.apiKey = apiKey;
  }

  @tool({
    provides: 'weather',
    description: 'Get current weather for a city',
    parameters: {
      type: 'object',
      properties: {
        city: {
          type: 'string',
          description: 'City name (e.g., "London", "New York")',
        },
        units: {
          type: 'string',
          enum: ['celsius', 'fahrenheit'],
          description: 'Temperature units',
          default: 'celsius',
        },
      },
      required: ['city'],
    },
  })
  async getWeather(
    params: { city: string; units?: string },
    ctx: Context
  ): Promise<WeatherResponse> {
    const response = await fetch(
      `https://api.weather.example/v1/current?city=${encodeURIComponent(params.city)}&units=${params.units || 'celsius'}`,
      {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      }
    );

    if (!response.ok) {
      throw new Error(`Weather API error: ${response.statusText}`);
    }

    return response.json();
  }

  @tool({
    description: 'Get weather forecast for the next 7 days',
    parameters: {
      type: 'object',
      properties: {
        city: { type: 'string' },
      },
      required: ['city'],
    },
  })
  async getForecast(params: { city: string }, ctx: Context) {
    const response = await fetch(
      `https://api.weather.example/v1/forecast?city=${encodeURIComponent(params.city)}&days=7`,
      {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      }
    );
    return response.json();
  }
}

// Usage
const agent = new BaseAgent({
  skills: [
    new WeatherSkill(process.env.WEATHER_API_KEY!),
    new OpenAISkill({ apiKey: '...' }),
  ],
});
```

### Database Skill

```typescript
import { Skill, tool, hook } from 'webagents';
import type { Context, HookData, HookResult } from 'webagents';

interface User {
  id: string;
  name: string;
  email: string;
}

class DatabaseSkill extends Skill {
  private db: Map<string, User> = new Map();

  async initialize() {
    // Connect to database
    console.log('Database connected');
  }

  async cleanup() {
    // Disconnect
    console.log('Database disconnected');
  }

  @tool({
    description: 'Find a user by ID',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'User ID' },
      },
      required: ['id'],
    },
  })
  async findUser(params: { id: string }, ctx: Context): Promise<User | null> {
    return this.db.get(params.id) || null;
  }

  @tool({
    description: 'Create a new user',
    scopes: ['admin'], // Only admins can create users
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string' },
        email: { type: 'string' },
      },
      required: ['name', 'email'],
    },
  })
  async createUser(
    params: { name: string; email: string },
    ctx: Context
  ): Promise<User> {
    const user: User = {
      id: crypto.randomUUID(),
      name: params.name,
      email: params.email,
    };
    this.db.set(user.id, user);
    return user;
  }

  @tool({
    description: 'List all users',
    parameters: { type: 'object', properties: {} },
  })
  async listUsers(params: {}, ctx: Context): Promise<User[]> {
    return Array.from(this.db.values());
  }
}
```

### Logging Skill with Hooks

```typescript
import { Skill, hook } from 'webagents';
import type { Context, HookData, HookResult } from 'webagents';

class LoggingSkill extends Skill {
  private logs: Array<{ timestamp: Date; event: string; data: unknown }> = [];

  @hook({ lifecycle: 'before_run', priority: 1 })
  async logRequest(data: HookData, ctx: Context): Promise<void> {
    this.log('request', { messages: data.messages });
  }

  @hook({ lifecycle: 'after_run', priority: 100 })
  async logResponse(data: HookData, ctx: Context): Promise<void> {
    this.log('response', { response: data.response?.substring(0, 100) });
  }

  @hook({ lifecycle: 'before_tool' })
  async logToolCall(data: HookData, ctx: Context): Promise<void> {
    this.log('tool_call', {
      tool: data.tool_name,
      params: data.tool_params,
    });
  }

  @hook({ lifecycle: 'after_tool' })
  async logToolResult(data: HookData, ctx: Context): Promise<void> {
    this.log('tool_result', {
      tool: data.tool_name,
      result: data.tool_result,
    });
  }

  @hook({ lifecycle: 'on_error' })
  async logError(data: HookData, ctx: Context): Promise<void> {
    this.log('error', { error: data.error?.message });
  }

  private log(event: string, data: unknown) {
    const entry = { timestamp: new Date(), event, data };
    this.logs.push(entry);
    console.log(`[${entry.timestamp.toISOString()}] ${event}:`, data);
  }

  getLogs() {
    return this.logs;
  }
}
```

## Server Examples

### HTTP Server

```typescript
import { BaseAgent } from 'webagents';
import { createAgentApp, serve } from 'webagents/server';
import { OpenAISkill } from 'webagents/skills/llm/openai';
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';

// Create agent
const agent = new BaseAgent({
  name: 'api-agent',
  instructions: 'You are a helpful API assistant.',
  skills: [
    new CompletionsTransportSkill(),
    new OpenAISkill({ apiKey: process.env.OPENAI_API_KEY! }),
  ],
});

// Set up completions transport
const completions = agent.skills.find(
  (s) => s.name === 'completions'
) as CompletionsTransportSkill;
completions.setAgent(agent);

// Create Hono app
const app = createAgentApp(agent);

// Start server
serve(app, { port: 3000 });
console.log('Server running on http://localhost:3000');

// Test with curl:
// curl http://localhost:3000/v1/chat/completions \
//   -H "Content-Type: application/json" \
//   -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```

### Fetch Handler (Cloudflare Workers)

```typescript
import { BaseAgent } from 'webagents';
import { createFetchHandler } from 'webagents/server';
import { OpenAISkill } from 'webagents/skills/llm/openai';

const agent = new BaseAgent({
  name: 'worker-agent',
  skills: [new OpenAISkill({ apiKey: '...' })],
});

const handler = createFetchHandler(agent);

// Cloudflare Workers
export default {
  fetch: handler,
};

// Bun
Bun.serve({ fetch: handler });
```

### Custom HTTP Endpoints

```typescript
import { Skill, http } from 'webagents';
import type { Context } from 'webagents';

class APISkill extends Skill {
  @http({ path: '/api/health', method: 'GET' })
  async health(req: Request, ctx: Context): Promise<Response> {
    return new Response(
      JSON.stringify({
        status: 'healthy',
        timestamp: new Date().toISOString(),
      }),
      { headers: { 'Content-Type': 'application/json' } }
    );
  }

  @http({ path: '/api/stats', method: 'GET' })
  async stats(req: Request, ctx: Context): Promise<Response> {
    return new Response(
      JSON.stringify({
        requests: 1234,
        tokens_used: 56789,
      }),
      { headers: { 'Content-Type': 'application/json' } }
    );
  }

  @http({ path: '/api/webhook', method: 'POST' })
  async webhook(req: Request, ctx: Context): Promise<Response> {
    const body = await req.json();
    console.log('Webhook received:', body);
    return new Response(JSON.stringify({ received: true }));
  }
}
```

## Multi-Agent Examples

### Agent Handoff

```typescript
import { BaseAgent, Skill, tool } from 'webagents';
import { OpenAISkill } from 'webagents/skills/llm/openai';

class SpecialistRouter extends Skill {
  private codeAgent: BaseAgent;
  private mathAgent: BaseAgent;

  constructor() {
    super();

    // Specialized agents
    this.codeAgent = new BaseAgent({
      name: 'code-expert',
      instructions: 'You are a coding expert.',
      skills: [new OpenAISkill({ apiKey: '...', model: 'gpt-4o' })],
    });

    this.mathAgent = new BaseAgent({
      name: 'math-expert',
      instructions: 'You are a math expert.',
      skills: [new OpenAISkill({ apiKey: '...', model: 'gpt-4o' })],
    });
  }

  @tool({
    description: 'Route to code expert for programming questions',
    parameters: {
      type: 'object',
      properties: {
        question: { type: 'string' },
      },
      required: ['question'],
    },
  })
  async askCodeExpert(params: { question: string }) {
    const response = await this.codeAgent.run([
      { role: 'user', content: params.question },
    ]);
    return response.content;
  }

  @tool({
    description: 'Route to math expert for mathematical questions',
    parameters: {
      type: 'object',
      properties: {
        question: { type: 'string' },
      },
      required: ['question'],
    },
  })
  async askMathExpert(params: { question: string }) {
    const response = await this.mathAgent.run([
      { role: 'user', content: params.question },
    ]);
    return response.content;
  }
}

// Router agent
const routerAgent = new BaseAgent({
  name: 'router',
  instructions: `You route questions to the appropriate expert.
    Use askCodeExpert for programming questions.
    Use askMathExpert for math questions.`,
  skills: [new SpecialistRouter(), new OpenAISkill({ apiKey: '...' })],
});
```

## Testing Examples

### Unit Testing Skills

```typescript
import { describe, it, expect, vi } from 'vitest';
import { WeatherSkill } from './weather-skill';

describe('WeatherSkill', () => {
  it('gets weather for a city', async () => {
    // Mock fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          city: 'London',
          temperature: 15,
          conditions: 'cloudy',
        }),
    });

    const skill = new WeatherSkill('test-api-key');
    const result = await skill.getWeather(
      { city: 'London' },
      {} as any
    );

    expect(result.city).toBe('London');
    expect(result.temperature).toBe(15);
  });
});
```

### Integration Testing Agents

```typescript
import { describe, it, expect } from 'vitest';
import { BaseAgent } from 'webagents';
import { Skill, tool, handoff } from 'webagents';
import { createResponseDoneEvent } from 'webagents/uamp';

// Mock LLM that returns predictable responses
class MockLLM extends Skill {
  @handoff({ name: 'mock-llm' })
  async *processUAMP(events: any[]) {
    yield createResponseDoneEvent('r1', [
      { type: 'text', text: 'Mock response' },
    ]);
  }
}

describe('Agent Integration', () => {
  it('processes messages through skills', async () => {
    const agent = new BaseAgent({
      skills: [new MockLLM()],
    });

    const response = await agent.run([
      { role: 'user', content: 'Hello' },
    ]);

    expect(response.content).toBe('Mock response');
  });
});
```
