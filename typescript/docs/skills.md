# Skills Guide

Skills are modular components that add capabilities to agents. They can provide tools, hooks, handoffs (LLM connections), and HTTP/WebSocket endpoints.

## Skill Architecture

```
Skill
├── Tools         - Functions the LLM can call
├── Hooks         - Lifecycle interceptors
├── Handoffs      - LLM processing (for LLM skills)
├── HTTP Endpoints - Custom API routes
└── WS Endpoints  - WebSocket handlers
```

## Built-in Skills

### LLM Skills

LLM skills process natural language using the `@handoff` decorator.

#### WebLLMSkill

In-browser LLM using WebGPU. Best for privacy-focused applications.

```typescript
import { WebLLMSkill } from 'webagents/skills/llm/webllm';

const skill = new WebLLMSkill({
  model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC',
  // Optional configuration
  temperature: 0.7,
  maxTokens: 2048,
});

// Initialize downloads the model (first time only)
await skill.initialize();
```

**Supported Models:**
- `Llama-3.1-8B-Instruct-q4f32_1-MLC`
- `Llama-3.2-3B-Instruct-q4f16_1-MLC`
- `Mistral-7B-Instruct-v0.3-q4f16_1-MLC`
- `Qwen2.5-7B-Instruct-q4f16_1-MLC`
- See [WebLLM models](https://webllm.mlc.ai) for full list

#### TransformersSkill

Hugging Face Transformers.js for browser/Node.js.

```typescript
import { TransformersSkill } from 'webagents/skills/llm/transformers';

const skill = new TransformersSkill({
  model: 'Xenova/Phi-3-mini-4k-instruct',
  // Uses WebGPU when available, falls back to WASM
});
```

#### OpenAISkill

OpenAI GPT models via API.

```typescript
import { OpenAISkill } from 'webagents/skills/llm/openai';

const skill = new OpenAISkill({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4o',
  // Optional
  temperature: 0.7,
  maxTokens: 4096,
  baseUrl: 'https://api.openai.com/v1', // For proxies
});
```

#### AnthropicSkill

Anthropic Claude models.

```typescript
import { AnthropicSkill } from 'webagents/skills/llm/anthropic';

const skill = new AnthropicSkill({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-3-5-sonnet-20241022',
  maxTokens: 8192,
});
```

#### GoogleSkill

Google Gemini models.

```typescript
import { GoogleSkill } from 'webagents/skills/llm/google';

const skill = new GoogleSkill({
  apiKey: process.env.GOOGLE_API_KEY,
  model: 'gemini-1.5-pro',
});
```

#### XAISkill

xAI Grok models (uses OpenAI-compatible API).

```typescript
import { XAISkill } from 'webagents/skills/llm/xai';

const skill = new XAISkill({
  apiKey: process.env.XAI_API_KEY,
  model: 'grok-beta',
});
```

### Transport Skills

Transport skills expose agents via different protocols.

#### CompletionsTransportSkill

OpenAI-compatible Chat Completions API.

```typescript
import { CompletionsTransportSkill } from 'webagents/skills/transport/completions';

const transport = new CompletionsTransportSkill({
  basePath: '/v1', // Optional, default
});

// Must set the agent after creation
transport.setAgent(agent);
```

**Endpoints provided:**
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List models

#### PortalTransportSkill

Native UAMP over WebSocket for portal/daemon connectivity.

```typescript
import { PortalTransportSkill } from 'webagents/skills/transport/portal';

const portal = new PortalTransportSkill({
  portalUrl: 'wss://portal.example.com/ws',
});

// Connect to portal
await portal.exposeAgent(agent);

// Call a remote agent
const response = await portal.callRemoteAgent('other-agent', [
  { role: 'user', content: 'Hello' }
]);
```

### Browser Skills

Browser skills provide access to Web APIs.

#### WakeLockSkill

Prevents the screen from sleeping.

```typescript
import { WakeLockSkill } from 'webagents/skills/browser/wakelock';

const skill = new WakeLockSkill();

// Tools provided:
// - acquire_wakelock(): Acquire wake lock
// - release_wakelock(): Release wake lock
// - wakelock_status(): Check status
```

#### NotificationsSkill

Web Notifications API.

```typescript
import { NotificationsSkill } from 'webagents/skills/browser/notifications';

const skill = new NotificationsSkill();

// Tools provided:
// - request_notification_permission(): Request permission
// - show_notification(title, body, icon?): Show notification
// - notification_permission_status(): Check permission
```

#### GeolocationSkill

Device location.

```typescript
import { GeolocationSkill } from 'webagents/skills/browser/geolocation';

const skill = new GeolocationSkill();

// Tools provided:
// - get_location(): Get current position
//   Returns: { latitude, longitude, accuracy, ... }
```

#### StorageSkill

localStorage and IndexedDB access.

```typescript
import { StorageSkill } from 'webagents/skills/browser/storage';

const skill = new StorageSkill();

// Tools provided:
// - storage_get(key): Get value
// - storage_set(key, value): Set value
// - storage_remove(key): Remove value
// - storage_get_json(key): Get and parse JSON
// - storage_set_json(key, value): Stringify and set JSON
```

#### CameraSkill

Video capture.

```typescript
import { CameraSkill } from 'webagents/skills/browser/camera';

const skill = new CameraSkill();

// Tools provided:
// - start_camera(facingMode?): Start camera
// - stop_camera(): Stop camera
// - capture_frame(): Capture image frame
//   Returns: base64 encoded image
```

#### MicrophoneSkill

Audio capture.

```typescript
import { MicrophoneSkill } from 'webagents/skills/browser/microphone';

const skill = new MicrophoneSkill();

// Tools provided:
// - start_microphone(): Start microphone
// - stop_microphone(): Stop microphone
// - start_recording(): Begin recording
// - stop_recording(): Stop and get audio
//   Returns: base64 encoded audio
```

## Creating Custom Skills

### Basic Skill with Tools

```typescript
import { Skill, tool } from 'webagents';
import type { Context } from 'webagents';

class WeatherSkill extends Skill {
  @tool({
    provides: 'weather',
    description: 'Get weather for a city',
    parameters: {
      type: 'object',
      properties: {
        city: { 
          type: 'string', 
          description: 'City name' 
        },
        units: { 
          type: 'string', 
          enum: ['celsius', 'fahrenheit'],
          default: 'celsius'
        }
      },
      required: ['city']
    }
  })
  async getWeather(
    params: { city: string; units?: string },
    ctx: Context
  ) {
    const response = await fetch(
      `https://api.weather.example/v1?city=${params.city}`
    );
    const data = await response.json();
    return {
      city: params.city,
      temperature: data.temp,
      conditions: data.conditions,
    };
  }
}
```

### Skill with Hooks

Hooks intercept the agent lifecycle:

```typescript
import { Skill, hook, tool } from 'webagents';
import type { Context, HookData, HookResult } from 'webagents';

class LoggingSkill extends Skill {
  @hook({ lifecycle: 'before_run', priority: 1 })
  async logRequest(
    data: HookData, 
    ctx: Context
  ): Promise<HookResult | void> {
    console.log('Request started:', data.messages);
    // Can modify or abort
    // return { abort: true, abort_reason: 'Blocked' };
  }

  @hook({ lifecycle: 'after_run', priority: 100 })
  async logResponse(
    data: HookData,
    ctx: Context
  ): Promise<HookResult | void> {
    console.log('Response:', data.response);
  }

  @hook({ lifecycle: 'before_tool' })
  async logToolCall(
    data: HookData,
    ctx: Context
  ): Promise<HookResult | void> {
    console.log(`Tool call: ${data.tool_name}`, data.tool_params);
  }

  @hook({ lifecycle: 'on_error' })
  async logError(
    data: HookData,
    ctx: Context
  ): Promise<HookResult | void> {
    console.error('Error:', data.error);
  }
}
```

**Hook Lifecycles:**
- `before_run` - Before processing messages
- `after_run` - After response generated
- `before_tool` - Before tool execution
- `after_tool` - After tool execution
- `before_handoff` - Before LLM processing
- `after_handoff` - After LLM processing
- `on_error` - On any error

### Skill with HTTP Endpoints

```typescript
import { Skill, http } from 'webagents';
import type { Context } from 'webagents';

class APISkill extends Skill {
  @http({ path: '/api/status', method: 'GET' })
  async getStatus(
    req: Request,
    ctx: Context
  ): Promise<Response> {
    return new Response(JSON.stringify({
      status: 'ok',
      uptime: process.uptime(),
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  @http({ path: '/api/data', method: 'POST' })
  async postData(
    req: Request,
    ctx: Context
  ): Promise<Response> {
    const body = await req.json();
    // Process data...
    return new Response(JSON.stringify({ received: body }));
  }
}
```

### Skill with WebSocket Endpoints

```typescript
import { Skill, websocket } from 'webagents';
import type { Context } from 'webagents';

class RealtimeSkill extends Skill {
  @websocket({ path: '/ws/stream', protocols: ['json'] })
  handleStream(ws: WebSocket, ctx: Context): void {
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Process and respond
      ws.send(JSON.stringify({ echo: data }));
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };
  }
}
```

### Scoped Tools

Restrict tools to authenticated users:

```typescript
class AdminSkill extends Skill {
  @tool({
    scopes: ['admin'],  // Requires admin scope
    description: 'Delete all data'
  })
  async deleteAll(params: {}, ctx: Context) {
    if (!ctx.hasScope('admin')) {
      throw new Error('Unauthorized');
    }
    // Perform admin action
  }

  @tool({
    scopes: ['user', 'admin'],  // Requires user OR admin
    description: 'View data'
  })
  async viewData(params: {}, ctx: Context) {
    return { data: '...' };
  }
}
```

### Skill Lifecycle

Override lifecycle methods for setup/cleanup:

```typescript
class DatabaseSkill extends Skill {
  private db: Database | null = null;

  async initialize(): Promise<void> {
    this.db = await Database.connect(process.env.DB_URL);
    console.log('Database connected');
  }

  async cleanup(): Promise<void> {
    await this.db?.close();
    console.log('Database disconnected');
  }

  @tool({ description: 'Query database' })
  async query(params: { sql: string }, ctx: Context) {
    return this.db?.query(params.sql);
  }
}
```

## Combining Skills

Mix and match skills for different capabilities:

```typescript
const agent = new BaseAgent({
  name: 'full-featured-agent',
  skills: [
    // LLM
    new WebLLMSkill({ model: '...' }),
    
    // Custom tools
    new WeatherSkill(),
    new CalculatorSkill(),
    
    // Browser APIs
    new WakeLockSkill(),
    new NotificationsSkill(),
    
    // Logging
    new LoggingSkill(),
  ]
});

// All tools from all skills are available
const response = await agent.run([
  { role: 'user', content: 'What\'s the weather in Tokyo?' }
]);
```

## Dynamic Skill Management

Add and remove skills at runtime:

```typescript
const agent = new BaseAgent({ ... });

// Add skill
const weatherSkill = new WeatherSkill();
agent.addSkill(weatherSkill);

// Check capabilities
console.log(agent.getCapabilities().provides);
// ['weather', ...]

// Remove skill
agent.removeSkill('WeatherSkill');

// Enable/disable individual tools
weatherSkill.setToolEnabled('getWeather', false);
```
