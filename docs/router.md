# UAMP Message Router

The Message Router is the central hub for capability-based message routing in WebAgents. It enables automatic wiring of handlers based on declared capabilities, supports custom event types, and provides extensibility through hooks.

## Design Principles

1. **Router is the hub** - All messages flow through the router
2. **Transports are endpoints** - WebSocket, HTTP, etc. connect as sources/sinks
3. **Auto-wiring by default** - Skills declare capabilities, router wires automatically
4. **Explicit wiring available** - `router.route()` for custom routes
5. **Loop prevention** - Three-layer protection against infinite loops
6. **Backward compatible** - New params are optional with sensible defaults

## Quick Start

### Simple Text Agent (Zero Config)

```typescript
import { BaseAgent, WebLLMSkill, WebSocketSink } from 'webagents-ts';

const agent = new BaseAgent({ id: 'chat-agent' });
agent.addSkill(new WebLLMSkill());  // Auto-becomes default handler

// Connect transport
const ws = new WebSocket('...');
agent.connectTransport(new WebSocketSink(ws, 'client-1'));

// Messages just work - router auto-wires everything
await agent.sendText('Hello!');
// -> Routes to LLM -> response.delta -> delivered to WebSocket
```

### Speech Pipeline (Auto-Wired)

```typescript
const agent = new BaseAgent({ id: 'voice-agent' });
agent.addSkill(new SpeechToTextSkill());  // subscribes: input.audio, produces: input.text
agent.addSkill(new WebLLMSkill());        // subscribes: input.text, produces: response.delta
agent.addSkill(new TextToSpeechSkill());  // subscribes: response.delta, produces: audio.delta

// Send audio - automatically flows through pipeline
await agent.sendAudio(audioData);
// input.audio -> STT -> input.text -> LLM -> response.delta -> TTS -> audio.delta -> client
```

### Custom Capability Routing

```typescript
import { NLISkill } from 'webagents-ts';

const agent = new BaseAgent({ id: 'smart-agent' });
agent.addSkill(new WebLLMSkill());

// Add NLI skill with custom capability
agent.addSkill(new NLISkill({
  agentUrl: 'https://emotions.webagents.ai',
  capability: 'analyze_emotion',
}));

// Messages with type 'analyze_emotion' auto-route to external agent
await agent.processMessage({
  id: 'msg-1',
  type: 'analyze_emotion',
  payload: { text: 'I feel great today!' }
});
```

## Handler Declaration

### Using `@handoff` Decorator

```typescript
import { Skill, handoff } from 'webagents-ts';

class MySkill extends Skill {
  @handoff({
    name: 'my-handler',
    subscribes: ['input.text'],      // Event types to consume
    produces: ['response.delta'],    // Event types emitted
    priority: 10,                    // Higher = preferred
  })
  async *processText(events, context) {
    for (const event of events) {
      yield {
        type: 'response.delta',
        event_id: 'resp-1',
        delta: { text: 'Hello!' },
      };
    }
  }
}
```

### Regex Pattern Matching

```typescript
@handoff({
  name: 'translator',
  subscribes: [/^translate\..+$/],  // Matches translate.en, translate.fr, etc.
  produces: ['response.delta'],
})
async *translate(events, context) {
  // event.type might be 'translate.en', 'translate.es', etc.
}
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subscribes` | `['input.text']` | Most handlers process text |
| `produces` | `['response.delta']` | Most handlers stream responses |
| `priority` | `0` | Higher = preferred |

## Observers (Non-Consuming)

Observers receive copies of events without consuming them:

```typescript
import { observe } from 'webagents-ts';

class LoggingSkill extends Skill {
  @observe({
    name: 'message-logger',
    subscribes: ['*'],  // Wildcard - sees everything
  })
  async onMessage(event) {
    console.log(`[${event.type}]`, event.payload);
    // Does NOT consume - message continues to handlers
  }
}
```

## Transport Sinks

### WebSocket Sink

```typescript
import { WebSocketSink } from 'webagents-ts';

const ws = new WebSocket('wss://example.com');
agent.connectTransport(new WebSocketSink(ws, 'ws-1'));
```

### SSE Sink

```typescript
import { SSESink } from 'webagents-ts';

// In an HTTP handler
app.get('/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  agent.connectTransport(new SSESink(res, 'sse-1'));
});
```

### Callback Sink (Testing)

```typescript
import { CallbackSink } from 'webagents-ts';

const events: ServerEvent[] = [];
agent.connectTransport(new CallbackSink(
  (event) => events.push(event),
  'test-sink'
));
```

### Buffer Sink (Collecting)

```typescript
import { BufferSink } from 'webagents-ts';

const sink = new BufferSink({ maxSize: 100 });
agent.connectTransport(sink);

// Later...
const allEvents = sink.getEvents();
```

## Loop Prevention

The router implements three-layer protection against infinite loops:

### 1. Source Tracking

Messages carry their source handler - the router won't route back to the producer:

```typescript
{
  id: 'msg-1',
  type: 'input.text',
  source: 'stt-handler',  // Won't be routed back to stt-handler
  payload: { text: 'Hello' }
}
```

### 2. Seen Set

Tracks which handlers have already processed a message:

```typescript
{
  id: 'msg-1',
  type: 'input.text',
  seen: ['stt-handler', 'llm-handler'],  // Won't go back to these
  payload: { text: 'Hello' }
}
```

### 3. TTL (Time-to-Live)

Maximum hops a message can traverse:

```typescript
{
  id: 'msg-1',
  type: 'input.text',
  ttl: 10,  // Decremented each hop, error when 0
  payload: { text: 'Hello' }
}
```

## Extensibility Hooks

### Error Handling

```typescript
agent.router.onError(async (error, event, handler, context) => {
  console.error(`Handler ${handler.name} failed:`, error);
  await alertOps(error);
});
```

### Unroutable Events

```typescript
agent.router.onUnroutable(async (event, context) => {
  console.warn(`No handler for ${event.type}`);
  await logToAnalytics('unroutable', event);
});
```

### Interceptors

```typescript
// Before routing - can modify or block
agent.router.beforeRoute(async (event, handler, context) => {
  // Return null to block, or modified event
  if (isBlacklisted(event)) return null;
  return { ...event, payload: { ...event.payload, enriched: true } };
});

// After routing - for logging/metrics
agent.router.afterRoute(async (event, handler, context) => {
  metrics.record('routed', handler.name);
  return event;
});
```

## System Events

| Event | Description |
|-------|-------------|
| `system.error` | Error occurred during processing |
| `system.stop` | Request to stop current processing |
| `system.cancel` | Cancel and cleanup resources |
| `system.ping` | Keep-alive request |
| `system.pong` | Keep-alive response |
| `system.unroutable` | No handler found for message |

## Direct Router API

```typescript
// Get the router instance
const router = agent.router;

// Register handler manually
router.registerHandler({
  name: 'custom-handler',
  subscribes: ['custom.event'],
  produces: ['response.delta'],
  priority: 0,
  process: async function* (event, context) {
    yield { type: 'response.delta', payload: { text: 'OK' } };
  },
});

// Add explicit route
router.route('special.event', 'custom-handler', 100);

// Register observer
router.registerObserver({
  name: 'logger',
  subscribes: ['*'],
  handler: async (event) => console.log(event),
});

// Send message directly
await router.send({
  id: 'msg-1',
  type: 'input.text',
  payload: { text: 'Hello' },
});
```

## Best Practices

1. **Use default values** - Only specify `subscribes`/`produces` when needed
2. **Keep handlers focused** - One handler, one responsibility
3. **Use observers for side effects** - Logging, analytics, debugging
4. **Set appropriate priorities** - Higher for specialized handlers
5. **Handle errors gracefully** - Use the `onError` hook

## Migration Guide

Existing code works unchanged. The new `subscribes` and `produces` parameters are optional:

```typescript
// Before (still works)
@handoff({ name: 'my-handler', priority: 10 })
async *process(events) { ... }

// Equivalent to:
@handoff({
  name: 'my-handler',
  priority: 10,
  subscribes: ['input.text'],      // default
  produces: ['response.delta'],    // default
})
async *process(events) { ... }
```
