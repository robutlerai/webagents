---
title: Message Router
description: Capability-based event routing — auto-wiring, observers, loop prevention, and extensibility hooks.
---

# Message Router

The Message Router is a central hub for capability-based message routing in WebAgents. It enables automatic wiring of handlers based on declared capabilities, supports custom event types, and provides extensibility through hooks.

## Overview

The router provides:

- **Auto-wiring** — handlers declare `subscribes` and `produces`, the router wires them automatically.
- **Priority-based selection** — preferred handlers run first.
- **Loop prevention** — three-layer protection (source tracking, seen set, TTL).
- **Observers** — non-consuming listeners for logging / analytics.
- **System events** — control flow (stop, cancel, error, ping/pong).
- **Extensibility hooks** — `onUnroutable`, `onError`, `beforeRoute`, `afterRoute`.

## Basic Usage

```typescript tab="TypeScript"
import { MessageRouter, BufferSink } from 'webagents';
import type { UAMPEvent, RouterContext } from 'webagents';

const router = new MessageRouter();

async function* processText(event: UAMPEvent, ctx: RouterContext) {
  yield {
    id: 'resp-1',
    type: 'response.delta',
    payload: { text: 'Hello!' },
  } satisfies UAMPEvent;
}

router.registerHandler({
  name: 'text-handler',
  subscribes: ['input.text'],
  produces: ['response.delta'],
  priority: 0,
  process: processText,
});

router.setDefault('text-handler');

const sink = new BufferSink();
router.registerSink(sink);
router.setActiveSink(sink.id);

await router.send({
  id: 'msg-1',
  type: 'input.text',
  payload: { text: 'Hello' },
});

console.log(sink.getEvents());
```

```python tab="Python"
from webagents.agents.core import MessageRouter, UAMPEvent, Handler, BufferSink

router = MessageRouter()

async def process_text(event, context):
    yield UAMPEvent(
        id='resp-1',
        type='response.delta',
        payload={'text': 'Hello!'},
    )

router.register_handler(Handler(
    name='text-handler',
    subscribes=['input.text'],
    produces=['response.delta'],
    priority=0,
    process=process_text,
))

router.set_default('text-handler')

sink = BufferSink()
router.register_sink(sink)
router.set_active_sink(sink.id)

await router.send(UAMPEvent(
    id='msg-1',
    type='input.text',
    payload={'text': 'Hello'},
))

print(sink.get_events())
```

## Handler Declaration

### Using `@handoff`

```typescript tab="TypeScript"
import { Skill, handoff } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @handoff({
    name: 'my-handler',
    subscribes: ['input.text'],
    produces: ['response.delta'],
    priority: 50,
  })
  async *process(events) {
    yield { type: 'response.delta', delta: 'Response' } as const;
  }
}
```

```python tab="Python"
from webagents.agents.tools.decorators import handoff

class MySkill(Skill):
    @handoff(
        name='my-handler',
        subscribes=['input.text'],     # Event types to consume
        produces=['response.delta'],   # Event types emitted
        priority=50,                   # Lower = higher priority
    )
    async def process(self, messages, **kwargs):
        return {'content': 'Response'}
```

### Regex Pattern Matching

```typescript tab="TypeScript"
@handoff({
  name: 'translator',
  subscribes: [/^translate\..+$/],   // matches translate.en, translate.fr
  produces: ['response.delta'],
})
async *translate(events) {
  // event.type might be 'translate.en', 'translate.es', etc.
  yield { type: 'response.delta', delta: '...' } as const;
}
```

```python tab="Python"
import re

@handoff(
    name='translator',
    subscribes=[re.compile(r'^translate\..+$')],
    produces=['response.delta'],
)
async def translate(self, messages, **kwargs):
    pass
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subscribes` | `['input.text']` | Most handlers process text |
| `produces` | `['response.delta']` | Most handlers stream responses |
| `priority` | `50` (Python) / `0` (TS) | Lower runs first; in TS with priority `0` and the higher-priority interpretation, see [`router.ts`](../../typescript/src/core/router.ts) |

## Observers

Observers receive copies of events without consuming them:

```typescript tab="TypeScript"
import { Skill, observe } from 'webagents';

class LoggingSkill extends Skill {
  readonly name = 'logging';

  @observe({ name: 'message-logger', subscribes: ['*'] })
  async logMessages(event) {
    console.log(`[${event.type}]`, event.payload);
    // Does NOT consume — message continues to handlers
  }
}
```

```python tab="Python"
from webagents.agents.tools.decorators import observe

class LoggingSkill(Skill):
    @observe(subscribes=['*'], name='message-logger')
    async def log_messages(self, event, context=None):
        print(f"[{event.type}] {event.payload}")
```

## Transport Sinks

### CallbackSink

```typescript tab="TypeScript"
import { CallbackSink } from 'webagents';

const events: unknown[] = [];
const sink = new CallbackSink((e) => events.push(e));
router.registerSink(sink);
```

```python tab="Python"
from webagents.agents.core import CallbackSink

events = []
sink = CallbackSink(lambda e: events.append(e))
router.register_sink(sink)
```

### BufferSink

```typescript tab="TypeScript"
import { BufferSink } from 'webagents';

const sink = new BufferSink({ maxSize: 100 });
router.registerSink(sink);

const allEvents = sink.getEvents();
```

```python tab="Python"
from webagents.agents.core import BufferSink

sink = BufferSink(max_size=100)
router.register_sink(sink)

all_events = sink.get_events()
```

## Loop Prevention

The router implements three-layer protection:

1. **Source tracking** — messages carry their source handler; the router won't route back to the producer.
2. **Seen set** — tracks which handlers have already processed a message.
3. **TTL (Time-to-Live)** — maximum hops a message can traverse (default: 10).

## Extensibility Hooks

### Error Handling

```typescript tab="TypeScript"
router.onError(async (error, event, handler, context) => {
  console.error(`Handler ${handler.name} failed:`, error);
});
```

```python tab="Python"
async def handle_error(error, event, handler, context):
    print(f"Handler {handler.name} failed: {error}")

router.on_error(handle_error)
```

### Unroutable Events

```typescript tab="TypeScript"
router.onUnroutable(async (event, context) => {
  console.warn(`No handler for ${event.type}`);
});
```

```python tab="Python"
async def handle_unroutable(event, context):
    print(f"No handler for {event.type}")

router.on_unroutable(handle_unroutable)
```

### Interceptors

```typescript tab="TypeScript"
router.beforeRoute(async (event, handler, context) => {
  if (isBlocked(event)) return null;  // Block
  return event;                       // Continue
});

router.afterRoute(async (event, handler, context) => {
  logMetric('routed', handler.name);
  return event;
});

function isBlocked(_: unknown) { return false; }
function logMetric(_: string, __: string) {}
```

```python tab="Python"
async def before(event, handler, context):
    if is_blocked(event):
        return None  # Block
    return event  # Continue

router.before_route(before)

async def after(event, handler, context):
    log_metric('routed', handler.name)
    return event

router.after_route(after)
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

## Backward Compatibility (Python)

The new `subscribes` / `produces` parameters are optional in Python. Existing code works unchanged.

```python tab="Python"
# Before (still works)
@handoff(name='my-handler', priority=10)
async def process(self, messages, **kwargs):
    pass

# Equivalent to:
@handoff(
    name='my-handler',
    priority=10,
    subscribes=['input.text'],
    produces=['response.delta'],
)
async def process(self, messages, **kwargs):
    pass
```

```typescript tab="TypeScript"
// In TypeScript, defaults are also applied automatically:
@handoff({ name: 'my-handler', priority: 10 })
async *process(events) {
  yield { type: 'response.delta', delta: '...' } as const;
}

// Equivalent to:
@handoff({
  name: 'my-handler',
  priority: 10,
  subscribes: ['input.text'],
  produces: ['response.delta'],
})
async *process(events) { /* ... */ }
```

## API Reference

### `UAMPEvent`

```typescript tab="TypeScript"
interface UAMPEvent {
  id: string;
  type: string;
  payload: Record<string, unknown>;
  source?: string;       // Handler that produced this
  ttl?: number;          // Time-to-live
  seen?: string[];       // Handlers that processed this
}
```

```python tab="Python"
@dataclass
class UAMPEvent:
    id: str                           # Unique message ID
    type: str                         # Event type
    payload: Dict[str, Any]           # Event payload
    source: Optional[str] = None      # Handler that produced this
    ttl: Optional[int] = None         # Time-to-live
    seen: Optional[Set[str]] = None   # Handlers that processed this
```

### `Handler`

```typescript tab="TypeScript"
interface Handler {
  name: string;
  subscribes: (string | RegExp)[];
  produces: string[];
  priority?: number;
  process: (event: UAMPEvent, context?: RouterContext) => AsyncGenerator<UAMPEvent>;
}
```

```python tab="Python"
@dataclass
class Handler:
    name: str
    subscribes: List[Union[str, Pattern]]
    produces: List[str]
    priority: int = 0
    process: Callable[..., AsyncGenerator] = None
```

### `TransportSink`

```typescript tab="TypeScript"
abstract class TransportSink {
  readonly id: string;
  readonly isActive: boolean;
  abstract send(event: ServerEvent): Promise<void>;
  abstract close(): void;
}
```

```python tab="Python"
class TransportSink(ABC):
    @property
    def id(self) -> str: ...

    @property
    def is_active(self) -> bool: ...

    async def send(self, event: Dict) -> None: ...

    def close(self) -> None: ...
```

### `MessageRouter`

```typescript tab="TypeScript"
class MessageRouter {
  send(event: UAMPEvent, context?: RouterContext): Promise<void>;
  registerHandler(handler: Handler): void;
  unregisterHandler(name: string): void;
  registerObserver(observer: Observer): void;
  unregisterObserver(name: string): void;
  route(eventType: string, handlerName: string, priority?: number): void;
  registerSink(sink: TransportSink): void;
  registerDefaultSink(sink: TransportSink): void;
  unregisterSink(sinkId: string): void;
  setActiveSink(sinkId: string): void;
  setDefault(handlerName: string): void;
  onUnroutable(handler: Function): void;
  onError(handler: Function): void;
  beforeRoute(interceptor: Function): void;
  afterRoute(interceptor: Function): void;
}
```

```python tab="Python"
class MessageRouter:
    async def send(event: UAMPEvent, context: RouterContext = None) -> None
    def register_handler(handler: Handler) -> None
    def unregister_handler(name: str) -> None
    def register_observer(observer: Observer) -> None
    def unregister_observer(name: str) -> None
    def route(event_type: str, handler_name: str, priority: int = None) -> None
    def register_sink(sink: TransportSink) -> None
    def register_default_sink(sink: TransportSink) -> None
    def unregister_sink(sink_id: str) -> None
    def set_active_sink(sink_id: str) -> None
    def set_default(handler_name: str) -> None
    def on_unroutable(handler: Callable) -> None
    def on_error(handler: Callable) -> None
    def before_route(interceptor: Callable) -> None
    def after_route(interceptor: Callable) -> None
```
