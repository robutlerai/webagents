# Message Router

The Message Router is a central hub for capability-based message routing in WebAgents. It enables automatic wiring of handlers based on declared capabilities, supports custom event types, and provides extensibility through hooks.

## Overview

The router provides:

- **Auto-wiring** - Handlers declare `subscribes` and `produces`, router wires automatically
- **Priority-based selection** - Higher priority handlers are preferred
- **Loop prevention** - Three-layer protection (source tracking, seen set, TTL)
- **Observers** - Non-consuming listeners for logging/analytics
- **System events** - Control flow (stop, cancel, error, ping/pong)
- **Extensibility hooks** - onUnroutable, onError, beforeRoute, afterRoute

## Basic Usage

```python
from webagents.agents.core import MessageRouter, UAMPEvent, Handler, BufferSink

# Create router
router = MessageRouter()

# Register a handler
async def process_text(event, context):
    yield UAMPEvent(
        id='resp-1',
        type='response.delta',
        payload={'text': 'Hello!'}
    )

router.register_handler(Handler(
    name='text-handler',
    subscribes=['input.text'],
    produces=['response.delta'],
    priority=0,
    process=process_text
))

# Set as default handler
router.set_default('text-handler')

# Connect a sink
sink = BufferSink()
router.register_sink(sink)
router.set_active_sink(sink.id)

# Send a message
await router.send(UAMPEvent(
    id='msg-1',
    type='input.text',
    payload={'text': 'Hello'}
))

# Check output
print(sink.get_events())
```

## Handler Declaration

### Using `@handoff` Decorator

```python
from webagents.agents.tools.decorators import handoff

class MySkill(Skill):
    @handoff(
        name='my-handler',
        subscribes=['input.text'],      # Event types to consume
        produces=['response.delta'],    # Event types emitted
        priority=50,                    # Lower = higher priority in Python
    )
    async def process(self, messages, **kwargs):
        # Process messages
        return {'content': 'Response'}
```

### Regex Pattern Matching

```python
import re

@handoff(
    name='translator',
    subscribes=[re.compile(r'^translate\..+$')],  # Matches translate.en, translate.fr
    produces=['response.delta'],
)
async def translate(self, messages, **kwargs):
    # event type might be 'translate.en', 'translate.es', etc.
    pass
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subscribes` | `['input.text']` | Most handlers process text |
| `produces` | `['response.delta']` | Most handlers stream responses |
| `priority` | `50` | Lower = higher priority |

## Observers

Observers receive copies of events without consuming them:

```python
from webagents.agents.tools.decorators import observe

class LoggingSkill(Skill):
    @observe(subscribes=['*'], name='message-logger')
    async def log_messages(self, event, context=None):
        print(f"[{event.type}] {event.payload}")
        # Does NOT consume - message continues to handlers
```

## Transport Sinks

### CallbackSink

```python
from webagents.agents.core import CallbackSink

events = []
sink = CallbackSink(lambda e: events.append(e))
router.register_sink(sink)
```

### BufferSink

```python
from webagents.agents.core import BufferSink

sink = BufferSink(max_size=100)
router.register_sink(sink)

# Later...
all_events = sink.get_events()
```

## Loop Prevention

The router implements three-layer protection:

### 1. Source Tracking

Messages carry their source handler - the router won't route back to the producer.

### 2. Seen Set

Tracks which handlers have already processed a message.

### 3. TTL (Time-to-Live)

Maximum hops a message can traverse (default: 10).

## Extensibility Hooks

### Error Handling

```python
async def handle_error(error, event, handler, context):
    print(f"Handler {handler.name} failed: {error}")

router.on_error(handle_error)
```

### Unroutable Events

```python
async def handle_unroutable(event, context):
    print(f"No handler for {event.type}")

router.on_unroutable(handle_unroutable)
```

### Interceptors

```python
# Before routing - can modify or block
async def before(event, handler, context):
    if is_blocked(event):
        return None  # Block
    return event  # Continue

router.before_route(before)

# After routing - for logging/metrics
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

## Backward Compatibility

Existing code works unchanged. The new `subscribes` and `produces` parameters are optional:

```python
# Before (still works)
@handoff(name='my-handler', priority=10)
async def process(self, messages, **kwargs):
    pass

# Equivalent to:
@handoff(
    name='my-handler',
    priority=10,
    subscribes=['input.text'],      # default
    produces=['response.delta'],    # default
)
async def process(self, messages, **kwargs):
    pass
```

## API Reference

### UAMPEvent

```python
@dataclass
class UAMPEvent:
    id: str                           # Unique message ID
    type: str                         # Event type
    payload: Dict[str, Any]           # Event payload
    source: Optional[str] = None      # Handler that produced this
    ttl: Optional[int] = None         # Time-to-live
    seen: Optional[Set[str]] = None   # Handlers that processed this
```

### Handler

```python
@dataclass
class Handler:
    name: str                                     # Handler name
    subscribes: List[Union[str, Pattern]]         # Event patterns
    produces: List[str]                           # Output event types
    priority: int = 0                             # Priority (lower = higher in Python)
    process: Callable[..., AsyncGenerator] = None # Handler function
```

### Observer

```python
@dataclass
class Observer:
    name: str                              # Observer name
    subscribes: List[Union[str, Pattern]]  # Event patterns
    handler: Callable[..., Awaitable]      # Handler function
```

### TransportSink

```python
class TransportSink(ABC):
    @property
    def id(self) -> str: ...
    
    @property
    def is_active(self) -> bool: ...
    
    async def send(self, event: Dict) -> None: ...
    
    def close(self) -> None: ...
```

### MessageRouter

```python
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
