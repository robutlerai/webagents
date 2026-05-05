---
title: Agent Hooks
description: Lifecycle integration points for skills — events, priority, and the unified request context.
---

# Agent Hooks

Hooks provide lifecycle integration points to react to events during request processing. Hooks can be defined in skills or as standalone functions.

Hooks are executed in priority order (lower numbers first) and receive the unified request context. Keep hooks small and deterministic; avoid blocking operations and always return the context.

## Hook Types

### Skill Hooks

Defined within skills using the `@hook` decorator:

```typescript tab="TypeScript"
import { Skill, hook } from 'webagents';
import type { HookData, Context } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @hook({ lifecycle: 'on_connection', priority: 10 })
  async setupRequest(data: HookData, ctx: Context) {
    ctx.set('custom_data', 'value');
    return data;
  }
}
```

```python tab="Python"
from webagents.agents.skills import Skill
from webagents.agents.skills.decorators import hook

class MySkill(Skill):
    @hook("on_connection", priority=10)
    async def setup_request(self, context):
        """Called when request starts"""
        context["custom_data"] = "value"
        return context
```

### Standalone Hooks

Decorated functions that can be passed to agents:

```typescript tab="TypeScript"
// In TypeScript, hooks are class members. Wrap standalone hook logic
// in a small Skill class and pass an instance to the agent.

import { BaseAgent, Skill, hook } from 'webagents';

class StandaloneHooks extends Skill {
  readonly name = 'standalone-hooks';

  @hook({ lifecycle: 'on_message', priority: 5 })
  async logMessages(data, ctx) {
    console.log('Message:', data.messages?.at(-1));
    return data;
  }

  @hook({ lifecycle: 'on_connection' })
  async setupAnalytics(data, ctx) {
    ctx.set('session_start', Date.now());
    return data;
  }
}

const agent = new BaseAgent({
  name: 'my-agent',
  model: 'openai/gpt-4o',
  skills: [new StandaloneHooks()],
});
```

```python tab="Python"
import time
from webagents.agents.skills.decorators import hook
from webagents.agents import BaseAgent

@hook("on_message", priority=5)
async def log_messages(context):
    """Log all messages"""
    print(f"Message: {context.messages[-1]}")
    return context

@hook("on_connection")
async def setup_analytics(context):
    """Initialize analytics tracking"""
    context["session_start"] = time.time()
    return context

agent = BaseAgent(
    name="my-agent",
    model="openai/gpt-4o",
    hooks=[log_messages, setup_analytics],
)
```

## Available Hooks

Hooks are executed in the following order during request processing:

1. **on_connection** — Once per request (initialization)
2. **before_llm_call** — Before each LLM call in the agentic loop
3. **after_llm_call** — After each LLM response in the agentic loop
4. **on_chunk** — For each streaming chunk (streaming only)
5. **before_toolcall** — Before each tool execution
6. **after_toolcall** — After each tool execution
7. **on_message** — Once per request (before finalization)
8. **finalize_connection** — Once per request (cleanup)

### `on_connection`

Called once when a new request connection is established.

Typical responsibilities:

- Authentication and identity extraction (e.g., `AuthSkill`)
- Payment token validation and minimum-balance checks (e.g., `PaymentSkill`)
- Request-scoped initialization (timers, correlation IDs)

```typescript tab="TypeScript"
@hook({ lifecycle: 'on_connection' })
async onConnection(data: HookData, ctx: Context) {
  const userId = ctx.auth?.userId;
  const isStreaming = data.stream === true;
  ctx.set('request_start', Date.now());
  return data;
}
```

```python tab="Python"
@hook("on_connection")
async def on_connection(self, context):
    """Initialize request processing"""
    user_id = context.peer_user_id
    is_streaming = context.stream

    context["request_start"] = time.time()
    return context
```

### `on_message`

Called for each message in the conversation.

```typescript tab="TypeScript"
@hook({ lifecycle: 'on_message' })
async onMessage(data: HookData, ctx: Context) {
  const message = data.messages?.at(-1);
  if (message?.role === 'user') {
    ctx.set('intent', this.analyzeIntent(String(message.content ?? '')));
  }
  return data;
}
```

```python tab="Python"
@hook("on_message")
async def on_message(self, context):
    """Process each message"""
    message = context.messages[-1]
    if message["role"] == "user":
        context["intent"] = self.analyze_intent(message["content"])
    return context
```

### `before_llm_call`

```typescript tab="TypeScript"
@hook({ lifecycle: 'before_llm_call', priority: 5 })
async beforeLlmCall(data: HookData, ctx: Context) {
  const messages = ctx.get('conversation_messages') ?? [];
  const processed = this.processMessages(messages as unknown[]);
  ctx.set('conversation_messages', processed);
  return data;
}
```

```python tab="Python"
@hook("before_llm_call", priority=5)
async def before_llm_call(self, context):
    """Preprocess messages before LLM"""
    messages = context.get('conversation_messages', [])
    processed_messages = self.process_messages(messages)
    context.set('conversation_messages', processed_messages)
    return context
```

### `after_llm_call`

```typescript tab="TypeScript"
@hook({ lifecycle: 'after_llm_call', priority: 10 })
async afterLlmCall(data: HookData, ctx: Context) {
  const response = ctx.get('llm_response') as { usage?: object } | undefined;
  await this.trackLlmUsage(response?.usage ?? {});
  return data;
}
```

```python tab="Python"
@hook("after_llm_call", priority=10)
async def after_llm_call(self, context):
    """Process LLM response"""
    response = context.get('llm_response')
    usage = response.get('usage', {})
    await self.track_llm_usage(usage)
    return context
```

### `before_toolcall`

```typescript tab="TypeScript"
@hook({ lifecycle: 'before_toolcall', priority: 1 })
async beforeToolcall(data: HookData, ctx: Context) {
  const fnName = data.tool_call?.function?.name;
  if (!this.isToolAllowed(fnName, ctx.auth?.userId)) {
    data.tool_call.function.name = 'tool_blocked';
    data.tool_call.function.arguments = '{}';
  }
  return data;
}
```

```python tab="Python"
@hook("before_toolcall", priority=1)
async def before_toolcall(self, context):
    """Validate tool execution"""
    tool_call = context["tool_call"]
    function_name = tool_call["function"]["name"]

    if not self.is_tool_allowed(function_name, context.peer_user_id):
        context["tool_call"]["function"]["name"] = "tool_blocked"
        context["tool_call"]["function"]["arguments"] = "{}"

    return context
```

### `after_toolcall`

```typescript tab="TypeScript"
@hook({ lifecycle: 'after_toolcall' })
async afterToolcall(data: HookData, ctx: Context) {
  const toolName = data.tool_call?.function?.name;
  await this.logToolUsage({
    tool: toolName,
    resultSize: String(data.tool_result ?? '').length,
    user: ctx.auth?.userId,
  });
  if (toolName === 'search') {
    data.tool_result = this.formatSearchResults(data.tool_result);
  }
  return data;
}
```

```python tab="Python"
@hook("after_toolcall")
async def after_toolcall(self, context):
    """Process tool results"""
    tool_result = context["tool_result"]
    tool_name = context["tool_call"]["function"]["name"]

    await self.log_tool_usage(
        tool=tool_name,
        result_size=len(tool_result),
        user=context.peer_user_id,
    )

    if tool_name == "search":
        context["tool_result"] = self.format_search_results(tool_result)

    return context
```

### `on_chunk`

```typescript tab="TypeScript"
@hook({ lifecycle: 'on_chunk' })
async onChunk(data: HookData, ctx: Context) {
  const content = String(data.content ?? '');
  if (this.containsSensitiveInfo(content)) {
    data.chunk.choices[0].delta.content = '[REDACTED]';
  }
  ctx.set('chunks_processed', Number(ctx.get('chunks_processed') ?? 0) + 1);
  return data;
}
```

```python tab="Python"
@hook("on_chunk")
async def on_chunk(self, context):
    """Process streaming chunks"""
    chunk = context["chunk"]
    content = context.get("content", "")

    if self.contains_sensitive_info(content):
        context["chunk"]["choices"][0]["delta"]["content"] = "[REDACTED]"

    context["chunks_processed"] = context.get("chunks_processed", 0) + 1
    return context
```

### `before_handoff`

```typescript tab="TypeScript"
@hook({ lifecycle: 'before_handoff' })
async beforeHandoff(data: HookData, ctx: Context) {
  const target = data.handoff_agent;
  ctx.set('handoff_metadata', {
    sourceAgent: ctx.metadata.agent_name,
    timestamp: Date.now(),
    reason: data.handoff_reason,
  });
  if (!this.canHandoffTo(target)) {
    throw new Error(`Cannot handoff to ${target}`);
  }
  return data;
}
```

```python tab="Python"
@hook("before_handoff")
async def before_handoff(self, context):
    """Prepare for agent handoff"""
    target_agent = context["handoff_agent"]

    context["handoff_metadata"] = {
        "source_agent": context.agent_name,
        "timestamp": time.time(),
        "reason": context.get("handoff_reason"),
    }

    if not self.can_handoff_to(target_agent):
        raise HandoffError(f"Cannot handoff to {target_agent}")

    return context
```

### `after_handoff`

```typescript tab="TypeScript"
@hook({ lifecycle: 'after_handoff' })
async afterHandoff(data: HookData, ctx: Context) {
  const result = data.handoff_result;
  const meta = ctx.get('handoff_metadata') as { timestamp: number };
  await this.logHandoff({
    target: data.handoff_agent,
    success: result?.success,
    duration: (Date.now() - meta.timestamp) / 1000,
  });
  return data;
}
```

```python tab="Python"
@hook("after_handoff")
async def after_handoff(self, context):
    """Process handoff results"""
    handoff_result = context["handoff_result"]

    await self.log_handoff(
        target=context["handoff_agent"],
        success=handoff_result.get("success"),
        duration=time.time() - context["handoff_metadata"]["timestamp"],
    )

    return context
```

### `finalize_connection`

```typescript tab="TypeScript"
@hook({ lifecycle: 'finalize_connection' })
async finalizeConnection(data: HookData, ctx: Context) {
  const start = Number(ctx.get('request_start') ?? Date.now());
  const duration = (Date.now() - start) / 1000;
  await this.logRequestComplete({
    requestId: ctx.metadata.completion_id,
    duration,
    tokens: ctx.get('usage') ?? {},
    chunks: ctx.get('chunks_processed') ?? 0,
  });
  this.cleanupRequestResources(String(ctx.metadata.completion_id));
  return data;
}
```

```python tab="Python"
@hook("finalize_connection")
async def finalize_connection(self, context):
    """Clean up and finalize"""
    duration = time.time() - context.get("request_start", time.time())

    await self.log_request_complete(
        request_id=context.completion_id,
        duration=duration,
        tokens=context.get("usage", {}),
        chunks=context.get("chunks_processed", 0),
    )

    self.cleanup_request_resources(context.completion_id)
    return context
```

## Hook Priority

Hooks execute in priority order (lower numbers first):

```typescript tab="TypeScript"
class SecuritySkill extends Skill {
  readonly name = 'security';
  @hook({ lifecycle: 'on_message', priority: 1 }) async securityCheck(d, c) { return d; }
}
class LoggingSkill extends Skill {
  readonly name = 'logging';
  @hook({ lifecycle: 'on_message', priority: 10 }) async logMessage(d, c)   { return d; }
}
class AnalyticsSkill extends Skill {
  readonly name = 'analytics';
  @hook({ lifecycle: 'on_message', priority: 20 }) async analyzeMessage(d, c) { return d; }
}
```

```python tab="Python"
class SecuritySkill(Skill):
    @hook("on_message", priority=1)  # Runs first
    async def security_check(self, context):
        return context

class LoggingSkill(Skill):
    @hook("on_message", priority=10)  # Runs second
    async def log_message(self, context):
        return context

class AnalyticsSkill(Skill):
    @hook("on_message", priority=20)  # Runs third
    async def analyze_message(self, context):
        return context
```

## Context Object

The context exposes:

| Field | Description |
|-------|-------------|
| `messages` | Conversation messages |
| `stream` | Streaming enabled |
| `auth.userId` (TS) / `peer_user_id` (Py) | Caller identifier |
| `metadata.completion_id` (TS) / `completion_id` (Py) | Request ID |
| `metadata.model` / `model` | Model name |
| `metadata.agent_name` / `agent_name` | Agent name |
| `metadata.usage` / `usage` | Token usage |
| `tool_call`, `tool_result`, `chunk`, `content` | Hook-specific fields |

In TypeScript, the data argument carries event-specific fields and the `Context` carries authentication, payment, and metadata. In Python, both live on the single `context` dict-like object.

## Practical Examples

### Rate Limiting

```typescript tab="TypeScript"
class RateLimitSkill extends Skill {
  readonly name = 'rate-limit';
  private requestCounts = new Map<string, number>();

  @hook({ lifecycle: 'on_connection', priority: 1 })
  async checkRateLimit(data, ctx) {
    const userId = String(ctx.auth?.userId ?? 'anonymous');
    const count = this.requestCounts.get(userId) ?? 0;
    if (count >= 100) {
      throw new Error('Rate limit exceeded');
    }
    this.requestCounts.set(userId, count + 1);
    return data;
  }
}
```

```python tab="Python"
class RateLimitSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.request_counts = {}

    @hook("on_connection", priority=1)
    async def check_rate_limit(self, context):
        user_id = context.peer_user_id

        count = self.request_counts.get(user_id, 0)
        if count >= 100:  # 100 requests per hour
            raise RateLimitError("Rate limit exceeded")

        self.request_counts[user_id] = count + 1
        return context
```

### Content Moderation

```typescript tab="TypeScript"
class ModerationSkill extends Skill {
  readonly name = 'moderation';

  @hook({ lifecycle: 'on_message', priority: 5 })
  async moderateInput(data, ctx) {
    const last = data.messages?.at(-1);
    if (last?.role === 'user' && this.isInappropriate(String(last.content))) {
      last.content = 'I cannot process inappropriate content.';
    }
    return data;
  }

  @hook({ lifecycle: 'on_chunk', priority: 5 })
  async moderateOutput(data, ctx) {
    const content = String(data.content ?? '');
    if (this.isInappropriate(content)) {
      data.chunk.choices[0].delta.content = '';
    }
    return data;
  }

  private isInappropriate(_: string) { return false; }
}
```

```python tab="Python"
class ModerationSkill(Skill):
    @hook("on_message", priority=5)
    async def moderate_input(self, context):
        """Filter inappropriate content"""
        message = context.messages[-1]

        if message["role"] == "user":
            if self.is_inappropriate(message["content"]):
                context.messages[-1]["content"] = "I cannot process inappropriate content."

        return context

    @hook("on_chunk", priority=5)
    async def moderate_output(self, context):
        """Filter streaming output"""
        content = context.get("content", "")

        if self.is_inappropriate(content):
            context["chunk"]["choices"][0]["delta"]["content"] = ""

        return context
```

### Analytics Collection

```typescript tab="TypeScript"
class AnalyticsSkill extends Skill {
  readonly name = 'analytics';

  @hook({ lifecycle: 'on_connection' })
  async startAnalytics(data, ctx) {
    ctx.set('analytics', { startTime: Date.now(), events: [] });
    return data;
  }

  @hook({ lifecycle: 'on_message' })
  async trackMessage(data, ctx) {
    const a = ctx.get('analytics') as any;
    a.events.push({ type: 'message', role: data.messages?.at(-1)?.role, timestamp: Date.now() });
    return data;
  }

  @hook({ lifecycle: 'before_toolcall' })
  async trackToolStart(data, ctx) {
    ctx.set('tool_start_time', Date.now());
    return data;
  }

  @hook({ lifecycle: 'after_toolcall' })
  async trackToolEnd(data, ctx) {
    const duration = (Date.now() - Number(ctx.get('tool_start_time') ?? Date.now())) / 1000;
    const a = ctx.get('analytics') as any;
    a.events.push({
      type: 'tool',
      name: data.tool_call?.function?.name,
      duration,
      timestamp: Date.now(),
    });
    return data;
  }

  @hook({ lifecycle: 'finalize_connection' })
  async sendAnalytics(data, ctx) {
    const a = ctx.get('analytics') as any;
    a.totalDuration = (Date.now() - a.startTime) / 1000;
    await this.sendToAnalyticsService(a);
    return data;
  }

  private async sendToAnalyticsService(_: unknown) {}
}
```

```python tab="Python"
class AnalyticsSkill(Skill):
    @hook("on_connection")
    async def start_analytics(self, context):
        context["analytics"] = {
            "start_time": time.time(),
            "events": [],
        }
        return context

    @hook("on_message")
    async def track_message(self, context):
        context["analytics"]["events"].append({
            "type": "message",
            "role": context.messages[-1]["role"],
            "timestamp": time.time(),
        })
        return context

    @hook("before_toolcall")
    async def track_tool_start(self, context):
        context["tool_start_time"] = time.time()
        return context

    @hook("after_toolcall")
    async def track_tool_end(self, context):
        duration = time.time() - context.get("tool_start_time", time.time())
        context["analytics"]["events"].append({
            "type": "tool",
            "name": context["tool_call"]["function"]["name"],
            "duration": duration,
            "timestamp": time.time(),
        })
        return context

    @hook("finalize_connection")
    async def send_analytics(self, context):
        analytics = context.get("analytics", {})
        analytics["total_duration"] = time.time() - analytics.get("start_time", time.time())
        await self.send_to_analytics_service(analytics)
        return context
```

## Best Practices

1. **Always return context (or `data`)** — hooks must return their input data so subsequent hooks see the mutations.
2. **Use priorities wisely** — order matters for dependent operations.
3. **Handle errors gracefully** — `finalize_connection` runs even if a prior hook throws; rely on it for cleanup.
4. **Keep hooks lightweight** — avoid heavy synchronous processing.
5. **Use context for state** — don't store request state on instance fields shared across requests.
