# Agent Hooks

Hooks provide lifecycle integration points to react to events during request processing. Hooks can be defined in skills or as standalone functions.

Hooks are executed in priority order (lower numbers first) and receive the unified request context. Keep hooks small and deterministic; avoid blocking operations and always return the context.

## Hook Types

### Skill Hooks

Defined within skills using the `@hook` decorator:

```python
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

```python
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

# Pass to agent
agent = BaseAgent(
    name="my-agent",
    model="openai/gpt-4o",
    hooks=[log_messages, setup_analytics]
)

## Available Hooks

Hooks are executed in the following order during request processing:

1. **on_connection** - Once per request (initialization)
2. **before_llm_call** - Before each LLM call in agentic loop
3. **after_llm_call** - After each LLM response in agentic loop
4. **on_chunk** - For each streaming chunk (streaming only)
5. **before_toolcall** - Before each tool execution
6. **after_toolcall** - After each tool execution
7. **on_message** - Once per request (before finalization)
8. **finalize_connection** - Once per request (cleanup)

### on_connection

Called once when a new request connection is established.

Typical responsibilities:
- Authentication and identity extraction (e.g., `AuthSkill`)
- Payment token validation and minimum balance checks (e.g., `PaymentSkill`)
- Request-scoped initialization (timers, correlation IDs)

```python
@hook("on_connection")
async def on_connection(self, context):
    """Initialize request processing"""
    # Access context data
    user_id = context.peer_user_id
    is_streaming = context.stream
    
    # Set up request-specific state
    context["request_start"] = time.time()
    
    return context
```

### on_message

Called for each message in the conversation.

Typical responsibilities:
- Lightweight analytics and message enrichment
- Intent detection, entity extraction
- Safety checks for input/output

```python
@hook("on_message")
async def on_message(self, context):
    """Process each message"""
    # Get current message
    message = context.messages[-1]
    
    if message["role"] == "user":
        # Analyze user input
        context["intent"] = self.analyze_intent(message["content"])
    
    return context
```

### before_llm_call

Called before each LLM call in the agentic loop.

Typical responsibilities:
- Message preprocessing and transformation
- Multimodal content formatting
- Conversation history manipulation

```python
@hook("before_llm_call", priority=5)
async def before_llm_call(self, context):
    """Preprocess messages before LLM"""
    messages = context.get('conversation_messages', [])
    
    # Transform messages (e.g., convert markdown images to multimodal format)
    processed_messages = self.process_messages(messages)
    context.set('conversation_messages', processed_messages)
    
    return context
```

### after_llm_call

Called after each LLM response in the agentic loop.

Typical responsibilities:
- Response post-processing
- Cost tracking per iteration
- Response validation

```python
@hook("after_llm_call", priority=10)
async def after_llm_call(self, context):
    """Process LLM response"""
    response = context.get('llm_response')
    
    # Track per-iteration costs
    usage = response.get('usage', {})
    await self.track_llm_usage(usage)
    
    return context
```

### before_toolcall

Called before executing a tool.

Typical responsibilities:
- Security and scope checks
- Argument validation/normalization
- Rate limiting and auditing

```python
@hook("before_toolcall", priority=1)
async def before_toolcall(self, context):
    """Validate tool execution"""
    tool_call = context["tool_call"]
    function_name = tool_call["function"]["name"]
    
    # Security check
    if not self.is_tool_allowed(function_name, context.peer_user_id):
        # Modify tool call to safe alternative
        context["tool_call"]["function"]["name"] = "tool_blocked"
        context["tool_call"]["function"]["arguments"] = "{}"
    
    return context
```

### after_toolcall

Called after tool execution completes.

Typical responsibilities:
- Post-processing tool results
- Adding usage metadata for priced tools
- Observability metrics

```python
@hook("after_toolcall")
async def after_toolcall(self, context):
    """Process tool results"""
    tool_result = context["tool_result"]
    tool_name = context["tool_call"]["function"]["name"]
    
    # Log usage
    await self.log_tool_usage(
        tool=tool_name,
        result_size=len(tool_result),
        user=context.peer_user_id
    )
    
    # Enhance result
    if tool_name == "search":
        context["tool_result"] = self.format_search_results(tool_result)
    
    return context
```

### on_chunk

Called for each streaming chunk.

Typical responsibilities:
- Realtime content filtering
- Inline analytics/telemetry

```python
@hook("on_chunk")
async def on_chunk(self, context):
    """Process streaming chunks"""
    chunk = context["chunk"]
    content = context.get("content", "")
    
    # Real-time content analysis
    if self.contains_sensitive_info(content):
        # Redact sensitive content
        context["chunk"]["choices"][0]["delta"]["content"] = "[REDACTED]"
    
    # Track streaming metrics
    context["chunks_processed"] = context.get("chunks_processed", 0) + 1
    
    return context
```

### before_handoff

Called before handing off to another agent.

```python
@hook("before_handoff")
async def before_handoff(self, context):
    """Prepare for agent handoff"""
    target_agent = context["handoff_agent"]
    
    # Add handoff metadata
    context["handoff_metadata"] = {
        "source_agent": context.agent_name,
        "timestamp": time.time(),
        "reason": context.get("handoff_reason")
    }
    
    # Validate handoff
    if not self.can_handoff_to(target_agent):
        raise HandoffError(f"Cannot handoff to {target_agent}")
    
    return context
```

### after_handoff

Called after handoff completes.

```python
@hook("after_handoff")
async def after_handoff(self, context):
    """Process handoff results"""
    handoff_result = context["handoff_result"]
    
    # Log handoff
    await self.log_handoff(
        target=context["handoff_agent"],
        success=handoff_result.get("success"),
        duration=time.time() - context["handoff_metadata"]["timestamp"]
    )
    
    return context
```

### finalize_connection

Called when request processing completes.

```python
@hook("finalize_connection")
async def finalize_connection(self, context):
    """Clean up and finalize"""
    # Calculate metrics
    duration = time.time() - context.get("request_start", time.time())
    
    # Log final metrics
    await self.log_request_complete(
        request_id=context.completion_id,
        duration=duration,
        tokens=context.get("usage", {}),
        chunks=context.get("chunks_processed", 0)
    )
    
    # Clean up resources
    self.cleanup_request_resources(context.completion_id)
    
    return context
```

## Hook Priority

Hooks execute in priority order (lower numbers first):

```python
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

The context object provides access to:

```python
context = {
    # Request data
    "messages": List[Dict],          # Conversation messages
    "stream": bool,                  # Streaming enabled
    "peer_user_id": str,            # User identifier
    "completion_id": str,           # Request ID
    "model": str,                   # Model name
    
    # Agent data
    "agent_name": str,              # Agent name
    "agent_skills": Dict[str, Skill], # Active skills
    
    # Execution state
    "usage": Dict,                  # Token usage
    "tool_calls": List,             # Tool executions
    
    # Hook-specific data
    "tool_call": Dict,              # Current tool (before/after_toolcall)
    "tool_result": str,             # Tool result (after_toolcall)
    "chunk": Dict,                  # Current chunk (on_chunk)
    "content": str,                 # Chunk content (on_chunk)
    
    # Custom data
    **custom_fields                 # Any custom fields added by hooks
}
```

## Practical Examples

### Rate Limiting

```python
class RateLimitSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.request_counts = {}
    
    @hook("on_connection", priority=1)
    async def check_rate_limit(self, context):
        user_id = context.peer_user_id
        
        # Check rate limit
        count = self.request_counts.get(user_id, 0)
        if count >= 100:  # 100 requests per hour
            raise RateLimitError("Rate limit exceeded")
        
        # Increment counter
        self.request_counts[user_id] = count + 1
        
        return context
```

### Content Moderation

```python
class ModerationSkill(Skill):
    @hook("on_message", priority=5)
    async def moderate_input(self, context):
        """Filter inappropriate content"""
        message = context.messages[-1]
        
        if message["role"] == "user":
            # Check content
            if self.is_inappropriate(message["content"]):
                # Replace with safe message
                context.messages[-1]["content"] = "I cannot process inappropriate content."
        
        return context
    
    @hook("on_chunk", priority=5)
    async def moderate_output(self, context):
        """Filter streaming output"""
        content = context.get("content", "")
        
        if self.is_inappropriate(content):
            # Replace chunk content
            context["chunk"]["choices"][0]["delta"]["content"] = ""
        
        return context
```

### Analytics Collection

```python
class AnalyticsSkill(Skill):
    @hook("on_connection")
    async def start_analytics(self, context):
        context["analytics"] = {
            "start_time": time.time(),
            "events": []
        }
        return context
    
    @hook("on_message")
    async def track_message(self, context):
        context["analytics"]["events"].append({
            "type": "message",
            "role": context.messages[-1]["role"],
            "timestamp": time.time()
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
            "timestamp": time.time()
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

1. **Always Return Context** - Hooks must return the context object
2. **Use Priorities Wisely** - Order matters for dependent operations
3. **Handle Errors Gracefully** - Don't break the request flow
4. **Keep Hooks Lightweight** - Avoid heavy processing
5. **Use Context for State** - Don't use instance variables for request state 