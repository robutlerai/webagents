# Agent Handoffs

!!! success "Unified Handoff System"

    The handoff system provides a unified interface for both local LLM completions and remote agent handoffs, with automatic streaming support and priority-based handler selection.

Handoffs enable seamless completion handling through a unified interface that supports:

- **Local LLM completions** (via LiteLLM, OpenAI, etc.)
- **Remote agent communication** (via NLI)
- **Automatic streaming/non-streaming adaptation**
- **Priority-based handler selection**
- **Dynamic prompt injection**

## Handoff System Overview

The new handoff system replaces the old hardcoded `primary_llm` pattern with a flexible, decorator-based approach:

```python
from webagents.agents.skills import Skill
from webagents.agents.tools.decorators import handoff

class CustomLLMSkill(Skill):
    """Custom LLM completion handler"""
    
    async def initialize(self, agent):
        # Register as handoff handler
        # NOTE: Register streaming function for best compatibility
        agent.register_handoff(
            Handoff(
                target="custom_llm",
                description="Custom LLM using specialized model",
                scope="all",
                metadata={
                    'function': self.chat_completion_stream,
                    'priority': 10,
                    'is_generator': True  # Streaming generator
                }
            ),
            source="custom_llm"
        )
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle LLM completion (streaming)"""
        async for chunk in self.my_streaming_llm_api(messages, tools):
            yield chunk
```

## Core Concepts

### Handoff Dataclass

```python
from webagents.agents.skills.base import Handoff

Handoff(
    target: str,              # Handler identifier
    description: str = "",    # Description/prompt for when to use
    scope: Union[str, List[str]] = "all",
    metadata: Dict[str, Any] = None  # Contains: function, priority, is_generator
)
```

### Priority System

Handoffs are selected based on priority (lower = higher priority):

- **Priority 10**: Local LLM handlers (default)
- **Priority 20**: Remote agent handlers
- **Priority 50+**: Custom/specialized handlers

The **first registered handoff** (lowest priority) becomes the **default completion handler**.

### Streaming vs Non-Streaming

The system automatically adapts handlers:

- **Async generators** (`async def func() -> AsyncGenerator`) = streaming native
- **Regular async functions** (`async def func() -> Dict`) = non-streaming native
- **Automatic adaptation** in both directions

## Using the @handoff Decorator

### Basic Handoff with Prompt

```python
from webagents.agents.tools.decorators import handoff

class SpecializedSkill(Skill):
    @handoff(
        name="specialist",
        prompt="Use this handler for complex mathematical computations requiring symbolic processing",
        priority=15
    )
    async def specialized_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        context=None,  # Auto-injected if present in signature
        **kwargs
    ) -> Dict[str, Any]:
        """Handle specialized completions"""
        result = await self.process_with_specialist(messages)
        return result
```

### Streaming Handoff

For streaming responses, use an async generator:

```python
class StreamingSkill(Skill):
    @handoff(
        name="streaming_llm",
        prompt="Streaming LLM handler for real-time responses",
        priority=10
    )
    async def streaming_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream LLM responses"""
        async for chunk in self.my_streaming_api(messages, tools):
            yield chunk
```

### Context Injection

The decorator automatically injects `context` if it's in your function signature:

```python
@handoff(name="context_aware", priority=10)
async def completion_with_context(
    self,
    messages: List[Dict[str, Any]],
    context=None,  # Auto-injected from request context
    **kwargs
) -> Dict[str, Any]:
    """Use context for billing, auth, etc."""
    user_id = context.auth.user_id if context else None
    return await self.process(messages, user_id=user_id)
```

## Built-in Handoff Skills

### LiteLLMSkill (Default)

LiteLLMSkill automatically registers as a handoff handler during initialization:

```python
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill

# In dynamic_factory.py or your agent setup
skills["litellm"] = LiteLLMSkill(model="openai/gpt-4o")

# LiteLLMSkill.initialize() automatically calls:
agent.register_handoff(
    Handoff(
        target="litellm_openai_gpt-4o",
        description="LiteLLM completion handler using openai/gpt-4o",
        metadata={'function': self.chat_completion_stream, 'priority': 10, 'is_generator': True}
    ),
    source="litellm"
)
# NOTE: Registers the streaming function for optimal compatibility in both modes
```

### AgentHandoffSkill (Remote Agents)

For handing off to remote agents via NLI with streaming support:

```python
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill

# Register remote agent handoff
skills["agent_handoff"] = AgentHandoffSkill({
    'agent_url': 'https://robutler.ai/agents/specialist'
})

# AgentHandoffSkill automatically registers with priority=20
# Supports streaming via NLI.stream_message()
```

## Manual Handoff Registration

You can also register handoffs manually without decorators:

```python
from webagents.agents.skills.base import Handoff

class MySkill(Skill):
    async def initialize(self, agent):
        # Register handoff manually
        # NOTE: This example shows non-streaming (is_generator=False)
        # For LLM handlers, prefer streaming (is_generator=True) as shown above
        agent.register_handoff(
            Handoff(
                target="my_handler",
                description="My custom completion handler",
                scope="owner",  # Only for owner
                metadata={
                    'function': self.my_completion_handler,
                    'priority': 15,
                    'is_generator': False  # Non-streaming example
                }
            ),
            source="my_skill"
        )
    
    async def my_completion_handler(self, messages, tools=None, **kwargs):
        # Non-streaming handler that returns a complete response
        return await self.process(messages)
```

## Handoff Execution Flow

### In BaseAgent

The agent's agentic loop uses the active handoff:

```python
# Non-streaming mode (agent.run())
response = await self._execute_handoff(
    self.active_handoff,
    messages=enhanced_messages,
    tools=available_tools,
    stream=False  # Consumes generators to single response
)

# Streaming mode (agent.run_streaming())
stream_gen = self._execute_handoff(
    self.active_handoff,
    messages=enhanced_messages,
    tools=available_tools,
    stream=True  # Wraps regular functions as generators
)

async for chunk in stream_gen:
    yield chunk
```

### Automatic Adaptation

The system handles adaptation automatically:

**Streaming Mode (stream=True):**
- Generator functions → called directly, yields chunks
- Regular functions → wrapped to yield single chunk

**Non-Streaming Mode (stream=False):**
- Regular functions → called directly, returns dict
- Generator functions → consumed to single dict response

!!! tip "Best Practice: Register Streaming Functions"
    For LLM handlers, **always register the streaming generator function** (`is_generator: True`). The system automatically adapts it for non-streaming mode by consuming the generator. This approach:
    
    - ✅ Works in both streaming and non-streaming modes
    - ✅ Provides real-time feedback when streaming
    - ✅ Handles tool calls correctly in both modes
    - ✅ Matches how LiteLLMSkill registers itself
    
    This is the pattern used by `LiteLLMSkill` which registers `chat_completion_stream` as the handoff function.

## Remote Agent Handoffs

### Using AgentHandoffSkill

```python
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill
from webagents.agents.skills.robutler.nli import NLISkill

# Setup NLI and handoff skills
skills = {
    "nli": NLISkill(),
    "agent_handoff": AgentHandoffSkill()
}

agent = BaseAgent(
    name="coordinator",
    instructions="Coordinate with specialist agents",
    skills=skills
)

# Agent can now hand off to remote agents automatically
# AgentHandoffSkill uses NLI.stream_message() for streaming
```

### NLI Streaming Support

The NLISkill provides `stream_message()` for SSE streaming from remote agents:

```python
# Used internally by AgentHandoffSkill
async for chunk in nli_skill.stream_message(
    agent_url="https://robutler.ai/agents/specialist",
    messages=messages,
    tools=tools,
    authorized_amount=0.50
):
    # OpenAI-compatible streaming chunks
    print(chunk)
```

## Dynamic Prompt Integration

Handoff prompts automatically integrate with agent system prompts:

```python
@handoff(
    name="math_expert",
    prompt="Use this handler for advanced mathematical problems requiring symbolic computation, calculus, or theorem proving",
    priority=15
)
async def math_completion(self, messages, **kwargs):
    return await self.math_engine.solve(messages)
```

The `prompt` parameter serves dual purposes:
1. **Description**: Explains when this handoff should be used
2. **Dynamic Prompt**: Added to agent's system prompt automatically

## Multi-Agent Workflows

### Conditional Remote Handoffs

```python
class RouterSkill(Skill):
    async def initialize(self, agent):
        self.agent = agent
        
        # Register conditional handoff
        agent.register_handoff(
            Handoff(
                target="specialist_router",
                description="Route to specialist agents based on query complexity",
                metadata={
                    'function': self.route_to_specialist,
                    'priority': 12,
                    'is_generator': True
                }
            ),
            source="router"
        )
    
    async def route_to_specialist(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Conditionally route to specialist agents"""
        last_message = messages[-1]['content']
        
        if self._needs_specialist(last_message):
            # Hand off to remote specialist
            specialist_url = await self._discover_specialist(last_message)
            
            async for chunk in self.agent.skills['agent_handoff'].remote_agent_handoff(
                messages=messages,
                agent_url=specialist_url,
                **kwargs
            ):
                yield chunk
        else:
            # Use local LLM
            async for chunk in self.agent.skills['litellm'].chat_completion_stream(
                messages=messages,
                **kwargs
            ):
                yield chunk
```

### Cascading Handoffs

```python
class CascadingSkill(Skill):
    """Try multiple handlers in order until one succeeds"""
    
    async def initialize(self, agent):
        agent.register_handoff(
            Handoff(
                target="cascading",
                description="Try multiple handlers with fallback",
                metadata={
                    'function': self.cascading_completion,
                    'priority': 8,  # Higher priority than defaults
                    'is_generator': True
                }
            ),
            source="cascading"
        )
    
    async def cascading_completion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Try handlers in priority order"""
        handlers = [
            ('specialist', 'https://robutler.ai/agents/specialist'),
            ('generalist', 'https://robutler.ai/agents/generalist'),
            ('local', None)  # Fallback to local LLM
        ]
        
        for name, url in handlers:
            try:
                if url:
                    # Try remote agent
                    async for chunk in self._stream_from_remote(url, messages):
                        yield chunk
                    return  # Success, exit
                else:
                    # Fallback to local
                    async for chunk in self._stream_from_local(messages):
                        yield chunk
                    return
            except Exception as e:
                self.logger.warning(f"Handler {name} failed: {e}")
                continue
        
        # All handlers failed
        yield self._create_error_response("All handlers failed")
```

## Migration Guide

### Legacy Pattern

```python
# Legacy: Hardcoded primary_llm
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill

skills = {
    "primary_llm": LiteLLMSkill(model="openai/gpt-4o"),
    # ... other skills
}

agent = BaseAgent(name="agent", skills=skills)
```

### Current Pattern

```python
# Current: Handoff-based (LiteLLMSkill auto-registers)
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill

skills = {
    "litellm": LiteLLMSkill(model="openai/gpt-4o"),  # Auto-registers as handoff
    # ... other skills
}

agent = BaseAgent(name="agent", skills=skills)
# agent.active_handoff is automatically set to LiteLLM (priority=10)
```

**Key Improvements:**
- ✅ No more `"primary_llm"` key required
- ✅ LiteLLMSkill self-registers during `initialize()`
- ✅ First handoff (lowest priority) = default handler
- ✅ Fully backward compatible for basic usage
- ✅ LiteLLM registers `chat_completion_stream` (streaming) for optimal compatibility
- ✅ Automatic adaptation between streaming/non-streaming modes

## Best Practices

1. **Priority Selection**
   - Reserve 1-10 for critical/high-priority handlers
   - Use 10-20 for standard local/remote handlers
   - Use 20+ for specialized/conditional handlers

2. **Streaming Support**
   - Use async generators for streaming-native handlers
   - System handles adaptation automatically
   - Don't mix streaming/non-streaming in one function

3. **Context Usage**
   - Add `context=None` to signature for auto-injection
   - Use for auth, billing, user preferences
   - Don't modify context, it's read-only

4. **Error Handling**
   - Always handle errors in custom handoffs
   - Provide fallback responses
   - Log failures for debugging

5. **Prompt Clarity**
   - Make handoff prompts specific and actionable
   - Describe when the handler should be used
   - Include examples of suitable queries

## Complete Example

```python
from webagents.agents import BaseAgent
from webagents.agents.skills import Skill
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill
from webagents.agents.skills.robutler.nli import NLISkill
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill
from webagents.agents.tools.decorators import handoff
from typing import List, Dict, Any, AsyncGenerator

class IntelligentRouterSkill(Skill):
    """Smart router with fallback chain"""
    
    async def initialize(self, agent):
        self.agent = agent
        
        # Register as high-priority handler
        agent.register_handoff(
            Handoff(
                target="intelligent_router",
                description="Intelligently route to best handler based on query analysis",
                metadata={
                    'function': self.route_completion,
                    'priority': 5,  # Highest priority
                    'is_generator': True
                }
            ),
            source="router"
        )
    
    async def route_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        context=None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Route to optimal handler"""
        
        query = messages[-1]['content']
        complexity = self._analyze_complexity(query)
        
        # Route based on complexity
        if complexity == 'expert':
            # Use remote specialist
            specialist_url = await self._find_specialist(query)
            handler = self.agent.skills['agent_handoff']
            
            async for chunk in handler.remote_agent_handoff(
                messages=messages,
                agent_url=specialist_url,
                tools=tools,
                context=context
            ):
                yield chunk
        
        else:
            # Use local LLM
            handler = self.agent.skills['litellm']
            async for chunk in handler.chat_completion_stream(
                messages=messages,
                tools=tools,
                **kwargs
            ):
                yield chunk

# Create agent with intelligent routing
agent = BaseAgent(
    name="smart-agent",
    instructions="You are a smart agent that routes queries optimally",
    skills={
        "router": IntelligentRouterSkill(),
        "litellm": LiteLLMSkill(model="openai/gpt-4o"),
        "nli": NLISkill(),
        "agent_handoff": AgentHandoffSkill()
    }
)

# The router takes priority, but falls back to LiteLLM when appropriate
```

## API Reference

See the [Handoff API Reference](../reference/agents/skills/base.md#handoff) for complete technical details.
