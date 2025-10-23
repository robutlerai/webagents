# Agent Handoffs

The handoff system provides a unified interface for both local LLM completions and remote agent handoffs, with automatic streaming support and priority-based handler selection.

Handoffs enable seamless completion handling through a unified interface that supports:

- **Local LLM completions** (via LiteLLM, OpenAI, etc.)
- **Remote agent handoffs** - Delegate to specialized agents with full streaming support
- **Automatic streaming/non-streaming adaptation**
- **Priority-based handler selection**
- **Dynamic prompt injection**

## Handoff System Overview

The handoff system provides a flexible, decorator-based approach for registering completion handlers:

```python
from webagents import Skill, handoff

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

## Dynamic Handoff Invocation

Skills can allow the LLM to explicitly choose to use their handoff during conversation, enabling dynamic switching between handlers:

### Using request_handoff Helper

```python
from webagents import Skill, tool, handoff

class SpecialistSkill(Skill):
    @handoff(name="specialist", prompt="Specialized handler", priority=15)
    async def specialist_handler(self, messages, tools, **kwargs):
        # Handle requests...
        async for chunk in process_with_specialist(messages):
            yield chunk
        
    @tool(description="Switch to specialist for advanced queries")
    async def use_specialist(self) -> str:
        return self.request_handoff("specialist")
```

When the LLM calls `use_specialist()`, the framework:
1. Detects the handoff request marker
2. Finds the registered `specialist` handoff
3. Executes it with the current conversation
4. Streams the response directly to the user

This works with both local and remote handoffs, enabling the LLM to dynamically route requests to the most appropriate handler.

### Handoff Chaining and Default Reset

When a dynamic handoff is invoked:

1. **The handoff executes** with the current conversation context
2. **Streaming is continuous** - the response streams directly to the user
3. **The agent resets to the default handoff** after the turn completes
4. **Next user message** uses the default handoff again (unless another dynamic handoff is requested)

This ensures that dynamic handoffs are **temporary switches** for specific requests, not permanent mode changes:

```python
# Turn 1: User: "Use specialist"
# → LLM calls use_specialist() → specialist handoff executes → response streams
# → After turn ends, active_handoff resets to default (e.g., litellm)

# Turn 2: User: "What about this?"
# → Uses default handoff (litellm) again
```

**Handoff chaining** is also supported - a handoff can request another handoff during its execution, allowing multi-stage processing within a single turn.

## Using the @handoff Decorator

### Basic Handoff with Prompt

```python
from webagents import handoff

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

## Remote Agent Handoffs

**AgentHandoffSkill** enables seamless handoffs to remote agents via NLI with full streaming support. This is essential for multi-agent systems where you want to delegate tasks to specialized agents.

### Basic Setup

```python
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill
from webagents.agents.skills.robutler.nli import NLISkill

# Setup skills - NLI is required for remote agent communication
skills = {
    "nli": NLISkill(),
    "agent_handoff": AgentHandoffSkill()
}

agent = BaseAgent(
    name="coordinator",
    instructions="Coordinate with specialist agents",
    skills=skills
)
```

### Default Agent Configuration

You can configure a default agent URL that will be used automatically:

```python
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill

# Register remote agent handoff with default agent
skills["agent_handoff"] = AgentHandoffSkill({
    'agent_url': 'https://robutler.ai/agents/specialist'
})

# This handoff will automatically use the configured agent
# Great for dedicated coordinator → specialist relationships
```

### Calling Remote Agents

Hand off to specific agents using their full URL (includes agent ID):

```python
# Direct handoff to a remote agent
async for chunk in agent.skills['agent_handoff'].remote_agent_handoff(
    agent_url="https://robutler.ai/agents/96f6d0ab-71d4-4035-a71d-94d1c2b72da3",
    messages=messages,
    tools=tools
):
    # Streams OpenAI-compatible chunks in real-time
    yield chunk
```

### Dynamic Agent Discovery

You can programmatically discover and call agents:

```python
from webagents import Skill, tool

class CoordinatorSkill(Skill):
    @tool(description="Delegate complex music tasks to the music specialist")
    async def delegate_to_music_agent(self, task: str) -> str:
        """Hand off music-related tasks to r-music agent"""
        # Discover agent (could be from database, config, or API)
        music_agent_url = "https://robutler.ai/agents/96f6d0ab-71d4-4035-a71d-94d1c2b72da3"
        
        # Request handoff (framework will stream the response)
        return self.request_handoff("agent_handoff", agent_url=music_agent_url)
```

### How It Works

1. **Automatic Registration**: AgentHandoffSkill registers itself with `priority=20` during initialization
2. **NLI Communication**: Uses `NLISkill.stream_message()` for SSE streaming from remote agents
3. **OpenAI Compatibility**: Returns OpenAI-compatible streaming chunks
4. **Tool Support**: Remote agents can use their own tools and skills
5. **Payment Integration**: Supports payment token authorization for paid agents

!!! tip "Agent URLs"
    Agent URLs must include the full agent ID: `https://robutler.ai/agents/{agent-id}`
    
    You can find agent IDs in the portal or via the agents API.

!!! info "Streaming by Default"
    Remote handoffs always stream responses using SSE (Server-Sent Events), providing real-time feedback to users even for long-running operations.

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

## Quick Start Example

```python
from webagents.agents import BaseAgent
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill
from webagents.agents.skills.robutler.nli import NLISkill
from webagents.agents.skills.robutler.handoff import AgentHandoffSkill

# Create agent with handoff support
agent = BaseAgent(
    name="coordinator",
    instructions="Coordinate tasks and hand off to specialists when needed",
    skills={
        "litellm": LiteLLMSkill(model="openai/gpt-4o"),
        "nli": NLISkill(),
        "agent_handoff": AgentHandoffSkill()
    }
)

# LiteLLMSkill automatically registers as the default handoff handler
# AgentHandoffSkill enables remote agent handoffs via NLI
```
