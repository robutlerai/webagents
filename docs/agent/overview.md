# Agent Overview

BaseAgent is the core class for creating AI agents in WebAgents. It uses a flexible, skill-based architecture so you can add exactly the capabilities you need. Agents speak OpenAI's Chat Completions dialect, so existing clients work out of the box. The [skill system](../skills/overview.md) adds platform features like [authentication](../skills/platform/auth.md), [payments](../skills/platform/payments.md), [discovery](../skills/platform/discovery.md), and multi-agent collaboration.

- Build an agent with a few lines of code
- Add capabilities via skills (tools, hooks, prompts, handoffs)
- Serve OpenAI-compatible endpoints with create_server

## Creating Agents

### Basic Agent

```python
from webagents.agents import BaseAgent

agent = BaseAgent(
    name="my-assistant",
    instructions="You are a helpful assistant",
    model="openai/gpt-4o"  # Smart model parameter
)
```

**New to WebAgents?** Check out the [Quickstart Guide](../quickstart.md) for a complete walkthrough.

### Agent with Skills

```python
from webagents.agents import BaseAgent
from webagents.agents.skills import ShortTermMemorySkill, DiscoverySkill

agent = BaseAgent(
    name="advanced-assistant",
    instructions="You are an advanced assistant with memory",
    model="openai/gpt-4o",
    skills={
        "memory": ShortTermMemorySkill({"max_messages": 50}),
        "discovery": DiscoverySkill()  # Find other agents
    }
)
```

!!! info "Skills"
    Explore available skills in the [Skills Repository](../skills/overview.md) or learn to [create custom skills](../skills/custom.md).

## Smart Model Parameter

The `model` parameter supports multiple formats. If you pass a provider-prefixed string (e.g., `openai/…`), the correct LLM skill is provisioned automatically. You can always pass a fully configured skill instance for custom behavior.

```python
# Explicit skill/model format
agent = BaseAgent(model="openai/gpt-4o")         # OpenAI GPT-4o
agent = BaseAgent(model="anthropic/claude-3")    # Anthropic Claude
agent = BaseAgent(model="litellm/gpt-4")         # Via LiteLLM proxy
agent = BaseAgent(model="xai/grok-beta")         # xAI Grok

# Custom skill instance
from webagents.agents.skills import OpenAISkill
agent = BaseAgent(model=OpenAISkill({
    "api_key": "sk-...",
    "temperature": 0.7
}))
```

See [LLM Skills](../skills/core/llm.md) for more configuration options.

## Running Agents

### Basic Conversation

```python
response = await agent.run([
    {"role": "user", "content": "Hello!"}
])
print(response.choices[0].message.content)
```

### Streaming Response

```python
async for chunk in agent.run_streaming([
    {"role": "user", "content": "Tell me a story"}
]):
    print(chunk.choices[0].delta.content, end="")
```

### With Tools

Attach additional tools per request using the OpenAI function-calling format:

```python
# External tools can be passed per request
response = await agent.run(
    messages=[{"role": "user", "content": "Calculate 42 * 17"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {...}
        }
    }]
)
```

Learn more about [creating tools](../skills/overview.md) and the [OpenAI function calling format](https://platform.openai.com/docs/guides/function-calling).

## Agent Capabilities

### Skills

Skills provide modular capabilities:

- **[LLM Skills](../skills/core/llm.md)** - Language model providers (OpenAI, Anthropic, LiteLLM)
- **[Memory Skills](../skills/core/memory.md)** - Conversation persistence and context management
- **[Platform Skills](../skills/platform/auth.md)** - WebAgents platform integration (auth, payments, discovery)
- **[Ecosystem Skills](../skills/ecosystem/filesystem.md)** - External services (database, filesystem, APIs)

### Tools

Tools are executable functions that extend agent capabilities:

```python
from webagents import tool

class MySkill(Skill):
    @tool
    def my_function(self, param: str) -> str:
        """Tool description"""
        return f"Result: {param}"
```

See comprehensive [tool examples and best practices](../skills/overview.md).

### Hooks

Lifecycle hooks enable event-driven behavior during request processing:

```python
from webagents.agents.skills.decorators import hook

class MySkill(Skill):
    @hook("on_message")
    async def process_message(self, context):
        """Process each message"""
        return context
```

Learn about [available hook events](../skills/overview.md) and the [agent lifecycle](lifecycle.md).

### Handoffs

Handoffs enable agents to delegate completions to specialized handlers or remote agents:

```python
from webagents import Skill, handoff

class SpecializedSkill(Skill):
    @handoff(
        name="math_expert",
        prompt="Use for advanced mathematical problems",
        priority=15
    )
    async def math_completion(self, messages, tools=None, **kwargs):
        """Handle math-focused completions"""
        async for chunk in self.specialized_math_llm(messages):
            yield chunk
```

Explore [handoff patterns](handoffs.md), [agent discovery](../skills/platform/discovery.md), and [remote agent communication](../skills/platform/nli.md).

## Context Management

!!! note "Context Management"
    Agents maintain a unified context object throughout execution via `contextvars`. Skills read and write to this thread-safe structure, avoiding globals while remaining fully async-compatible.

```python
# Within a skill
context = self.get_context()
user_id = context.peer_user_id
messages = context.messages
streaming = context.stream
```

## Agent Registration

Register agents with the server to make them available via HTTP endpoints:

```python
from webagents.server.core.app import create_server
import uvicorn

# Create server with your agents
server = create_server(agents=[agent])

# Or multiple agents
server = create_server(agents=[agent1, agent2])

# Run the server
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Learn about [server deployment](../server.md), [dynamic agents](../dynamic-agents.md), and [server architecture](../server-architecture.md).

## Best Practices

1. **Start Simple** - Begin with a basic agent, add skills as you go
2. **Use Dependencies** - Some skills auto-require others (e.g., [payments](../skills/platform/payments.md) depends on [auth](../skills/platform/auth.md))
3. **Scope Appropriately** - Use tool scopes (see [Skills Overview](../skills/overview.md)) for access control
4. **Test Thoroughly** - Treat skills as units; test hooks and tools independently
5. **Monitor Performance** - Track usage and latency; payments will use `context.usage`

## Next Steps

- **[Quickstart Guide](../quickstart.md)** - Build your first agent in 5 minutes
- **[Skills Repository](../skills/overview.md)** - Explore available skills and create custom ones
- **[Agent Lifecycle](lifecycle.md)** - Understand the complete request processing flow
- **[Server Deployment](../server.md)** - Deploy your agents to production
- **[Contributing](../developers/contributing.md)** - Contribute to the WebAgents ecosystem