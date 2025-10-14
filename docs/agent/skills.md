Skills are modular capability packages that extend a `BaseAgent` with tools, prompts, hooks, handoffs, and optional HTTP endpoints. They're first-class, composable building blocks that keep business logic organized and reusable across agents.

- Tools: executable functions registered via `@tool`
- Prompts: guidance for the LLM, optionally prioritized or scoped
- Hooks: lifecycle callbacks (e.g., `on_message`, `before_toolcall`)
- Handoffs: completion handlers (local LLM or remote agents) registered during initialization
- HTTP endpoints: register custom REST handlers via `@http`
- Dependencies: declare other skills your skill requires (e.g., memory)

### Add Skills to an Agent

Consistent with the Quickstart, you attach skills when creating your agent:

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.nli.skill import NLISkill
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill
from webagents.agents.skills.robutler.payments.skill import PaymentSkill

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini",  # Automatically provisions LLM skill
    skills={
        "nli": NLISkill(),            # Natural language communication with agents
        "auth": AuthSkill(),          # Authentication & scoped access control
        "discovery": DiscoverySkill(),# Real-time agent discovery (intent-based)
        "payments": PaymentSkill()    # Monetization via priced tools
    }
)
```

This mirrors the examples in the Quickstart and Index pages. After skills are attached, your agent can immediately use their tools, prompts, hooks, HTTP endpoints, and handoffs during requests.

### Skill Anatomy (Minimal Example)

```python
from webagents import Skill, tool, handoff
from webagents.agents.skills.base import Handoff
from typing import Dict, Any, AsyncGenerator

class MySkill(Skill):
    def __init__(self, config=None):
        super().__init__(
            config=config,
            scope="all",               # all | owner | admin
            dependencies=["memory"],   # ensure memory is present if needed
        )

    async def initialize(self, agent):
        """Called after skill is attached to agent"""
        self.agent = agent
        
        # Register handoff if this skill provides completion handling
        # See Agent Handoffs documentation for details
    
    @tool
    def summarize(self, text: str, max_len: int = 200) -> str:
        """Summarize input text to a target length."""
        return text[:max_len]

    @hook("on_message")
    async def on_message(self, context):
        # Inspect/augment request context before tools or model
        return context
    
    @handoff(
        name="custom_handler",
        prompt="Use for specialized processing",
        priority=15
    )
    async def custom_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Custom completion handler (streaming)"""
        # Process and yield chunks
        yield {"choices": [{"delta": {"content": "Processing..."}}]}
```

- Register execution logic with `@tool`
- Guide LLM behavior with prompts (see Skills Framework for full patterns)
- React to request lifecycle via `@hook`
- Provide completion handlers with `@handoff` (for LLM or remote agent routing)

### HTTP Endpoints in Skills

Register custom REST endpoints with the `@http` decorator. These are mounted under your agent’s base path when served.

```python
from webagents import http

@http("/weather", method="get", scope="owner")
def get_weather(location: str, units: str = "celsius") -> dict:
    return {"location": location, "temperature": 25, "units": units}

@http("/data", method="post")
async def post_data(payload: dict) -> dict:
    return {"received": payload, "status": "processed"}
```

- `path`: endpoint path relative to the agent root (e.g., `/assistant/weather`)
- `method`: one of `get`, `post`, etc. (default is `post` if omitted)
- `scope`: optional access control (`all`, `owner`, `admin`)

When the agent is served, these endpoints are available immediately.

### Using Skill Tools in a Request

Tools you register are available to the agent at runtime. You can also pass external tools per request (OpenAI function-calling compatible):

```python
response = await agent.run([
    {"role": "user", "content": "Summarize: ..."}
])

# Or include additional, ad-hoc tools for a single call
response = await agent.run(
    messages=[{"role": "user", "content": "Calculate 42 * 17"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
        }
    }]
)
```

### Serving an Agent with Skills

Follow the Quickstart approach to serve your agent over HTTP:

```python
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

## Dynamic Skill Management

Agent owners can add and remove skills dynamically through conversation using the Control Skill's management tools. Skills are persisted to the portal database and take effect immediately via cache invalidation.

### Adding Skills

Agent owners can add skills by talking to their agent:

```
You: "I want to add the OpenAI skill"

Agent: "✓ OpenAI Workflows skill added successfully!

Next step: Configure your credentials at http://localhost:2224/agents/my-agent/setup/openai"
```

Skills requiring setup (like OpenAI Workflows) will provide a setup URL where credentials can be configured.

### Listing Available Skills

To see what skills can be added:

```
You: "What skills can I add?"

Agent: "Available skills:

- openai: OpenAI Workflows - Execute OpenAI hosted agents and workflows (requires setup) [○ Available]"
```

The status indicator shows whether each skill is enabled:
- `✓ ENABLED`: Currently active on this agent
- `○ Available`: Can be added

### Removing Skills

To remove a skill:

```
You: "Remove the OpenAI skill"

Agent: "✓ Skill 'openai' removed successfully and will take effect on the next message."
```

Note: Core skills (`litellm`, `auth`, `payment`, `control`) cannot be removed as they provide essential functionality.

### Available Skills

Currently, the following skills are available for dynamic addition:

- **openai**: OpenAI Workflows - Execute OpenAI hosted agents and workflows (requires setup)

More skills will be added to the registry as they become available for dynamic management.

### How It Works

1. **Owner-Only**: Only the agent owner can manage skills (enforced by the `scope="owner"` decorator)
2. **Database Persistence**: Skills are stored in the portal database's `skills` JSON field
3. **Immediate Effect**: Cache invalidation ensures the agent is recreated with new skills on the next message
4. **Setup Flow**: Skills requiring credentials provide a setup URL for secure configuration via KV storage

### For Skill Developers

To make a skill available for dynamic addition, add it to the skill registry at `agents/skills/registry.py`:

```python
AVAILABLE_DYNAMIC_SKILLS = {
    "my_skill": {
        "class": "webagents.agents.skills.my_package.MySkill",
        "name": "My Skill",
        "description": "What this skill does",
        "requires_setup": True,  # If credentials needed
        "setup_path": "/setup/my_skill",
        "config": {}  # Default configuration
    }
}
```

Then update the dynamic factory's `_create_agent_skills` method to instantiate your skill when its type is detected in the database.
