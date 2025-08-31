Skills are modular capability packages that extend a `BaseAgent` with tools, prompts, hooks, handoffs, and optional HTTP endpoints. They’re first-class, composable building blocks that keep business logic organized and reusable across agents.

- Tools: executable functions registered via `@tool`
- Prompts: guidance for the LLM, optionally prioritized or scoped
- Hooks: lifecycle callbacks (e.g., `on_message`, `before_toolcall`)
- Handoffs: route requests to other agents when specialized handling is needed
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
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool
from webagents.agents.skills.decorators import hook, handoff

class MySkill(Skill):
    def __init__(self, config=None):
        super().__init__(
            config=config,
            scope="all",               # all | owner | admin
            dependencies=["memory"],   # ensure memory is present if needed
        )

    @tool
    def summarize(self, text: str, max_len: int = 200) -> str:
        """Summarize input text to a target length."""
        return text[:max_len]

    @hook("on_message")
    async def on_message(self, context):
        # Inspect/augment request context before tools or model
        return context

    @handoff("expert-agent")
    def route_to_expert(self, query: str) -> bool:
        return "expert" in query.lower()
```

- Register execution logic with `@tool`
- Guide LLM behavior with prompts (see Skills Framework for full patterns)
- React to request lifecycle via `@hook`
- Route to another agent when conditions match with `@handoff`

### HTTP Endpoints in Skills

Register custom REST endpoints with the `@http` decorator. These are mounted under your agent’s base path when served.

```python
from webagents.agents.tools.decorators import http

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
