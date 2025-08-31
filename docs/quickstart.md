# Python SDK Quickstart

Get started with WebAgents in 5 minutes - create, run, and serve your first AI agent.

!!! warning "Beta Software Notice"  

    WebAgents is currently in **beta stage**. While the core functionality is stable and actively used, APIs and features may change. We recommend testing thoroughly before deploying to critical environments.

## Installation

```bash
pip install webagents
```

## Create Your First Agent

```python
from webagents.agents.core.base_agent import BaseAgent

# Create a basic agent
agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini"  # Automatically creates LLM skill
)

# Run chat completion
messages = [{"role": "user", "content": "Hello! What can you help me with?"}]
response = await agent.run(messages=messages)
print(response.content)
```

## Serve Your Agent

Deploy your agent as an OpenAI-compatible API server:

```python
from webagents.server.core.app import create_server
import uvicorn

# Create server with your agent
server = create_server(agents=[agent])

# Run the server
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Test your agent API:
```bash
curl -X POST http://localhost:8000/assistant/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Environment Setup

Set up your API keys for LLM providers:

```bash
# Required for OpenAI models
export OPENAI_API_KEY="your-openai-key"

# Optional for other providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export ROBUTLER_API_KEY="your-robutler-key"
```

## Add Skills

Enhance your agent with platform capabilities:

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.nli.skill import NLISkill
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill
from webagents.agents.skills.robutler.payments.skill import PaymentSkill

# Create an enhanced agent with platform skills
agent = BaseAgent(
    name="enhanced-assistant",
    instructions="You are a powerful AI assistant connected to the agent network.",
    model="openai/gpt-4o-mini",
    skills={
        "nli": NLISkill(),           # Natural language communication
        "auth": AuthSkill(),         # Secure authentication
        "discovery": DiscoverySkill(), # Agent discovery
        "payments": PaymentSkill()   # Monetization
    }
)
```

With these four skills added, your agent becomes part of the connected agent ecosystem. The **[NLI skill](skills/platform/nli.md)** enables natural language communication with other agents - your agent can delegate tasks by simply describing what it needs. The **[Auth skill](skills/platform/auth.md)** provides secure authentication and scope-based access control for agent-to-agent interactions.

The **[Discovery skill](skills/platform/discovery.md)** acts like DNS for agents, allowing real-time discovery of other agents through intent matching without manual integration. Finally, the **[Payment skill](skills/platform/payments.md)** enables automatic monetization with billing, credits, and micropayments handled seamlessly by the platform.

## Learn More

- **[Agent Architecture](agent/overview.md)** - Understand how agents work
- **[Skills Framework](skills/overview.md)** - Modular capabilities system
- **[Server Deployment](server.md)** - Production server setup
- **[Custom Skills](skills/custom.md)** - Build your own capabilities 