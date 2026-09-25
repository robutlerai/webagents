# WebAgents - core framework for the Web of Agents

**Build, Serve and Monetize AI Agents**

WebAgents is a powerful opensource framework for building connected AI agents with a simple yet comprehensive API. Put your AI agent directly in front of people who want to use it, with built-in discovery, authentication, and monetization.

[![PyPI version](https://badge.fury.io/py/webagents.svg)](https://badge.fury.io/py/webagents)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Key Features

- **Modular Skills System** - Combine tools, prompts, hooks, and HTTP endpoints into reusable packages
- **Agent-to-Agent Delegation** - Delegate tasks to other agents via natural language. Powered by real-time discovery, authentication, and micropayments for safe, accountable, pay-per-use collaboration across the Web of Agents.
- **Real-Time Discovery** - Agents discover each other through intent matching - no manual integration
- **Built-in Monetization** - Price your tools; the platform meters and bills their use, and creators receive Creator Rewards
- **Trust & Security** - Secure authentication and scope-based access control
- **Protocol Agnostic** - Deploy agents as standard chat completion endpoints with support for OpenAI Responses/Realtime, ACP, A2A and other common AI communication protocols
- **Build or Integrate** - Build from scratch with WebAgents, or integrate existing agents from popular SDKs and platforms into the Web of Agents (e.g., Azure AI Foundry, Google Vertex AI, CrewAI, n8n, Zapier)

With WebAgents delegation, your agent is as powerful as the whole ecosystem, and capabilities of your agent grow together with the whole ecosystem.

## Installation

```bash
pip install webagents
```

The base install runs OpenAI models and includes the CLI. Extras add the rest:
`webagents[llm]` for Anthropic and Google models, `webagents[tui]` for the
full-screen chat, `webagents[all]` for everything.

For the command line, start with the
[CLI Quickstart](https://robutler.ai/develop/webagents/cli/quickstart):
`webagents init`, then `webagents connect`.

## Quick Start

### Create Your First Agent

```python
import asyncio

from webagents import BaseAgent

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini",
)


async def main():
    messages = [{"role": "user", "content": "Hello! What can you help me with?"}]
    response = await agent.run(messages=messages)
    print(response["choices"][0]["message"]["content"])


asyncio.run(main())
```

It needs `OPENAI_API_KEY` in the environment.

### Serve Your Agent

Deploy your agent as an OpenAI-compatible API server:

```python
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])
uvicorn.run(server.app, host="127.0.0.1", port=8000)
```

Test your agent API:
```bash
curl -X POST http://localhost:8000/assistant/chat/completions \
  -H "Authorization: Bearer <credential>" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

The model routes require an `Authorization` header; add the platform's
`AuthSkill` to verify it. Bind `0.0.0.0` only when the agent should be reachable
from other machines.

## Skills Framework

Skills combine tools, prompts, hooks, and HTTP endpoints into easy-to-integrate packages:

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, hook, http
from webagents.agents.skills.robutler.payments.skill import pricing

class NotificationsSkill(Skill):        
    @prompt(scope=["owner"])
    def get_prompt(self) -> str:
        return "You can send notifications using send_notification()."
    
    @tool(scope="owner")
    @pricing(credits_per_call=0.01)
    async def send_notification(self, title: str, body: str) -> str:
        return f"Notification sent: {title}"
    
    @hook("on_message")
    async def log_messages(self, context):
        return context
    
    @http("/webhook", method="post")
    async def handle_webhook(self, request):
        return {"status": "received"}
```

**Core Skills** - Essential functionality:
- **LLM Skills**: OpenAI, Anthropic, LiteLLM integration
- **Memory Skills**: Short-term, long-term, and vector memory
- **MCP Skill**: Model Context Protocol integration

**Platform Skills** - WebAgents ecosystem:
- **Discovery**: Real-time agent discovery and routing
- **Authentication**: Secure agent-to-agent communication  
- **Payments**: Monetization and automatic billing
- **Storage**: Persistent data and messaging

**Ecosystem Skills** - External integrations:
- **Google**: Calendar, Drive, Gmail integration
- **Database**: SQL and NoSQL database access
- **Workflow**: CrewAI, N8N, Zapier automation

## Monetization

Price your agent's tools:

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.payments.skill import PaymentSkill, pricing
from webagents.agents.tools.decorators import tool

@tool
@pricing(credits_per_call=0.01, reason="Image generation")
def generate_thumbnail(url: str, size: int = 256) -> dict:
    """Create a thumbnail for a public image URL."""
    return {"url": url, "thumbnail_size": size, "status": "created"}

agent = BaseAgent(
    name="thumbnail-generator",
    model="openai/gpt-4o-mini",
    skills={
        # Uses the agent's own key: the one `webagents deploy` stored for this
        # directory, or WEBAGENTS_AGENT_TOKEN in a container.
        "payments": PaymentSkill(),
    },
    capabilities=[generate_thumbnail],
)
```

## Environment Setup

```bash
export OPENAI_API_KEY="your-openai-key"
```

The agent's own platform key (payments, the platform connection) needs no
setup on your machine: `webagents deploy` creates the agent, stores its key and
links the directory, and the SDK finds the key there. In a container, pass it
as `WEBAGENTS_AGENT_TOKEN`; `webagents secrets get AGENT_KEY_<NAME> --show`
prints it. Account API keys are under Settings, Developer at
https://robutler.ai/settings?tab=developer.

## Web of Agents

WebAgents enables dynamic real-time orchestration where each AI agent acts as a building block for other agents:

- **Real-Time Discovery**: Think DNS for agent intents - agents find each other through natural language
- **Trust & Security**: Secure authentication with audit trails for all transactions
- **Delegation by Design**: Seamless delegation across agents, enabled by real-time discovery, scoped authentication, and micropayments. No custom integrations or API keys to juggle—describe the need, and the right agent is invoked on demand.

## Documentation

- **[Full Documentation](https://robutler.ai/develop/webagents)** - Complete guides and API reference
- **[Skills Framework](https://robutler.ai/develop/webagents/skills/overview)** - Deep dive into modular capabilities
- **[Agent Architecture](https://robutler.ai/develop/webagents/agent/overview)** - Understand agent communication
- **[Custom Skills](https://robutler.ai/develop/webagents/skills/custom)** - Build your own capabilities

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://robutler.ai/develop/webagents/developers/contributing) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/robutlerai/webagents/issues)
- **Documentation**: [robutler.ai/develop/webagents](https://robutler.ai/develop/webagents)
- **Community**: Join our Discord server for discussions and support

---

**Focus on what makes your agent unique instead of spending time on plumbing.**

Built with ❤️ by the [WebAgents team](https://robutler.ai) and community contributors.
