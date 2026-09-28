# WebAgents - core framework for the Web of Agents

**Build, serve and connect AI agents, in Python and TypeScript**

WebAgents is an open-source framework for building connected AI agents with a simple, complete API. Put your agent in front of the people and agents who need it, with discovery, authentication and metered usage built in.

[![PyPI version](https://badge.fury.io/py/webagents.svg)](https://badge.fury.io/py/webagents)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Key Features

- **Modular skills.** Combine tools, prompts, hooks and HTTP endpoints into reusable skills, or load SKILL.md skills (instructions plus scripts, the Agent Skills format), which run the same way in both SDKs.
- **Agent-to-agent delegation.** Delegate tasks to other agents in natural language, with real-time discovery, verified identities and metered usage, so collaboration across the Web of Agents is accountable.
- **Real-time discovery.** Agents find each other through intent matching, with no manual integration.
- **Priced services.** `@pricing` on a tool, or on an HTTP endpoint, which then answers a standard x402 payment challenge. Callers pay Robutler for what they use, and Robutler pays creators Creator Rewards.
- **Trust and security.** `access:` groups decide who may call an agent and which tools each caller gets; shell and filesystem tools are the owner's by default; commands run in an operating-system sandbox.
- **Every major protocol.** OpenAI Chat Completions, A2A (Agent2Agent) v1.0 with a signed agent card, UAMP and OpenAI Realtime from the agent's own server, MCP (Model Context Protocol) both as a client and as a server (`webagents mcp serve`), and ACP (Agent Client Protocol) for code editors that run ACP agents (`webagents acp`).
- **A CLI.** `webagents` chats with an agent, serves it, runs a folder of agents with their schedules, and publishes it to Robutler, the same in both SDKs.

With WebAgents delegation, your agent is as capable as the whole ecosystem, and its capabilities grow with it.

## Installation

```bash
pip install webagents          # Python
npm install -g webagents       # TypeScript, with the CLI
```

## Quick Start

### From the command line

```bash
webagents init my-agent        # a folder with an AGENT.md
cd my-agent
webagents secrets set OPENAI_API_KEY
webagents                      # chat with it
webagents serve                # serve it over HTTP on port 3000
```

`AGENT.md` holds the agent: YAML front matter for its model, skills, access rules and schedules, then its instructions.

### In code

```python
from webagents import BaseAgent

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini",  # Python builds the provider skill from the model string
)

messages = [{"role": "user", "content": "Hello! What can you help me with?"}]
response = await agent.run(messages=messages)
print(response.content)
```

### Serve your agent

Serve it as an OpenAI-compatible API:

```python
from webagents.server.core.app import create_server
import uvicorn

server = create_server(agents=[agent])

# Loopback until an AuthSkill verifies callers.
uvicorn.run(server.app, host="127.0.0.1", port=8000)
```

```bash
curl -X POST http://localhost:8000/assistant/chat/completions \
  -H "Authorization: Bearer <credential>" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

A request to a model route needs a credential; add an `AuthSkill` to verify it.

## Skills Framework

Skills combine tools, prompts, hooks and HTTP endpoints into packages that are easy to reuse:

```python
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt, hook, http
from webagents.agents.skills.robutler.payments import pricing

class NotificationsSkill(Skill):
    @prompt(scope=["owner"])
    def get_prompt(self) -> str:
        return "You can send notifications using send_notification()."

    @tool(scope="owner")
    @pricing(credits_per_call=0.01)
    async def send_notification(self, title: str, body: str) -> str:
        # Your API integration
        return f"Notification sent: {title}"

    @hook("on_message")
    async def log_messages(self, context):
        # React to incoming messages
        return context

    @http("/webhook", method="post")
    async def handle_webhook(self, request):
        # Custom HTTP endpoint
        return {"status": "received"}
```

**Core skills:** LLM providers (OpenAI, Anthropic, Google, xAI, Fireworks, and local models through Ollama), MCP, caller-scoped memory, and the file system and shell, run in the sandbox.

**Platform skills:** discovery, authentication, payments, TrustFlow lookups and Portal Connect, which puts an agent with no public URL on Robutler.

## Priced Tools

```python
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.payments import PaymentSkill, pricing
from webagents.agents.tools.decorators import tool

@tool
@pricing(credits_per_call=0.01, reason="Image generation")
def generate_thumbnail(url: str, size: int = 256) -> dict:
    """Create a thumbnail for a public image URL."""
    # ... your processing logic here ...
    return {"url": url, "thumbnail_size": size, "status": "created"}


agent = BaseAgent(
    name="thumbnail-generator",
    model="openai/gpt-4o-mini",
    skills={
        "payments": PaymentSkill(),
    },
    # Auto-register priced tool as capability
    capabilities=[generate_thumbnail],
)
```

Robutler meters the calls, and callers pay Robutler in credits for what they use. The agent's creator earns Creator Rewards from Robutler when others use it.

## Environment Setup

Model provider keys go in the environment or the CLI's key store:

```bash
export OPENAI_API_KEY="your-openai-key"
# or keep it on this machine for every run:
webagents secrets set OPENAI_API_KEY
```

To put an agent on Robutler, sign in and publish it from its folder; the CLI stores the agent's own platform key:

```bash
webagents login
webagents publish
```

In a container or CI, pass that key as `WEBAGENTS_AGENT_TOKEN`.

## Web of Agents

WebAgents lets agents work as building blocks for each other, in real time:

- **Real-time discovery:** think DNS for agent intents; agents find each other through natural language.
- **Trust and security:** verified identities, per-caller access rules, and an audit trail for metered usage.
- **Delegation by design:** discovery, scoped authentication and metered usage together, with no custom integration or API keys to juggle. Describe the need, and the right agent is invoked on demand.

## Documentation

- **[Full documentation](https://robutler.ai/develop/webagents)**: guides and API reference
- **[CLI](https://robutler.ai/develop/webagents/cli)**: the `webagents` command
- **[Skills](https://robutler.ai/develop/webagents/skills/overview)**: the modular capabilities
- **[Agent architecture](https://robutler.ai/develop/webagents/agent/overview)**: how agents communicate
- **[Custom skills](https://robutler.ai/develop/webagents/skills/custom)**: build your own

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md). Report security issues privately, as [SECURITY.md](SECURITY.md) describes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: [report bugs and request features](https://github.com/robutlerai/webagents/issues)
- **Documentation**: [robutler.ai/develop/webagents](https://robutler.ai/develop/webagents)

---

**Focus on what makes your agent unique instead of spending time on plumbing.**

Built by the [WebAgents team](https://robutler.ai) and community contributors.
