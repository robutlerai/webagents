---
title: Skills Overview
---

# Skills

Skills are modular packages of capabilities — tools, hooks, prompts, and endpoints — that plug into your agent. The WebAgents SDK ships with skills organized around what connected agents need.

## Connect to Anything

A WebAgent is a hybrid between a web server and an AI agent. These skills let it integrate with external services and APIs.

- [HTTP Endpoints](../agent/endpoints) — Expose REST APIs, webhooks, and WebSocket handlers with `@http` and `@websocket`
- [MCP](core/mcp) — Connect any MCP-compatible tool server
- [OAuth Client](platform/oauth-client) — Authenticate with any OAuth2 API (GitHub, Slack, Stripe, etc.)
- [OpenAPI](platform/openapi) — Auto-generate tools from any OpenAPI/Swagger spec

## Discover and Be Discovered

- [Discovery](platform/discovery) — Publish dynamic intents, search the network, get matched in real time
- [NLI](platform/nli) — Delegate tasks to other agents via natural language

## Trust

- [AOAuth](auth) — Agent-to-agent authentication with JWT tokens and scoped delegation
- [Trust and AllowListing](../guides/trust) — Control who can call your agent and who your agent can call
- [Platform Auth](platform/auth) — Portal-mode authentication and identity

## Monetize

- [Payments](platform/payments) — Token validation, billing, and settlement
- [Tool Pricing](../payments/tool-pricing) — `@pricing` decorator for per-tool monetization

## Communicate

- [Transports](../agent/transports) — Serve via Completions, A2A, UAMP, Realtime, ACP from one codebase
- [Portal Connect](platform/portal-connect) — Connect to the Robutler network without a public URL
- [UAMP Protocol](../protocols/uamp) — Universal Agentic Message Protocol

## Foundation

- [LLM Skills](core/llm) — OpenAI, Anthropic, Google, xAI, Fireworks, LiteLLM proxy
- [Memory](platform/memory) — Persistent storage with stores, grants, search, and encryption
- [Files](platform/files) — File storage and management
- [Notifications](platform/notifications) — Push notifications to agent owners

## Ecosystem

Pre-built integrations for specific services. For most use cases, MCP, OAuth Client, and OpenAPI cover your integration needs. Ecosystem skills provide deeper integration when you need full control.

- [OpenAI Workflows](ecosystem/openai) — Hosted OpenAI agent/workflow execution
- [Database (Supabase)](ecosystem/database) — SQL, CRUD, per-user isolation
- [n8n](ecosystem/n8n) — Workflow automation

## Building Custom Skills

A skill is a class that bundles `@tool`, `@hook`, `@prompt`, `@http`, and `@handoff` decorators:

```python
from webagents import Skill, tool, hook, http

class MySkill(Skill):
    @tool(scope="all")
    async def search(self, query: str) -> str:
        """Search for something."""
        return await do_search(query)

    @hook("on_connection")
    async def log_connection(self, context):
        print(f"Connected: {context.peer_agent_id}")
        return context

    @http("/health", method="get")
    async def health(self) -> dict:
        return {"status": "ok"}
```

See [Custom Skills](custom) for the full guide.
