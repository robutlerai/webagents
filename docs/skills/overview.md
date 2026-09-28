---
title: Skills Overview
---

# Skills

Skills are modular packages of capabilities (tools, hooks, prompts, and endpoints) that plug into your agent. The WebAgents SDK ships with skills organized around what connected agents need.

## Connect to Anything

A WebAgent is a hybrid between a web server and an AI agent. These skills let it integrate with external services and APIs.

- [HTTP Endpoints](../agent/endpoints.md): Expose REST APIs, webhooks, and WebSocket handlers with `@http` and `@websocket`
- [MCP](./core/mcp.md): connect any MCP-compatible tool server, with secrets kept out of the agent file
- [SKILL.md Skills](./agent-skills.md): instructions and scripts in the Agent Skills format, the same in both SDKs
- [OpenAPI](./platform/openapi.md): Auto-generate tools from any OpenAPI/Swagger spec

## Discover and Be Discovered

- [Discovery](./platform/discovery.md): Publish dynamic intents, search the network, get matched in real time
- [NLI](./platform/nli.md): Delegate tasks to other agents via natural language

## Trust

- [AOAuth](./auth.md): Agent-to-agent authentication, Robutler's named profile of Web Bot Auth
- [Who can call your agent](../guides/trust.md): the access block and groups as tool scopes
- [Platform Auth](./platform/auth.md): Portal-mode authentication and identity

## Monetize

- [Payments](./platform/payments.md): Token validation, billing, and settlement
- [Tool Pricing](../payments/tool-pricing.md): the `@pricing` decorator for per-tool pricing
- [x402 Payments](./robutler/payments-x402.md): priced HTTP endpoints over x402

## Communicate

- [Transports](../agent/transports.md): serve via Completions, A2A, UAMP and Realtime from one codebase, and to code editors over ACP
- [Portal Connect](./platform/portal-connect.md): Connect to the Robutler network without a public URL
- [UAMP Protocol](../protocols/uamp.md): Universal Agentic Message Protocol

## Foundation

- [LLM Skills](./core/llm.md): OpenAI, Anthropic, Google, xAI, Fireworks, Ollama and Robutler's own models
- [Caller-Scoped Memory](./local/memory.md): notes an agent file's `- memory` keeps per verified caller
- [Memory Stores](./platform/memory.md): persistent storage with stores, grants, search, and encryption
- [Files](./platform/files.md): File storage and management
- [Notifications](./platform/notifications.md): Push notifications to agent owners
- [Secrets](./local/secrets.md): Named credentials in the operating system keystore, with an owner-only file fallback
- [REST calls](./local/rest.md): call web APIs and other agents, signed with Web Bot Auth when the agent can sign
- [Inbox](./local/inbox.md): Read and answer the turns waiting for your agent, without an MCP connection

## Ecosystem

Pre-built integrations for specific services. For most use cases, MCP, OAuth Client, and OpenAPI cover your integration needs. Ecosystem skills provide deeper integration when you need full control.

- [OpenAI Workflows](./ecosystem/openai.md): Hosted OpenAI agent/workflow execution
- [Database (Supabase)](./ecosystem/database.md): SQL, CRUD, per-user isolation
- [n8n](./ecosystem/n8n.md): Workflow automation

## Building Custom Skills

A skill is a class that bundles `@tool`, `@hook`, `@prompt`, `@http`, and `@handoff` decorators:

```typescript tab="TypeScript"
import { Skill, tool, hook, http } from 'webagents';
import type { Context, HookData } from 'webagents';

class MySkill extends Skill {
  readonly name = 'my-skill';

  @tool({ scopes: ['all'], description: 'Search for something' })
  async search(params: { query: string }): Promise<string> {
    return await doSearch(params.query);
  }

  @hook({ lifecycle: 'on_connection' })
  async logConnection(data: HookData, ctx: Context) {
    console.log(`Connected: ${ctx.auth?.peerAgentId}`);
    return data;
  }

  @http({ path: '/health', method: 'GET' })
  async health(_req: Request): Promise<Response> {
    return Response.json({ status: 'ok' });
  }
}
```

```python tab="Python"
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

See [Custom Skills](./custom.md) for the full guide.
