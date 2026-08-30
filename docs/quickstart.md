---
title: Quickstart
description: Build, serve, and connect your first agent.
---

# Quickstart

This guide builds the same agent in TypeScript and Python. Pick a tab — your choice persists across every page.

## Installation

```bash tab="TypeScript"
npm install webagents
```

```bash tab="Python"
pip install webagents
```

## Create an Agent

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful AI assistant.',
  model: 'openai/gpt-4o-mini',
});

const response = await agent.run([
  { role: 'user', content: 'Hello!' },
]);

console.log(response.content);
```

```python tab="Python"
import asyncio
from webagents import BaseAgent

agent = BaseAgent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
    model="openai/gpt-4o-mini",
)

async def main():
    response = await agent.run(messages=[{"role": "user", "content": "Hello!"}])
    print(response["choices"][0]["message"]["content"])

asyncio.run(main())
```

## Serve as an API

Build an agent, build a server, run it. There is no wrapper in between: the
server serves the OpenAI-compatible endpoint AND the platform registration
surface — the agent card at `/.well-known/agent.json` (at the ORIGIN as well
as under the agent prefix) carrying `metadata.publicKey` as an SPKI PEM,
`/.well-known/jwks.json`, and a 60s presence heartbeat. The snippets below are
generated from runnable, test-executed example files: edit the examples and
run `scripts/sync_doc_examples.py`, never this page.

<!-- BEGIN GENERATED: typescript/examples/own-url-minimal.ts -->
```typescript tab="TypeScript"
import { BaseAgent, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
```
<!-- END GENERATED -->

<!-- BEGIN GENERATED: python/examples/own_url_minimal.py -->
```python tab="Python"
import uvicorn

from webagents import BaseAgent, create_server

agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```
<!-- END GENERATED -->

Test it:

```bash
curl -X POST http://localhost:8000/mini/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $WEBAGENTS_API_KEY" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

(The TypeScript server mounts the same endpoint under its `basePath`:
`POST http://localhost:8000/agents/mini/chat/completions`.)

Your agent now speaks the OpenAI Completions protocol. Any compatible client
can talk to it.

The `Authorization` header is required: this endpoint runs the model on YOUR
credit, so a request with no credential is refused with `401` before the model
is reached. Both SDKs enforce the same floor and accept the credential in any
of `Authorization`, `X-Api-Key` or `X-Owner-Assertion`. Add an `AuthSkill` to
the agent to have the credential actually verified (api key, owner assertion,
or the platform's service token) rather than merely required — the floor only
guarantees that a served port is not an anonymous, billable model endpoint.

The floor is not specific to this one URL. It covers every `POST` path that
reaches the model, on every server the SDKs offer — `chat/completions`,
`v1/chat/completions`, `uamp`, `uamp/stream` and `uamp/completions`, whether
they are served by a built-in route or by a transport skill's own `@http`
handler mounted at the same subpath — plus the `uamp` WebSocket, where the
credential may also be given as `?token=` because a browser cannot set headers
on a handshake. `GET` requests and CORS preflights are never gated: nothing
about them costs money.

The signing key is persisted (`WEBAGENTS_KEYS_DIR`, default
`~/.webagents/keys`) and MUST survive restarts: registration pins the public
key it read from the card and verifies every later token against that copy.

## Connect Without a Public URL

No inbound port, no DNS, no TLS: add `PortalConnectSkill` and the agent dials
the platform instead. It is the same agent and the same server — one more
skill, and the server's own lifecycle opens the socket.

<!-- BEGIN GENERATED: typescript/examples/portal-connect-minimal.ts -->
```typescript tab="TypeScript"
import { BaseAgent, PortalConnectSkill, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  skills: [new PortalConnectSkill()],
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
```
<!-- END GENERATED -->

<!-- BEGIN GENERATED: python/examples/portal_connect_minimal.py -->
```python tab="Python"
import uvicorn

from webagents import BaseAgent, create_server
from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill

agent = BaseAgent(
    name="mini",
    instructions="You are helpful.",
    model="openai/gpt-4o-mini",
    skills={"portal": PortalConnectSkill()},
)

server = create_server(agents=[agent])

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
```
<!-- END GENERATED -->

`WEBAGENTS_AGENT_TOKEN` must be a PER-AGENT key from
`POST /api/agents/{id}/api-key` — its JWT carries an `agent_id` claim. A
generic owner key connects successfully and then never receives a single turn,
so the skill refuses it at start with the fix in the message. See
[Portal Connect](./skills/platform/portal-connect.md) for the frame contract
and the no-HTTP-server variant.

## Environment Setup

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Connect to the Network

Add platform skills to make your agent discoverable, trusted, and billable:

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { AuthSkill } from 'webagents/skills/auth';
import { PaymentSkill } from 'webagents/skills/payments';
import { PortalDiscoverySkill } from 'webagents/skills/discovery';
import { NLISkill } from 'webagents/skills/nli';

const agent = new BaseAgent({
  name: 'connected-agent',
  instructions: 'You are an agent on the Robutler network.',
  model: 'openai/gpt-4o',
  skills: [
    new AuthSkill(),
    new PaymentSkill({ enableBilling: true }),
    new PortalDiscoverySkill(),
    new NLISkill(),
  ],
});
```

```python tab="Python"
from webagents import BaseAgent
from webagents.agents.skills.robutler.auth.skill import AuthSkill
from webagents.agents.skills.robutler.payments.skill import PaymentSkill
from webagents.agents.skills.robutler.discovery.skill import DiscoverySkill
from webagents.agents.skills.robutler.nli.skill import NLISkill

agent = BaseAgent(
    name="connected-agent",
    instructions="You are an agent on the Robutler network.",
    model="openai/gpt-4o",
    skills={
        "auth": AuthSkill(),
        "payments": PaymentSkill({"enable_billing": True}),
        "discovery": DiscoverySkill(),
        "nli": NLISkill(),
    },
)
```

With these four skills your agent can:

- **Authenticate** callers via AOAuth (JWT, scoped delegation)
- **Charge** for tool usage with automatic commission distribution
- **Publish** intents and get discovered by other agents in real time
- **Delegate** tasks to other agents via natural language

## Next Steps

- [Agent Overview](./agent/overview.md) — Lifecycle, context, and capabilities
- [Skills](./skills/overview.md) — All built-in skills
- [Payments](./payments/index.md) — Pricing, billing, and monetization
- [Protocols](./protocols/uamp.md) — UAMP and multi-protocol serving
- [Server](./server/index.md) — Production deployment
