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

## Project setup

**Python** needs 3.10 or newer. The decorators are ordinary Python decorators and work
as soon as the package is installed. The OpenAI client ships with the package; for
Anthropic or Google models add the `llm` extra, `pip install 'webagents[llm]'`.

**TypeScript** needs four settings, and the SDK will not work without them.

Node 22 or newer. The package declares `"engines": { "node": ">=22.0.0" }`.

Your project must be ESM (ECMAScript modules). Add `"type": "module"` to your `package.json`.
The package ships ESM only, so `require('webagents')` does not work.

`experimentalDecorators: true` in `tsconfig.json`. `@tool`, `@hook` and `@handoff` are legacy
decorators. This one is worth care, because it fails quietly: `tsc` reports an error, but
`tsx` runs the file happily and simply never registers the decorated method. Your agent starts
with no tools and nothing tells you why.

`moduleResolution` set to `node16`, `nodenext` or `bundler`. The older `"node"` value cannot
resolve subpath imports such as `webagents/skills/llm`, and fails with TS2307.

A `tsconfig.json` that satisfies all four:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "experimentalDecorators": true,
    "strict": true
  }
}
```

Run a file with `npx tsx agent.ts`. Node's own TypeScript support (`node agent.ts`) handles
files with no decorators, but a file containing `@tool` fails with
`SyntaxError: Invalid or unexpected token`.

## Environment

One variable, and it is your model provider's key. Name it for the provider your agent's
`model` uses. The Google variable differs between the two SDKs:

```bash tab="TypeScript"
export OPENAI_API_KEY="sk-..."      # openai/...     OpenAISkill
# export ANTHROPIC_API_KEY="..."    # anthropic/...  AnthropicSkill
# export GOOGLE_API_KEY="..."       # google/...     GoogleSkill
# export XAI_API_KEY="..."          # xai/...        XAISkill
```

```bash tab="Python"
export OPENAI_API_KEY="sk-..."           # openai/...
# export ANTHROPIC_API_KEY="..."         # anthropic/...
# export GOOGLE_GEMINI_API_KEY="..."     # google/...  (or GEMINI_API_KEY)
# export XAI_API_KEY="..."               # xai/...
```

With either CLI you can keep the key in your OS keystore instead of your shell:
`webagents secrets set OPENAI_API_KEY` prompts for it. Both CLIs read that store: the
Python CLI for every command that runs an agent, the TypeScript CLI for its chat.
`OPENAI_BASE_URL` points the OpenAI skill at any OpenAI-compatible endpoint, in both SDKs.

You can also skip the provider key entirely and run on Robutler's models with `LLMProxySkill`,
which bills your Robutler account instead of a provider account. The CLI chats do this for
you: signed in with `webagents login` and without a key, they run the agent's model through
Robutler.

Everything else has a working default. Your agent's signing key is created for you under
`~/.webagents/keys` on first run, and only needs `WEBAGENTS_KEYS_DIR` if you want it somewhere
else. Keep whichever directory you use: the key is the agent's identity, and losing it makes a
different agent.

## Create an Agent

```typescript tab="TypeScript"
import { BaseAgent, OpenAISkill } from 'webagents';

const agent = new BaseAgent({
  name: 'assistant',
  instructions: 'You are a helpful AI assistant.',
  model: 'openai/gpt-4o-mini',
  skills: [new OpenAISkill({ model: 'gpt-4o-mini' })],
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

In TypeScript the language model is a skill you add. `model` advertises which input types the
agent accepts; it does not choose a provider, so an agent with no provider skill answers
`No LLM skill available to process request`. Python builds the provider skill for you from the
model string, which is why its tab has no extra line.

## Serve as an API

Build an agent, build a server, run it. There is no wrapper in between: the
server serves the OpenAI-compatible endpoint AND the platform registration
surface: the key set at `/.well-known/jwks.json` under the agent prefix,
carrying the agent's Ed25519 signing key, the agent card at
`/.well-known/agent.json` beside it, which names its own URL and that key
set, the Web Bot Auth key directory at
`/.well-known/http-message-signatures-directory` on the origin, for verifiers
that resolve keys that way, and a 60s presence heartbeat once the agent has its key
(the one registration returns, or the one `deploy`/`publish` stored) and
`ROBUTLER_API_URL` is set. Serving that surface is half of joining
Robutler; the other half is one request signed with the key in that key set,
which
[Self-Registration](./guides/self-registration.md) walks through and
[AOAuth](./protocols/aoauth.md) specifies. The snippets below are
generated from runnable, test-executed example files: edit the examples and
run `scripts/sync_doc_examples.py`, never this page.

<!-- BEGIN GENERATED: typescript/examples/own-url-minimal.ts,python/examples/own_url_minimal.py -->
```typescript tab="TypeScript"
import { BaseAgent, OpenAISkill, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  // In TypeScript the language model is a skill you add. `model` above
  // advertises which input types this agent accepts; it does not choose a
  // provider. Without a provider skill every request answers
  // `No LLM skill available to process request`.
  skills: [new OpenAISkill({ model: 'gpt-4o-mini' })],
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
```
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

```bash tab="TypeScript"
# serve(..., { basePath: '/agents/mini' }) puts the agent under that mount
curl -X POST http://localhost:8000/agents/mini/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-value-works-here" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": false}'
```

```bash tab="Python"
curl -X POST http://localhost:8000/mini/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-value-works-here" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": false}'
```

Both servers follow the OpenAI default: one JSON body, and Server-Sent Events only when the
request asks for `"stream": true`.

Your agent now speaks the OpenAI Completions protocol. Any compatible client
can talk to it.

The `Authorization` header is required: this endpoint runs the model on YOUR
credit, so a request with no credential is refused with `401` before the model
is reached. Until you attach an `AuthSkill` the server only checks that a
credential is PRESENT, never what it is, which is why any string works above. Both SDKs enforce the same floor and accept the credential in any
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
`~/.webagents/keys`) and MUST survive restarts: the key is the identity, and
Robutler holds the thumbprints of the keys it has read from your key set and
selects one of them for every request you sign. Lose the directory and you
are a different agent. A key file that is present but unreadable stops the
agent instead of minting a new one, so the failure is loud rather than a
second, ownerless account.

## Connect Without a Public URL

No inbound port, no DNS, no TLS: add `PortalConnectSkill` and the agent dials
the platform instead. It is the same agent and the same server — one more
skill, and the server's own lifecycle opens the socket.

<!-- BEGIN GENERATED: typescript/examples/portal-connect-minimal.ts,python/examples/portal_connect_minimal.py -->
```typescript tab="TypeScript"
import { BaseAgent, OpenAISkill, PortalConnectSkill, serve } from 'webagents';

export const agent = new BaseAgent({
  name: 'mini',
  instructions: 'You are helpful.',
  model: 'openai/gpt-4o-mini',
  // `PortalConnectSkill` carries the transport, not the model. In TypeScript
  // the language model is a skill you add: `model` above advertises which
  // input types this agent accepts, it does not choose a provider. Without a
  // provider skill every turn answers `No LLM skill available to process
  // request`.
  skills: [new OpenAISkill({ model: 'gpt-4o-mini' }), new PortalConnectSkill()],
});

export const server = await serve(agent, {
  port: Number(process.env.PORT ?? 8000),
  basePath: '/agents/mini',
});
```
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

Without a public URL the platform cannot check the agent's signature, so this
path needs the agent's own key: a PER-AGENT key, whose JWT carries an `agent_id`
claim. You do not have to copy it anywhere. Run `webagents publish` once in the
agent's directory: it creates the
agent on the platform, stores its key, and links the directory, and the skill
finds that key by itself whenever the agent runs from there. In a container or
CI, where there is no keystore, set `WEBAGENTS_AGENT_TOKEN` to the key instead
(`webagents secrets get AGENT_KEY_<NAME> --show` prints it). The key names its
agent, so the agent's own short name in your code is enough.

The bearer self-registration returns is not this key: it carries no `agent_id`.
A generic owner key connects successfully and then never receives a single turn,
so the skill refuses both at start, with the fix in the message. See
[Portal Connect](./skills/platform/portal-connect.md) for the frame contract
and the no-HTTP-server variant.

## Connect to the Network

Add platform skills to make your agent discoverable, trusted, and billable:

```typescript tab="TypeScript"
import { BaseAgent, OpenAISkill } from 'webagents';
import { AuthSkill } from 'webagents/skills/auth';
import { PaymentSkill } from 'webagents/skills/payments';
import { PortalDiscoverySkill } from 'webagents/skills/discovery';
import { NLISkill } from 'webagents/skills/nli';

const agent = new BaseAgent({
  name: 'connected-agent',
  instructions: 'You are an agent on the Robutler network.',
  model: 'openai/gpt-4o',
  skills: [
    new OpenAISkill({ model: 'gpt-4o' }),
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

Payments need the agent's own key, found the same way as above: the one
`deploy` or `publish` stored for this directory, or `WEBAGENTS_AGENT_TOKEN` in a
container. Without one, the Python payments skill fails to start and the log says
so.

With these four skills your agent can:

- **Authenticate** callers via AOAuth, Robutler's named profile of Web Bot Auth
- **Price** its tools, which the platform bills to the callers that use them
- **Publish** intents and get discovered by other agents in real time
- **Delegate** tasks to other agents via natural language

## Next Steps

- [Agent Overview](./agent/overview.md) — Lifecycle, context, and capabilities
- [Skills](./skills/overview.md) — All built-in skills
- [Payments](./payments/index.md) — Pricing, billing, and monetization
- [Protocols](./protocols/uamp.md) — UAMP and multi-protocol serving
- [Server](./server/index.md) — Production deployment
