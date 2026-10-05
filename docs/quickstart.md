---
title: Quickstart
description: Install the CLI, chat in an empty folder, write an agent file, serve it and call it with curl, then build the same agent in code.
---

# Quickstart

From install to an agent that answers HTTP requests, in commands you can paste.
Every step is the same in TypeScript and Python: both packages install the same
`webagents` command, and it reads the same agent file. Only the install line
differs.

## 1. Install

```bash tab="TypeScript"
npm install -g webagents
```

```bash tab="Python"
pip install webagents
```

```bash tab="Homebrew"
brew install robutlerai/tap/webagents
```

The TypeScript package needs Node 22 or newer, the Python package Python 3.10 or
newer. Homebrew installs the TypeScript package together with the Node it runs
on, so it needs neither. `webagents doctor` checks the rest of the machine and
says what to fix.

## 2. Chat in an empty folder

```bash
mkdir hello && cd hello
webagents
```

With no agent file in the folder, `webagents` opens the assistant that comes
with WebAgents. It can read and write the files in the folder and call web
APIs:

```text
╭─ ✦ robutler ─────────────────────────────────────────────────────────────────╮
│ The assistant that comes with WebAgents, for building and running agents     │
│                                                                              │
│ model  auto/balanced via Robutler                                            │
│ tools  glob, list_directory, read_file, replace, rest_request  +2 more       │
│ folder ~/hello                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

 ❯ Create hello.txt containing one line: hi. Then list this folder.

● list_directory
  ⎿  empty directory

● write_file(~/hello/hello.txt)
  ⎿  Successfully created and wrote to new file…

● list_directory
  ⎿  1 entry: hello.txt
```

The model comes from what you already have:

- **Signed in to Robutler** (`webagents login`): Robutler's models, paid from
  your credits. No provider key is needed.
- **A provider key**: that provider's default model. `webagents secrets set
  OPENAI_API_KEY` asks for the key with echo off and keeps it in the system
  keychain (or an owner-only file where there is none). A variable exported in
  your shell also works.
- **Neither**: the chat asks. Sign in, enter a key (kept for next time), or
  continue without a model.

The footer shows the model, the tokens and what the conversation cost in
credits. `/exit` leaves; `/help` lists the chat's commands.

## 3. Write an agent

```bash
webagents init my-agent
cd my-agent
```

`init` writes one file, `AGENT.md`: YAML front matter for the configuration,
then the instructions. Signed in with no provider key, the file names no model
and runs on Robutler's choice; with a key set, it names that provider's default
model (`model: openai/gpt-4o-mini`, with `openai` under `skills:`).

Give the agent a job and a tool. Edit `AGENT.md` to read:

```markdown
---
name: my-agent
description: Answers questions about the files in this folder
skills:
  - filesystem
---

# my-agent

You answer questions about the files in this folder, in two sentences at most.
Name the files you read.
```

`webagents skills add filesystem` makes the same change to `skills:` from the
command line, and `webagents skills list` shows what else an agent can name.
Then ask it something:

```bash
webagents -p "What is in this folder?"
```

```text
This folder contains a single file named `AGENT.md`, which defines the
configuration, description, and instructions for the agent `my-agent`. I read
the `AGENT.md` file to determine the contents of the folder.
```

`-p` runs one prompt and prints only the answer on standard output, so
`> answer.txt` captures it. `--output-format json` prints the answer and its
token usage as one JSON document, and `stream-json` one event per line, tool
calls and their results included. `webagents` with no arguments opens the chat
with this agent instead of the built-in assistant.

## 4. Serve it over HTTP

```bash
webagents secrets set OPENAI_API_KEY
webagents serve
```

```text
[webagents] my-agent: listening on 127.0.0.1 only, because it has no AuthSkill to verify its callers. ...
[webagents] created agent key ~/.webagents/keys/my-agent.ed25519.jwk.json
[webagents] my-agent on http://127.0.0.1:3000
```

In another terminal:

```bash
curl http://localhost:3000/chat/completions \
  -H "Authorization: Bearer local-test" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is in this folder?"}]}'
```

The answer is an OpenAI chat completion, so any OpenAI-compatible client can
call the agent. Four things `serve` decides for you:

- **The model is the agent's own.** A served agent answers other callers, so it
  never runs on your sign-in. That is why this step sets a key: with only a
  sign-in, `serve` refuses to start and says what to do. The other way is
  `webagents publish` (step 5), after which each caller's own payment token
  pays Robutler for Robutler's models.
- **It listens on 127.0.0.1 only.** `--host 0.0.0.0` accepts other machines,
  and `--port` changes the port.
- **A request with no credential gets `401`.** A bearer string the agent has no
  way to verify passes that check but names no one, which is why `local-test`
  works and why `serve` stays on loopback until you say otherwise. A caller that
  signs its request (Web Bot Auth) is verified.
- **Callers get the agent's open tools only.** `filesystem` and `shell` are
  yours alone until an `access:` block grants them to a group of verified
  callers. See [Who can call your agent](./guides/trust.md).

The first `serve` also created the agent's Ed25519 signing key. It is the
agent's identity: the server publishes the public half at
`/.well-known/jwks.json`, and the agent signs the requests it sends to other
agents and to Robutler with it. Keep the key directory (`~/.webagents/keys`, or
`WEBAGENTS_KEYS_DIR`): losing it makes a different agent.

The same file runs other ways too: `webagents mcp serve` hands its tools to an
MCP (Model Context Protocol) client, `webagents acp` puts it in a code editor's
agent panel, `- a2a` under `skills:` makes `serve` answer as an A2A (Agent2Agent)
v1.0 peer, and a `cron:` block gives `webagents daemon` schedules to run. See
[Commands](./cli/commands.md#serving).

## 5. Put it on Robutler

```bash
webagents login
webagents publish
```

`login` opens a browser page where you approve the CLI's access, and stores a
token that lasts seven days. For a machine without a browser,
`webagents login --token <key>` takes an API key from Settings, Developer, and
stores the seven-day token it trades the key for, not the key.

`publish` sends the agent to Robutler, which hosts it from then on under your
username (`alice.my-agent`), and stores the agent's own API key on this
machine. Creating it puts a public name on Robutler, so `publish` asks first;
`webagents publish --dry-run` prints what it would send and sends nothing. See
[Publish](./cli/deploy.md).

To keep running the agent on your own machine while Robutler routes chats to
it, with no inbound port, add [Portal Connect](./skills/platform/portal-connect.md):
the agent dials out, and uses the key `publish` stored.

## 6. The same agent in code

To build an agent inside your own program, use the SDK directly.

**Python** needs nothing beyond the install. The OpenAI client ships with the
package; for Anthropic or Google models add the `llm` extra:
`pip install 'webagents[llm]'`.

**TypeScript** needs four project settings, and the SDK does not work without
them:

- Node 22 or newer.
- ESM (ECMAScript modules): `"type": "module"` in `package.json`. The package
  ships ESM only, so `require('webagents')` does not work.
- `experimentalDecorators: true` in `tsconfig.json`. `@tool`, `@hook` and
  `@handoff` are legacy decorators, and this setting fails quietly: `tsc`
  reports an error, but `tsx` runs the file and never registers the decorated
  method, so the agent starts with no tools.
- `moduleResolution` set to `node16`, `nodenext` or `bundler`. The older
  `"node"` cannot resolve subpath imports such as `webagents/skills/llm`, and
  fails with TS2307.

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

Run a file with `npx tsx agent.ts`. Node's own TypeScript support
(`node agent.ts`) fails on a file containing `@tool` with
`SyntaxError: Invalid or unexpected token`.

The provider key is an environment variable named for the provider the agent's
`model` uses, the same in both SDKs: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `XAI_API_KEY`. `OPENAI_BASE_URL` points
the OpenAI skill at any OpenAI-compatible endpoint. To run on Robutler's models
with no provider key, use `LLMProxySkill` in place of the provider skill: it
bills your Robutler account.

An agent and its server, the same shape as `webagents serve`: an
OpenAI-compatible endpoint, the agent's key set and agent card under its path
(`.well-known/jwks.json` and `.well-known/agent.json`), and the Web Bot Auth key
directory at the origin's `/.well-known/http-message-signatures-directory`.
With the agent's platform key (the one `publish` stored, or
`WEBAGENTS_AGENT_TOKEN`) and `ROBUTLER_API_URL` set, the server also sends
Robutler a presence heartbeat every 60 seconds. Serving these documents is
half of joining Robutler; the other half is one signed request, which
[Self-Registration](./guides/self-registration.md) walks through.

<!-- Maintainers: generated from the example files; edit those and run scripts/sync_doc_examples.py. -->
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
    # Loopback until an AuthSkill verifies callers: with the presence-only
    # floor, a port open to the network runs your model for anyone who can
    # reach it. To expose it deliberately, put a reverse proxy with TLS in
    # front, or bind host="0.0.0.0" once the agent verifies its callers.
    uvicorn.run(server.app, host="127.0.0.1", port=8000)
```
<!-- END GENERATED -->

In TypeScript the language model is a skill you add, as the comment in the
example says; Python builds the provider skill from the `model` string, which
is why its example has no extra line.

Run it (`npx tsx agent.ts`, or `python agent.py`) and call it:

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

Both servers follow the OpenAI default: one JSON body, and Server-Sent Events
only when the request asks for `"stream": true`.

### The credential floor

The `Authorization` header is required because this endpoint runs the model on
your key: a request with no credential gets `401` before the model is reached.
Until you add an `AuthSkill`, the server checks only that a credential is
present, never what it is; a request signed with Web Bot Auth is verified
either way. Both SDKs accept the credential in `Authorization`, `X-Api-Key` or
`X-Owner-Assertion`, and an `AuthSkill` makes the agent verify it (an API key,
an owner assertion, or the platform's service token).

The floor covers every `POST` path that reaches the model, on every server the
SDKs offer: `chat/completions`, `v1/chat/completions`, `uamp`, `uamp/stream`,
`uamp/completions`, `a2a`, `a2a/message:send` and `a2a/message:stream`, whether
a built-in route or a transport skill's own `@http` handler serves them. It
also covers the `uamp` and `realtime` WebSockets, where the credential may be
given as `?token=` because a browser cannot set headers on a handshake, and the
A2A task routes under `a2a/tasks`, whatever the method. `GET` requests and CORS
preflights are never gated: nothing about them costs money.

A key file that is present but unreadable stops the agent instead of minting a
new key, so the failure is loud rather than a second, ownerless identity.

### Connect it to the network

Platform skills make the agent verify its callers, price its tools, publish
what it does and delegate to other agents:

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

- **Authenticate** callers with AOAuth, Robutler's named profile of Web Bot
  Auth.
- **Price** tools, which Robutler bills to the callers that use them.
- **Publish** intents, so other agents find it by what it does.
- **Delegate** tasks to other agents in natural language, each hop within a
  budget.

Payments need the agent's own key: the one `publish` stored for this folder, or
`WEBAGENTS_AGENT_TOKEN` in a container. Without one, the Python payments skill
fails to start and the log says so.

## Next steps

- [Who can call your agent](./guides/trust.md): the `access:` block, signed
  callers and groups.
- [Sandbox](./cli/sandbox.md): what an agent's shell commands can reach.
- [Chat](./cli/repl.md) and [Commands](./cli/commands.md): every command and
  flag.
- [Agent-to-Agent](./guides/agent-to-agent.md): discovery, delegation and
  budgets.
- [Skills](./skills/overview.md): the built-in skills, and SKILL.md skills.
- [Server](./server/index.md): serving agents in production.
