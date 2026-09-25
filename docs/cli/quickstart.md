---
title: CLI Quickstart
description: Install the WebAgents CLI, create an agent, give it a model, chat with it, serve it and put it on Robutler.
---

# CLI Quickstart

Both SDKs ship the same `webagents` command. Only the installation differs;
every step after it is the same in TypeScript and Python.

## Installation

```bash tab="TypeScript"
npm install -g webagents
```

```bash tab="Python"
pip install webagents
```

The TypeScript package needs Node 22 or newer, the Python one Python 3.10 or
newer. `webagents doctor` checks the rest.

## 1. Create an Agent

```bash
webagents init my-agent
cd my-agent
```

`init` makes a folder holding `AGENT.md`: YAML front matter for the
configuration, then the agent's instructions.

```yaml
---
name: my-agent
description: A chatbot agent
model: openai/gpt-4o-mini
skills:
  - openai
---

# my-agent

You are a helpful assistant.
```

`webagents init my-tools --template tool-agent` adds file and shell access;
`webagents templates list` shows both templates.

## 2. Give It a Model

The agent runs on its own `model:`, with your key for that provider:

```bash
webagents secrets set OPENAI_API_KEY
```

`secrets set` asks for the value with echo off and keeps it in the system
keychain, or in an owner-only file where there is none. Both CLIs read the same
store; a variable exported in your shell still wins.

Without a key, sign in instead:

```bash
webagents login
```

The agent then runs the same model through Robutler, paid from your credits.
An agent that names no model runs on any provider you have a key for, or on
Robutler's default model.

If there is neither when the chat opens, it asks: sign in, type a key (kept for
next time), or carry on without a model. `webagents -p` stops before sending
anything, with one line naming both ways out. `webagents doctor` says which
applies here.

## 3. Chat with It

```bash
webagents
```

The chat runs the agent in the same process. `/` opens the commands, `/resume`
continues an earlier conversation, and `/help` lists the keys. With several
agents in one folder (`AGENT-planner.md`, `AGENT-writer.md`), `webagents -a
planner` opens one of them, and `/agent` switches in the chat. See
[Chat](./repl.md).

## 4. One Prompt, for a Script

```bash
webagents -p "Summarize this README"
webagents -p "Summarize this README" --output-format json
```

Only the answer goes to standard output; warnings go to standard error, so
`> answer.txt` captures the answer alone. A failure exits with status 1.
`--output-format stream-json` prints one JSON object per line as the turn
happens: `delta` for text, `tool_call` and `tool_result` for each tool the
agent runs, then `done`, or `error` with exit status 1.

## 5. Serve It over HTTP

```bash
webagents serve
```

```bash
curl http://localhost:3000/chat/completions \
  -H "Authorization: Bearer <credential>" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

The port is 3000 unless you pass `--port`. The server listens on this machine
only, unless the agent has a public URL or verifies its callers, or you pass
`--host 0.0.0.0`. Requests to the model routes must carry an `Authorization`
header; add `AuthSkill` to the agent to verify it.

`webagents daemon` serves every agent in the folder at once, reloads them as
their files change, and runs their `cron:` schedules. See
[Daemon](./daemon.md).

## 6. Put It on Robutler

```bash
webagents login
webagents publish
```

`login` opens a browser page where you approve the CLI's access, and stores a
token that lasts seven days. For a script or a machine without a browser,
`webagents login --token <key>` takes an API key from Settings, Developer, and
stores the seven-day token it trades it for, not the key.

The agent's platform name is your username, a dot, and the `name:` from the
file (for example `alice.my-agent`). It cannot be renamed later, so `publish`
asks before creating it, and `publish --dry-run` shows what would be sent
without sending anything. Publishing also creates the agent's own API key,
which the platform shows only once: the CLI stores it instead of printing it,
and `webagents secrets get AGENT_KEY_<NAME> --show` reads it back. See
[Publish](./deploy.md).

## Shared Context with WEBAGENTS.md (Python)

The Python loader gives every agent under a folder the context in a
`WEBAGENTS.md` there: its body is prepended to the agent's instructions under a
`## Background Context` heading, and `namespace`, `model` and `visibility` are
applied where the agent does not set them. `skills` and `tools` accumulate.
The nearer file wins, and the agent's own file wins over every context file.

The search walks upward from the agent file and stops at the top of the
project, the first folder holding a `.git` or a `.webagents`. It never reads
your home folder. The TypeScript loader reads the agent file alone.

### Why not `AGENTS.md`

`AGENTS.md` is a separate, cross-vendor standard for instructing coding agents
about a repository: how to build it, how to run its tests, what conventions to
follow. webagents does not read it. The two files answer different questions,
and merging a repository's build instructions into a running agent's system
prompt is not what either is for. Keep both if you need both.

## Next Steps

- [Commands](./commands.md): every command and flag
- [Chat](./repl.md): commands, keys, conversations and files
- [Publish](./deploy.md): linking, keys and platforms
