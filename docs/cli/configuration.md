---
title: Configuration
description: Configure agents through `AGENT.md` front matter, the CLI through `config.json`, and model access through environment variables.
---

# Configuration

Three places, three jobs: `AGENT.md` describes an agent, `config.json`
configures the CLI, and environment variables carry keys.

## CLI Settings (`config.json`)

The CLI reads `./.webagents/config.json` in the current directory first, then
`~/.webagents/config.json`, then its built-in defaults. Under `--profile
<name>` the global file is `~/.webagents-<name>/config.json` instead.

```bash
webagents config show                                  # every key, and where it came from
webagents config set platform.url https://robutler.ai  # global
webagents config validate                              # unknown keys and bad values
webagents config path                                  # which files are read
```

| Key | What it sets |
|-----|--------------|
| `platform.url` | The platform `login` and `deploy` talk to. `ROBUTLER_API_URL` outranks it; see [Deploy](./deploy.md#which-platform) |
| `daemon.host`, `daemon.port` | Where the local daemon listens, and where the chat and every `daemon` command look for it (default `127.0.0.1:8765`). A `--host` or `--port` flag outranks them. `daemon.host` must be an address on this machine (`127.0.0.1`, `::1` or `localhost`); see [Daemon](./daemon.md#configuration) |
| `link.agentId`, `link.agentName` | The platform agent this directory deploys to, written by `deploy`, `link` and the TypeScript `publish` |

Values can reference environment variables as `${VAR}` or `${VAR:-fallback}`.
The TypeScript CLI reads the same files, with `webagents config get|set|unset|validate|path`.

## Agent Configuration (`AGENT.md`)

YAML front matter configures the agent, and the markdown below it is the
agent's instructions:

```yaml
---
name: my-assistant
description: Answers questions about the project's data
model: openai/gpt-4o-mini
intents:
  - answer questions about the sales data
skills:
  - filesystem:
      whitelist:
        - ./data
  - shell
  - mcp:
      sqlite:
        command: uvx
        args: ["mcp-server-sqlite", "--db-path", "./data.db"]
sandbox:
  preset: strict
  allowed_folders:
    - ./data
---

# Instructions

You are a helpful assistant...
```

A skill's settings go under its own entry in `skills:`, as with `filesystem`
and `mcp` above. For `mcp`, each key is a server name. `sandbox:` confines what
the agent's shell commands can read, write and reach; see [Sandbox](./sandbox.md).

Unknown top-level keys are an error rather than silently ignored, so a typo such
as `skils:` is reported where it is. The keys that take effect are `name`,
`description`, `namespace`, `intents`, `model`, `skills`, `scopes`, `cron` and
`sandbox`. A few more (`tools`, `visibility`, `version`, `author`, `tags`,
`mcp_servers`, `watch`) are accepted so that older files still load, but they do
nothing, and `webagents doctor` says so.

## Environment Variables

Model keys, one per provider. An exported variable always wins;
`webagents secrets set <VAR>` keeps one for later runs instead (both CLIs read
it; the TypeScript CLI in its chat, not in `serve`). Signed in with
`webagents login` and without the key, the chats run the agent's model through
Robutler instead.

| Provider | Variable |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google, Python SDK | `GOOGLE_GEMINI_API_KEY` or `GEMINI_API_KEY` |
| Google, TypeScript SDK | `GOOGLE_API_KEY` |
| xAI | `XAI_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |

`OPENAI_BASE_URL` points the OpenAI skill at any OpenAI-compatible endpoint,
such as a local model server. `ROBUTLER_API_URL` selects the platform.
`WEBAGENTS_DEBUG=1` adds tracebacks to CLI errors.

`WEBAGENTS_AGENT_TOKEN` is the agent's own platform key, for containers and CI.
On a machine where you ran `webagents publish` in the agent's
directory it is not needed: the SDKs find the key that command stored. The
older names `WEBAGENTS_API_KEY` (platform skills) and `ROBUTLER_API_KEY`
(TypeScript payments) are still read.
