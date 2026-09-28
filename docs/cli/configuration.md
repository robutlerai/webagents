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
<name>` the global file is `~/.webagents-<name>/config.json` instead, and the
chat's history, sessions, checkpoints and log (`logs/repl.log`) live in that
folder too. An agent's signing keys stay in `~/.webagents/keys` whatever the
profile: the key is the agent's identity on this machine, not a platform
setting (`WEBAGENTS_KEYS_DIR` moves it).

```bash
webagents config get                                   # every key, as JSON, after every layer
webagents config set platform.url https://robutler.ai  # global
webagents config validate                              # unknown keys and bad values
webagents config path                                  # which files are read
```

| Key | What it sets |
|-----|--------------|
| `platform.url` | The platform `login` and `publish` talk to. `ROBUTLER_API_URL` outranks it; see [Publish](./deploy.md#which-platform) |
| `daemon.host`, `daemon.port` | Where the local daemon listens, and where the chat and every `daemon` command look for it (default `127.0.0.1:8765`). A `--host` or `--port` flag outranks them. `daemon.host` must be an address on this machine (`127.0.0.1`, `::1` or `localhost`); see [Daemon](./daemon.md#the-address) |
| `link.agentId`, `link.agentName` | The platform agent this directory publishes to, written by `link` and `publish` |

Values can reference environment variables as `${VAR}` or `${VAR:-fallback}`.
Both CLIs read the same files.

## Agent Configuration (`AGENT.md`)

YAML front matter configures the agent, and the markdown below it is the
agent's instructions:

```yaml
---
name: my-assistant
description: Answers questions about the project's data
model: openai/gpt-4o-mini
fallback_models:
  - anthropic/claude-haiku-4-5
intents:
  - answer questions about the sales data
skills:
  - openai
  - anthropic
  - filesystem:
      whitelist:
        - ./data
  - shell
  - memory
  - mcp:
      sqlite:
        command: uvx
        args: ["mcp-server-sqlite", "--db-path", "./data.db"]
agent_skills:
  - ./skills
access:
  groups:
    analysts:
      - domain:analytics.example
  tools:
    analysts: [filesystem]
sandbox:
  preset: strict
  allowed_folders:
    - ./data
observability:
  otel: true
---

# Instructions

You are a helpful assistant...
```

| Key | What it does |
|-----|--------------|
| `name` | The agent's name; on Robutler, the part after your username |
| `description` | One line about the agent, shown on its card |
| `namespace` | A grouping label for the agent |
| `intents` | What the agent can do, for discovery on Robutler |
| `model` | `provider/model`; see [Models](./models.md) |
| `fallback_models` | Models to try, in order, when `model` does not answer; see [Models](./models.md#failover) |
| `skills` | The coded skills the agent loads, by the names `webagents skills list` shows, each with its own settings |
| `agent_skills` | Folders of SKILL.md skills beyond `.agents/skills/`; see [SKILL.md skills](../skills/agent-skills.md) |
| `access` | Who may call the agent and which tools each group gets; see [Who can call your agent](../guides/trust.md) |
| `sandbox` | What the agent's shell commands can read, write and reach (`preset`, `allowed_folders`, `network`, `env_passthrough`); see [Sandbox](./sandbox.md) |
| `cron` | Schedules the daemon runs; see [Daemon](./daemon.md#schedules) |
| `max_tool_rounds` | The tool rounds one turn may run, a whole number from 1 to 1000 (default 50). At the limit the agent makes one last call with tools off and answers from what it gathered; `--max-tool-rounds` and the chat's `/rounds` take precedence |
| `observability` | `{otel: true}` records runs as OpenTelemetry spans; see [Observability](#observability) |
| `scopes` | The agent's own scopes (Python) |

A skill's settings go under its own entry in `skills:`, as with `filesystem`
and `mcp` above. For `mcp`, each key is a server name.

The loaders are strict, so a mistake is reported where it is instead of
running as something else:

- An unknown top-level key is an error with a "did you mean": `skils:` is
  reported, not ignored. So is a mistyped key inside `sandbox:`,
  `observability:`, a `cron:` schedule, or the `memory` skill's settings.
- Front matter that is not valid YAML is refused with the line and column.
- The old string form of `cron:` is refused with the list form it takes.
- An agent file that is a symbolic link is refused: an agent file must be a
  regular file in its folder.

A few keys (`tools`, `visibility`, `version`, `author`, `tags`,
`mcp_servers`, `watch`) are accepted so that older files still load, but they
do nothing, and `webagents doctor` says so.

## Observability

`observability: {otel: true}` (or `observability: true`) records each run as
OpenTelemetry spans that follow the GenAI semantic conventions:
`invoke_agent <agent>` for the run, `chat <model>` for each model call (with
the provider, the model and the token counts), `execute_tool <tool>` for each
tool call, and `settle_payment` for a settle (with the credits). The metrics
`gen_ai.client.token.usage` and `gen_ai.client.operation.duration` are
recorded beside them. No message text, tool arguments or tool results are
ever put in an attribute.

`WEBAGENTS_OTEL=1` (or `true`, `on`, `yes`) switches it on without touching
the file; the file wins when it says something. The SDK adds no dependency:
spans are recorded when the OpenTelemetry API is installed
(`@opentelemetry/api` in TypeScript, `opentelemetry-api` in Python) and a
tracer provider is configured, as any OpenTelemetry exporter setup does, and
nothing happens otherwise.

## Environment Variables

Model keys, one per provider. An exported variable always wins;
`webagents secrets set <VAR>` keeps one for later runs instead, and each CLI
uses the keys it stored in the chat, `-p`, `serve`, the daemon, `cron run`,
`acp` and `mcp serve`. Signed in with `webagents login` and without the key, the
chats run the agent's model through Robutler instead.

| Provider | Variable |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY`, else `GOOGLE_GEMINI_API_KEY`, else `GEMINI_API_KEY` |
| xAI | `XAI_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Ollama | none; `OLLAMA_BASE_URL` if the server is not at `http://localhost:11434/v1` |

`OPENAI_BASE_URL` points the OpenAI skill at any OpenAI-compatible endpoint
when the file names none; `- openai: {base_url: ...}` does the same for one
agent. `ROBUTLER_API_URL` selects the platform. `WEBAGENTS_DEBUG=1` adds
tracebacks to CLI errors.

`WEBAGENTS_AGENT_TOKEN` is the agent's own platform key, for containers and CI.
On a machine where you ran `webagents publish` in the agent's
directory it is not needed: the SDKs find the key that command stored. The
older names `WEBAGENTS_API_KEY` (platform skills) and `ROBUTLER_API_KEY`
(TypeScript payments) are still read.
