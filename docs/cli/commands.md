---
title: Commands
description: Reference for the webagents command, the same in the TypeScript and Python SDKs, and the chat's slash commands.
---

# Commands

One command surface for both SDKs: the same subcommands, arguments, flags and
messages. Global flags go before the command.

## Chat and Prompts

```bash
webagents                                  # chat with this folder's agent
webagents -a writer                        # the agent called writer (AGENT-writer.md, or its name:)
webagents -m anthropic/claude-sonnet-4     # another model for this run
webagents -p "Summarize README.md"         # one answer on stdout, then exit
webagents -p "..." --output-format json    # {"content": ..., "usage": {...}}
webagents -p "..." --output-format stream-json   # one JSON object per line
webagents --no-streaming                   # each reply appears whole
```

`chat` and `connect` are the same command by name. `-a` takes an agent's name
or the `<name>` of its `AGENT-<name>.md`; a name that matches nothing is refused
with the agents that are there. With no `-a`, the chat opens `AGENT.md`, else
the only `AGENT-<name>.md`, else the built-in assistant, `robutler`.

A prompt with no model to run on stops before anything is sent, names both ways
out, and exits 1. `--output-format json` and `stream-json` apply to `-p` only.

## Serving

```bash
webagents serve [path]              # one agent over HTTP (--port, default 3000)
webagents serve --host 0.0.0.0      # accept other machines
webagents daemon                    # every agent in this folder, reloaded as files change
webagents daemon --watch ./agents --port 8766 --no-cron
webagents mcp serve [path]          # the agent's tools to an MCP client, over stdio
webagents mcp serve --http 3001     # the same over Streamable HTTP at /mcp
webagents acp [path]                # the agent to a code editor, over ACP on stdio
```

`serve`, `daemon` and `mcp serve --http` listen on this machine only unless
`--host` says otherwise. A port that is already taken is one sentence and exit
1. See [Daemon](./daemon.md).

`mcp serve` over stdio is how Claude Code, Codex and OpenCode start a local
MCP (Model Context Protocol) server, and the caller is you, the agent's owner.
Over HTTP a caller must send a credential; the agent's auth skills and
`access:` block say who it is, and it lists and calls only the tools it may
use. See [MCP](../skills/core/mcp.md#serving-an-agent-over-mcp).

`acp` is how a code editor runs the agent in its agent panel, over the Agent
Client Protocol (ACP) on stdin and stdout. See
[Transports](../agent/transports.md#acp-agent-client-protocol-code-editors)
for the Zed and JetBrains settings.

## Schedules

```bash
webagents cron list                        # the schedules the agents in this folder declare
webagents cron run <agent> <schedule>      # run one now, and deliver it as configured
```

Both take `--watch <dir>` for another folder. See
[Daemon](./daemon.md#schedules) for the `cron:` block.

## Account and Publishing

```bash
webagents login                     # sign in through the browser
webagents login --token <key>       # with an API key, for a script
webagents login --url <portal>      # sign in to another portal, and use it from now on
webagents logout
webagents whoami
webagents link [name]               # link this folder to one of your agents
webagents link --show
webagents unlink
webagents publish [path]            # create the agent, or update the linked one
webagents publish --dry-run         # show what would be sent; needs no sign-in
webagents publish --yes             # create without asking
webagents budget <token_id>         # the budget tree of a run's payment token
```

See [Publish](./deploy.md).

## Setup and Checks

```bash
webagents init [name]               # a project folder with AGENT.md (default my-agent)
webagents init tools -t tool-agent  # with file and shell access, for you alone
webagents templates list
webagents doctor                    # runtime, agent, model, sign-in, keys, keychain, sandbox, skills, MCP, config
webagents doctor -a writer          # the same for another agent in this folder
webagents models                    # model providers, and which are ready here
webagents sandbox setup             # whether the sandbox runs on this machine, and what it lacks
```

`doctor` reports ten checks, one row each in `--json`; the `keychain` row is
explained in [Keychain dialogs on macOS](./keychain.md). `models` lists every
provider with the key or address it needs; `ollama` is ready when an Ollama
server answers at `OLLAMA_BASE_URL`. See [Models](./models.md).

`sandbox setup` checks rather than installs: it runs one confined `true`
through the sandbox engine and exits 1 when shell commands would be refused,
naming what this machine lacks and how to add it. See
[Sandbox](./sandbox.md#where-the-engine-comes-from).

## Skills

```bash
webagents skills list                          # every skill an agent file can name, and the SKILL.md skills here
webagents skills add shell todo                # add coded skills to this folder's AGENT.md (-a <agent> for another)
webagents skills remove shell                  # take one out
webagents skills add robutlerai/webagents --skill word-docx   # install a SKILL.md skill from GitHub
webagents skills add https://gitlab.com/group/repo.git
webagents skills add ./my-skills               # or from a folder
webagents skills remove word-docx              # remove an installed SKILL.md skill
```

Given names, `skills add` and `skills remove` change only the `skills:` list:
comments, the other keys, a skill's own settings and the instructions stay as
you wrote them. A name the list does not know is refused with a suggestion, and
nothing is written. After an add, the command says what a skill still needs on
this machine, such as a provider's key (`secrets set`) or a sign-in (`login`).
A file that is a symbolic link, lies outside the folder or belongs to another
user is never written.

Given a source instead (`owner/repo`, a git URL, a `tree` or `blob` URL, a
`file://` URL or a folder), `skills add` installs SKILL.md skills into
`.agents/skills/`: it clones at one commit, lists every file with scripts and
binaries flagged, asks before it installs (`--yes` where nothing can answer),
and records the source, commit and a digest in `.webagents/skills.lock`.
`remove` takes out only what the lock recorded. See
[SKILL.md skills](../skills/agent-skills.md).

## Keys and Secrets

```bash
webagents secrets set OPENAI_API_KEY    # asked for with echo off, or read from a pipe
webagents secrets list                  # stored keys, and which this shell sets
webagents secrets get NAME              # whether it is stored
webagents secrets get NAME --show       # its value, bare, for a script
webagents secrets remove NAME
```

The value is never taken as an argument, because an argument lands in your
shell history and the process list. Keys live in the system keychain, or an
owner-only file where there is none. Each CLI keeps its own keychain items
(see [Keychain dialogs on macOS](./keychain.md)), and a variable exported in
the shell wins over a stored one.

The same store holds the secrets an MCP server's settings name as
`${secret:NAME}`; see [MCP](../skills/core/mcp.md#secrets-in-server-settings).
`secrets unset` still works as another name for `remove`.

## Configuration

```bash
webagents config get                # every key, as JSON
webagents config get daemon.port    # one value, after every layer
webagents config set daemon.port 8766
webagents config set platform.url https://example.com --project
webagents config unset daemon.port
webagents config validate
webagents config path
```

An unknown key is refused, and a value is typed by its key's default. See
[Configuration](./configuration.md).

## Global Flags

| Flag | What it does |
|------|--------------|
| `--json` | One JSON document on stdout: `{"ok": true, "data": ...}` or `{"ok": false, "error": {"code", "message", "fix"}}` |
| `--profile <name>` | Separate settings, keys, sign-in and chat history, under `~/.webagents-<name>` |
| `--token <token>` | A platform token for this run, instead of the stored sign-in |
| `--max-tool-rounds <n>` | The tool rounds one turn may run, from 1 to 1000, for the chat, `-p`, `serve` and the daemon; above the agent file's `max_tool_rounds` |
| `-V`, `--version` | The version |
| `-h`, `--help` | Help, for any command |

`WEBAGENTS_PROFILE` and `WEBAGENTS_TOKEN` do the same as the flags.

## Chat Commands

Inside a chat, `/` opens the commands, grouped the way `/help` shows them:

- **This agent:** `/agent`, `/skills`, `/reload`, `/model`, `/tools`, `/mcp`,
  `/access`, `/cron`, `/memory`, `/sandbox`, `/status`
- **Conversation:** `/new`, `/clear`, `/resume`, `/undo`, `/rewind`
- **Account:** `/login`, `/logout`, `/keys`, `/secrets`, `/publish`
- **Chat:** `/help`, `/exit`

They are the same, in the same words, in both CLIs. See
[Chat](./repl.md#commands) for what each one does, and the keys.
