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
```

Both listen on this machine only unless `--host` says otherwise. See
[Daemon](./daemon.md).

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
```

See [Publish](./deploy.md).

## Setup and Checks

```bash
webagents init [name]               # a project folder with AGENT.md (default my-agent)
webagents init tools -t tool-agent  # with file and shell access
webagents templates list
webagents doctor                    # runtime, agent, model, sign-in, keys, sandbox, config
webagents models                    # providers, and which have a key here
webagents skills list               # the skills an agent file can name
webagents skills add shell todo     # add to this folder's AGENT.md (-a <agent> for another)
webagents skills remove shell       # take one out
```

`skills add` and `skills remove` change only the `skills:` list: comments, the
other keys, a skill's own settings and the instructions stay as you wrote them.
A name the list does not know is refused with a suggestion, and nothing is
written. After an add, the command says what a skill still needs on this
machine, such as a provider's key (`secrets set`) or a sign-in (`login`).

## Keys

```bash
webagents secrets set OPENAI_API_KEY    # asked for with echo off
webagents secrets list                  # stored keys, and which this shell sets
webagents secrets get NAME [--show]     # whether it is stored, or its value
webagents secrets unset NAME
```

Keys live in the system keychain, or an owner-only file where there is none.
Both CLIs read the same store. A variable exported in the shell wins over a
stored one.

## Configuration

```bash
webagents config get [key]          # one value, or every key as JSON
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
| `--profile <name>` | Separate settings, keys and sign-in, under `~/.webagents-<name>` |
| `--token <token>` | A platform token for this run, instead of the stored sign-in |
| `-V`, `--version` | The version |
| `-h`, `--help` | Help, for any command |

`WEBAGENTS_PROFILE` and `WEBAGENTS_TOKEN` do the same as the flags.

## Chat Commands

Inside a chat, `/` opens the commands: `/help`, `/new`, `/resume`, `/undo`, `/model`,
`/agent`, `/tools`, `/status`, `/login`, `/keys`, `/sandbox`, `/publish` and
the rest. They are the same, in the same words, in both CLIs. See
[Chat](./repl.md#commands) for the full list and the keys.
