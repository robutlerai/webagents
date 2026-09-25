---
title: WebAgents CLI
description: The webagents command, the same in the TypeScript and Python SDKs - chat, one-shot prompts, serving, the local daemon, signing in and publishing.
---

# WebAgents CLI

Both SDKs install a `webagents` command, and it is the same command: the same
subcommands, arguments, flags and messages, reading the same agent file
(`AGENT.md`) and the same settings, keys and sign-in. A script or a habit
written against one works against the other.

Start with the [Quickstart](./quickstart.md).

## Installation

```bash tab="TypeScript"
npm install -g webagents
```

```bash tab="Python"
pip install webagents
```

## Everyday Commands

```bash
webagents init my-agent              # a project folder with AGENT.md
webagents                            # chat with this folder's agent
webagents -p "Summarize this"        # one prompt, answer on stdout
webagents doctor                     # what stands between this folder and a running agent
webagents serve                      # this agent over HTTP, on port 3000
webagents publish                    # send it to Robutler
```

## Command Tree

```
webagents               Chat (the default command; also `chat` and `connect`)
├── serve [path]        One agent over HTTP (--port, --host)
├── daemon              Every agent in a folder, reloaded as files change (--port, --host, --watch, --no-cron)
├── init [name]         A project folder with AGENT.md (--template chatbot|tool-agent)
├── publish [path]      Create the agent on Robutler, or update the linked one (--yes, --dry-run)
├── login, logout       Your Robutler account (--url, --token)
├── whoami              Who you are signed in as
├── link [name]         Link this folder to one of your agents (--show)
├── unlink              Forget that link
├── doctor              Check this setup and say what to fix
├── models              Model providers, and which have a key here
├── skills list         Skills an agent file can name
├── templates list      What `init --template` can make
├── config              get, set, unset, validate, path
└── secrets             list, set, unset, get: keys kept on this machine
```

Global flags go before the command: `--json` (one JSON document on standard
output), `--profile <name>` (separate settings, keys and sign-in) and
`--token <token>` (a platform token for this run only). `-V` prints the
version and `-h` the help, for any command.

## Chat

`webagents` opens a chat with the agent in the current folder, or the built-in
assistant where there is none: it works with the files in the folder and calls
web APIs. The chat has the same commands, keys and conversation files in both
SDKs. See [Chat](./repl.md).

## Where the SDKs Differ

The command is the same; a few things underneath are not.

- **Sandbox.** The Python SDK confines an agent's shell commands with the
  operating system (Seatbelt on macOS, bubblewrap on Linux) when the agent
  file declares `sandbox:`. The TypeScript SDK has no sandbox yet, and
  `webagents doctor` says so. See [Sandbox](./sandbox.md).
- **Shared context.** The Python loader merges `WEBAGENTS.md` context files
  into the agents below them. The TypeScript loader reads the agent file alone.
- **Skills.** The two SDKs ship different skill sets; `webagents skills list`
  names what each can load.

## Configuration

Settings are read in this order, first match wins:

1. command-line flags
2. environment variables, for the settings that have one (`ROBUTLER_API_URL`,
   `WEBAGENTS_PROFILE`, `WEBAGENTS_TOKEN`)
3. `./.webagents/config.json` in the current folder
4. `~/.webagents/config.json`
5. built-in defaults

Agent behavior itself lives in the `AGENT.md` front matter. See
[Configuration](./configuration.md) for the keys, and
[Publish](./deploy.md#which-platform) for how the platform address is chosen.
