---
title: WebAgents CLI
description: The webagents command, the same in the TypeScript and Python SDKs - chat, one-shot prompts, serving, the local daemon, signing in and publishing.
---

# WebAgents CLI

Both SDKs install a `webagents` command, and it is the same command: the same
subcommands, arguments, flags and messages, reading the same agent file
(`AGENT.md`) and the same settings, keys and sign-in. A script or a habit
written against one works against the other.

Start with the [Quickstart](../quickstart.md).

## Installation

```bash tab="TypeScript"
npm install -g webagents
```

```bash tab="Python"
pip install webagents
```

```bash tab="Homebrew"
brew install robutlerai/tap/webagents
```

Every install brings two commands: `webagents`, and `robutler`, the chat with
the assistant that comes with WebAgents. With Homebrew,
`brew install robutlerai/tap/robutler` installs the same two commands under
the other name; install one of the two, not both.

## Everyday Commands

```bash
webagents init my-agent              # a project folder with AGENT.md
webagents                            # chat with this folder's agent
webagents -c                         # continue the last conversation here
webagents -p "Summarize this"        # one prompt, answer on stdout
webagents doctor                     # what stands between this folder and a running agent
webagents serve                      # this agent over HTTP, on port 3000
webagents publish                    # send it to Robutler
```

## Command Tree

`webagents --help` shows the commands in these groups:

```
webagents                 Chat with this folder's agent (the default command, also `chat`)
                          -c continues the last conversation, -r [number] an earlier one

Build
├── init [name]           A folder with a new agent in it (--template chatbot|tool-agent, --list)
├── skills                list, add, remove: coded skills by name, SKILL.md skills from git or a folder
├── mcp                   list, add, remove: MCP servers the agent uses, from other apps;
│                         serve [path]: the agent as an MCP server (--http <port>, --host)
├── cron                  list, run: the schedules the agents in a folder declare (--watch)
└── doctor                Check this setup and say what to fix (-a <agent>)

Run
├── serve [path]          One agent over HTTP (--port, --host)
├── daemon                Every agent in a folder, with schedules, reloaded as files change
│                         (--port, --host, --watch, --no-cron)
└── acp [path]            The agent to a code editor, over the Agent Client Protocol on stdio

Robutler
├── login, logout         Your Robutler account (--url, --token)
├── whoami                Who you are signed in as
├── publish [path]        Publish the agent to Robutler, or update it (--yes, --dry-run)
├── link [name]           Link this folder to one of your agents (--show)
├── unlink                Forget that link
└── budget <token_id>     A run's budget tree, from its payment token

This machine
├── conversations         list, delete, prune: the conversations the chat keeps
├── secrets               list, set, remove, get: keys and secrets kept on this machine
├── models                Model providers, and which are ready here
├── sandbox setup         Whether the sandbox runs here, and what it lacks
├── config                get, set, unset, validate, path
└── help [command]
```

`connect` (the old name of `chat`) and `templates list` (now `init --list`)
still work and are not listed.

Global flags go before the command: `--json` (one JSON document on standard
output), `--profile <name>` (separate settings, keys and sign-in),
`--max-tool-rounds <n>`, `--token <token>` (a platform token for this run
only) and `--no-sandbox`. `-V` prints the version and `-h` the help, for any
command.

## Chat

`webagents` opens a chat with the agent in the current folder, or the built-in
assistant where there is none: it works with the files in the folder and calls
web APIs. The chat has the same commands, keys and conversation files in both
SDKs, and it can make and change an agent as you talk to it: `/agent new`,
`/agent edit`, `/skills add`, `/model --save`. Long conversations are
compacted to fit the model's context, and `/resume` continues an earlier one.
See [Chat](./repl.md).

## Where the SDKs Differ

The command is the same; a few things underneath are not.

- **Shared context.** The Python loader merges `WEBAGENTS.md` context files
  into the agents below them (see
  [Configuration](./configuration.md#shared-context-with-webagentsmd-python)).
  The TypeScript loader reads the agent file alone.
- **Skills.** The two SDKs ship different coded skills; `webagents skills list`
  names what each can load. SKILL.md skills load and run the same way in both.
- **The shell tool's name.** The `shell` skill's tool is `run_command` in
  Python and `runCommand` in TypeScript. An `access:` rule that names the skill
  (`shell`) covers it in both.
- **srt.** Both packages bring the sandbox runtime: TypeScript as a
  dependency, Python inside the package, with node from your PATH or from
  the `nodejs-wheel-binaries` package pip installs with it (see
  [Sandbox](./sandbox.md#where-the-engine-comes-from)).
- **The keychain.** Each CLI keeps its own sign-in and keys in the system
  keychain, so sign in, and store each key, once in each CLI you use. The
  owner-only file used where there is no keychain is shared (see
  [Keychain dialogs on macOS](./keychain.md)).

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
