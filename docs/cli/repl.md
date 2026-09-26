---
title: Chat
description: Chatting with an agent in the terminal, the same in the TypeScript and Python CLIs. Commands, keys, conversations, files and models.
---

# Chat

`webagents` opens a chat with the agent in the current folder: its `AGENT.md`,
or the only `AGENT-<name>.md`. With no agent file, it opens the built-in
assistant, which can read and edit the files in the folder and call web APIs
(the `filesystem` and [`rest`](../skills/local/rest.md) skills). In the chat you
are the agent's owner, so owner-only tools are yours to use. The chat looks and
works the same in the TypeScript and Python
CLIs: the same commands in the same words, the same keys, and the same
conversation files, so a conversation started in one can be continued in the
other.

```bash
webagents                               # this folder's agent
webagents -a writer                     # AGENT-writer.md
webagents -m anthropic/claude-sonnet-4  # another model for this chat
webagents -p "Summarize README.md"      # one answer, then exit
```

## The Model

The agent runs on its own `model:`, with your key for that provider. Without
the key, and signed in with `webagents login`, it runs the same model through
Robutler, paid from your credits. An agent with no `model:` uses any provider
you have a key for, or Robutler's default model when you have none.

When there is neither a key nor a sign-in, the chat asks before the first
message: sign in in the browser, type a key (kept for next time, with input
hidden), or carry on without a model. `/login`, `/keys` and `/model` do the
same later, without leaving the chat.

## Commands

Type `/` for the menu: `↑` `↓` choose, `tab` completes, `enter` runs.

| Command | What it does |
|---------|--------------|
| `/help [command]` | Show the commands and keys |
| `/new` | Start a new conversation |
| `/clear` | Start a new conversation and clear the screen |
| `/resume [number]` | Continue an earlier conversation in this folder |
| `/undo` | Put back the files your last message changed |
| `/rewind [number]` | Put the folder back as it was before an earlier message |
| `/model [provider/model]` | Show or switch the model |
| `/agent [name]` | List this folder's agents, or switch to one |
| `/tools` | List what the agent can use |
| `/status` | Account, agent, model, sandbox and folder |
| `/login` | Sign in to Robutler |
| `/logout` | Sign out of Robutler |
| `/keys [set\|unset NAME]` | Model provider keys, and where each comes from |
| `/sandbox` | What the agent's commands are allowed to do |
| `/publish` | Publish this agent to Robutler, or update it |
| `/exit` | Leave the chat |

## Keys

| Key | Action |
|-----|--------|
| `enter` | Send |
| `alt+enter` | New line (or end a line with `\`) |
| `↑` `↓` | Earlier messages |
| `tab` | Complete a command |
| `esc` | Stop a reply; twice in the box, clear it |
| `ctrl+c` | Clear the box; twice, leave |

## Conversations

Every reply is saved as it arrives. `/resume` lists this folder's earlier
conversations with the agent, newest first, and `/resume 2` continues the
second one, showing its last few exchanges first. `/new` starts over; the old
conversation stays in the list.

Conversations are kept under your profile, in
`~/.webagents/sessions/<folder>/<agent>/`, never in the project, so nothing is
left in a folder you chat in. The files are readable only by you. With `session: {backend: robutler}` in the agent file they are also kept on
Robutler, and `/resume` lists both. `/undo` and `/rewind` take back what an
agent changed in the folder. See [Conversations](./session.md).

## Files

`@path/to/file` in a message includes that file's contents:

```
❯ Summarize @README.md
❯ Compare @src/old.py with @src/new.py
```

A word after `@` that is not a file in the folder is left as typed.

## The Sandbox

`/sandbox` says what the agent's shell commands are allowed to do, and
`/status` shows the same in one line. See [Sandbox](./sandbox.md) for the
presets and how to change them.

## Scripts and Pipes

When the input is not a terminal, the chat reads one line at a time and prints
plain text, so a script can drive it:

```bash
printf '/status\n/exit\n' | webagents
```

For one answer and nothing else, use `webagents -p "..."`, with
`--output-format json` or `stream-json` for machine-readable output.
