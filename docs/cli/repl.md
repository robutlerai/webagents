---
title: Chat
description: Chatting with an agent in the terminal, the same in the TypeScript and Python CLIs. Commands, keys, changing the agent from the chat, conversations, files, models and cost.
---

# Chat

`webagents` opens a chat with the agent in the current folder: its `AGENT.md`,
or the only `AGENT-<name>.md`. With no agent file, it opens the built-in
assistant, which can read and edit the files in the folder and call web APIs
(the `filesystem` and [`rest`](../skills/local/rest.md) skills), and says how
to make an agent of your own: `/agent new <name>`. In the chat you are the
agent's owner, so owner-only tools are yours to use. The chat looks and works
the same in the TypeScript and Python CLIs: the same commands in the same
words, the same keys, the same conversation files and the same history, so a
conversation started in one can be continued in the other.

```bash
webagents                               # this folder's agent
webagents -c                            # continue the last conversation here
webagents -r                            # list the earlier ones; -r 2 continues the second
webagents -a writer                     # AGENT-writer.md
webagents -m anthropic/claude-sonnet-4  # another model for this chat
webagents -p "Summarize README.md"      # one answer, then exit
```

## The Model

The agent runs on its own `model:`, with your key for that provider. Without
the key, and signed in with `webagents login`, it runs the same model through
Robutler, paid from your credits. An agent with no `model:` uses any provider
you have a key for, or Robutler's default model when you have none. When the
file lists `fallback_models`, the chat moves to the next one when a model does
not answer, and says so. See [Models](./models.md).

When there is neither a key nor a sign-in, the chat asks before the first
message: sign in in the browser, type a key (kept for next time, with input
hidden), or carry on without a model. `/login`, `/keys` and `/model` do the
same later, without leaving the chat.

## Commands

Type `/` for the menu: `↑` `↓` choose, `tab` completes, `enter` runs.
`/help <command>` shows a command's forms. The menu opens above the input
box, over the last lines of the conversation, and puts them back when it
closes. At the top of a cleared screen, where there is nothing above the box
to open over, it opens under the box. `/help` lists the commands in the five
groups below.

The menu searches as you type. It finds a command by its name or, from two
letters on, by a word of what it does: `/sign` finds `/login` and `/logout`.
After a command it lists what that command takes, and what you type narrows
the list by name or by the words in each row, several words at once and in
any order: `/resume budget sheet`, `/model 4.1`, `/help sign in`. A word
matches the start of a word, so `bud` finds "budget".

A command takes at most one word naming what to do (`/mcp add`, `/resume
delete`, `/keys remove`), then its arguments; there is never a second level.

### Conversation

| Command | What it does |
|---------|--------------|
| `/new` | Start a new conversation |
| `/resume [number \| delete <number>]` | Continue an earlier conversation, or delete one |
| `/compact [focus]` | Summarize the conversation so far, to make room |
| `/context` | How full the model's context is |
| `/undo` | Put back the files your last message or command changed |
| `/rewind [number]` | Put the folder back as it was before an earlier message |
| `/clear` | Start a new conversation and clear the screen |

### Agent

| Command | What it does |
|---------|--------------|
| `/agent [name]` | List this folder's agents, switch to one, or make one |
| `/agent new <name> [chatbot\|tool-agent]` | Make an agent in this folder from a template |
| `/agent edit [name]` | Open an agent's file in your editor, then use it |
| `/model [provider/model] [--save]` | Show or switch the model; `--save` keeps it in the agent file |
| `/skills [list\|add\|remove]` | This agent's skills, and adding or removing one |
| `/tools` | List what the agent can use, and who else may |
| `/mcp [list\|add\|remove]` | The MCP servers this agent uses; add or remove one |
| `/memory [forget <key>]` | What this agent remembers |
| `/cron [run <name>]` | This agent's schedules; run one now |
| `/reload` | Read the agent file again and use it |

### Limits

| Command | What it does |
|---------|--------------|
| `/access` | Who may call this agent, and what each caller gets |
| `/sandbox` | What the agent's commands are allowed to do |
| `/rounds [n] [--save]` | Show or set the tool rounds one turn may run; `--save` keeps it in the agent file as `max_tool_rounds` |

A turn may run 50 tool rounds unless `/rounds`, `--max-tool-rounds` or the agent file's `max_tool_rounds` says otherwise. At the limit the agent makes one last call with tools off and answers from what it gathered, and the chat asks `Used 50 tool rounds. Keep going? [Y/n]`: yes goes on with a fresh budget, no ends the turn. A turn that makes the same tool call three times in a row with the same arguments, and gets the same result each time, stops the same way, and the chat says which tool it repeated; any other call in between (an edit, another command) or a different result starts that count over, so an edit-and-rerun cycle is not a loop. `-p`, `serve`, the daemon and ACP never ask: they end with the answer and the reason (`tool_round_limit` or `tool_loop`).

### Account

| Command | What it does |
|---------|--------------|
| `/login` | Sign in to Robutler |
| `/logout` | Sign out of Robutler |
| `/keys [set\|remove NAME]` | Model provider keys, and where each comes from |
| `/secrets [set\|remove NAME]` | Secrets for MCP servers, and adding or removing one |
| `/publish [--dry-run]` | Publish this agent to Robutler, or update it |

### Chat

| Command | What it does |
|---------|--------------|
| `/status` | Account, agent, model, sandbox, folder and Robutler |
| `/help [command]` | Show the commands and keys |
| `/exit` | Leave the chat |

A command that fails says why and the chat goes on. A command given
arguments it does not take shows its usage and does nothing. A mistyped name
gets a "did you mean", and `/edit` points to `/agent edit`. `/keys unset NAME`
still works as `/keys remove NAME`.

## Changing the Agent from the Chat

`/agent new`, `/agent edit`, `/skills add` and `/skills remove`, `/model
--save` and `/publish` change a file or something on Robutler, so they follow
the same rules:

- They run only from lines you type at a terminal. When the chat reads from a
  pipe, they refuse and name the CLI command that does the same in a script.
- They change only the file the chat runs (or creates), in its folder. A file
  that is a symbolic link, lies outside the folder or belongs to another user
  is never written.
- They show the change and ask once: `Make this change? [y/N]`.
- A change is taken in a snapshot first, so `/undo` puts it back.
- Afterwards the chat reloads the agent, so the next message uses the new
  version.

```text
/agent new support-bot              # AGENT.md here, or AGENT-support-bot.md beside another agent
/agent new ops tool-agent           # from the tool-agent template
/agent edit                         # this agent's file in $VISUAL or $EDITOR
/skills add memory todo             # coded skills, by name
/skills add robutlerai/webagents --skill word-docx   # a SKILL.md skill from GitHub
/skills remove todo
/model openai/gpt-4.1 --save        # switch, and keep it in the file
/publish --dry-run                  # what publishing would send, sending nothing
```

`/agent new` writes into the chat's folder, never into your home folder, with
the provider the chat is running on, so the new agent answers at once.
`/skills add` with a source (`owner/repo`, a git URL or a folder) lists every
file it would install and asks; the chat never accepts `--yes`. `/model` only
offers models the agent's own model skill can run: an agent that names the
`openai` skill runs `openai/` models until you add another provider's skill.

When the file changes outside the chat, the chat notices before your next
message and says so once; `/reload` shows what changed and asks before it uses
a new `access:`, `sandbox:`, `cron:`, skill list or `mcp.json`. The chat never
reloads unasked.

`/publish` creates the agent on Robutler, or updates the one this folder is
linked to. Before it updates a linked agent it asks
`Update <agent> on <host> from <file>? [y/N]`; from a pipe it never updates.
`/publish --dry-run` shows the method, the route and the body.

## Looking at the Agent

- `/tools` lists every tool with who may use it: `only you`, `every caller`,
  or `you and <groups>`.
- `/access` reads the agent's `access:` block: who is refused, the groups and
  their members, what a caller in no group gets, and which tools each group
  may use. See [Who can call your agent](../guides/trust.md).
- `/mcp` lists the MCP servers, from the agent file or `mcp.json`, with each
  one's transport and tools, or why it did not connect (secrets masked), and
  how to serve this agent to an MCP client. `/mcp list` shows the servers
  Claude Desktop, Claude Code, Cursor, VS Code and Windsurf use, and `/mcp add
  <name>` copies one into this agent, its keys into your secrets; `/reload`
  connects it. `/mcp remove <name>` takes one out of the agent file or
  `mcp.json`, and `/reload` stops it; the secrets it read stay stored until
  `/secrets remove`. See [MCP](../skills/core/mcp.md#servers-other-apps-use).
- `/cron` shows the agent's schedules the way `webagents cron list` does, and
  `/cron run <name>` runs one now after telling you where it delivers.
- `/memory` shows where the agent keeps its notes, how many are yours, shared
  and per caller, and your newest keys; `/memory forget <key>` removes one of
  yours after asking.
- `/sandbox` says what the agent's commands are allowed to do, and warns when
  the sandbox is declared but cannot run here, in which case every command is
  refused. See [Sandbox](./sandbox.md).
- `/status` shows the account, profile, agent, model, sandbox and folder, the
  conversation's messages, tokens and cost, and whether the agent is published
  (`Published as <name>. /publish updates it.`).

## Keys

| Key | Action |
|-----|--------|
| `enter` | Send |
| `alt+enter` | New line (or end a line with `\`) |
| `↑` `↓` | Earlier messages typed in this folder |
| `→` or `tab` | Take the grey suggestion from history (`alt+f`: one word of it) |
| `tab` | Complete a command |
| `esc` | Stop a reply, and any command it is running; twice in the box, clear it |
| `ctrl+c` | During a reply, stop it as `esc` does; in the box, clear it; twice, leave |

Stopping a reply stops the command it was running too, with everything that
command started, and the chat says `Interrupted`. A command never reads
what you type into the chat: its input is empty.

After `/resume `, `/rewind `, `/agent `, `/model `, `/skills `, `/help `,
`/keys `, `/cron `, `/memory ` or `/mcp `, the menu stays open and offers
what the command takes: this folder's earlier conversations and snapshots,
its agents, the models the agent can switch to, the forms (`delete`, `new`,
`edit`, `add`, `remove`, `run`, `forget`, `list`), the skills an agent file
can name, provider key names, the agent's schedules, your notes, the MCP
servers other apps use and, after `/mcp remove`, this agent's own.

`enter` on a row that completes the command runs it: a conversation to
continue, a snapshot to go back to, a model or an agent to switch to, a
command to explain, a key to set, a schedule to run, a note to forget. None
of those changes a file or what is stored without asking you first. `enter`
on any other row puts it in the box and the line goes on: a form, a skill
(several may follow), and the rows of `/mcp add`, `/mcp remove` and `/keys
remove`, which act without asking, so a second `enter` sends them. `tab`
always puts the row in the box. `/resume` and `/rewind` chosen in the menu
open their list rather than printing it.

## History

The lines you type are kept in `history` under your profile folder
(`~/.webagents/history`, or `~/.webagents-<name>/history` under `--profile`),
readable only by you (folder 0700, file 0600), the last 1,000 of them. Both
CLIs read and write the same file, so `↑` finds a line typed in either.

Each line is kept with the folder it was typed in, and a chat offers only the
lines typed in its own folder: `↑` and the grey suggestion never bring back
another project's prompts. A line with no folder recorded is not offered. A
recalled line that starts with `/` does not open the command menu,
so `↑` and `↓` keep walking the history; edit the line or move the cursor and
the menu opens.

## Conversations

Every reply is saved as it arrives. A chat starts a new conversation; when
the last one here was used in the past day, the chat says so under the
banner (`Last conversation here 2h ago (14 messages): /resume 1 continues
it.`) and waits for you. `webagents -c` opens the chat already in it.

`/resume` opens this folder's earlier conversations with the agent in the
menu, newest first, each with when it was last used, how many messages it has
and how it began. Type to narrow the list by what was said (`/resume budget`),
`↑` `↓` to choose, `enter` to continue one; the chat shows its last few
exchanges first. `/resume 2` continues the second one directly, and
`webagents -r 2` does the same from the shell. `/resume delete` chooses one to
delete the same way, and asks first. `/new` starts over; the old conversation
stays in the list. Outside the chat, `webagents conversations` lists, deletes
and prunes them.

Conversations are kept under your profile, in
`~/.webagents/sessions/<folder>/<agent>/`, never in the project, so nothing is
left in a folder you chat in. The files are readable only by you. When the
agent file lists `- session: {backend: robutler}` under `skills:`, they are
also kept on Robutler, and `/resume` lists both; deleting one here leaves its
copy on Robutler. `/undo` and `/rewind` take back what an agent changed in the
folder. See [Conversations](./session.md).

## Context and Compaction

A conversation grows until it no longer fits the model's context. The chat
keeps it within bounds by compacting it: first the long outputs of old tool
calls are cleared (the call stays, with a one-line note of what it returned),
and when that is not enough, everything before the latest exchanges becomes
one summary made by the agent's own model. By default this happens before a
message would take the conversation past 80% of the model's context, and the
chat says so:

```text
Compacted the conversation: 38 earlier messages became a summary, the last 6 stay as they were. Context 31% full.
```

- `/compact` does it now: everything before the latest exchange becomes a
  summary. `/compact the budget decisions` tells the summary what to keep.
- `/context` says how full the context is, whether the conversation carries a
  summary, and when it compacts on its own.
- The footer shows `context 64%` once the conversation passes half.
- The file keeps the whole conversation as it happened (`transcript`), so a
  compaction never loses what was said; the model is sent the compacted one.
  `/resume` continues the compacted one.
- An agent with the `memory` skill keeps each summary as an episode in its
  memory, so what was compacted can still be searched.

The agent file's `compaction:` block changes the defaults, and `auto: false`
turns the automatic part off. See
[Context compaction](./configuration.md#context-compaction).

## Cost

After each reply the footer shows the conversation's tokens and, when it is
known, what they cost in credits: `2.5k tokens, ~0.0006 credits`. On
Robutler's models the chat shows the cost the platform reports with a reply,
and tokens alone when it reports none. With your own provider key it is an
estimate from the provider's list price, which is why it
carries a `~` (cache reads and long-context tiers are not counted). A model
the price table does not know, and a local Ollama model, show tokens alone.
`/status` shows the conversation's totals, and the line you see on leaving
shows this chat's.

## Files

`@path/to/file` in a message includes that file's contents:

```
❯ Summarize @README.md
❯ Compare @src/old.py with @src/new.py
```

A word after `@` that is not a file in the folder is left as typed.

## Scripts and Pipes

When the input is not a terminal, the chat reads one line at a time and prints
plain text, so a script can drive it:

```bash
printf '/status\n/exit\n' | webagents
```

Every question answers no from a pipe, and the commands that change a file
refuse. For one answer and nothing else, use `webagents -p "..."`, with
`--output-format json` or `stream-json` for machine-readable output.
