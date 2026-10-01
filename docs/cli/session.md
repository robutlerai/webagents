---
title: Conversations
description: Where the chat keeps conversations, how to continue, delete and prune them, and how to undo what an agent changed.
---

# Conversations

Every chat reply is saved as it arrives, the same way in both SDKs, so a
conversation started in one CLI can be continued in the other.

A chat starts a new conversation. When the last one in this folder was used in
the past day, the chat says so under the banner and waits:

```text
Last conversation here 2h ago (14 messages): /resume 1 continues it.
```

## Continuing One

```bash
webagents -c         # open the chat in the last conversation here
webagents -r         # open the chat with the list of earlier ones
webagents -r 2       # open the chat in the second one
```

In the chat:

```text
/resume              # this folder's earlier conversations with this agent, newest first
/resume budget       # the ones whose words start "budget", to choose from
/resume 2            # continue the second one
/new                 # start over; the old one stays in the list
```

`/resume` lists the conversations in the menu: type to narrow them by what
was said, several words at once, then `↑` `↓` and `enter` to continue one.
The chat shows the last few exchanges of that conversation, then carries on
from there. `-c` and `-r` open the chat, so they do not go with `-p`.

## Deleting and Pruning

```text
/resume delete 2     # in the chat: delete the second one, after asking
```

```bash
webagents conversations list                     # this folder's, newest first
webagents conversations list --all               # every folder's
webagents conversations delete 5b1f2c9a          # by the start of its id, after asking
webagents conversations prune --older-than 30d   # last used more than 30 days ago
webagents conversations prune --older-than 2w --dry-run
```

`list` shows each conversation's id (its first 8 characters are enough),
when it was last used, how many messages it has and how it began. `delete`
and `prune` ask first; from a script they need `--yes`, and `--json` gives a
machine-readable answer. `--older-than` takes a number and a unit: `90m`,
`12h`, `30d` or `2w`. A conversation that is also on Robutler keeps its copy
there; delete that one on Robutler.

## Long Conversations

When a conversation grows past most of the model's context, the chat compacts
it (see [Context and Compaction](./repl.md#context-and-compaction)). The file
keeps both: `messages`, the conversation the model is sent, with its earlier
part as a summary, and `transcript`, the whole conversation as it happened.
`/resume` continues from `messages`; the lists, `/status` and the start hint
count and preview it from `transcript`, so a compacted conversation keeps its
size and its first line there.

## Where They Live

Under your profile, never in the project:

```
~/.webagents/sessions/<folder>/<agent>/<id>.json
```

`<folder>` is the project folder's full path with every character outside
`A-Z a-z 0-9 . _ -` replaced by `-`. The files are readable only by you
(mode 0600). Under `--profile <name>` they move to `~/.webagents-<name>/`.

Nothing is written into a folder you chat in, so a conversation cannot end up
committed to a repository by accident.

## On Robutler

When the agent file lists `- session: {backend: robutler}` under `skills:`,
every conversation is also kept on Robutler, as your chat with the agent there. You need to be
signed in (`webagents login`) and the agent published (`webagents publish`).
`/resume` then lists this machine's conversations and Robutler's together;
one that is only on Robutler, started on the web or on another machine, is
marked `Robutler:` and continues here like any other. `/status` says when the
current one is also on Robutler. See [Session](../skills/local/session.md).

## Undo

When the agent can change files (it has `filesystem` or `shell`), the chat
takes a snapshot of its folder before each message you send. `/undo` puts back
what your last message changed: files it edited or deleted come back, and
files it made are removed. It shows the list and asks before touching
anything.

```
/undo            # put back what the last message changed; again for the one before
/rewind          # this folder's snapshots, newest first, to choose from
/rewind 3        # put the folder back as the third one has it
```

A restore takes a snapshot of how things were first, so `/rewind` can take
it back too. Snapshots cover every file in the folder except `.git`,
`.webagents`, `node_modules`, `.venv`, `venv` and `__pycache__`; a file over
10 MB is left out, and a restore leaves it alone. Symlinks are kept as links
and never followed. The last 50 are kept, under your profile and never in the
folder:

```
~/.webagents/checkpoints/<folder>/
```

Undo is off when the chat runs in your home folder, or a folder above it:
start the chat in a project folder instead. Both CLIs read and write the same
snapshots.
