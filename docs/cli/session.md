---
title: Conversations
description: Where the chat keeps conversations, how to continue one, and how to undo what an agent changed.
---

# Conversations

Every chat reply is saved as it arrives, the same way in both SDKs, so a
conversation started in one CLI can be continued in the other.

```
/resume          # this folder's earlier conversations with this agent, newest first
/resume 2        # continue the second one
/new             # start over; the old one stays in the list
```

`/resume 2` shows the last few exchanges of that conversation, then carries on
from there.

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

When the agent file names `session: {backend: robutler}`, every conversation
is also kept on Robutler, as your chat with the agent there. You need to be
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
/rewind          # this folder's snapshots, newest first
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
