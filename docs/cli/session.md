---
title: Conversations
description: Where the chat keeps conversations, how to continue one, and how conversations differ from checkpoints.
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

## Conversations and Checkpoints

A conversation is the messages exchanged with an agent. A checkpoint is a
snapshot of the agent's working files, restorable later, and comes from the
`checkpoint` skill, not from the CLI. See [Checkpoint](../skills/local/checkpoint.md).
