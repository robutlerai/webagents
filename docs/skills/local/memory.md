---
title: Caller-Scoped Memory
description: The memory skill an agent file names. Notes that last between conversations, kept apart for each verified caller, on this machine and on Robutler, with automatic compaction of long conversations.
---

# Caller-Scoped Memory

`- memory` in an agent file gives the agent notes that last between
conversations, kept apart for every caller it serves, the same in both SDKs.
An agent that serves many people and other agents never lets one caller's
notes into another caller's conversation.

```yaml
skills:
  - openai
  - memory                                  # notes on this machine
```

```yaml
skills:
  - openai
  - memory: {local: true, portal: true}     # and on Robutler too
```

| Setting | Default | What it does |
|---|---|---|
| `local` | `true` | Keep notes in the agent's folder |
| `portal` | `false` | Keep notes on Robutler as well, the durable and shared tier |
| `notes_budget` | `4000` | Characters of notes put into the prompt |
| `compaction` | `{threshold: 60000, keep: 12}` | Summarize a conversation that grows past `threshold` tokens, keeping the last `keep` messages as they are |

At least one of `local` and `portal` must be true. An unknown setting is
refused with the ones it takes.

## Whose Notes Are Whose

Every note belongs to one namespace, and the namespace comes only from who
the platform or the agent verified the caller to be:

| Caller | Namespace | Reads | Writes |
|---|---|---|---|
| The owner | `owner` | every namespace | `owner` and `shared` |
| A verified caller | `caller:<identity>`, from its first verified identity | its own and `shared` | its own |
| A channel sender | `caller:channel:<type>:<id>` | its own and `shared` | its own |
| Nobody verified | none | `shared` | nothing |

No argument the model passes can widen that. The owner files a note under
`shared` when every caller should read it. An identity that cannot name a
namespace (one with a comma in it, for example) gets none, and reads `shared`
alone.

## The Tools

| Tool | Does |
|---|---|
| `memory_search` | Search the notes and earlier-conversation summaries the caller may read |
| `memory_write` | Save or replace a note by key (a short slug such as `preferences`) |
| `memory_forget` | Remove a note |
| `memory_list` | List the notes the caller may read |

Each conversation also starts with a `## Memory` block: the caller's own notes
and the shared ones, within `notes_budget`. The block is fixed for the
conversation, so the prompt stays the same from turn to turn and the model
provider's prompt cache keeps working; a note written mid-conversation shows
up in the next one, or through `memory_search` at once.

## Compaction

When a conversation grows past `compaction.threshold` tokens, the older turns
become one summary, written by the agent's own model, and the last
`compaction.keep` messages stay as they are. A tool call is never separated
from its result. The summary is also saved as an `episode-...` entry in the
caller's namespace, so `memory_search` finds it later, but it is not added to
the notes block.

## Where Notes Live

**On this machine**, notes are Markdown files with front matter, under
`.webagents/memory/` in the agent's folder: `owner/`, `shared/`, and one folder
per caller under `callers/`. Files are readable only by you (0600, folders
0700), and you can edit them by hand. A note you write yourself, a plain `.md`
file with no front matter in `owner/` or `shared/`, is loaded too: its file
name is the key, and it gets front matter the next time the agent writes it.
A full-text index is rebuilt from the files each time the agent starts.

**On Robutler**, the same notes are kept in the platform's memory store, with
the same namespaces and semantic search, and synced with this machine by
entry. The agent needs its platform credential (a published agent, or
`WEBAGENTS_AGENT_TOKEN`). On Robutler the owner can search, edit, forget and
export the entries in the agent's settings, and set how long each namespace
keeps them.

## In the Chat

`/memory` shows where the notes are kept, how many are yours, shared and per
caller, and your newest keys. `/memory forget <key>` removes one of yours after
asking.

## Serving an Agent with Memory

Memory is only as separate as the identities the agent can verify. A served
agent with no `AuthSkill` and no `access:` block verifies no one, so its
callers read `shared` and write nothing; `webagents serve` says so when it
starts. Callers relayed by Robutler over Portal Connect arrive verified. See
[Who can call your agent](../../guides/trust.md).
