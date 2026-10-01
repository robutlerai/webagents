---
title: Caller-Scoped Memory
description: The memory skill an agent file names. Notes that last between conversations, kept apart for each verified caller, on this machine and on Robutler, with an index of them in the prompt.
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
| `notes_budget` | `4000` | Characters of the index put into the prompt |
| `compaction` | none | The older way to say when a conversation is compacted (`threshold` tokens); the agent file's own [`compaction:`](../../cli/configuration.md#context-compaction) block replaces it |

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
| `memory_read` | Read one note in full: its content, description and when it changed |
| `memory_write` | Save or replace a note by key (a short slug such as `preferences`), with a one-line description |
| `memory_forget` | Remove a note |
| `memory_list` | List the notes the caller may read, with their descriptions |

## The Index in the Prompt

Each conversation starts with a `## Memory` block that is an index of the
notes, as Claude Code's `MEMORY.md` is: one line per note the caller may read,
its key and its description, newest first, the caller's own notes and the
shared ones.

```text
## Memory
One line per note you keep, newest first. memory_read gives a note in full; memory_write keeps one, with a one-line description.
Your notes (owner):
- launch: When and where the launch is
- tone: Tone
(12 more notes; memory_list shows them all.)
```

A note written without a description shows its first line instead (without a
heading's or a list's mark), cut to 120 characters. The lines fit within
`notes_budget`; the notes that do not are counted, and `memory_list` shows
them all. The model reads a note in full with `memory_read` when it needs it,
so an agent can keep many notes without carrying them all in every prompt.

The block is fixed for the conversation, so the prompt stays the same from
turn to turn and the model provider's prompt cache keeps working; a note
written mid-conversation shows up in the next one, or through `memory_search`
and `memory_read` at once.

## Compaction

Compacting a long conversation is the agent's, with or without this skill (see
[Context compaction](../../cli/configuration.md#context-compaction)). This
skill keeps each summary as an `episode-...` entry in the caller's namespace,
so `memory_search` finds it later; episodes are not listed in the index.

## Where Notes Live

**On this machine**, notes are Markdown files with front matter, under
`.webagents/memory/` in the agent's folder: `owner/`, `shared/`, and one folder
per caller under `callers/`. Files are readable only by you (0600, folders
0700), and you can edit them by hand. A note's description is the `description:` line of its front matter. A note you write yourself, a plain `.md`
file with no front matter in `owner/` or `shared/`, is loaded too: its file
name is the key, and it gets front matter the next time the agent writes it.
A full-text index is rebuilt from the files each time the agent starts.

**On Robutler**, the same notes are kept in the platform's memory store, with
the same namespaces, descriptions and semantic search, and synced with this
machine by entry. The agent needs its platform credential (a published agent, or
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
