---
title: Memory Skills
---
# Memory Skills

WebAgents provides two layers of memory: **core memory** for short-term conversation context, and **platform memory** for persistent storage with access control.

## Short-Term Memory (Core)

The `ShortTermMemorySkill` maintains conversation context within a session. It keeps a rolling window of recent messages and injects them into the LLM context automatically.

```python
from webagents import BaseAgent
from webagents.agents.skills.core.memory.short_term.skill import ShortTermMemorySkill

agent = BaseAgent(
    name="memory-agent",
    model="openai/gpt-4o",
    skills={
        "memory": ShortTermMemorySkill({"max_messages": 50}),
    },
)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | int | `50` | Maximum messages to retain in the rolling window |

Short-term memory is ephemeral — it exists only for the lifetime of the agent process. For persistent storage across sessions, use the platform Memory skill.

## Persistent Memory (Platform)

The [Memory Skill](../platform/memory.md) provides durable, UUID-based storage with access control, grants, full-text search, and encryption. It supports both portal-backed (PostgreSQL) and local (SQLite) backends.

Key capabilities:

- **Store-based model** — Data keyed by `(store_id, owner_id, namespace, key)`, with stores for agents, chats, and users
- **Access grants** — Share stores between agents at `search`, `read`, or `readwrite` levels
- **Full-text search** — PostgreSQL `tsvector` (portal) or FTS5 (local)
- **In-context vs not-in-context** — Control whether the LLM can see an entry
- **Encryption** — Client-side encrypted entries stored as opaque blobs

```python
from webagents.agents.skills.robutler.kv import MemorySkill

agent = BaseAgent(
    name="persistent-agent",
    model="openai/gpt-4o",
    skills={
        "memory": MemorySkill(agent_id="my-agent-uuid"),
    },
)
```

See [Memory Skill](../platform/memory.md) for the full reference — tool actions, access control cascade, store concepts, and configuration.

## Choosing a Memory Strategy

| Need | Skill | Backend |
|------|-------|---------|
| Conversation context within a session | `ShortTermMemorySkill` | In-memory |
| Persistent key-value across sessions | `MemorySkill` | Portal (PostgreSQL) |
| Persistent key-value, self-hosted | `LocalMemorySkill` | SQLite |
| Cross-agent shared memory | `MemorySkill` with grants | Portal |
| Skill-internal secrets (API keys, tokens) | `MemorySkill` with `in_context=false` | Portal or SQLite |

## Using Memory in Custom Skills

```python
from webagents import Skill, tool

class NoteSkill(Skill):
    @tool
    async def save_note(self, title: str, content: str) -> str:
        """Save a note to persistent memory."""
        memory = self.agent.skills["memory"]
        await memory.setInternal(self.agent.name, title, content)
        return f"Saved: {title}"

    @tool
    async def search_notes(self, query: str) -> str:
        """Search saved notes."""
        memory = self.agent.skills["memory"]
        results = await memory.search(query)
        return str(results)
```
