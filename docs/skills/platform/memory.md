---
title: Memory Skill
---
# Memory Skill

Persistent agent memory with UUID-based stores, grants, text search, and encryption.

## Overview

The Memory Skill replaces the legacy KV Skill with a richer storage model. Every piece of data is attached to a **store** (any UUID -- an agent, chat, user, or any entity). Access is controlled by a cascading check (`canAccessStore`), and agents can share stores with each other via **grants**.

### Key Features

- **Store-based model**: Data is keyed by `(store_id, owner_id, namespace, key)` -- agents can have memory about anything with a UUID
- **Access grants**: Share stores between agents at three levels: `search`, `read`, `readwrite`
- **Full-text search**: `tsvector`-powered search across one store or all accessible stores
- **In-context vs not-in-context**: Control whether the LLM can see an entry or only skills can
- **Encryption support**: Client-side encrypted entries are stored as opaque blobs and excluded from search
- **TTL**: Time-to-live for automatic expiry of entries and grants

## Usage

### Portal-backed (TypeScript)

```typescript
import { RobutlerMemorySkill } from 'webagents';

const skill = new RobutlerMemorySkill({
  portalUrl: 'https://robutler.ai',
  apiKey: process.env.PLATFORM_SERVICE_KEY,
  agentId: 'my-agent-uuid',
});
```

### Portal-backed (Python)

```python
from webagents.agents.skills.robutler.kv import MemorySkill

skill = MemorySkill(agent_id="my-agent-uuid")
```

### Local (SQLite-backed, TypeScript)

```typescript
import { LocalMemorySkill } from 'webagents';

const skill = new LocalMemorySkill({
  agentId: 'my-agent',
  storagePath: './.webagents/memory.db',
});
```

### Local (SQLite-backed, Python)

```python
from webagents.agents.skills.local.memory import LocalMemorySkill

skill = LocalMemorySkill(agent_id="my-agent", storage_path="./.webagents/memory.db")
```

## Tool Reference

The `memory` tool uses a **file-system metaphor** aligned with Anthropic's native `memory_20250818`. It exposes 9 commands. Paths are of the form `/memories/<key>` for the agent's own store, or `/memories/shared/<store_id>/<key>` for a granted store; a path ending in `/` refers to a directory listing.

### `view(path)`

Read a single entry, or — when `path` ends in `/` — list the entries in that store. Requires `read` access. Only returns `in_context=true` entries when listing.

### `create(path, content)`

Create or overwrite a memory entry at `path` with `content` (a string or JSON-serializable value). Requires `readwrite` access. Optional implicit TTL is configured per-skill (`defaultTtl`).

### `edit(path, old_str, new_str)`

In-place `str_replace` on a stored value: GET the current value, replace the first occurrence of `old_str` with `new_str`, PUT the result. Requires `readwrite` access.

### `delete(path)`

Remove an entry. Only the entry creator (`owner_id`) can delete it. Requires `readwrite` access.

### `rename(path, new_str)`

Move an entry from `path` to `new_str` — implemented as `view` + `create` + `delete`. Requires `readwrite` access on both source and destination stores.

### `search(query)`

Hybrid full-text + semantic search across all accessible stores. Results include `key`, `value`, and `storeId`. Portal: PostgreSQL `tsvector` + Milvus E5 vectors merged via Reciprocal Rank Fusion. Local: FTS5 only.

### `share(path?, agent, level?)`

Grant another `agent` access to a store. `level` is one of `search` (search-only — values returned, but `list`/`view` blocked, ideal for paid lookup), `read` (`view` + `search`), or `readwrite` (full). Defaults to `read`. If `path` is omitted, shares the agent's own store; otherwise the store id is extracted from `/memories/shared/<store_id>/...`. Requires `readwrite` access to the store being shared.

### `unshare(path?, agent)`

Revoke a previously granted access. Mirrors `share` for store resolution. Requires `readwrite` access.

### `stores()`

List all stores the agent can access: self store, granted stores, and contextual stores (chat, user).

### Authentication

The skill always authenticates as the **owner agent** — the JWT supplied via `apiKey` in the constructor — regardless of which user or chat triggered the call. Referring identities (chat id, user id, optional referring agent id) are forwarded only as **scope** query/body parameters and never as authentication credentials.

## Store Concept

A store is identified by a UUID. Common patterns:

| Store | UUID Source | Description |
|-------|-----------|-------------|
| Self | Agent's own UUID | Persistent memory across all conversations |
| Chat | Chat UUID | Conversation-scoped, shared with participants |
| User | User UUID | Per-user personalization |
| Shared | Any UUID | Granted via `share` action |

## Access Control

Every store access runs through `canAccessStore()`:

1. **Self store**: `store_id == agentId` → `readwrite`
2. **Explicit grant**: `memory_grants` row exists → level from grant
3. **Chat participant**: Agent is in `chatParticipants` for the chat UUID → `readwrite`
4. **Session user**: `store_id == session.userId` → `readwrite`
5. **Denied**: None of the above match → `403`

## In-Context vs Not-In-Context

- `in_context=true` (default): Entries visible to the LLM via the `memory` tool
- `in_context=false`: Entries only accessible via `getInternal`/`setInternal` -- for skill-internal data like API keys

## Internal API

For skill developers, direct access without going through the LLM:

```typescript
// Get any entry (including inContext=false)
const value = await memorySkill.getInternal(storeId, key);

// Set an entry not visible to LLM
await memorySkill.setInternal(storeId, key, value, {
  encrypted: true,
  ttl: 3600,
});
```

## Security

- `owner_id` is always set server-side (never LLM-controlled)
- UUIDs are not bearer tokens -- access is always verified
- Grants require `readwrite` access to create
- Encrypted entries are excluded from search indexing
- Chat/user contextual access is ephemeral (real-time participant check)

## Configuration

Memory is configured per-agent in the UI under **Settings > Memory**:

- **Master toggle**: Enable/disable memory for the agent
- **Self store**: Agent's own persistent memory
- **Chat store**: Per-conversation shared memory
- **User store**: Per-user personalization memory

## Migration from KV Skill

The old `kv_set`/`kv_get`/`kv_delete` tools are still available via the `RobutlerKVSkill` / `KVSkill` aliases (deprecated). To migrate:

1. Replace `KVSkill` with `MemorySkill` (Python) or `RobutlerMemorySkill` (TypeScript)
2. The new `memory` tool uses an `action` parameter instead of separate tool names
3. Add `store` parameter (defaults to agent's self store for backward compatibility)
