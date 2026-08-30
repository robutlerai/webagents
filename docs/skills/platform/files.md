---
title: File Storage Skill
description: Store, retrieve, and manage files through the Robutler content API.
---

# File Storage Skill

Store, retrieve, and manage files through the Robutler content API.

> [!NOTE]
> The dedicated `RobutlerFilesSkill` is currently **Python-only**. TypeScript agents handle binary content via `StoreMediaSkill` (`webagents/skills/media`) for inline / generated media. Track parity at [internal/python-typescript-parity.md](../../internal/python-typescript-parity.md).

## Usage

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
// In TypeScript, use StoreMediaSkill from `webagents/skills/media` for
// resolving and persisting media content. For arbitrary file storage,
// POST multipart to the platform's `/api/content/upload` endpoint:
//
// import { BaseAgent } from 'webagents';
// const agent = new BaseAgent({ name: 'file-agent', model: 'openai/gpt-4o-mini' });
// // Then upload via fetch(`${base}/api/content/upload`, { method: 'POST', body: form })
```

```python tab="Python"
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.storage.files.skill import RobutlerFilesSkill

agent = BaseAgent(
    name="file-agent",
    model="openai/gpt-4o-mini",
    skills={
        "files": RobutlerFilesSkill(),
    },
)
```

## Tool Reference

### `store_file_from_url`

Download and store a file from a URL. Scope: `owner`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | str | Yes | — | URL to download |
| `filename` | str | No | auto-detected | Custom filename |
| `description` | str | No | — | File description |
| `tags` | list | No | — | Tags for the file |
| `visibility` | str | No | `private` | `public`, `private`, or `shared` |

Returns JSON with `id`, `filename`, `url`, `size`, `content_type`, `visibility`.

### `store_file_from_base64`

Store a file from base64 encoded data. Scope: `owner`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `filename` | str | Yes | — | File name |
| `base64_data` | str | Yes | — | Base64 encoded content |
| `content_type` | str | No | `application/octet-stream` | MIME type |
| `description` | str | No | — | File description |
| `tags` | list | No | — | Tags for the file |
| `visibility` | str | No | `private` | `public`, `private`, or `shared` |

### `list_files`

List accessible files. Scope: `all` (results filtered by ownership).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `scope` | str | No | all | `public`, `private`, or omit for all |

Pricing: 0.005 credits per call.

- **Owner** sees all files (public + private) or filtered by scope.
- **Non-owner** sees only public files.

## Configuration

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
```

```python tab="Python"
files_skill = RobutlerFilesSkill({
    "portal_url": "https://robutler.ai",
    "chat_base_url": "https://chat.robutler.ai",
    "api_key": "your-api-key",
})
```

Environment variables:

| Variable | Read as | Notes |
|----------|---------|-------|
| `ROBUTLER_INTERNAL_API_URL` | portal base URL (first) | in-cluster deployments |
| `ROBUTLER_API_URL` | portal base URL (second) | public API host |
| `ROBUTLER_CHAT_URL` | chat frontend base URL | public content links |
| `WEBAGENTS_API_KEY` | API key | falls back to the agent's own `api_key` |

There is no placeholder fallback. With no key configured anywhere, the skill
LOGS a warning at `initialize()` and each of its tools fails with that same
message when called — it never raises during initialization, because skills
initialize lazily on the agent's first run and raising there would take the
whole request down instead of just this skill. The minted per-agent key is an
RS256 JWT (from `POST /api/agents/{id}/api-key`), not a `rok_`-prefixed opaque
string.

### One principal for storing and listing

Uploads go to `POST /api/content/upload`, which files the row under the
BEARER'S SUBJECT. The listing therefore asks for that same principal
(`GET /api/agents/{principal}/content` — the `{id}` there is a principal, and
the route answers "content reachable by it"). Listing under the agent id while
uploading under the key's subject returns an empty list for every file the
agent ever stored: a per-agent key carries `agent_id` as a claim but keeps the
OWNER as its subject.

## File Naming

Uploaded files are automatically prefixed with the agent name to prevent conflicts: `image.jpg` becomes `my-agent_image.jpg`.
