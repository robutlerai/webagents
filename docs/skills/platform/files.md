---
title: File Storage Skill
---
# File Storage Skill

Store, retrieve, and manage files through the Robutler content API.

## Usage

```python
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

- **Owner** sees all files (public + private) or filtered by scope
- **Non-owner** sees only public files

## Configuration

```python
files_skill = RobutlerFilesSkill({
    "portal_url": "https://robutler.ai",
    "chat_base_url": "https://chat.robutler.ai",
    "api_key": "your-api-key",
})
```

Environment variables: `ROBUTLER_API_URL`, `ROBUTLER_CHAT_URL`, `WEBAGENTS_API_KEY`.

## File Naming

Uploaded files are automatically prefixed with the agent name to prevent conflicts: `image.jpg` becomes `my-agent_image.jpg`.
