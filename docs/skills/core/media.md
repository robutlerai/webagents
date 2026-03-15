---
title: Media Skill
---
# Media Skill

The MediaSkill handles multi-modal content resolution and saving across all LLM providers. It ensures that images, audio, video, and documents are converted to the format each provider expects (base64 inline data or URL reference) before an LLM call, and that generated media (e.g., inline images from Gemini) is saved to persistent storage after the call.

## How It Works

MediaSkill uses two UAMP lifecycle hooks:

### `before_llm_call` — Content Resolution

Before an LLM call, MediaSkill scans all messages for content URLs (e.g., `/api/content/{uuid}`). For each URL, it:

1. Reads the adapter's `mediaSupport` declaration for the target provider.
2. If the provider expects `base64`, resolves the URL to base64 data via the configured `MediaResolver`.
3. If the provider expects `url`, passes through (or converts to an HMAC-signed URL for access control).
4. Caches resolved content to avoid redundant fetches across turns.

Resolved content is stored on `context._resolved_images` for the adapter to use.

### `after_llm_call` — Generated Media Saving

After an LLM call, MediaSkill checks `context._inline_images` (populated by adapters when they encounter inline media in the provider response, e.g., Gemini's `inlineData`). For each generated image:

1. Saves the base64 data to persistent storage via the configured `MediaSaver`.
2. Stores the resulting content URLs on `context._saved_media_urls`.

These URLs can then be included in the stored message for the user to view.

## Configuration

MediaSkill requires two injectable dependencies:

```typescript
import { MediaSkill } from 'webagents/skills/media';

const mediaSkill = new MediaSkill({
  resolver: myMediaResolver,  // implements MediaResolver
  saver: myMediaSaver,        // implements MediaSaver
});
```

### MediaResolver Interface

```typescript
interface MediaResolver {
  resolve(url: string, userId?: string): Promise<{ data: string; mimeType: string } | null>;
}
```

The resolver fetches content from a URL and returns it as base64 data. The optional `userId` parameter enables access control checks (e.g., verifying the user owns the content).

### MediaSaver Interface

```typescript
interface MediaSaver {
  save(data: string, mimeType: string, metadata?: Record<string, string>): Promise<string>;
}
```

The saver persists base64 media data and returns a URL where the content can be accessed.

## Platform Integration

On the Robutler platform, MediaSkill is automatically configured with:

- **`PortalMediaResolver`**: Resolves content URLs by reading from the platform's content storage. Enforces ownership checks (`canAccessContent(contentId, userId)`) to prevent unauthorized content access.
- **`PortalMediaSaver`**: Saves generated media to the platform's content storage system and returns a `/api/content/{uuid}` URL.

These are wired via `PortalMediaFactory` in `lib/agents/factories.ts`.

## Provider-Aware Resolution

The key value of MediaSkill is that it adapts content format to match each provider's requirements:

| Provider | Image Format | Audio Format | Document Format |
|----------|-------------|--------------|-----------------|
| Google Gemini | base64 `inlineData` | base64 `inlineData` | base64 `inlineData` |
| OpenAI | URL `image_url` | base64 `input_audio` | — |
| Anthropic | base64 `source.data` | — | base64 `source.data` |
| xAI / Fireworks | URL `image_url` | — | — |

Without MediaSkill, each LLM skill would need its own content resolution logic. With it, all content handling is centralized and consistent.

## Caching

MediaSkill caches resolved content in-memory keyed by `(url, format)` pairs. This avoids redundant disk reads when the same image appears in multiple conversation turns. The cache is scoped to the skill instance lifetime.

## Security

- **Content IDOR Prevention**: `PortalMediaResolver` always verifies that the requesting user owns the content before resolving it. Content IDs that fail the ownership check return `null`.
- **No Raw URLs to Providers**: When a provider requires base64, the content URL is never sent to the external LLM provider — only the resolved base64 data is transmitted.
