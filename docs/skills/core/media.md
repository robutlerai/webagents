---
title: StoreMediaSkill
---
# StoreMediaSkill (Media Skill)

The StoreMediaSkill handles multi-modal content resolution, storage, and URL management across all LLM providers. It acts as the **portal content boundary**: tools produce raw UAMP content_items (base64 or temp CDN URLs), and this skill intercepts via hooks to upload them to `/content` and replace references with `/api/content/UUID` URLs.

## How It Works

StoreMediaSkill uses three UAMP lifecycle hooks:

### `before_llm_call` -- Content Resolution

Before an LLM call, StoreMediaSkill scans all messages for content URLs (e.g., `/api/content/{uuid}`). For each URL, it:

1. Reads the adapter's `mediaSupport` declaration for the target provider.
2. If the provider expects `base64`, resolves the URL to base64 data via the configured `MediaResolver`.
3. If the provider expects `url`, passes through (or converts to an HMAC-signed URL for access control).
4. Caches resolved content to avoid redundant fetches across turns.

Resolved content is stored on `context._resolved_images` for the adapter to use.

### `after_tool` -- Content Storage (Portal Boundary)

After a tool call returns a `StructuredToolResult` with `content_items`, StoreMediaSkill:

1. Detects non-`/content` media (base64 data URIs, external temp CDN URLs).
2. Downloads external URLs or extracts base64 data.
3. Uploads via the configured `MediaSaver` to get a `MediaSaverResult { url, content_id }`.
4. Replaces the content_item's media field with the new URL and sets `content_id`.
5. Returns structured `content_items` (no URLs appended to text — text purity rule).

This hook runs at priority 10, before the payment hook (priority 20).

### `after_llm_call` -- Generated Media Saving

After an LLM call, StoreMediaSkill checks `context._inline_images` (populated by adapters when they encounter inline media in the provider response, e.g., Gemini's `inlineData`). For each generated image:

1. Saves the base64 data to persistent storage via the configured `MediaSaver`.
2. Stores the resulting content URLs on `context._saved_media_urls`.

## Architecture

```
Tools/LLM skills produce raw UAMP content_items (base64 or temp CDN URLs)
                       |
                       v
         StoreMediaSkill (portal-specific)
         - after_tool: uploads base64/temp URLs to /content
         - Replaces content_items with /api/content/UUID + content_id
         - Exposes _media_saver for save_content built-in tool
                       |
                       v
         LLM sees content via structured content_items
         present() controls display; save_content() persists external content
         UI renders via content_items in response.done
```

Standalone (non-portal) agents: no StoreMediaSkill = content stays as base64 UAMP, everything still works.

## Configuration

StoreMediaSkill requires two injectable dependencies:

```typescript
import { StoreMediaSkill } from 'webagents/skills/media';

const mediaSkill = new StoreMediaSkill({
  resolver: myMediaResolver,  // implements MediaResolver
  saver: myMediaSaver,        // implements MediaSaver
});
```

### MediaResolver Interface

```typescript
interface MediaResolver {
  resolve(url: string, mode: 'base64' | 'url', userId?: string): Promise<ResolvedMedia | null>;
}
```

The resolver fetches content from a URL and returns it as base64 data or a signed URL. The optional `userId` parameter enables access control checks.

### MediaSaver Interface

```typescript
interface MediaSaver {
  save(base64: string, mimeType: string, meta?: { chatId?: string; agentId?: string; userId?: string }): Promise<string>;
}
```

The saver persists base64 media data and returns a `MediaSaverResult { url, content_id }` where the content can be accessed.

## Platform Integration

On the Robutler platform, StoreMediaSkill is automatically configured with:

- **`PortalMediaResolver`**: Resolves content URLs by reading from the platform's content storage. Enforces ownership checks (`canAccessContent(contentId, userId)`) to prevent unauthorized content access.
- **`PortalMediaSaver`**: Saves generated media to the platform's content storage system and returns a `/api/content/{uuid}` URL. Falls back to a generated UUID for `ownerId` when neither `userId` nor `agentId` are available (e.g. in anonymous or system contexts).

These are wired via `PortalStoreMediaFactory` in `lib/agents/factories.ts`.

## Content Reference Model

- All portal content eventually gets a `/api/content/UUID` URL.
- Content is referenced via structured `content_items` with `content_id` fields — not URLs in text (text purity rule).
- The delegate tool accepts content IDs or bare UUIDs in its `attachments` parameter.
- The `present` tool controls what content is displayed to the user; `save_content` persists external content.

## Cross-Turn Visual Context

The platform preserves media content across conversation turns. When an LLM generates or receives an image, that content remains available to subsequent LLM calls in the same chat:

1. **Chat history reconstruction** (`chatHistoryToOpenAIMessages`): When loading conversation history for a new LLM call, media `content_items` attached to *both* user and assistant messages are preserved and signed. This ensures the LLM can "see" images from prior turns.
2. **Content resolution** (`resolveContentMedia` in `uamp-proxy.ts`): Before sending messages to the LLM provider, `/api/content/UUID` references in `content_items` are resolved to base64 `inlineData` (for Gemini) or signed URLs (for other providers). This applies to all message roles including tool results.
3. **Tool result media**: When a tool call returns `content_items` (e.g., from `generate_image` or `delegate`), these are included as inline media parts in subsequent Gemini requests, allowing the LLM to reference generated images in its response.

This enables workflows like: "generate a unicorn with Flux, then delegate to nano-banana to make it green" — nano-banana receives the original image as visual context alongside the editing instruction.

## Provider-Aware Resolution

The key value of StoreMediaSkill is that it adapts content format to match each provider's requirements:

| Provider | Image Format | Audio Format | Video Format | Document Format |
|----------|-------------|--------------|--------------|-----------------|
| Google Gemini | base64 `inlineData` | base64 `inlineData` | base64 `inlineData` | base64 `inlineData` (PDF, text/*, JSON, XML, code) |
| OpenAI | URL `image_url` | base64 `input_audio` | placeholder | base64 `file` part (PDF, DOCX, XLSX, PPTX, text/*, JSON, code) |
| Anthropic | base64 `source.data` | placeholder | placeholder | base64 `document` block (PDF, DOCX, XLSX, PPTX, text/*, CSV, HTML, MD) |
| xAI / Fireworks | URL `image_url` | -- | -- | -- |

Without StoreMediaSkill, each LLM skill would need its own content resolution logic. With it, all content handling is centralized and consistent.

## Caching

StoreMediaSkill caches resolved content in-memory keyed by `(url, format)` pairs. This avoids redundant disk reads when the same image appears in multiple conversation turns. The cache is scoped to the skill instance lifetime.

## Security

- **Content IDOR Prevention**: `PortalMediaResolver` always verifies that the requesting user owns the content before resolving it. Content IDs that fail the ownership check return `null`.
- **No Raw URLs to Providers**: When a provider requires base64, the content URL is never sent to the external LLM provider -- only the resolved base64 data is transmitted.
