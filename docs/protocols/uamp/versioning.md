# UAMP Protocol Versioning

## Current Version

**UAMP 1.0**

## Version Negotiation

Version is exchanged during session creation:

```typescript
// Client sends
{ type: 'session.create', uamp_version: '1.0', session: { ... } }

// Server responds
{ type: 'session.created', uamp_version: '1.0', session: { ... } }
```

### Negotiation Flow

1. Client sends `session.create` with `uamp_version` it supports
2. Server responds with `session.created` with its `uamp_version`
3. If versions are incompatible, server sends `error` event:

```typescript
{
  type: 'response.error',
  error: {
    code: 'version_mismatch',
    message: 'Server requires UAMP 1.0, client sent 2.0',
    details: {
      server_version: '1.0',
      client_version: '2.0',
      supported_versions: ['1.0', '1.1']
    }
  }
}
```

## Versioning Strategy

### Major Version (1.x → 2.x)

Breaking changes that require code updates:

- Removing required fields
- Changing field types
- Changing event semantics
- Removing event types

### Minor Version (1.0 → 1.1)

Additive, backward-compatible changes:

- New optional fields
- New event types
- New enum values for extensible types

### Extensions Field

The `extensions` field allows experimentation without version bumps:

```typescript
session: {
  modalities: ['text'],
  extensions: {
    experimental_feature: { enabled: true }
  }
}
```

## Compatibility Rules

1. **Servers** should support multiple minor versions within a major version
2. **Clients** should ignore unknown fields (forward compatibility)
3. **Unknown event types** should be logged but not cause errors
4. **Required fields** are only required at their minimum version

## Version History

### UAMP 1.0 (Current)

Initial release with:

- Text, audio, image, video, file modalities
- Session management
- Tool calling
- Streaming responses
- Usage tracking
- Progress events
- Thinking events (extended reasoning)

### Known Limitations (UAMP 1.0)

- ~33% overhead for audio due to base64 (acceptable for most use cases)
- No automatic session recovery (client must track and replay)
- Service Workers cannot maintain WebSocket (use HTTP+SSE fallback)

### Future Considerations (v2+)

- Binary WebSocket frames for audio-heavy apps
- Server-side session checkpointing with resume tokens
- WebTransport support for better multiplexing
- Compression for large payloads
