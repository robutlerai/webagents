# Cross-Language Interoperability

## Protocol Compatibility

The WebAgents framework ensures full interoperability between Python and TypeScript agents through standardized protocols:

### UAMP (Universal Agentic Message Protocol)

Both implementations share the same 45+ event types:

- **Session**: `session.create`, `session.created`, `session.update`, `session.updated`, `session.end`, `session.error`
- **Input**: `input.text`, `input.audio`, `input.image`, `input.video`, `input.file`, `input.typing`, `input.audio_committed`
- **Response**: `response.create`, `response.cancel`, `response.created`, `response.delta`, `response.done`, `response.error`, `response.cancelled`
- **Tools**: `tool.call`, `tool.result`, `tool.call_done`
- **Audio**: `audio.delta`, `audio.done`, `transcript.delta`, `transcript.done`
- **Payment**: `payment.required`, `payment.submit`, `payment.accepted`, `payment.balance`, `payment.error`
- **Conversation**: `conversation.item.create`, `conversation.item.delete`, `conversation.item.truncate`
- **Utility**: `ping`, `pong`, `rate_limit`, `progress`, `thinking`, `usage.delta`, `presence.typing`, `capabilities`, `capabilities.query`, `client.capabilities`

### Completions API

Both expose OpenAI-compatible `/chat/completions`:

```bash
curl -X POST https://agent.example.com/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

### A2A Protocol

Both support Google's Agent-to-Agent protocol:
- `/.well-known/agent.json` — Agent card
- `/a2a` — JSON-RPC endpoint (`tasks/send`, `tasks/get`, `tasks/cancel`)

### NLI (Natural Language Interface)

Agents communicate via `@name` resolution:

```typescript
// TypeScript calling Python agent
const response = await nli.nli({
  agent_url: '@python-analyzer',
  message: 'Analyze this data',
});
```

```python
# Python calling TypeScript agent
response = await nli.nli("@typescript-processor", "Process this text")
```

## Payment Token Exchange

JWT payment tokens work across runtimes:

1. User creates token via portal API (either runtime)
2. Token is passed as `X-Payment-Token` header
3. Receiving agent verifies using shared JWKS endpoint
4. Lock/settle operations work via the portal's payment API

## Testing Interoperability

Run the cross-language test suite:

```bash
# Start Python agent
cd python && webagentsd serve --port 8000

# Start TypeScript agent
cd typescript && webagents serve --port 3000

# Run interop tests
PYTHON_AGENT_URL=http://localhost:8000 TS_AGENT_URL=http://localhost:3000 \
  npx vitest run tests/interop/
```
