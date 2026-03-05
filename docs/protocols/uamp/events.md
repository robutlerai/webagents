# Events

All UAMP communication is event-based. Events flow bidirectionally between clients and servers.

## Event Categories

### Client → Server Events

| Event Type | Description |
|------------|-------------|
| `session.create` | Create new session |
| `session.end` | End a session |
| `session.update` | Update session configuration |
| `capabilities.query` | Query server capabilities |
| `client.capabilities` | Announce client capabilities |
| `input.text` | Text input |
| `input.audio` | Audio input |
| `input.image` | Image input |
| `input.video` | Video input |
| `input.file` | File input |
| `tool.result` | Tool execution result |
| `response.create` | Request response generation |
| `response.cancel` | Cancel in-progress response |
| `input.typing` | User typing indicator |
| `voice.invite` | Initiate voice session |
| `voice.accept` | Accept voice session |
| `voice.decline` | Decline voice session |
| `voice.end` | End voice session (bidirectional) |
| `payment.submit` | Submit payment token/proof |
| `ping` | Connection keepalive |

### Server → Client Events

| Event Type | Description |
|------------|-------------|
| `session.created` | Session creation confirmed |
| `session.end` | End a session (bidirectional) |
| `session.error` | Session-level error |
| `session.updated` | Session update confirmed |
| `capabilities` | Server capabilities |
| `response.created` | Response started |
| `response.delta` | Streaming content |
| `response.done` | Response complete |
| `response.error` | Error occurred |
| `tool.call` | Tool execution request |
| `tool.call_done` | Tool call completed |
| `audio.delta` | Streaming audio output |
| `transcript.delta` | Real-time transcription |
| `usage.delta` | Usage statistics update |
| `progress` | Progress update |
| `thinking` | Reasoning content |
| `rate_limit` | Rate limit notification |
| `presence.typing` | Typing indicator from another user |
| `payment.required` | Payment required to continue |
| `payment.accepted` | Payment accepted |
| `payment.balance` | Balance update notification |
| `payment.error` | Payment error |
| `pong` | Keepalive response |
| `message.created` | New chat message (chat UI use cases) |
| `message.read` | Read receipt |
| `presence.online` | User/agent came online |
| `presence.offline` | User/agent went offline |

## Base Event Structure

All events share a common base structure:

```typescript
interface BaseEvent {
  type: string;              // Event type identifier
  event_id: string;          // Unique event ID (UUID)
  timestamp?: number;        // Unix timestamp in milliseconds
  session_id?: string;       // Session scope (required for multiplexed connections)
}
```

## Session Events

### SessionCreateEvent

Sent by client to create a new session. Can include `client_capabilities` to inform the agent what the client can render/handle.

```typescript
interface SessionCreateEvent extends BaseEvent {
  type: 'session.create';
  uamp_version: string;       // Required: '1.0', '1.1', etc.
  session: {
    id?: string;              // Client-suggested ID
    modalities: Modality[];   // ['text', 'audio', 'image']
    instructions?: string;    // System instructions
    tools?: ToolDefinition[];
    voice?: VoiceConfig;
    input_audio_format?: AudioFormat;
    output_audio_format?: AudioFormat;
    turn_detection?: TurnDetectionConfig;
    response_format?: ResponseFormat;  // Structured output format
    extensions?: {            // Provider-specific extensions
      openai?: { model?: string; temperature?: number; };
      anthropic?: { thinking?: boolean; };
      google?: { safety_settings?: any; };
      [provider: string]: any;
    };
  };
  
  // Multiplexed session fields (used when one connection hosts multiple sessions)
  agent?: string;             // Agent name/ID this session targets
  chat?: string;              // Chat ID when session is chat-scoped
  token?: string;             // Per-session auth token (e.g. AOAuth JWT)
  payment_token?: string;     // Per-session payment token

  // Optional: Inform agent what client can handle
  client_capabilities?: ClientCapabilities;
}
```

### SessionCreatedEvent

Sent by server to confirm session creation.

```typescript
interface SessionCreatedEvent extends BaseEvent {
  type: 'session.created';
  uamp_version: string;       // Server's supported version
  session: Session;           // Full session object
  session_id?: string;        // Optional: for multiplexed sessions
  chat?: string;              // Optional: chat ID when session is chat-scoped
  agent?: string;             // Optional: agent name when session is agent-scoped
}
```

### SessionUpdateEvent (token refresh)

Client updates auth or payment context for a session without reconnecting. Used for token refresh (e.g. AOAuth JWT, Robutler JWT, X402 payment token).

```typescript
interface SessionUpdateEvent extends BaseEvent {
  type: 'session.update';
  session_id?: string;        // Omit for connection-level update (e.g. browser JWT refresh)
  token?: string;             // New auth token for this session
  payment_token?: string;     // New payment token for this session
}
```

### SessionEndEvent

Sent by either side to end a session.

```typescript
interface SessionEndEvent extends BaseEvent {
  type: 'session.end';
  reason?: string;            // 'user_left', 'timeout', 'daemon_takeover', etc.
}
```

### SessionErrorEvent

Sent by server when a session-level error occurs.

```typescript
interface SessionErrorEvent extends BaseEvent {
  type: 'session.error';
  error: {
    code: string;             // 'agent_offline', 'rate_limited', 'unauthorized', 'timeout'
    message: string;
    details?: any;
  };
}
```

### SessionUpdatedEvent

Server confirms the session was updated.

```typescript
interface SessionUpdatedEvent extends BaseEvent {
  type: 'session.updated';
  session_id?: string;
}
```

### CapabilitiesQueryEvent

Client requests capability information. Optional - servers may also send capabilities automatically after session creation.

```typescript
interface CapabilitiesQueryEvent extends BaseEvent {
  type: 'capabilities.query';
  model?: string;             // Query for specific model, or current if omitted
}
```

### ClientCapabilitiesEvent

Client announces its capabilities using the unified `Capabilities` format (same as model/agent).

**When sent:**
- During `session.create` (via `client_capabilities` field)
- Anytime during session when capabilities change

```typescript
interface ClientCapabilitiesEvent extends BaseEvent {
  type: 'client.capabilities';
  capabilities: Capabilities;  // Same format as model capabilities
}
```

See [Types](types.md#unified-capabilities) for the full `Capabilities` definition.

**Example:**

```typescript
// Client announces capabilities (same format as model!)
→ {
    type: 'client.capabilities',
    capabilities: {
      id: 'web-app',
      provider: 'robutler',
      modalities: ['text', 'image', 'audio'],
      supports_streaming: true,
      widgets: ['chart', 'table', 'form'],
      extensions: { supports_html: true, platform: 'web' }
    }
  }
```

### CapabilitiesEvent

Server announces model/agent capabilities. Enables clients to adapt their UI (show/hide image upload, audio button, etc.).

**When sent:**
- After `session.created` (recommended)
- In response to `capabilities.query`
- When model/backend changes mid-session

```typescript
interface CapabilitiesEvent extends BaseEvent {
  type: 'capabilities';
  capabilities: ModelCapabilities;
}
```

See [Types](types.md#model-capabilities) for the full `ModelCapabilities` definition.

**Example flow:**

```typescript
// Client creates session
→ { type: 'session.create', session: { modalities: ['text', 'image'] } }

// Server confirms and announces capabilities
← { type: 'session.created', session: { id: 'sess_123' } }
← { 
    type: 'capabilities', 
    capabilities: {
      model_id: 'gpt-4o',
      modalities: ['text', 'image'],
      image: { formats: ['jpeg', 'png'], detail_levels: ['auto', 'low', 'high'] },
      file: { supports_pdf: true },
      tools: { supports_tools: true, built_in_tools: ['web_search'] }
    }
  }

// Client can now show image upload button, PDF support indicator, etc.
```

## Input Events

### InputTextEvent

```typescript
interface InputTextEvent extends BaseEvent {
  type: 'input.text';
  text: string;
  role?: 'user' | 'system' | 'assistant';
  
  // Stateless conversation context (platform sends full history per request)
  messages?: Message[];       // Full conversation history for LLM context
  payment_token?: string;     // Payment token for this interaction
  context?: {                 // Routing and broadcast metadata
    chat_id?: string;         // Chat this message belongs to
    sender_id?: string;       // User who sent the message
    [key: string]: any;       // Extensible
  };
}
```

See [Types](types.md#message) for the `Message` definition.

### InputAudioEvent

```typescript
interface InputAudioEvent extends BaseEvent {
  type: 'input.audio';
  audio: string;              // Base64 encoded
  format: AudioFormat;
  is_final?: boolean;         // End of audio stream
}
```

### InputImageEvent

```typescript
interface InputImageEvent extends BaseEvent {
  type: 'input.image';
  image: string | { url: string };  // Base64 or URL
  format?: 'jpeg' | 'png' | 'webp' | 'gif';
  detail?: 'low' | 'high' | 'auto';
}
```

### InputVideoEvent

```typescript
interface InputVideoEvent extends BaseEvent {
  type: 'input.video';
  video: string | { url: string };  // Base64 or URL
  format?: 'mp4' | 'webm';
}
```

### InputFileEvent

```typescript
interface InputFileEvent extends BaseEvent {
  type: 'input.file';
  file: string | { url: string };   // Base64 or URL
  filename: string;
  mime_type: string;
}
```

## Response Events

### ResponseCreateEvent

Request the agent to generate a response.

```typescript
interface ResponseCreateEvent extends BaseEvent {
  type: 'response.create';
  response?: {
    modalities?: Modality[];  // Override session modalities
    instructions?: string;    // Override instructions for this response
    tools?: ToolDefinition[];
  };
  response_format?: ResponseFormat;  // Override output format for this response
}
```

### ResponseCancelEvent

Cancel an in-progress response.

```typescript
interface ResponseCancelEvent extends BaseEvent {
  type: 'response.cancel';
  response_id?: string;       // If not provided, cancels current response
}
```

### ResponseCreatedEvent

Confirms a response has started.

```typescript
interface ResponseCreatedEvent extends BaseEvent {
  type: 'response.created';
  response_id: string;
}
```

### ResponseDeltaEvent

Streaming content delta.

```typescript
interface ResponseDeltaEvent extends BaseEvent {
  type: 'response.delta';
  response_id: string;
  delta: {
    type: 'text' | 'audio' | 'tool_call';
    text?: string;
    audio?: string;           // Base64 chunk
    tool_call?: {
      id: string;
      name: string;
      arguments: string;      // Partial JSON
    };
  };
}
```

### ResponseDoneEvent

Response completed.

```typescript
interface ResponseDoneEvent extends BaseEvent {
  type: 'response.done';
  response_id: string;
  response: {
    id: string;
    status: 'completed' | 'cancelled' | 'failed';
    output: ContentItem[];
    usage?: UsageStats;       // Optional - provider may not provide
  };
  signature?: string;         // Optional RS256 JWT signing the response (non-repudiation)
}
```

The optional `signature` field contains an RS256 JWT produced by the responding agent, enabling cryptographic non-repudiation. The JWT claims include `response_hash` (SHA-256 of the full response text) and `request_hash` (SHA-256 of the original request). Callers can verify the signature against the agent's JWKS endpoint. Agents that do not implement signing omit this field.

### ResponseErrorEvent

Response error.

```typescript
interface ResponseErrorEvent extends BaseEvent {
  type: 'response.error';
  response_id?: string;
  error: {
    code: string;
    message: string;
    details?: any;
  };
}
```

## Tool Events

### ToolCallEvent

Agent requesting tool execution.

```typescript
interface ToolCallEvent extends BaseEvent {
  type: 'tool.call';
  call_id: string;
  name: string;
  arguments: string;          // JSON string
}
```

### ToolResultEvent

Client providing tool result.

```typescript
interface ToolResultEvent extends BaseEvent {
  type: 'tool.result';
  call_id: string;
  result: string;             // JSON string
  is_error?: boolean;
}
```

### ToolCallDoneEvent

Tool call completed on server side.

```typescript
interface ToolCallDoneEvent extends BaseEvent {
  type: 'tool.call_done';
  call_id: string;
}
```

## Progress Events

### ProgressEvent

Status updates for long-running operations.

```typescript
interface ProgressEvent extends BaseEvent {
  type: 'progress';
  target: 'tool' | 'response' | 'upload' | 'reasoning';
  target_id?: string;         // e.g., tool_call_id or response_id
  stage?: string;             // "downloading", "analyzing", "thinking"
  message?: string;           // Human-readable status
  percent?: number;           // 0-100, optional
  step?: number;              // Current step number
  total_steps?: number;       // Total steps
}
```

### ThinkingEvent

Reasoning content (like Anthropic's extended thinking).

```typescript
interface ThinkingEvent extends BaseEvent {
  type: 'thinking';
  content: string;            // The reasoning text
  stage?: string;             // "analyzing", "planning", "reflecting"
  redacted?: boolean;         // If content is hidden for safety
  is_delta?: boolean;         // true = append, false = complete thought
}
```

## Audio Events

### AudioDeltaEvent

Streaming audio output.

```typescript
interface AudioDeltaEvent extends BaseEvent {
  type: 'audio.delta';
  response_id: string;
  audio: string;              // Base64 encoded chunk
}
```

### TranscriptDeltaEvent

Real-time transcription.

```typescript
interface TranscriptDeltaEvent extends BaseEvent {
  type: 'transcript.delta';
  response_id: string;
  transcript: string;
}
```

## Usage Events

### UsageDeltaEvent

Incremental usage updates during streaming.

```typescript
interface UsageDeltaEvent extends BaseEvent {
  type: 'usage.delta';
  response_id: string;
  delta: Partial<UsageStats>;
}
```

## Utility Events

### PingEvent / PongEvent

Connection keepalive.

```typescript
interface PingEvent extends BaseEvent {
  type: 'ping';
}

interface PongEvent extends BaseEvent {
  type: 'pong';
}
```

### RateLimitEvent

Rate limit notification.

```typescript
interface RateLimitEvent extends BaseEvent {
  type: 'rate_limit';
  limit: number;
  remaining: number;
  reset_at: number;           // Unix timestamp
}
```

## Presence Events

### InputTypingEvent

Client indicates the user has started or stopped typing.

```typescript
interface InputTypingEvent extends BaseEvent {
  type: 'input.typing';
  is_typing: boolean;           // true = started typing, false = stopped
  chat_id?: string;             // Optional: for multi-chat contexts
}
```

### PresenceTypingEvent

Server broadcasts typing status from another participant (for multi-user scenarios).

```typescript
interface PresenceTypingEvent extends BaseEvent {
  type: 'presence.typing';
  user_id: string;              // ID of the user who is typing
  username?: string;            // Optional display name
  is_typing: boolean;           // true = typing, false = stopped
  chat_id?: string;             // Optional: for multi-chat contexts
}
```

**Example flow:**

```typescript
// User A starts typing
→ { type: 'input.typing', event_id: 'evt_1', is_typing: true, chat_id: 'chat_abc' }

// Server broadcasts to User B
← { type: 'presence.typing', event_id: 'evt_2', user_id: 'user_a', username: 'alice', is_typing: true, chat_id: 'chat_abc' }

// User A stops typing (sends message or timeout)
→ { type: 'input.typing', event_id: 'evt_3', is_typing: false, chat_id: 'chat_abc' }

// Server broadcasts to User B
← { type: 'presence.typing', event_id: 'evt_4', user_id: 'user_a', username: 'alice', is_typing: false, chat_id: 'chat_abc' }
```

### MessageCreatedEvent

Server notifies that a new chat message was created (for chat UI use cases). Used when fanning out to subscribers of a chat.

```typescript
interface MessageCreatedEvent extends BaseEvent {
  type: 'message.created';
  session_id?: string;
  message: object;            // Message payload (id, content, sender, etc.)
  chat_id?: string;
}
```

### MessageReadEvent

Client or server indicates messages in a chat were read (read receipt).

```typescript
interface MessageReadEvent extends BaseEvent {
  type: 'message.read';
  session_id?: string;
  chat_id: string;
  user_id: string;
  last_read_at?: number;      // Unix timestamp
}
```

### PresenceOnlineEvent / PresenceOfflineEvent

Server broadcasts user or agent presence changes.

```typescript
interface PresenceOnlineEvent extends BaseEvent {
  type: 'presence.online';
  user_id: string;
  username?: string;
}

interface PresenceOfflineEvent extends BaseEvent {
  type: 'presence.offline';
  user_id: string;
  username?: string;
}
```

## Payment Events

Payment events enable real-time token balance management and payment negotiation during agent conversations. The protocol supports simple session tokens and is extensible for UCP/AP2 commerce flows.

### PaymentRequiredEvent

Server indicates payment is required to continue processing.

```typescript
interface PaymentRequiredEvent extends BaseEvent {
  type: 'payment.required';
  response_id?: string;         // If blocking a specific response
  requirements: {
    amount: string;             // Required amount (string for precision)
    currency: string;           // 'USD', 'EUR', 'ETH', etc.
    schemes: Array<{
      scheme: 'token' | 'crypto' | 'card' | 'ap2';
      network?: string;         // 'robutler', 'ethereum', 'base', etc.
      address?: string;         // For crypto payments
      min_amount?: string;
      max_amount?: string;
    }>;
    expires_at?: number;        // Unix timestamp
    reason?: string;            // 'llm_usage', 'tool_call', 'api_access'
    
    // UCP/AP2 extension for commerce flows
    ap2?: {
      mandate_uri?: string;           // Payment mandate endpoint
      credential_types?: string[];    // Accepted credential types
      checkout_session_uri?: string;  // UCP checkout session
    };
  };
}
```

### PaymentSubmitEvent

Client submits payment token or proof.

```typescript
interface PaymentSubmitEvent extends BaseEvent {
  type: 'payment.submit';
  payment: {
    scheme: string;             // 'token', 'crypto', 'ap2'
    network: string;            // 'robutler', 'ethereum', etc.
    token?: string;             // For token-based payments
    proof?: string;             // For crypto/verifiable payments
    amount: string;
    ap2_credential?: object;    // UCP/AP2 verifiable credential
  };
}
```

### PaymentAcceptedEvent

Server confirms payment was accepted.

```typescript
interface PaymentAcceptedEvent extends BaseEvent {
  type: 'payment.accepted';
  payment_id: string;
  balance_remaining?: string;
  expires_at?: number;          // Token expiry timestamp
}
```

### PaymentBalanceEvent

Server sends balance update (proactive notification).

```typescript
interface PaymentBalanceEvent extends BaseEvent {
  type: 'payment.balance';
  balance: string;
  currency: string;
  low_balance_warning?: boolean;   // true when balance < threshold
  estimated_remaining?: number;    // Estimated messages/requests remaining
  expires_at?: number;             // Token expiry timestamp
}
```

### PaymentErrorEvent

Server reports payment error.

```typescript
interface PaymentErrorEvent extends BaseEvent {
  type: 'payment.error';
  code: 'insufficient_balance' | 'token_expired' | 'token_invalid' | 
        'payment_failed' | 'rate_limited' | 'mandate_revoked';
  message: string;
  balance_required?: string;
  balance_current?: string;
  can_retry: boolean;
}
```

## Voice Events

### VoiceInviteEvent

Client or server initiates a voice session.

```typescript
interface VoiceInviteEvent extends BaseEvent {
  type: 'voice.invite';
  voice?: VoiceConfig;        // Proposed voice configuration
  offer?: object;             // WebRTC SDP offer or similar
}
```

### VoiceAcceptEvent

Accepting party confirms the voice session.

```typescript
interface VoiceAcceptEvent extends BaseEvent {
  type: 'voice.accept';
  voice?: VoiceConfig;        // Negotiated voice configuration
  answer?: object;            // WebRTC SDP answer or similar
}
```

### VoiceDeclineEvent

Declining the voice session invitation.

```typescript
interface VoiceDeclineEvent extends BaseEvent {
  type: 'voice.decline';
  reason?: string;            // 'busy', 'unsupported', 'user_declined', etc.
}
```

### VoiceEndEvent

Either side ends the voice session.

```typescript
interface VoiceEndEvent extends BaseEvent {
  type: 'voice.end';
  reason?: string;            // 'completed', 'error', 'user_hangup', etc.
  duration_ms?: number;       // Duration of the voice session
}
```

**Example payment flow:**

```typescript
// Session created with initial balance
← { type: 'session.created', session: { id: 'sess_123' } }
← { type: 'payment.balance', balance: '10.00', currency: 'USD' }

// Normal usage
→ { type: 'input.text', text: 'Hello' }
← { type: 'response.delta', delta: { type: 'text', text: 'Hi!' } }
← { type: 'response.done', response: { ... } }
← { type: 'payment.balance', balance: '9.95', currency: 'USD' }

// Low balance warning
← { type: 'payment.balance', balance: '1.00', currency: 'USD', low_balance_warning: true }

// Balance exhausted
→ { type: 'input.text', text: 'Continue' }
← { type: 'payment.required', requirements: { 
      amount: '10.00', currency: 'USD',
      schemes: [{ scheme: 'token', network: 'robutler' }],
      reason: 'llm_usage'
   }}

// Client submits new token
→ { type: 'payment.submit', payment: { scheme: 'token', network: 'robutler', token: 'tok_xxx', amount: '10.00' } }
← { type: 'payment.accepted', payment_id: 'pay_123', balance_remaining: '10.00' }

// Processing continues
← { type: 'response.delta', delta: { type: 'text', text: 'Continuing...' } }
```

## Multiplexed Sessions

Multiplexing is a protocol-level concept: multiple concurrent sessions share a single transport connection. All events in multiplexed mode include a `session_id` field that scopes the event to a specific session.

### Single vs. Multiplexed Mode

- **Single-session (default):** Omit `session_id`. One connection = one session. Backwards-compatible with existing implementations.
- **Multiplexed:** Client sends multiple `session.create` messages; server responds with `session.created` including a unique `session_id`. All subsequent events for that session include `session_id`.

### Interaction-Scoped Sessions

For platform-routed interactions (e.g. a chat platform routing user messages to agents), `session_id` is **interaction-scoped**: a unique UUID generated per request. This enables:

- **Concurrent multi-chat support:** Multiple independent conversations over a single connection, each identified by a unique `session_id`.
- **Request-response correlation:** The agent echoes `session_id` on all response events, allowing the platform to match responses to the originating request.
- **Stateless agents:** The platform sends full conversation context (`messages` array) with each `input.text`, so agents don't need to maintain per-chat state.

### Use Cases

- **Multi-agent daemons:** One daemon hosts N agents. One WebSocket to the server, one session per agent. Each `session.create` includes a per-session `token` (e.g. AOAuth JWT for that agent).
- **Browser chat UIs:** One WebSocket per user, one session per active chat. Joining a chat = `session.create { chat: "chat_abc" }`. Leaving = `session.end { session_id }`.
- **Platform routing:** One WebSocket to an agent daemon, interaction-scoped `session_id` per request. The platform sends `input.text` with `session_id`, `messages`, and `context`. The agent echoes `session_id` on all response events.

### Per-Session Auth

On `session.create`, the client may include a `token` (and optionally `payment_token`) that applies only to that session. The server associates that auth context with the returned `session_id`.

### Token Refresh

The client sends `session.update` with `session_id` and new `token` or `payment_token`. The server responds with `session.updated`. The connection stays open.

### Example

```typescript
// Multiplexed: create two sessions on one WS
ws.send(JSON.stringify({ type: 'session.create', agent: 'alice', token: 'aoauth-jwt-alice' }));
// ← { type: 'session.created', session_id: 'sess_1', agent: 'alice' }

ws.send(JSON.stringify({ type: 'session.create', agent: 'bob', token: 'aoauth-jwt-bob' }));
// ← { type: 'session.created', session_id: 'sess_2', agent: 'bob' }

// All events scoped by session_id
// ← { type: 'input.text', session_id: 'sess_1', text: 'Hi Alice' }
ws.send(JSON.stringify({ type: 'response.delta', session_id: 'sess_1', delta: { type: 'text', text: 'Hello!' } }));

// Refresh token for session (no reconnect)
ws.send(JSON.stringify({ type: 'session.update', session_id: 'sess_1', token: 'new-aoauth-jwt' }));
// ← { type: 'session.updated', session_id: 'sess_1' }
```
