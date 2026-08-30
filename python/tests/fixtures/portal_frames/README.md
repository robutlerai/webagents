# Recorded portal frames

Every fixture EXCEPT `input_text_unmapped_session.json` is the byte-shape a
REAL portal emitter puts on the agent socket today (tokens redacted, ids
randomized). That one is a PROPOSED shape — see its row and the note below.
Provenance:

| Fixture | Emitter (lib/ws/server.ts unless noted) | Shape |
|---|---|---|
| `input_text.json` | `sendInputToAgentSession` → `sendToSession` | UNWRAPPED command; carries `session_id`, full `messages` (incl. `content_items` and `role:"tool"` rows from `chatHistoryToOpenAIMessages`), `payment_token` |
| `broadcast_message_created.json` | `broadcastToChat` Redis envelope | WRAPPED `{event, chatId, _origin, _portal}` — a copy of what happened, never a work order |
| `emit_to_user_local.json` | `emitToUser` local branch | UNWRAPPED broadcast event |
| `emit_to_user_redis.json` | `emitToUser` Redis branch | WRAPPED — same logical event, different shape by pod |
| `session_created.json` | agent-branch `createSessionCreated` | echoes the caller's `agent` string verbatim; adds `agent_id`/`agent_username`; **no `token` key** |
| `session_error.json` | `sendSessionError` | `{type, event_id, timestamp, error:{code,message}}` |
| `response_cancel.json` | disconnect-grace fan-out | minimal `{type, session_id}` — deliberately no event_id (that is what the wire carries) |
| `payment_submit.json` | `createPaymentSubmit` via approval resolution | `payment: {scheme, amount, token}` + `session_id` |
| `outbound_input_text.json` | `lib/agents/uamp-client.ts` `sendInput` + messages attach | outbound WS tier; **no session_id** |
| `input_text_unmapped_session.json` | **NOT YET EMITTED** — proposed multi-pod dispatch (M5) | per-request `session_id` never announced by a local `session.created`; carries `agent` for the fallback |

Do NOT hand-edit these to match parser expectations — that is exactly how
the previous suite stayed green while the parser dropped all production
traffic. When the portal emitters change shape, re-record and let
`test_portal_frame_contract.py` fail until the parser follows.

## `agent` on `input.text` is not emitted yet

Both SDKs fall back to the frame's own `agent` field when a `session_id` was
never announced. NOTHING SETS THAT FIELD TODAY: `sendInputToAgentSession`
(lib/ws/server.ts) builds the frame without it, and the cross-pod publish in
`dispatchInputToAgent` carries `agentId` on the Redis WRAPPER, not on the
frame the daemon receives. So the fallback is INERT until the M5 dispatch hop
adds `agent` to both emitters; a cross-pod turn with an unannounced sid is
still dropped (with an error log rather than silence). The fixture exists so
the parser side is ready and tested when that lands.
