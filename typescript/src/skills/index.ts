/**
 * Skills Module
 * 
 * All available skills for WebAgents.
 */

// LLM Skills
export * from './llm/index';

// Transport Skills
export * from './transport/index';

// Browser Skills (legacy: tab-multiplexed adapter for in-extension agent)
export * from './browser/index';

// Browser-control (Plan v3-02): unified skill + 3 backends
// (browserbase / browser-use / chrome-extension tab-provider).
// NOTE on naming: the legacy `./browser/index` module already exports
// a class named `BrowserControlSkill`. The new module exports a
// DIFFERENT class with the same name, so we re-export the new module
// under an explicit alias to avoid the symbol collision under
// `export *`. Consumers wanting the new skill should import from
// `webagents/skills/browser-control` directly.
export {
  BrowserbaseBackend,
  BrowserUseBackend,
  ChromeBrowserBackend,
  forwardChildLiveBlock,
} from './browser-control/index';
export type {
  BrowserBackend,
  SessionHandle,
  BackendActionResult,
} from './browser-control/index';

// Speech Skills
export * from './speech/index';

// Voice Skills (Plan v3-03 — RealtimeLLMSkill + re-exported RealtimeTransportSkill).
// Re-exports `RealtimeTransportSkill` from `./transport/realtime/`, which already
// appears under the `Transport Skills` umbrella above. The voice barrel re-exports
// it under the same name, so consumers can import it from `webagents/skills/voice`
// to stay shaped around the voice domain. The names are identical — re-exporting
// the type alias from two places is fine; webagents currently uses
// `export *` everywhere so the duplicate `RealtimeTransportSkill` token from the
// voice barrel shadows the transport one with the same value (and is `===`).
export * from './voice/index';

// NLI Skill (Agent-to-Agent Communication)
export * from './nli/index';

// Portal Discovery Skill
export * from './discovery/index';

// Test Runner Skill (Compliance Testing)
export * from './testrunner/index';

// Auth Skill (JWT verification via JWKS)
export * from './auth/index';

// Notification Skill (pluggable approval + operator notifications;
// LocalNotificationSkill default for standalone SDK).
export * from './notification/index';

// Payment x402 Skill
export * from './payments/index';

// Filesystem Skill (sandboxed file operations)
export * from './filesystem/index';

// Shell Skill (sandboxed command execution)
export * from './shell/index';

// MCP Skill (Model Context Protocol client)
export * from './mcp/index';

// Dynamic Routing Skill (agent-to-agent discovery and delegation)
export * from './routing/index';

// Storage Skills (KV, JSON, Files)
export * from './storage/index';

// Session Skill (conversational state management)
export * from './session/index';

// Checkpoint Skill (file system snapshots)
export * from './checkpoint/index';

// Todo Skill (task management)
export * from './todo/index';

// RAG Skill (retrieval-augmented generation)
export * from './rag/index';

// Sandbox Skill (Docker code execution)
export * from './sandbox/index';

// Plugin Skill (dynamic skill loading)
export * from './plugin/index';

// Social Skills (Chats, Notifications, Publish, Portal Connect/WS)
export * from './social/index';

// Media Skill (content resolution, storage, URL management)
export * from './media/index';

// OpenAPI Skill (REST API integration via OpenAPI specs)
export * from './openapi/index';

// Messaging Skills (Telegram, Twilio, Slack, Discord, WhatsApp,
// Messenger, Instagram, LinkedIn, Bluesky, Reddit)
export * from './messaging/index';

// Functions skills — substrate (FunctionRuntimeSkill) plus consumer skills
// (cron, custom_http, custom_tools) and host-self-edit.
export * from './functions/index';
export * from './cron/index';
export * from './custom-http/index';
export * from './custom-tools/index';
export * from './host-self-edit/index';
