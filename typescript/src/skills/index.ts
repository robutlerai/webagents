/**
 * Skills Module
 * 
 * All available skills for WebAgents.
 */

// LLM Skills
export * from './llm/index';

// Transport Skills
export * from './transport/index';

// Browser Skills
export * from './browser/index';

// Speech Skills
export * from './speech/index';

// NLI Skill (Agent-to-Agent Communication)
export * from './nli/index';

// Portal Discovery Skill
export * from './discovery/index';

// Test Runner Skill (Compliance Testing)
export * from './testrunner/index';

// Auth Skill (JWT verification via JWKS)
export * from './auth/index';

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
