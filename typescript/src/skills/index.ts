/**
 * Skills Module
 * 
 * All available skills for WebAgents.
 */

// LLM Skills
export * from './llm/index.js';

// Transport Skills
export * from './transport/index.js';

// Browser Skills
export * from './browser/index.js';

// Speech Skills
export * from './speech/index.js';

// NLI Skill (Agent-to-Agent Communication)
export * from './nli/index.js';

// Portal Discovery Skill
export * from './discovery/index.js';

// Test Runner Skill (Compliance Testing)
export * from './testrunner/index.js';

// Auth Skill (JWT verification via JWKS)
export * from './auth/index.js';

// Payment x402 Skill
export * from './payments/index.js';

// Filesystem Skill (sandboxed file operations)
export * from './filesystem/index.js';

// Shell Skill (sandboxed command execution)
export * from './shell/index.js';

// MCP Skill (Model Context Protocol client)
export * from './mcp/index.js';

// Dynamic Routing Skill (agent-to-agent discovery and delegation)
export * from './routing/index.js';

// Storage Skills (KV, JSON, Files)
export * from './storage/index.js';

// Session Skill (conversational state management)
export * from './session/index.js';

// Checkpoint Skill (file system snapshots)
export * from './checkpoint/index.js';

// Todo Skill (task management)
export * from './todo/index.js';

// RAG Skill (retrieval-augmented generation)
export * from './rag/index.js';

// Sandbox Skill (Docker code execution)
export * from './sandbox/index.js';

// Plugin Skill (dynamic skill loading)
export * from './plugin/index.js';

// Social Skills (Chats, Notifications, Publish, Portal Connect/WS)
export * from './social/index.js';

// Media Skill (content resolution, storage, URL management)
export * from './media/index.js';
