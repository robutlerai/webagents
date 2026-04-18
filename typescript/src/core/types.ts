/**
 * Core Framework Types
 * 
 * Type definitions for tools, hooks, handoffs, HTTP/WebSocket configs, and context.
 */

import type { JSONSchema, Capabilities, Message, UsageStats, ContentItem, ToolDefinition } from '../uamp/types';
import type { ClientEvent, ServerEvent } from '../uamp/events';

// ============================================================================
// Tool Types
// ============================================================================

/**
 * Tool configuration for the @tool decorator
 */
export interface ToolConfig {
  /** Tool name (defaults to function name) */
  name?: string;
  /** Tool description */
  description?: string;
  /** JSON Schema for parameters */
  parameters?: JSONSchema;
  /** Capability this tool provides */
  provides?: string;
  /** Required scopes to access this tool */
  scopes?: string[];
  /** Whether tool is enabled */
  enabled?: boolean;
}

/**
 * Registered tool metadata
 */
export interface Tool {
  /** Tool name */
  name: string;
  /** Tool description */
  description?: string;
  /** Parameter schema */
  parameters?: JSONSchema;
  /** Capability provided */
  provides?: string;
  /** Required scopes */
  scopes?: string[];
  /** Whether enabled */
  enabled: boolean;
  /** The handler function */
  handler: ToolHandler;
  /** Pricing config for payment skill hooks (alternative to @pricing decorator) */
  pricing?: PricingConfig;
}

/**
 * Tool handler function signature
 */
export type ToolHandler = (
  params: Record<string, unknown>,
  context: Context
) => Promise<unknown>;

// ============================================================================
// Hook Types
// ============================================================================

/**
 * Hook lifecycle points
 */
export type HookLifecycle =
  | 'before_run'
  | 'after_run'
  | 'before_tool'
  | 'after_tool'
  | 'before_toolcall'
  | 'after_toolcall'
  | 'before_handoff'
  | 'after_handoff'
  | 'on_connection'
  | 'finalize_connection'
  | 'before_llm_call'
  | 'after_llm_call'
  | 'on_message'
  | 'on_chunk'
  | 'on_error';

/**
 * Hook configuration for the @hook decorator
 */
export interface HookConfig {
  /** Lifecycle point */
  lifecycle: HookLifecycle;
  /** Hook priority (lower = earlier) */
  priority?: number;
  /** Whether hook is enabled */
  enabled?: boolean;
}

/**
 * Registered hook metadata
 */
export interface Hook {
  /** Lifecycle point */
  lifecycle: HookLifecycle;
  /** Priority */
  priority: number;
  /** Whether enabled */
  enabled: boolean;
  /** The handler function */
  handler: HookHandler;
}

/**
 * Hook handler function signature
 */
export type HookHandler = (
  data: HookData,
  context: Context
) => Promise<HookResult | void>;

/**
 * Data passed to hooks
 */
export interface HookData {
  /** Messages in conversation */
  messages?: Message[];
  /** Tool name (for tool hooks) */
  tool_name?: string;
  /** Tool parameters */
  tool_params?: Record<string, unknown>;
  /** Tool result */
  tool_result?: unknown;
  /** Handoff target */
  handoff_target?: string;
  /** Error (for error hooks) */
  error?: Error;
  /** Response content */
  response?: string | unknown;
  /** HTTP request object (for on_connection hooks) */
  request?: unknown;
  /** Connection metadata (path, headers, transport, etc.) */
  metadata?: Record<string, unknown>;
  /** WebSocket connection (for realtime hooks) */
  ws?: unknown;
  /** Streaming chunk (for on_chunk hooks) */
  chunk?: unknown;
  /** The raw server event (for on_chunk hooks) */
  event?: unknown;
  /** Current agentic loop iteration (for before/after_llm_call) */
  iteration?: number;
}

/**
 * Hook result to modify behavior
 */
export interface HookResult {
  /** Modified messages */
  messages?: Message[];
  /** Modified tool params */
  tool_params?: Record<string, unknown>;
  /** Modified tool result */
  tool_result?: unknown;
  /** Skip remaining hooks */
  skip_remaining?: boolean;
  /** Abort operation */
  abort?: boolean;
  /** Abort reason */
  abort_reason?: string;
}

// ============================================================================
// Handoff Types
// ============================================================================

/**
 * Handoff configuration for the @handoff decorator
 */
export interface HandoffConfig {
  /** Handoff target name */
  name: string;
  /** Description for LLM */
  description?: string;
  /** Priority (higher = preferred, default: 0) */
  priority?: number;
  /** Required scopes */
  scopes?: string[];
  /** Whether enabled */
  enabled?: boolean;
  /** Event types/patterns this handoff subscribes to (default: ['input.text']) */
  subscribes?: (string | RegExp)[];
  /** Event types this handoff produces (default: ['response.delta']) */
  produces?: string[];
}

/**
 * Registered handoff metadata
 */
export interface Handoff {
  /** Target name */
  name: string;
  /** Description */
  description?: string;
  /** Priority */
  priority: number;
  /** Required scopes */
  scopes?: string[];
  /** Whether enabled */
  enabled: boolean;
  /** Event types/patterns this handoff subscribes to */
  subscribes: (string | RegExp)[];
  /** Event types this handoff produces */
  produces: string[];
  /** The handler function */
  handler: HandoffHandler;
}

/**
 * Handoff handler function signature
 */
export type HandoffHandler = (
  events: ClientEvent[],
  context: Context
) => AsyncGenerator<ServerEvent, void, unknown>;

// ============================================================================
// Observer Types
// ============================================================================

/**
 * Observer configuration for the @observe decorator
 */
export interface ObserveConfig {
  /** Observer name */
  name: string;
  /** Event types/patterns to observe ('*' for all) */
  subscribes: (string | RegExp)[];
}

/**
 * Registered observer metadata
 */
export interface Observer {
  /** Observer name */
  name: string;
  /** Event types/patterns to observe */
  subscribes: (string | RegExp)[];
  /** Whether enabled */
  enabled: boolean;
  /** The handler function */
  handler: ObserverHandler;
}

/**
 * Observer handler function signature (non-consuming)
 */
export type ObserverHandler = (
  event: { type: string; payload: Record<string, unknown> },
  context: Context
) => Promise<void>;

// ============================================================================
// HTTP/WebSocket Types
// ============================================================================

/**
 * HTTP method types
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'OPTIONS' | 'HEAD';

/**
 * HTTP endpoint configuration for the @http decorator
 */
export interface HttpConfig {
  /** URL path */
  path: string;
  /** HTTP method */
  method?: HttpMethod;
  /** Required scopes */
  scopes?: string[];
  /** Content type */
  content_type?: string;
  /** Whether enabled */
  enabled?: boolean;
}

/**
 * Registered HTTP endpoint metadata
 */
export interface HttpEndpoint {
  /** URL path */
  path: string;
  /** HTTP method */
  method: HttpMethod;
  /** Required scopes */
  scopes?: string[];
  /** Content type */
  content_type?: string;
  /** Whether enabled */
  enabled: boolean;
  /** The handler function */
  handler: HttpHandler;
}

/**
 * HTTP handler function signature
 */
export type HttpHandler = (
  request: Request,
  context: Context
) => Promise<Response>;

/**
 * WebSocket endpoint configuration for the @websocket decorator
 */
export interface WebSocketConfig {
  /** URL path */
  path: string;
  /** Required scopes */
  scopes?: string[];
  /** Subprotocols */
  protocols?: string[];
  /** Whether enabled */
  enabled?: boolean;
}

/**
 * Registered WebSocket endpoint metadata
 */
export interface WebSocketEndpoint {
  /** URL path */
  path: string;
  /** Required scopes */
  scopes?: string[];
  /** Subprotocols */
  protocols?: string[];
  /** Whether enabled */
  enabled: boolean;
  /** The handler function */
  handler: WebSocketHandler;
}

/**
 * WebSocket handler function signature
 */
export type WebSocketHandler = (
  ws: WebSocket,
  context: Context
) => void | Promise<void>;

// ============================================================================
// Context Types
// ============================================================================

// ============================================================================
// Auth Scopes
// ============================================================================

export enum AuthScope {
  ADMIN = 'admin',
  OWNER = 'owner',
  USER = 'user',
  ALL = 'all',
}

/**
 * Authentication information
 */
export interface AuthInfo {
  /** Whether authenticated */
  authenticated: boolean;
  /** User ID */
  user_id?: string;
  /** User email */
  email?: string;
  /** Agent ID (for agent-scoped auth) */
  agent_id?: string;
  /** Agent ID alias (camelCase convenience) */
  agentId?: string;
  /** Active scopes */
  scopes?: string[];
  /** Auth scope level */
  scope?: AuthScope;
  /** Auth provider */
  provider?: string;
  /** Raw token claims */
  claims?: Record<string, unknown>;
  /** Owner assertion claims (if present) */
  assertion?: Record<string, unknown>;
}

/**
 * Payment/billing information
 */
export interface PaymentInfo {
  /** Whether payment is valid */
  valid: boolean;
  /** Payment token */
  token?: string;
  /** Remaining balance */
  balance?: number;
  /** Currency */
  currency?: string;
  /** Lock ID for the current payment lock */
  lockId?: string;
  /** Locked amount in dollars */
  lockedAmount?: number;
  /** Whether settlement succeeded */
  settled?: boolean;
  /** BYOK user ID (from token claims) */
  byokUserId?: string;
  /** BYOK provider keys */
  byokKeys?: Record<string, unknown>;
  /** Request a refreshed/topped-up payment token (used by PaymentSkill for negotiation) */
  refreshToken?: (opts: { amount: string }) => Promise<string | null>;
}

// ============================================================================
// Pricing Types
// ============================================================================

/**
 * Configuration for the @pricing decorator on tools.
 */
export interface PricingConfig {
  /** Fixed credits charged per tool call */
  creditsPerCall?: number;
  /** Lock amount: fixed number OR function of tool params (dollars).
   *  When a function, receives the tool's input params and returns
   *  the dollar amount to lock before execution. */
  lock?: number | ((params: Record<string, unknown>) => number);
  /** Settlement function: receives tool result + params, returns
   *  the dollar amount to charge. Overrides _billing metadata parsing. */
  settle?: (result: unknown, params: Record<string, unknown>) => number;
  /** Reason for the charge (shown to user) */
  reason?: string;
  /** Callback on successful tool execution */
  onSuccess?: (result: unknown, context: Context) => void | Promise<void>;
  /** Callback on tool execution failure */
  onFail?: (error: Error, context: Context) => void | Promise<void>;
}

/**
 * Pricing info returned by dynamic pricing tools.
 * A tool can return `[result, PricingInfo]` tuple for dynamic pricing.
 */
export interface PricingInfo {
  /** Credits to charge for this specific invocation */
  credits: number;
  /** Reason for the charge */
  reason?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Session state
 */
export interface SessionState {
  /** Session ID */
  id: string;
  /** Creation time */
  created_at: number;
  /** Last activity time */
  last_activity: number;
  /** Custom state data */
  data: Record<string, unknown>;
}

/**
 * Request context passed to handlers
 */
export interface Context {
  /** Session state */
  session: SessionState;
  /** Authentication info */
  auth: AuthInfo;
  /** Payment info */
  payment: PaymentInfo;
  /** Client capabilities */
  client_capabilities?: Capabilities;
  /** Agent capabilities */
  agent_capabilities?: Capabilities;
  /** Request metadata */
  metadata: Record<string, unknown>;
  /** Abort signal for cancellation propagation */
  signal?: AbortSignal;
  
  // Methods
  /** Get session data */
  get<T = unknown>(key: string): T | undefined;
  /** Set session data */
  set<T = unknown>(key: string, value: T): void;
  /** Delete session data */
  delete(key: string): void;
  /** Check if user has scope */
  hasScope(scope: string): boolean;
  /** Check if user has all scopes */
  hasScopes(scopes: string[]): boolean;
}

// ============================================================================
// Run Options
// ============================================================================

/**
 * Options for agent.run()
 */
export interface RunOptions {
  /** Model to use */
  model?: string;
  /** System instructions */
  instructions?: string;
  /** Temperature */
  temperature?: number;
  /** Max output tokens */
  max_tokens?: number;
  /** Available tools */
  tools?: Tool[];
  /** Enable streaming */
  stream?: boolean;
  /** Timeout in ms */
  timeout?: number;
  /** Additional context */
  context?: Partial<Context>;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
  /** Payment token JWT for LLM proxy authorization */
  paymentToken?: string;
  /** Client capabilities to announce in session.create */
  client_capabilities?: Capabilities;
}

/**
 * Response from agent.run()
 */
export interface RunResponse {
  /** Response content */
  content: string;
  /** Content items (multimodal) */
  content_items?: import('../uamp/types.js').ContentItem[];
  /** Tool calls made */
  tool_calls?: import('../uamp/types.js').ToolCall[];
  /** Usage statistics */
  usage?: UsageStats;
  /** Response metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Streaming chunk from agent.runStreaming()
 */
export interface StreamChunk {
  /** Chunk type */
  type: 'delta' | 'tool_call' | 'tool_result' | 'tool_progress' | 'file' | 'thinking' | 'done' | 'error';
  /** Text delta */
  delta?: string;
  /** Tool call */
  tool_call?: import('../uamp/types.js').ToolCall;
  /** Tool result (for internal tool execution progress) */
  tool_result?: {
    call_id: string;
    result: string;
    is_error?: boolean;
    content_items?: import('../uamp/types.js').ContentItem[];
    /** Optional structured metadata forwarded from `StructuredToolResult.data`.
     *  The NLI delegate skill uses this to surface `subChatId` so the
     *  parent's `<DelegateSubChatPreview />` knows which sub-chat to
     *  subscribe to (plan §4 step 1). */
    data?: Record<string, unknown>;
  };
  /** Incremental text from a running tool (e.g. delegation streaming) */
  tool_progress?: {
    call_id: string;
    text: string;
    replace?: boolean;
    media_type?: string;
    status?: string;
    progress_percent?: number;
    estimated_duration_ms?: number;
  };
  /** Thinking/reasoning content from the model */
  thinking?: { content: string; stage?: string };
  /** Final response (for 'done') */
  response?: RunResponse;
  /** Error (for 'error') */
  error?: Error;
}

// ============================================================================
// Prompt Types
// ============================================================================

/**
 * Configuration for the @prompt decorator.
 */
export interface PromptConfig {
  /** Execution priority (lower numbers execute first, default: 50) */
  priority?: number;
  /** Access scope: "all", "owner", "admin", or an array of scopes */
  scope?: string | string[];
}

/**
 * Registered prompt metadata.
 * Prompt handlers contribute dynamic content to the system message before each LLM call.
 */
export interface Prompt {
  /** Prompt name (defaults to method name) */
  name: string;
  /** Execution priority (lower = first) */
  priority: number;
  /** Access scope for filtering */
  scope: string | string[];
  /** Handler that returns prompt text to append to the system message */
  handler: (context: Context) => string | Promise<string>;
}

// ============================================================================
// Skill Types
// ============================================================================

/**
 * Skill configuration
 */
export interface SkillConfig {
  /** Skill name */
  name?: string;
  /** Whether skill is enabled */
  enabled?: boolean;
  /** Skill-specific config */
  [key: string]: unknown;
}

/**
 * Base interface for skills
 */
export interface ISkill {
  /** Skill name */
  readonly name: string;
  /** Whether skill is enabled */
  enabled: boolean;
  /** Registered tools */
  readonly tools: Tool[];
  /** Registered hooks */
  readonly hooks: Hook[];
  /** Registered handoffs */
  readonly handoffs: Handoff[];
  /** Registered prompts (dynamic system message contributors) */
  readonly prompts?: Prompt[];
  /** Registered HTTP endpoints */
  readonly httpEndpoints: HttpEndpoint[];
  /** Registered WebSocket endpoints */
  readonly wsEndpoints: WebSocketEndpoint[];
  /** Initialize the skill */
  initialize?(): Promise<void>;
  /** Cleanup resources */
  cleanup?(): Promise<void>;
}

// ============================================================================
// Agent Types
// ============================================================================

/**
 * Agent configuration
 */
export interface AgentConfig {
  /** Agent name */
  name?: string;
  /** Agent description */
  description?: string;
  /** System instructions */
  instructions?: string;
  /** Skills to load */
  skills?: ISkill[];
  /** Default model */
  model?: string;
  /** Agent capabilities */
  capabilities?: Partial<Capabilities>;
  /** Max tool execution iterations before stopping the agentic loop (default: 10) */
  maxToolIterations?: number;
}

// ============================================================================
// Agentic Loop Types
// ============================================================================

/**
 * A message in the conversation history, compatible with OpenAI's chat format.
 * Used internally by the agentic loop to track the evolving conversation
 * across multiple LLM invocations and tool executions.
 */
export interface AgenticMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  content_items?: ContentItem[];
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
  /** Ephemeral flag: media in this message should be inlined for LLM consumption.
   *  Set only by `read_content` tool. Never persisted or serialised over the wire. */
  _inline_for_llm?: boolean;
}

/**
 * Structured return from tools that produce multimodal content.
 */
export interface StructuredToolResult {
  text: string;
  content_items?: ContentItem[];
  /** Optional follow-up messages to append to the agent's conversation
   *  after the `role: 'tool'` result row for this call. Used by tools like
   *  `read_content` to add an `_inline_for_llm` user message AFTER the
   *  tool_result, so providers like Anthropic and OpenAI that require
   *  `tool_use` to be immediately followed by `tool_result` are not broken. */
  _post_messages?: AgenticMessage[];
  /** Optional structured metadata that flows through to the rendered tool
   *  result envelope on the parent side. The NLI delegate skill uses this
   *  to surface `subChatId` so the parent's `<DelegateSubChatPreview />`
   *  knows which sub-chat to subscribe to (plan §4 step 1). Kept loose so
   *  other tools can attach their own structured metadata without expanding
   *  this interface. */
  data?: Record<string, unknown>;
}

/**
 * Base interface for agents
 */
export interface IAgent {
  /** Agent name */
  readonly name: string;
  /** Agent description */
  readonly description?: string;
  /** Get agent capabilities */
  getCapabilities(): Capabilities;
  /** Process UAMP events */
  processUAMP(events: ClientEvent[]): AsyncGenerator<ServerEvent, void, unknown>;
  /** Run with messages */
  run(messages: Message[], options?: RunOptions): Promise<RunResponse>;
  /** Run with streaming */
  runStreaming(messages: Message[], options?: RunOptions): AsyncGenerator<StreamChunk, void, unknown>;
  /** Get tool definitions */
  getToolDefinitions?(): ToolDefinition[];
  /** Add a skill to the agent */
  addSkill?(skill: ISkill): void;
  /** Look up an HTTP endpoint handler by path and method */
  getHttpHandler?(path: string, method: string): HttpEndpoint | undefined;
  /** Look up a WebSocket endpoint handler by path */
  getWebSocketHandler?(path: string): WebSocketEndpoint | undefined;
}
