/**
 * A2A v1.0 wire shapes (a2aproject/A2A v1.0.1, `specification/a2a.proto`,
 * package `lf.a2a.v1`, rendered as ProtoJSON: camelCase fields, enums as
 * `TASK_STATE_*` / `ROLE_*` strings, timestamps ISO 8601 UTC with `Z`).
 *
 * v1.0 broke the v0.3 wire (plan item 1.3, 2026-09-26): `kind` is gone from
 * parts and tasks, `mimeType` became `mediaType`, states are SCREAMING_SNAKE,
 * the `final` flag on stream events is gone (the stream closes at a terminal
 * or interrupted state) and the card's `url` / `preferredTransport` became
 * `supportedInterfaces`. The shapes here are v1.0; `protocol.ts` reads the
 * v0.3 spellings tolerantly on the way in and never writes them.
 */

export const A2A_VERSION = '1.0';
export const A2A_VERSION_HEADER = 'A2A-Version';
export const A2A_VERSIONS_ACCEPTED: readonly string[] = ['1.0', '1.0.0', '1'];

export type TaskState =
  | 'TASK_STATE_UNSPECIFIED'
  | 'TASK_STATE_SUBMITTED'
  | 'TASK_STATE_WORKING'
  | 'TASK_STATE_COMPLETED'
  | 'TASK_STATE_FAILED'
  | 'TASK_STATE_CANCELED'
  | 'TASK_STATE_INPUT_REQUIRED'
  | 'TASK_STATE_REJECTED'
  | 'TASK_STATE_AUTH_REQUIRED';

export const TERMINAL_STATES: ReadonlySet<TaskState> = new Set<TaskState>([
  'TASK_STATE_COMPLETED',
  'TASK_STATE_FAILED',
  'TASK_STATE_CANCELED',
  'TASK_STATE_REJECTED',
]);

export const INTERRUPTED_STATES: ReadonlySet<TaskState> = new Set<TaskState>([
  'TASK_STATE_INPUT_REQUIRED',
  'TASK_STATE_AUTH_REQUIRED',
]);

/** Whether a task in `state` will not change again on its own. */
export function isSettled(state: TaskState): boolean {
  return TERMINAL_STATES.has(state) || INTERRUPTED_STATES.has(state);
}

export type Role = 'ROLE_USER' | 'ROLE_AGENT';

/** Exactly one of `text`, `raw` (base64), `url`, `data`; no `kind`. */
export interface Part {
  text?: string;
  raw?: string;
  url?: string;
  data?: unknown;
  mediaType?: string;
  filename?: string;
  metadata?: Record<string, unknown>;
}

export interface Message {
  messageId: string;
  contextId?: string;
  taskId?: string;
  role: Role;
  parts: Part[];
  metadata?: Record<string, unknown>;
  extensions?: string[];
  referenceTaskIds?: string[];
}

export interface TaskStatus {
  state: TaskState;
  message?: Message;
  timestamp?: string;
}

export interface Artifact {
  artifactId: string;
  name?: string;
  description?: string;
  parts: Part[];
  metadata?: Record<string, unknown>;
  extensions?: string[];
}

export interface Task {
  id: string;
  contextId: string;
  status: TaskStatus;
  artifacts: Artifact[];
  history: Message[];
  metadata?: Record<string, unknown>;
}

export interface SendMessageConfiguration {
  acceptedOutputModes?: string[];
  taskPushNotificationConfig?: unknown;
  historyLength?: number;
  returnImmediately?: boolean;
}

export interface SendMessageRequest {
  tenant?: string;
  message: Message;
  configuration?: SendMessageConfiguration;
  metadata?: Record<string, unknown>;
}

export type SendMessageResponse = { task: Task } | { message: Message };

export interface TaskStatusUpdateEvent {
  taskId: string;
  contextId: string;
  status: TaskStatus;
  metadata?: Record<string, unknown>;
}

export interface TaskArtifactUpdateEvent {
  taskId: string;
  contextId: string;
  artifact: Artifact;
  append?: boolean;
  lastChunk?: boolean;
  metadata?: Record<string, unknown>;
}

/** Exactly one member set. */
export type StreamResponse =
  | { task: Task }
  | { message: Message }
  | { statusUpdate: TaskStatusUpdateEvent }
  | { artifactUpdate: TaskArtifactUpdateEvent };

export interface ListTasksResponse {
  tasks: Task[];
  nextPageToken: string;
  pageSize: number;
  totalSize: number;
}

// ---------------------------------------------------------------------------
// Errors (section 3.4 of the spec pack; A2A v1.0 section 7)
// ---------------------------------------------------------------------------

export const A2A_ERROR_DOMAIN = 'a2a-protocol.org';
export const ERROR_INFO_TYPE = 'type.googleapis.com/google.rpc.ErrorInfo';

export interface A2AErrorSpec {
  /** JSON-RPC code. */
  code: number;
  /** `google.rpc.ErrorInfo.reason`. */
  reason: string;
  /** HTTP status on the REST binding. */
  http: number;
  message: string;
}

export const A2A_ERRORS = {
  PARSE_ERROR: { code: -32700, reason: 'PARSE_ERROR', http: 400, message: 'Invalid JSON payload' },
  INVALID_REQUEST: { code: -32600, reason: 'INVALID_REQUEST', http: 400, message: 'Request payload validation error' },
  METHOD_NOT_FOUND: { code: -32601, reason: 'METHOD_NOT_FOUND', http: 404, message: 'Method not found' },
  INVALID_PARAMS: { code: -32602, reason: 'INVALID_PARAMS', http: 400, message: 'Invalid parameters' },
  INTERNAL_ERROR: { code: -32603, reason: 'INTERNAL_ERROR', http: 500, message: 'Internal error' },
  TASK_NOT_FOUND: { code: -32001, reason: 'TASK_NOT_FOUND', http: 404, message: 'Task not found' },
  TASK_NOT_CANCELABLE: { code: -32002, reason: 'TASK_NOT_CANCELABLE', http: 400, message: 'Task cannot be canceled' },
  PUSH_NOTIFICATION_NOT_SUPPORTED: { code: -32003, reason: 'PUSH_NOTIFICATION_NOT_SUPPORTED', http: 400, message: 'Push Notification is not supported' },
  UNSUPPORTED_OPERATION: { code: -32004, reason: 'UNSUPPORTED_OPERATION', http: 400, message: 'This operation is not supported' },
  CONTENT_TYPE_NOT_SUPPORTED: { code: -32005, reason: 'CONTENT_TYPE_NOT_SUPPORTED', http: 400, message: 'Incompatible content types' },
  INVALID_AGENT_RESPONSE: { code: -32006, reason: 'INVALID_AGENT_RESPONSE', http: 500, message: 'Invalid agent response' },
  EXTENDED_AGENT_CARD_NOT_CONFIGURED: { code: -32007, reason: 'EXTENDED_AGENT_CARD_NOT_CONFIGURED', http: 400, message: 'Extended agent card is not configured' },
  EXTENSION_SUPPORT_REQUIRED: { code: -32008, reason: 'EXTENSION_SUPPORT_REQUIRED', http: 400, message: 'A required extension is not supported' },
  VERSION_NOT_SUPPORTED: { code: -32009, reason: 'VERSION_NOT_SUPPORTED', http: 400, message: 'Protocol version not supported' },
} as const satisfies Record<string, A2AErrorSpec>;

export type A2AErrorName = keyof typeof A2A_ERRORS;

/** `google.rpc.Code` names for the REST error envelope, by HTTP status. */
export function rpcStatusName(http: number): string {
  switch (http) {
    case 400:
      return 'INVALID_ARGUMENT';
    case 401:
      return 'UNAUTHENTICATED';
    case 403:
      return 'PERMISSION_DENIED';
    case 404:
      return 'NOT_FOUND';
    case 409:
      return 'ABORTED';
    case 501:
      return 'UNIMPLEMENTED';
    default:
      return http >= 500 ? 'INTERNAL' : 'UNKNOWN';
  }
}

/** A protocol error, carrying everything both bindings need to answer it. */
export class A2AError extends Error {
  readonly code: number;
  readonly reason: string;
  readonly http: number;
  readonly metadata: Record<string, string>;

  constructor(name: A2AErrorName, message?: string, metadata: Record<string, string> = {}) {
    const spec = A2A_ERRORS[name];
    super(message ?? spec.message);
    this.name = 'A2AError';
    this.code = spec.code;
    this.reason = spec.reason;
    this.http = spec.http;
    this.metadata = metadata;
  }

  /** The `google.rpc.ErrorInfo` detail every A2A error carries. */
  errorInfo(): Record<string, unknown> {
    return {
      '@type': ERROR_INFO_TYPE,
      reason: this.reason,
      domain: A2A_ERROR_DOMAIN,
      ...(Object.keys(this.metadata).length ? { metadata: this.metadata } : {}),
    };
  }

  /** The JSON-RPC `error` member. */
  jsonRpc(): { code: number; message: string; data: unknown[] } {
    return { code: this.code, message: this.message, data: [this.errorInfo()] };
  }

  /** The REST binding's error body. */
  rest(): { error: { code: number; status: string; message: string; details: unknown[] } } {
    return { error: { code: this.http, status: rpcStatusName(this.http), message: this.message, details: [this.errorInfo()] } };
  }
}

export function isA2AError(value: unknown): value is A2AError {
  return value instanceof A2AError || (value as { name?: string } | null)?.name === 'A2AError';
}
