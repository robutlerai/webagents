/**
 * The Agent Client Protocol (ACP v1) as this SDK speaks it: the constants,
 * the error codes, what `initialize` answers, the kind a tool name maps to,
 * and which kinds ask the client for permission (gap-closure plan item 1.6,
 * 2026-09-26).
 *
 * WHY A SEPARATE MODULE. The Python SDK carries the same decisions in
 * `agents/skills/core/transport/acp/protocol.py`, and both are pinned by one
 * fixture, `python/tests/fixtures/acp/acp_protocol.json`, so an editor sees
 * the same agent whichever SDK runs it. Keeping the decisions in one small
 * file, with no transport code around them, is what keeps that comparison
 * honest.
 *
 * ACP IS STDIO. The editor spawns `webagents acp` and talks JSON-RPC 2.0 over
 * the process's stdin and stdout, one message per line. This name used to be
 * a homegrown "Agent Commerce Protocol" stub whose four tools fetched any
 * URL the caller named (S-249); that stub is gone, and `acp` is the Agent
 * Client Protocol in both SDKs.
 */

/** The one protocol version this agent speaks (an integer; ACP v2 is a draft). */
export const PROTOCOL_VERSION = 1;

/** The `name` of `agentInfo`; the agent's own name goes in `title`. */
export const AGENT_INFO_NAME = 'webagents';

/** Session ids start with this, so a client can tell ours apart in a log. */
export const SESSION_ID_PREFIX = 'sess_';

/** JSON-RPC 2.0 and ACP error codes (`errors` in the fixture). */
export const PARSE_ERROR = -32700;
export const INVALID_REQUEST = -32600;
export const METHOD_NOT_FOUND = -32601;
export const INVALID_PARAMS = -32602;
export const INTERNAL_ERROR = -32603;
export const AUTH_REQUIRED = -32000;
export const RESOURCE_NOT_FOUND = -32002;
export const REQUEST_CANCELLED = -32800;

/**
 * What `initialize` advertises: only what the agent answers. `loadSession`
 * because `session/load` replays history; `list` because `session/list`
 * answers; embedded context because a `resource` block's text reaches the
 * model; no image or audio prompts; MCP servers over stdio (always), HTTP
 * and SSE, the transports the `mcp` skill serves.
 */
export const AGENT_CAPABILITIES = {
  loadSession: true,
  promptCapabilities: { image: false, audio: false, embeddedContext: true },
  mcpCapabilities: { http: true, sse: true },
  sessionCapabilities: { list: {} },
} as const;

/**
 * The registry requires at least one auth method of type `agent` or
 * `terminal`. `terminal` args REPLACE the normal args, so the client runs
 * `webagents login` in a terminal of its own.
 */
export const AUTH_METHODS = [
  {
    id: 'login',
    name: 'Log in to Robutler',
    description: 'Signs in with `webagents login`. A provider key set with `webagents secrets set` works without signing in.',
    type: 'terminal',
    args: ['login'],
  },
] as const;

export type ToolKind = 'read' | 'edit' | 'delete' | 'move' | 'search' | 'execute' | 'think' | 'fetch' | 'other';

/** Tool kinds that ask the client before the tool runs. */
export const PERMISSION_KINDS: ReadonlySet<ToolKind> = new Set<ToolKind>(['edit', 'delete', 'move', 'execute']);

/** The choices offered with `session/request_permission`. */
export const PERMISSION_OPTIONS = [
  { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
  { optionId: 'reject', name: 'Reject', kind: 'reject_once' },
] as const;

/** What the model is told when the user rejects a tool, or the prompt was cancelled while the question was open. */
export const rejected = (tool: string): string => `Rejected by the user: ${tool} was not run.`;
export const cancelled = (tool: string): string => `Cancelled: ${tool} was not run.`;

export const STOP_END_TURN = 'end_turn';
/**
 * A turn the agent's tool budget ended (2026-09-28, `core/tool-budget.ts`):
 * its rounds ran out, or it repeated one call. The precise reason goes in the
 * response's `_meta.webagents_finish`.
 */
export const STOP_MAX_TURN_REQUESTS = 'max_turn_requests';
export const STOP_CANCELLED = 'cancelled';

/**
 * The kind of a tool from its name (the fixture's `tool_kinds.rules`, in
 * order; the first match wins). `execute` before `read` so `run_command` is
 * not a read; `search` before `read` so `search_file_content` is not a read.
 */
const KIND_RULES: ReadonlyArray<{ kind: ToolKind; starts: string[]; contains?: string[] }> = [
  { kind: 'execute', starts: ['run_', 'exec', 'shell', 'terminal', 'bash', 'spawn'], contains: ['command'] },
  { kind: 'search', starts: ['search', 'grep', 'find', 'glob', 'discover'], contains: ['search'] },
  { kind: 'delete', starts: ['delete', 'remove', 'rm_', 'unlink', 'drop'] },
  { kind: 'move', starts: ['move', 'rename', 'mv_'] },
  { kind: 'edit', starts: ['write', 'edit', 'replace', 'create', 'put_', 'append', 'patch', 'update', 'set_'], contains: ['_write'] },
  { kind: 'fetch', starts: ['fetch', 'http', 'web_', 'rest_', 'curl', 'download', 'get_url'] },
  { kind: 'think', starts: ['todo', 'think', 'plan', 'note'] },
  { kind: 'read', starts: ['read', 'list', 'get_', 'cat', 'view', 'show', 'ls_', 'stat'] },
];

/** Between an MCP server's name and its tool's name (`mcp/config.ts`). */
const MCP_SEPARATOR = '__';

/** The ACP `kind` of a tool; an MCP tool (`<server>__<tool>`) is judged on the part after the separator. */
export function toolKind(name: string): ToolKind {
  let bare = (name ?? '').toLowerCase();
  if (bare.includes(MCP_SEPARATOR)) bare = bare.slice(bare.indexOf(MCP_SEPARATOR) + MCP_SEPARATOR.length);
  for (const rule of KIND_RULES) {
    if (rule.starts.some((prefix) => bare.startsWith(prefix))) return rule.kind;
    if (rule.contains?.some((part) => bare.includes(part))) return rule.kind;
  }
  return 'other';
}

export function needsPermission(kind: ToolKind): boolean {
  return PERMISSION_KINDS.has(kind);
}

/** A JSON-RPC error answer: `code`, `message` and optional `data`. */
export class AcpError extends Error {
  constructor(
    readonly code: number,
    message: string,
    readonly data?: unknown,
  ) {
    super(message);
    this.name = 'AcpError';
  }

  toJSON(): { code: number; message: string; data?: unknown } {
    return { code: this.code, message: this.message, ...(this.data !== undefined ? { data: this.data } : {}) };
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

/**
 * The user message a `session/prompt` carries: its text blocks, an embedded
 * `resource`'s text under its uri, and a `resource_link` by uri. Image and
 * audio blocks are refused, because `initialize` said so.
 */
export function promptText(blocks: unknown): string {
  if (!Array.isArray(blocks) || blocks.length === 0) {
    throw new AcpError(INVALID_PARAMS, 'prompt must be a non-empty array of content blocks');
  }
  const parts: string[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) throw new AcpError(INVALID_PARAMS, 'each prompt block must be an object');
    const kind = block.type;
    if (kind === 'text') {
      parts.push(String(block.text ?? ''));
    } else if (kind === 'resource') {
      const resource = isRecord(block.resource) ? block.resource : {};
      const uri = String(resource.uri ?? '');
      parts.push(typeof resource.text === 'string' ? `[Resource: ${uri}]\n${resource.text}` : `[Resource: ${uri}]`);
    } else if (kind === 'resource_link') {
      parts.push(`[Resource link: ${String(block.uri ?? '')}]`);
    } else if (kind === 'image' || kind === 'audio') {
      throw new AcpError(INVALID_PARAMS, `${kind} prompt blocks are not supported by this agent`);
    } else {
      throw new AcpError(INVALID_PARAMS, `unknown prompt block type: ${JSON.stringify(kind)}`);
    }
  }
  return parts.join('\n\n');
}

/** A tool call's arguments as the client shows them (`rawInput`): the JSON object, or an empty one. */
export function parseArguments(args: unknown): Record<string, unknown> {
  if (isRecord(args)) return args;
  if (typeof args === 'string' && args.trim()) {
    try {
      const parsed: unknown = JSON.parse(args);
      return isRecord(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }
  return {};
}

/** A tool call's `content` entry for a text result. */
export function textContent(text: string): { type: 'content'; content: { type: 'text'; text: string } } {
  return { type: 'content', content: { type: 'text', text } };
}

function pairs(value: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (isRecord(value)) {
    for (const [k, v] of Object.entries(value)) out[k] = String(v);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      if (isRecord(item) && typeof item.name === 'string') out[item.name] = String(item.value ?? '');
    }
  }
  return out;
}

/**
 * The `mcp` skill's config for the `mcpServers` a session names: a stdio entry
 * is `{name, command, args, env: [{name, value}]}`; an `http` or `sse` entry is
 * `{type, name, url, headers: [{name, value}]}`. Entries that name neither a
 * command nor a url are left out (the skill would refuse them by name anyway).
 */
export function mcpServersConfig(entries: unknown): Record<string, Record<string, unknown>> {
  const servers: Record<string, Record<string, unknown>> = {};
  for (const entry of Array.isArray(entries) ? entries : []) {
    if (!isRecord(entry) || typeof entry.name !== 'string' || !entry.name) continue;
    const kind = entry.type;
    if (kind === 'http' || kind === 'sse' || (kind === undefined && typeof entry.url === 'string')) {
      if (typeof entry.url !== 'string') continue;
      servers[entry.name] = { url: entry.url, headers: pairs(entry.headers), transport: kind ?? 'http' };
    } else if (typeof entry.command === 'string' && entry.command) {
      servers[entry.name] = {
        command: entry.command,
        args: Array.isArray(entry.args) ? entry.args.map(String) : [],
        env: pairs(entry.env),
      };
    }
  }
  return servers;
}

export interface PlanEntry {
  content: string;
  priority: 'high' | 'medium' | 'low';
  status: 'pending' | 'in_progress' | 'completed';
}

/** A todo list as an ACP plan: `critical` reads as `high`, `cancelled` items are left out. */
export function planEntries(items: unknown): PlanEntry[] | null {
  if (!Array.isArray(items)) return null;
  const entries: PlanEntry[] = [];
  for (const item of items) {
    if (!isRecord(item)) continue;
    let status = String(item.status ?? 'pending');
    if (status === 'cancelled') continue;
    if (!['pending', 'in_progress', 'completed'].includes(status)) status = 'pending';
    let priority = String(item.priority ?? 'medium');
    if (priority === 'critical') priority = 'high';
    if (!['high', 'medium', 'low'].includes(priority)) priority = 'medium';
    entries.push({ content: String(item.content ?? ''), priority: priority as PlanEntry['priority'], status: status as PlanEntry['status'] });
  }
  return entries;
}
