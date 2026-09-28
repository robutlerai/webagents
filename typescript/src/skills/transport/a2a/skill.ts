/**
 * A2A v1.0 transport (a2aproject/A2A v1.0.1), the same shape as the Python
 * `A2ATransportSkill` and pinned by one fixture both suites replay
 * (`python/tests/fixtures/a2a/vectors.json`). Plan item 1.3, 2026-09-26.
 *
 * WHAT IT SERVES, under the agent's prefix:
 *   * `POST /a2a`: JSON-RPC, the v1.0 PascalCase methods and the v0.3 dotted
 *     aliases (`protocol.ts` has the table and the version rules that keep
 *     OpenClaw, which sends no `A2A-Version`, and Hermes, which sends 1.0,
 *     both talking to us);
 *   * the HTTP+JSON binding at the same prefix: `POST /a2a/message:send`,
 *     `POST /a2a/message:stream`, `GET /a2a/tasks`, `GET /a2a/tasks/{id}`,
 *     `POST /a2a/tasks/{id}:cancel`, `GET|POST /a2a/tasks/{id}:subscribe`
 *     (the spec's prose says POST for subscribe and its proto says GET, so
 *     both are served);
 *   * `GET /.well-known/agent-card.json`: the v1.0 card, signed with the
 *     agent's Ed25519 identity when it has one (`card.ts`). The registration
 *     card at `/.well-known/agent.json` is the server's and is not touched:
 *     the platform reads it and refuses one that does not name itself.
 *
 * THE RUN IS THE AGENT'S NORMAL RUN, UNDER THE CALLER'S VERIFIED IDENTITY.
 * Every request is identified the way a scoped endpoint is
 * (`endpoint-gate.ts`: `identificationContext`, then `identifyCaller`, the
 * auth skills and the access block), refused with the gate's own 401/403 when
 * that fails, and the send then goes through `agent.run` / `runStreaming`
 * with the request's credential headers on the run metadata and the raw
 * request in session data, exactly what `/chat/completions` does. So
 * `access:` applies, payment applies, and an A2A caller can do nothing a
 * chat caller could not. `contextId` is the conversation (the run's chat
 * id); parts become content items; the output becomes the task's status
 * message and one artifact.
 *
 * TASKS ARE PER CALLER (`tasks.ts`): a caller reads, lists, subscribes to and
 * cancels only what it created, and another caller's task id is not found,
 * never forbidden. `returnImmediately` answers the task WORKING at once and
 * keeps working; the default blocks up to `blocking_timeout_seconds` and
 * answers whatever state the task is in by then, which is what OpenClaw's
 * 120 s poll and Hermes' 120 s wait expect. Push notifications and the
 * extended card are declined with the spec's own codes.
 *
 * WHAT WENT AWAY: the pre-v1 `tasks/send` JSON-RPC method and the old
 * `/.well-known/agent.json` handler this skill mounted. The latter is the one
 * to know about: under `serve()` a skill's `@http` route is mounted before
 * the fetch handler, so this skill's old card SHADOWED the self-naming
 * registration card for any agent that attached it, and that card could not
 * register. The server's card is the only one at that path now.
 */

import { Skill } from '../../../core/skill';
import { http } from '../../../core/decorators';
import type { Context, IAgent, RunOptions } from '../../../core/types';
import type { ToolDefinition } from '../../../uamp/types';
import { identificationContext, inboundRequest, refusalResponse, type InboundRequestShape } from '../../../server/endpoint-gate';
import { replyText as safeReplyText } from '../../../server/error-reply';
import { TrustLookup, platformCredentialFor } from '../../../trustflow/trust-lookup';
import { withTrustRecordExtension } from '../../../trustflow/trust-record';
import { envVar } from '../../platform-url';
import {
  AGENT_CARD_WELL_KNOWN_SUFFIX,
  A2A_RPC_SUBPATH,
  buildA2AAgentCard,
  identityCardSigner,
  signAgentCard,
  skillsFromTools,
  type AgentProvider,
} from './card';
import {
  callerKey,
  checkVersion,
  messageToRunMessage,
  nowIso,
  outputToParts,
  readConfiguration,
  readMessage,
  requestedVersion,
  resolveMethod,
  runMetadata,
  type Operation,
  type SendConfiguration,
} from './protocol';
import { TaskStore, taskView, type ListFilter, type TaskRecord } from './tasks';
import { A2AError, isA2AError, isSettled, type Message, type StreamResponse, type Task, type TaskState } from './types';
import { callAgent, type CallOptions, type CallResult } from './a2a-client';

export { AGENT_CARD_WELL_KNOWN_SUFFIX, A2A_RPC_SUBPATH };

// ---------------------------------------------------------------------------
// Configuration: the agent file's `a2a` entry, the same keys as Python
// (`python/tests/fixtures/a2a/config_shapes.json`)
// ---------------------------------------------------------------------------

export interface A2APeer {
  /** The bearer this peer gets, configured out of band. */
  token?: string;
}

export interface A2ATransportConfig {
  name?: string;
  enabled?: boolean;
  /** Peer agent URL (or origin) to what it gets: the longest matching prefix wins. */
  peers?: Record<string, A2APeer>;
  /** How long a finished task can be read back. Default 3600. */
  task_ttl_seconds?: number;
  taskTtlSeconds?: number;
  /** How long a blocking send waits before answering the task as it is. Default 60. */
  blocking_timeout_seconds?: number;
  blockingTimeoutSeconds?: number;
  /** Card fields. */
  version?: string;
  provider?: AgentProvider | null;
  documentation_url?: string | null;
  documentationUrl?: string | null;
  icon_url?: string | null;
  iconUrl?: string | null;
  /** The agent URL for the card when the agent has no identity and no WEBAGENTS_PUBLIC_URL. */
  public_url?: string;
  publicUrl?: string;
  /**
   * Carry this agent's signed TrustFlow record in the card (plan item 2.7):
   * fetched from the platform as this agent, as the extension
   * `trustflow/trust-record.ts` defines, and covered by the card signature.
   */
  trust_record?: boolean;
  trustRecord?: boolean;
  [key: string]: unknown;
}

/** The settings as both SDKs read them back (snake_case, the fixture's names). */
export interface A2ASettings {
  task_ttl_seconds: number;
  blocking_timeout_seconds: number;
  peers: Record<string, A2APeer>;
  version: string;
  provider: AgentProvider | null;
  documentation_url: string | null;
  icon_url: string | null;
  trust_record: boolean;
}

export const DEFAULT_TASK_TTL_SECONDS = 3600;
export const DEFAULT_BLOCKING_TIMEOUT_SECONDS = 60;

function positiveNumber(...candidates: unknown[]): number | undefined {
  for (const value of candidates) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value;
    if (typeof value === 'string' && value.trim() && Number.isFinite(Number(value)) && Number(value) > 0) return Number(value);
  }
  return undefined;
}

function optionalString(...candidates: unknown[]): string | null {
  for (const value of candidates) if (typeof value === 'string' && value.trim()) return value;
  return null;
}

/** The `a2a` entry's settings; unknown keys and the loaders' own are ignored. */
export function resolveA2ASettings(config: A2ATransportConfig = {}): A2ASettings {
  const peers: Record<string, A2APeer> = {};
  if (config.peers && typeof config.peers === 'object' && !Array.isArray(config.peers)) {
    for (const [url, entry] of Object.entries(config.peers)) {
      if (!url || typeof url !== 'string') continue;
      const token = entry && typeof entry === 'object' && typeof (entry as A2APeer).token === 'string' ? (entry as A2APeer).token : undefined;
      peers[url] = token ? { token } : {};
    }
  }
  const provider = config.provider;
  const validProvider =
    provider && typeof provider === 'object' && typeof provider.organization === 'string' && typeof provider.url === 'string'
      ? { organization: provider.organization, url: provider.url }
      : null;
  return {
    task_ttl_seconds: positiveNumber(config.task_ttl_seconds, config.taskTtlSeconds) ?? DEFAULT_TASK_TTL_SECONDS,
    blocking_timeout_seconds: positiveNumber(config.blocking_timeout_seconds, config.blockingTimeoutSeconds) ?? DEFAULT_BLOCKING_TIMEOUT_SECONDS,
    peers,
    version: optionalString(config.version) ?? '1.0.0',
    provider: validProvider,
    documentation_url: optionalString(config.documentation_url, config.documentationUrl),
    icon_url: optionalString(config.icon_url, config.iconUrl),
    trust_record: config.trust_record === true || config.trustRecord === true,
  };
}

/**
 * The bearer for `url`: the token of the longest configured peer that is the
 * URL itself or a path prefix of it (`https://peer.example` matches
 * `https://peer.example/agents/x/a2a` and not `https://peer.example.evil`).
 */
export function peerTokenFor(url: string, peers: Record<string, A2APeer>): string | null {
  const target = url.replace(/\/+$/, '');
  let best: { length: number; token: string } | null = null;
  for (const [peer, entry] of Object.entries(peers)) {
    const prefix = peer.replace(/\/+$/, '');
    if (!prefix || !entry.token) continue;
    if (target !== prefix && !target.startsWith(`${prefix}/`)) continue;
    if (!best || prefix.length > best.length) best = { length: prefix.length, token: entry.token };
  }
  return best?.token ?? null;
}

// ---------------------------------------------------------------------------
// The skill
// ---------------------------------------------------------------------------

const JSON_RPC = '2.0';
const REST_CONTENT_TYPE = 'application/a2a+json';
const CARD_CACHE_CONTROL = 'public, max-age=300';

interface Admitted {
  inbound: InboundRequestShape;
  context: Context;
  owner: string;
}

type JsonRpcId = string | number | null;

export class A2ATransportSkill extends Skill {
  readonly settings: A2ASettings;
  private agent: IAgent | null = null;
  private readonly store: TaskStore;

  constructor(config: A2ATransportConfig = {}) {
    super({ ...config, name: config.name || 'a2a-transport' });
    this.settings = resolveA2ASettings(config);
    this.store = new TaskStore(this.settings.task_ttl_seconds * 1000);
  }

  setAgent(agent: IAgent): void {
    this.agent = agent;
  }

  /** The bearer a client sends `url`, from the configured peers. */
  peerTokenFor(url: string): string | null {
    return peerTokenFor(url, this.settings.peers);
  }

  /**
   * Call a peer over A2A v1.0 as this agent (`a2a-client.ts`: its card, the
   * first JSON-RPC 1.0 interface, one `SendMessage`, the task polled to a
   * settled state), with the bearer the agent file configures for it
   * (`peers`); `options.token` overrides it.
   */
  async callPeer(url: string, input: string | Message, options: CallOptions = {}): Promise<CallResult> {
    const token = options.token ?? this.peerTokenFor(url) ?? undefined;
    return callAgent(url, input, { ...options, ...(token ? { token } : {}) });
  }

  /** How many tasks the store holds (tests). */
  get taskCount(): number {
    return this.store.size;
  }

  override async cleanup(): Promise<void> {
    this.store.clear();
  }

  // ===========================================================================
  // The card
  // ===========================================================================

  @http({ path: AGENT_CARD_WELL_KNOWN_SUFFIX, method: 'GET' })
  async handleAgentCard(request: Request, _context: Context): Promise<Response> {
    const agent = this.agent;
    if (!agent) return json({ error: { code: 'not_ready', message: 'No agent attached' } }, 503);
    const card = await this.buildCard(agent, request);
    const body = JSON.stringify(card);
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(body));
    const etag = `"${Array.from(new Uint8Array(digest).slice(0, 16), (b) => b.toString(16).padStart(2, '0')).join('')}"`;
    if (request.headers.get('if-none-match') === etag) {
      return new Response(null, { status: 304, headers: { ETag: etag, 'Cache-Control': CARD_CACHE_CONTROL } });
    }
    return new Response(body, {
      headers: { 'Content-Type': 'application/json', 'Cache-Control': CARD_CACHE_CONTROL, ETag: etag },
    });
  }

  /** The v1.0 card for `agent`, signed when the agent holds an identity. */
  async buildCard(agent: IAgent, request?: Request): Promise<Record<string, unknown>> {
    const principal = this.principalFor(agent, request);
    const card = buildA2AAgentCard(agent, {
      principal,
      version: this.settings.version,
      provider: this.settings.provider,
      documentationUrl: this.settings.documentation_url,
      iconUrl: this.settings.icon_url,
      skills: skillsFromTools(await this.publicTools(agent)),
    });
    let carded = card as unknown as Record<string, unknown>;
    if (this.settings.trust_record) {
      const record = await this.ownTrustRecord(agent, principal);
      if (record) carded = withTrustRecordExtension(carded, record);
    }
    const identity = agent.identity;
    if (identity && typeof identity.getHeldKeys === 'function') {
      try {
        return await signAgentCard(carded, identityCardSigner(identity));
      } catch {
        // An identity that is not initialised yet signs nothing; the card is still served.
      }
    }
    return carded;
  }

  /** The platform lookup the record comes from; a test sets a stub. */
  trustLookup?: Pick<TrustLookup, 'record'>;
  private heldTrustRecord?: { record: string; expires: number };
  private trustRecordRetryAt = 0;

  /**
   * This agent's signed TrustFlow record from the platform (plan item 2.7),
   * held until an hour before it expires and at most a day, so a fresh one
   * follows the batch; when the platform cannot answer the card is served
   * without it and the ask is repeated a minute later, not per request.
   */
  async ownTrustRecord(agent: IAgent, principal: string): Promise<string | null> {
    const now = Date.now();
    if (this.heldTrustRecord && this.heldTrustRecord.expires > now) return this.heldTrustRecord.record;
    if (this.trustRecordRetryAt > now) return null;
    try {
      const lookup =
        this.trustLookup ??
        (this.trustLookup = new TrustLookup({
          credential: () =>
            platformCredentialFor({
              identity: agent.identity,
              apiKey: envVar('WEBAGENTS_AGENT_TOKEN') ?? envVar('WEBAGENTS_API_KEY'),
            }),
        }));
      const { record, payload } = await lookup.record(principal);
      const exp = typeof payload.exp === 'number' ? payload.exp * 1000 : now + 3_600_000;
      this.heldTrustRecord = { record, expires: Math.min(exp - 3_600_000, now + 86_400_000) };
      return record;
    } catch {
      this.trustRecordRetryAt = now + 60_000;
      return null;
    }
  }

  /**
   * The agent URL the card's interfaces are built on: the identity's issuer
   * (what `serve()` composes from `publicUrl + basePath`), else the
   * configured or environment public URL plus the served prefix, else the
   * request's own origin plus that prefix. The last is a Host-header guess
   * the registration card refuses to make; it is made here because this
   * card's readers (OpenClaw, Hermes) post to the interface URL as written
   * and cannot resolve a relative one, and the card was fetched from exactly
   * that origin a moment ago.
   */
  principalFor(agent: IAgent, request?: Request): string {
    const identity = agent.identity;
    if (identity?.issuer) return identity.issuer.replace(/\/+$/, '');
    let basePath = '';
    let origin = '';
    if (request) {
      try {
        const url = new URL(request.url);
        origin = url.origin;
        basePath = url.pathname.endsWith(AGENT_CARD_WELL_KNOWN_SUFFIX) ? url.pathname.slice(0, -AGENT_CARD_WELL_KNOWN_SUFFIX.length) : url.pathname;
      } catch {
        // relative or malformed: no origin to derive
      }
    }
    const configured = optionalString(
      (this.config as A2ATransportConfig).public_url,
      (this.config as A2ATransportConfig).publicUrl,
      typeof process !== 'undefined' ? process.env?.WEBAGENTS_PUBLIC_URL : undefined,
    );
    const base = (configured ?? origin).replace(/\/+$/, '');
    return `${base}${basePath}`.replace(/\/+$/, '') || basePath || '/';
  }

  /**
   * The tools a default-group caller may use, never owner-only ones: listed
   * inside a run bound to an anonymous caller in the access block's default
   * group (when the block names one), the way an MCP listing is scoped.
   */
  async publicTools(agent: IAgent): Promise<ToolDefinition[]> {
    const scopes: string[] = [];
    const accessSkill = ((agent as unknown as { skills?: unknown[] }).skills ?? []).find(
      (s) => !!(s as { policy?: unknown })?.policy && typeof (s as { name?: unknown }).name === 'string',
    ) as { policy?: { default?: string | null } } | undefined;
    if (accessSkill?.policy?.default) scopes.push(`group:${accessSkill.policy.default}`);
    try {
      if (typeof agent.listTools === 'function') {
        return await agent.listTools({ auth: { authenticated: false, scopes } });
      }
    } catch {
      // fall through to the unscoped listing, which the base context already keeps anonymous
    }
    return agent.getToolDefinitions?.() ?? [];
  }

  // ===========================================================================
  // JSON-RPC
  // ===========================================================================

  @http({ path: A2A_RPC_SUBPATH, method: 'POST' })
  async handleJsonRpc(request: Request, context: Context): Promise<Response> {
    const raw = new Uint8Array(await request.arrayBuffer());
    let body: unknown;
    try {
      body = JSON.parse(new TextDecoder().decode(raw));
    } catch {
      return rpcResponse(null, { error: new A2AError('PARSE_ERROR').jsonRpc() });
    }
    if (!isRecord(body)) return rpcResponse(null, { error: new A2AError('INVALID_REQUEST', 'Request must be a JSON object').jsonRpc() });
    const id: JsonRpcId = typeof body.id === 'string' || typeof body.id === 'number' ? body.id : null;
    if (typeof body.method !== 'string' || !body.method) {
      return rpcResponse(id, { error: new A2AError('INVALID_REQUEST', 'Request has no method').jsonRpc() });
    }
    const resolved = resolveMethod(body.method);
    if (!resolved) return rpcResponse(id, { error: new A2AError('METHOD_NOT_FOUND', `Method not found: ${body.method}`).jsonRpc() });
    try {
      checkVersion(requestedVersion(request.headers, safeUrl(request)));
    } catch (error) {
      if (isA2AError(error)) return rpcResponse(id, { error: error.jsonRpc() });
      throw error;
    }
    if (resolved.op === 'push_config') return rpcResponse(id, { error: new A2AError('PUSH_NOTIFICATION_NOT_SUPPORTED').jsonRpc() });
    if (resolved.op === 'extended_card') return rpcResponse(id, { error: new A2AError('EXTENDED_AGENT_CARD_NOT_CONFIGURED').jsonRpc() });

    const admitted = await this.admit(request, context, raw);
    if (admitted instanceof Response) return admitted;
    const params = isRecord(body.params) ? body.params : {};
    try {
      switch (resolved.op) {
        case 'send': {
          const task = await this.send(admitted, params);
          return rpcResponse(id, { result: { task } });
        }
        case 'stream': {
          const record = await this.startSend(admitted, params);
          return sseResponse(this.events(record), (event) => ({ jsonrpc: JSON_RPC, id, result: event }));
        }
        case 'get': {
          const record = this.requireTask(admitted.owner, params.id);
          return rpcResponse(id, { result: { task: taskView(record.task, { historyLength: historyLengthOf(params.historyLength) }) } });
        }
        case 'list':
          return rpcResponse(id, { result: this.store.list(admitted.owner, listFilter(params)) });
        case 'cancel': {
          const record = this.requireTask(admitted.owner, params.id);
          return rpcResponse(id, { result: { task: this.cancel(record) } });
        }
        case 'subscribe': {
          const record = this.requireTask(admitted.owner, params.id);
          return sseResponse(this.events(record), (event) => ({ jsonrpc: JSON_RPC, id, result: event }));
        }
        default:
          return rpcResponse(id, { error: new A2AError('UNSUPPORTED_OPERATION').jsonRpc() });
      }
    } catch (error) {
      if (isA2AError(error)) return rpcResponse(id, { error: error.jsonRpc() });
      throw error;
    }
  }

  // ===========================================================================
  // HTTP+JSON
  // ===========================================================================

  @http({ path: `${A2A_RPC_SUBPATH}/message:send`, method: 'POST' })
  async handleRestSend(request: Request, context: Context): Promise<Response> {
    return this.rest(request, context, 'send');
  }

  @http({ path: `${A2A_RPC_SUBPATH}/message:stream`, method: 'POST' })
  async handleRestStream(request: Request, context: Context): Promise<Response> {
    return this.rest(request, context, 'stream');
  }

  @http({ path: `${A2A_RPC_SUBPATH}/tasks`, method: 'GET' })
  async handleRestList(request: Request, context: Context): Promise<Response> {
    return this.rest(request, context, 'list');
  }

  @http({ path: `${A2A_RPC_SUBPATH}/tasks/{id}`, method: 'GET' })
  async handleRestGet(request: Request, context: Context): Promise<Response> {
    // `{id}` also matches `{id}:subscribe` when the more specific route is
    // not the one the host dispatched to; the suffix decides.
    const suffix = taskPathSuffix(request);
    return this.rest(request, context, suffix === 'subscribe' ? 'subscribe' : 'get');
  }

  @http({ path: `${A2A_RPC_SUBPATH}/tasks/{id}:cancel`, method: 'POST' })
  async handleRestCancel(request: Request, context: Context): Promise<Response> {
    return this.rest(request, context, 'cancel');
  }

  @http({ path: `${A2A_RPC_SUBPATH}/tasks/{id}:subscribe`, method: 'POST' })
  async handleRestSubscribePost(request: Request, context: Context): Promise<Response> {
    return this.rest(request, context, 'subscribe');
  }

  @http({ path: `${A2A_RPC_SUBPATH}/tasks/{id}:subscribe`, method: 'GET' })
  async handleRestSubscribeGet(request: Request, context: Context): Promise<Response> {
    return this.rest(request, context, 'subscribe');
  }

  private async rest(request: Request, context: Context, op: Operation): Promise<Response> {
    const raw = new Uint8Array(await request.arrayBuffer());
    try {
      checkVersion(requestedVersion(request.headers, safeUrl(request)));
    } catch (error) {
      if (isA2AError(error)) return restError(error);
      throw error;
    }
    const admitted = await this.admit(request, context, raw);
    if (admitted instanceof Response) return admitted;
    try {
      switch (op) {
        case 'send': {
          const task = await this.send(admitted, parseRestBody(raw));
          return restJson({ task });
        }
        case 'stream': {
          const record = await this.startSend(admitted, parseRestBody(raw));
          return sseResponse(this.events(record), (event) => event);
        }
        case 'get': {
          const url = safeUrl(request);
          const record = this.requireTask(admitted.owner, taskIdOf(request));
          return restJson({ ...taskView(record.task, { historyLength: historyLengthOf(url?.searchParams.get('historyLength')) }) });
        }
        case 'list': {
          const url = safeUrl(request);
          const query = url ? Object.fromEntries(url.searchParams.entries()) : {};
          return restJson(this.store.list(admitted.owner, listFilter(query)));
        }
        case 'cancel': {
          const record = this.requireTask(admitted.owner, taskIdOf(request));
          return restJson({ ...this.cancel(record) });
        }
        case 'subscribe': {
          const record = this.requireTask(admitted.owner, taskIdOf(request));
          return sseResponse(this.events(record), (event) => event);
        }
        default:
          return restError(new A2AError('UNSUPPORTED_OPERATION'));
      }
    } catch (error) {
      if (isA2AError(error)) return restError(error);
      throw error;
    }
  }

  // ===========================================================================
  // Who is calling
  // ===========================================================================

  /**
   * Identify the caller the way a scoped endpoint does, or answer the gate's
   * own refusal. The owner key for the task store comes from the result.
   */
  private async admit(request: Request, context: Context, raw: Uint8Array): Promise<Admitted | Response> {
    const inbound = inboundRequest(request, raw);
    const identified = identificationContext(context, inbound);
    try {
      await this.agent?.identifyCaller?.(identified);
    } catch (error) {
      const refusal = refusalResponse(error);
      if (refusal) return json(refusal.body, refusal.status);
      throw error;
    }
    return { inbound, context: identified, owner: await callerKey(identified.auth, inbound) };
  }

  // ===========================================================================
  // Tasks
  // ===========================================================================

  private requireTask(owner: string, id: unknown): TaskRecord {
    if (typeof id !== 'string' || !id) throw new A2AError('INVALID_PARAMS', 'A task id is required');
    const record = this.store.get(owner, id);
    if (!record) throw new A2AError('TASK_NOT_FOUND', `Task not found: ${id}`, { taskId: id });
    return record;
  }

  /** A blocking or immediate send: the task as it stands when the answer goes out. */
  private async send(admitted: Admitted, params: Record<string, unknown>): Promise<Task> {
    const configuration = readConfiguration(params.configuration);
    const record = await this.startSend(admitted, params, { streaming: false, configuration });
    if (!configuration.returnImmediately && !isSettled(record.task.status.state)) {
      await Promise.race([record.settled, sleep(this.settings.blocking_timeout_seconds * 1000)]);
    }
    return taskView(record.task, { historyLength: configuration.historyLength });
  }

  /** Create the task, record its first event and start the run. */
  private async startSend(
    admitted: Admitted,
    params: Record<string, unknown>,
    options: { streaming: boolean; configuration?: SendConfiguration } = { streaming: true },
  ): Promise<TaskRecord> {
    const agent = this.agent;
    if (!agent) throw new A2AError('INTERNAL_ERROR', 'No agent attached');
    // Push delivery is refused before any task exists (`readConfiguration`).
    if (!options.configuration) readConfiguration(params.configuration);
    const message = readMessage(params.message);
    if (message.taskId) {
      const existing = this.store.get(admitted.owner, message.taskId);
      if (!existing) throw new A2AError('TASK_NOT_FOUND', `Task not found: ${message.taskId}`, { taskId: message.taskId });
      throw new A2AError(
        'INVALID_PARAMS',
        isSettled(existing.task.status.state)
          ? 'That task is finished; send a new message with the same contextId to continue the conversation'
          : 'That task is still running',
      );
    }
    const contextId = message.contextId ?? crypto.randomUUID();
    const taskId = crypto.randomUUID();
    const inboundMessage: Message = { ...message, contextId, taskId };
    const task: Task = {
      id: taskId,
      contextId,
      status: { state: 'TASK_STATE_SUBMITTED', timestamp: nowIso() },
      artifacts: [],
      history: [inboundMessage],
    };
    const record = this.store.create(admitted.owner, task);
    this.store.emit(record, { task: taskView(task) });
    this.store.setStatus(record, { state: 'TASK_STATE_WORKING', timestamp: nowIso() });
    void this.execute(agent, record, admitted.inbound, options.streaming);
    return record;
  }

  /**
   * The run, under the caller's identity; the task's terminal state is its
   * outcome. A streaming send runs `runStreaming` and records each delta as
   * an artifact chunk; a blocking one runs `run` and records the whole
   * artifact at once, media included.
   */
  private async execute(agent: IAgent, record: TaskRecord, inbound: InboundRequestShape, streaming: boolean): Promise<void> {
    const controller = new AbortController();
    record.abort = () => controller.abort();
    const options: RunOptions = {
      metadata: runMetadata(inbound),
      sessionData: { _inboundRequest: inbound },
      chatId: record.task.contextId,
      signal: controller.signal,
      // This caller's own token, or none: it must not fall back to whatever
      // the agent's base context carries from an earlier caller (S-285,
      // 2026-09-26). `runMetadata` also carries it as `x-payment-token`, which
      // the payment skill reads after `payment_token`; passing it here seeds
      // the run context's `payment_token` through the session extensions
      // (`core/agent.ts`), so the caller who sent no token runs on no token.
      ...(inbound.headers['x-payment-token'] ? { paymentToken: inbound.headers['x-payment-token'] } : {}),
    };
    const messages = record.task.history.map(messageToRunMessage);
    const artifactId = crypto.randomUUID();
    let text = '';
    let first = true;
    try {
      if (streaming && typeof agent.runStreaming === 'function') {
        for await (const chunk of agent.runStreaming(messages, options)) {
          if (controller.signal.aborted) return;
          if (chunk.type === 'delta' && chunk.delta) {
            text += chunk.delta;
            this.store.addArtifactChunk(record, { artifactId, name: 'response', parts: [{ text: chunk.delta }] }, !first, false);
            first = false;
          }
        }
        if (controller.signal.aborted) return;
        const parts = outputToParts(text, undefined);
        const artifact = record.task.artifacts.find((a) => a.artifactId === artifactId);
        if (artifact) artifact.parts = parts;
        else record.task.artifacts.push({ artifactId, name: 'response', parts });
        this.complete(record, parts);
        return;
      }
      const result = await agent.run(messages, options);
      if (controller.signal.aborted) return;
      const parts = outputToParts(result.content ?? '', result.content_items);
      this.store.addArtifactChunk(record, { artifactId, name: 'response', parts }, false, true);
      this.complete(record, parts);
    } catch (error) {
      if (controller.signal.aborted || isSettled(record.task.status.state)) return;
      const refusal = refusalResponse(error);
      const status = (error as { statusCode?: unknown; status_code?: unknown })?.statusCode ?? (error as { status_code?: unknown })?.status_code;
      const httpStatus = refusal?.status ?? (typeof status === 'number' ? status : undefined);
      const text = refusal ? refusal.body.error.message : safeReplyText(error, `${agent.name} a2a`);
      const reply: Message = {
        messageId: crypto.randomUUID(),
        contextId: record.task.contextId,
        taskId: record.task.id,
        role: 'ROLE_AGENT',
        parts: [{ text }],
      };
      this.store.setStatus(
        record,
        { state: 'TASK_STATE_FAILED', message: reply, timestamp: nowIso() },
        { error: { code: refusal?.body.error.code ?? 'run_failed', message: text, ...(httpStatus ? { httpStatus } : {}) } },
      );
    }
  }

  private complete(record: TaskRecord, parts: Message['parts']): void {
    const reply: Message = {
      messageId: crypto.randomUUID(),
      contextId: record.task.contextId,
      taskId: record.task.id,
      role: 'ROLE_AGENT',
      parts,
    };
    this.store.addHistory(record, reply);
    this.store.setStatus(record, { state: 'TASK_STATE_COMPLETED', message: reply, timestamp: nowIso() });
  }

  private cancel(record: TaskRecord): Task {
    if (isSettled(record.task.status.state)) {
      throw new A2AError('TASK_NOT_CANCELABLE', `Task ${record.task.id} is ${record.task.status.state}`, {
        taskId: record.task.id,
        state: record.task.status.state,
      });
    }
    record.abort?.();
    this.store.setStatus(record, { state: 'TASK_STATE_CANCELED', timestamp: nowIso() });
    return taskView(record.task);
  }

  /** Every event of `record` so far and then live, until a settled state. */
  private async *events(record: TaskRecord): AsyncGenerator<StreamResponse> {
    let index = 0;
    let wake: (() => void) | null = null;
    const listener = () => wake?.();
    record.listeners.add(listener);
    try {
      for (;;) {
        while (index < record.events.length) {
          const event = record.events[index++];
          yield event;
          if (endsStream(event)) return;
        }
        if (isSettled(record.task.status.state)) return;
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
        wake = null;
      }
    } finally {
      record.listeners.delete(listener);
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function safeUrl(request: Request): URL | undefined {
  try {
    return new URL(request.url);
  } catch {
    return undefined;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function endsStream(event: StreamResponse): boolean {
  if ('statusUpdate' in event) return isSettled(event.statusUpdate.status.state);
  return false;
}

function historyLengthOf(value: unknown): number | undefined {
  if (typeof value === 'number' && value >= 0) return Math.floor(value);
  if (typeof value === 'string' && value.trim() && Number.isFinite(Number(value)) && Number(value) >= 0) return Math.floor(Number(value));
  return undefined;
}

function listFilter(params: Record<string, unknown>): ListFilter {
  const pageSize = typeof params.pageSize === 'number' ? params.pageSize : typeof params.pageSize === 'string' ? Number(params.pageSize) : undefined;
  const status = typeof params.status === 'string' && params.status.startsWith('TASK_STATE_') ? (params.status as TaskState) : undefined;
  return {
    ...(typeof params.contextId === 'string' && params.contextId ? { contextId: params.contextId } : {}),
    ...(status ? { status } : {}),
    ...(pageSize && Number.isFinite(pageSize) ? { pageSize } : {}),
    ...(typeof params.pageToken === 'string' && params.pageToken ? { pageToken: params.pageToken } : {}),
    ...(historyLengthOf(params.historyLength) !== undefined ? { historyLength: historyLengthOf(params.historyLength) } : {}),
    ...(params.includeArtifacts === false || params.includeArtifacts === 'false' ? { includeArtifacts: false } : {}),
  };
}

/** The `{id}` of a `/a2a/tasks/{id}[:verb]` path, decoded. */
function taskIdOf(request: Request): string {
  const url = safeUrl(request);
  const path = url?.pathname ?? '';
  const marker = `${A2A_RPC_SUBPATH}/tasks/`;
  const at = path.lastIndexOf(marker);
  if (at === -1) return '';
  let rest = path.slice(at + marker.length);
  const colon = rest.lastIndexOf(':');
  if (colon !== -1 && ['cancel', 'subscribe'].includes(rest.slice(colon + 1))) rest = rest.slice(0, colon);
  try {
    return decodeURIComponent(rest);
  } catch {
    return rest;
  }
}

function taskPathSuffix(request: Request): string | null {
  const path = safeUrl(request)?.pathname ?? '';
  const colon = path.lastIndexOf(':');
  if (colon === -1 || path.lastIndexOf('/') > colon) return null;
  return path.slice(colon + 1);
}

function parseRestBody(raw: Uint8Array): Record<string, unknown> {
  if (raw.length === 0) return {};
  let body: unknown;
  try {
    body = JSON.parse(new TextDecoder().decode(raw));
  } catch {
    throw new A2AError('INVALID_PARAMS', 'Body is not valid JSON');
  }
  if (!isRecord(body)) throw new A2AError('INVALID_PARAMS', 'Body must be a JSON object');
  return body;
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });
}

function rpcResponse(id: JsonRpcId, body: { result?: unknown; error?: unknown }): Response {
  return json({ jsonrpc: JSON_RPC, id, ...body });
}

function restJson(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': REST_CONTENT_TYPE } });
}

function restError(error: A2AError): Response {
  return restJson(error.rest(), error.http);
}

function sseResponse<T>(events: AsyncGenerator<StreamResponse>, wrap: (event: StreamResponse) => T): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const event of events) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(wrap(event))}\n\n`));
        }
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
    cancel() {
      void events.return(undefined);
    },
  });
  return new Response(stream, {
    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' },
  });
}
