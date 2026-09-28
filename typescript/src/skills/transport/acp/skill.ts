/**
 * ACP (Agent Client Protocol) transport: the agent a code editor spawns
 * (gap-closure plan item 1.6, 2026-09-26). https://agentclientprotocol.com/
 *
 * `webagents acp [path]` builds the agent `serve` would build and hands it to
 * `ACPTransportSkill.serveStdio`: JSON-RPC 2.0 over the process's stdin and
 * stdout, one message per line, stdout reserved for protocol messages (the
 * CLI sends `console.log` to stderr before the agent file is read, and this
 * skill writes the wire straight to stdout). The Python twin is
 * `agents/skills/core/transport/acp/skill.py`; both are pinned by
 * `python/tests/fixtures/acp/acp_protocol.json` and driven end to end, as
 * spawned processes, by `python/tests/fixtures/acp/acp_transcripts.json`.
 *
 * WHAT IT ANSWERS: `initialize` (echoes protocol version 1, advertises only
 * what is served, and a `terminal` auth method that runs `webagents login`),
 * `authenticate`, `session/new` (the required `mcpServers` are attached to
 * the agent through the `mcp` client skill, in the session's `cwd`),
 * `session/prompt`, `session/cancel`, `$/cancel_request`, `session/load` (the
 * whole history replayed before the answer) and `session/list`. Sessions are
 * kept on disk under `sessionsDir` (the CLI passes the profile directory's
 * `acp/sessions`), so an editor that restarts the agent can load them again.
 * Requests and notifications are told apart by the PRESENCE of `id` (0 is an
 * id); anything else is `-32601`, the client's own `fs/*` and `terminal/*`
 * methods included: an agent calls those on its client, and serving them is
 * what S-269 was in the Python SDK.
 *
 * THE TURN, on the wire: the agent runs as the local owner (the person whose
 * editor this is), and its stream becomes `session/update` notifications:
 * text is `agent_message_chunk`, thinking is `agent_thought_chunk`, a tool
 * call is `tool_call` (with its `kind` from `protocol.toolKind`), then
 * `tool_call_update` as it runs and finishes, and a todo tool's list is the
 * `plan`. A tool whose kind edits, deletes, moves or executes asks the client
 * first (`session/request_permission`, after its `tool_call`); a refusal is
 * what the model is told, through the loop's `before_tool` abort, so the
 * turn goes on. `session/cancel` aborts the run's signal and the prompt
 * answers `stopReason: cancelled` after the last update it forwarded. A
 * failed run is the JSON-RPC error on `session/prompt`.
 *
 * NOT THE "AGENT COMMERCE PROTOCOL". This file used to be a homegrown stub
 * of that name, with four tools that fetched any URL the caller gave
 * (S-249). It is retired: `acp` is the Agent Client Protocol in both SDKs.
 */

import { Skill } from '../../../core/skill';
import { agentFinishOf, isAgentFinish } from '../../../core/tool-budget';
import { hook } from '../../../core/decorators';
import type { Context, HookData, HookResult, IAgent, ISkill, RunOptions, SkillConfig, StreamChunk } from '../../../core/types';
import { LOCAL_OWNER } from '../../../access/caller';
import * as P from './protocol';
import { AcpError } from './protocol';

export interface ACPTransportConfig extends SkillConfig {
  /** Where sessions are kept (`sessions_dir` in an agent file); unset: `~/.webagents/acp/sessions`. */
  sessionsDir?: string;
}

/** The settings as both SDKs read them back (snake_case, the fixture's names). */
export interface AcpSettings {
  sessions_dir: string | null;
}

export interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
}

/** A session as it is written to disk, the same bytes the Python SDK writes. */
export interface AcpSessionRecord {
  sessionId: string;
  cwd: string;
  agent: string;
  createdAt: string;
  updatedAt: string;
  title?: string;
  messages: SessionMessage[];
}

/** A session id an editor may hand back: never joined into a path otherwise. */
const SESSION_ID = /^[A-Za-z0-9._-]{1,128}$/;

/** A tool result that starts like this failed (`core/agent.ts`). */
const ERROR_PREFIXES = ['Tool execution error:', 'Error parsing tool arguments:'];

const now = (): string => new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const isAbsolutePath = (value: string): boolean => value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value);
const shortId = (): string => crypto.randomUUID().replace(/-/g, '').slice(0, 12);

type Lifecycle = { initialize?: () => Promise<void>; cleanup?: () => Promise<void> };
type SkillHost = { skills?: ISkill[]; addSkill?: (skill: ISkill) => void };

/** One editor session: where it works, and the conversation so far. */
export class AcpSession {
  title?: string;
  messages: SessionMessage[] = [];
  mcpAttached = false;

  constructor(
    readonly sessionId: string,
    public cwd: string,
    readonly agent: string,
    readonly createdAt: string,
    public updatedAt: string,
  ) {}

  toRecord(): AcpSessionRecord {
    return {
      sessionId: this.sessionId,
      cwd: this.cwd,
      agent: this.agent,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      ...(this.title ? { title: this.title } : {}),
      messages: this.messages.map((m) => ({ role: m.role, content: m.content })),
    };
  }

  static fromRecord(record: AcpSessionRecord): AcpSession {
    const session = new AcpSession(
      String(record.sessionId),
      String(record.cwd ?? ''),
      String(record.agent ?? ''),
      String(record.createdAt ?? now()),
      String(record.updatedAt ?? now()),
    );
    if (typeof record.title === 'string') session.title = record.title;
    session.messages = (Array.isArray(record.messages) ? record.messages : [])
      .filter(isRecord)
      .map((m) => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: String(m.content ?? '') }));
    return session;
  }

  listing(): Record<string, unknown> {
    return { sessionId: this.sessionId, cwd: this.cwd, updatedAt: this.updatedAt, ...(this.title ? { title: this.title } : {}) };
  }
}

/** Sessions as files, one per id, under `directory` (null keeps them in memory only). */
export class SessionStore {
  constructor(readonly directory: string | null) {}

  pathOf(sessionId: string): string | null {
    if (this.directory === null || !SESSION_ID.test(sessionId ?? '')) return null;
    return `${this.directory}/${sessionId}.json`;
  }

  async save(session: AcpSession): Promise<void> {
    const target = this.pathOf(session.sessionId);
    if (target === null) return;
    const fs = await import('node:fs/promises');
    await fs.mkdir(this.directory as string, { recursive: true });
    const tmp = `${target}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(session.toRecord(), null, 2), 'utf-8');
    await fs.rename(tmp, target);
  }

  async load(sessionId: string): Promise<AcpSession | null> {
    const target = this.pathOf(sessionId);
    if (target === null) return null;
    try {
      const fs = await import('node:fs/promises');
      const record: unknown = JSON.parse(await fs.readFile(target, 'utf-8'));
      if (!isRecord(record) || record.sessionId !== sessionId) return null;
      return AcpSession.fromRecord(record as unknown as AcpSessionRecord);
    } catch {
      return null;
    }
  }

  async listAll(): Promise<AcpSession[]> {
    if (this.directory === null) return [];
    let names: string[];
    try {
      const fs = await import('node:fs/promises');
      names = await fs.readdir(this.directory);
    } catch {
      return [];
    }
    const sessions: AcpSession[] = [];
    for (const name of names) {
      if (!name.endsWith('.json')) continue;
      const loaded = await this.load(name.slice(0, -'.json'.length));
      if (loaded) sessions.push(loaded);
    }
    return sessions;
  }
}

/**
 * One JSON-RPC connection: sends messages, and matches the client's answers
 * to the requests this agent made (`session/request_permission`).
 */
export class AcpConnection {
  private nextId = 0;
  private readonly pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();

  constructor(private readonly writeLine: (line: string) => void) {}

  send(message: Record<string, unknown>): void {
    this.writeLine(JSON.stringify(message));
  }

  respond(id: unknown, result: unknown): void {
    this.send({ jsonrpc: '2.0', id, result });
  }

  fail(id: unknown, error: AcpError): void {
    this.send({ jsonrpc: '2.0', id, error: error.toJSON() });
  }

  notify(method: string, params: Record<string, unknown>): void {
    this.send({ jsonrpc: '2.0', method, params });
  }

  request(method: string, params: Record<string, unknown>): Promise<unknown> {
    const id = ++this.nextId;
    return new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ jsonrpc: '2.0', id, method, params });
    });
  }

  /** A response from the client to one of this agent's requests. */
  resolve(message: Record<string, unknown>): void {
    const id = message.id;
    const waiting = typeof id === 'number' ? this.pending.get(id) : undefined;
    if (!waiting) return;
    this.pending.delete(id as number);
    if (message.error !== undefined && message.error !== null) {
      const error = isRecord(message.error) ? message.error : {};
      waiting.reject(new AcpError(Number(error.code ?? P.INTERNAL_ERROR), String(error.message ?? 'error'), error.data));
    } else {
      waiting.resolve(message.result);
    }
  }

  close(): void {
    for (const waiting of this.pending.values()) waiting.reject(new AcpError(P.INTERNAL_ERROR, 'The client closed the connection.'));
    this.pending.clear();
  }
}

/** A `session/prompt` in flight. */
interface PromptRun {
  session: AcpSession;
  requestId: unknown;
  controller: AbortController;
  cancelled: boolean;
  /** `$/cancel_request` answers `-32800`; `session/cancel` answers `stopReason: cancelled`. */
  cancelIsError: boolean;
  /** toolCallId -> tool name, for every call already announced. */
  announced: Map<string, string>;
  answer: string;
  error?: string;
  /** The turn's finish when the agent's tool budget ended it. */
  finish?: Record<string, unknown>;
  whenCancelled: Promise<void>;
  markCancelled: () => void;
}

/** This package's version, for `agentInfo`; `0.0.0` when it cannot be read. */
async function sdkVersion(): Promise<string> {
  try {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const here = path.dirname(fileURLToPath(import.meta.url));
    const pkg = JSON.parse(await fs.readFile(path.join(here, '..', '..', '..', '..', 'package.json'), 'utf-8')) as { version?: unknown };
    return typeof pkg.version === 'string' ? pkg.version : '0.0.0';
  } catch {
    return '0.0.0';
  }
}

export class ACPTransportSkill extends Skill {
  readonly settings: AcpSettings;
  private agent: IAgent | null = null;
  private connection: AcpConnection | null = null;
  private storeInstance: SessionStore | null = null;
  private readonly sessions = new Map<string, AcpSession>();
  private readonly runs = new Map<string, PromptRun>();
  /**
   * What the client said it can do in `initialize` (`fs`, `terminal`). Read
   * before this agent would call a client method; it calls none today, so a
   * client that advertises nothing loses nothing.
   */
  clientCapabilities: Record<string, unknown> = {};
  private readonly mcpSkills: Array<ISkill & Lifecycle> = [];
  private readonly tasks = new Set<Promise<void>>();

  constructor(config: ACPTransportConfig = {}) {
    super({ ...config, name: config.name || 'acp-transport' });
    this.settings = { sessions_dir: typeof config.sessionsDir === 'string' && config.sessionsDir ? config.sessionsDir : null };
  }

  /** `BaseAgent.addSkill` calls this. */
  setAgent(agent: IAgent): void {
    this.agent = agent;
  }

  get store(): SessionStore {
    if (!this.storeInstance) {
      const configured = this.settings.sessions_dir;
      const home = typeof process !== 'undefined' ? (process.env.HOME || process.env.USERPROFILE || '') : '';
      this.storeInstance = new SessionStore(configured ?? (home ? `${home}/.webagents/acp/sessions` : null));
    }
    return this.storeInstance;
  }

  // ---------------------------------------------------------------------
  // Serving
  // ---------------------------------------------------------------------

  /** Serve `agent` over `stdin` and `stdout` until the client closes stdin. */
  async serveStdio(agent: IAgent, stdin: NodeJS.ReadableStream = process.stdin, stdout: NodeJS.WritableStream = process.stdout): Promise<void> {
    const readline = await import('node:readline');
    const lines = readline.createInterface({ input: stdin, crlfDelay: Infinity });
    await this.serve(agent, lines, (line) => {
      stdout.write(`${line}\n`);
    });
  }

  /** Serve `agent` over any line source and sink (the stdio pair, or a test's). */
  async serve(agent: IAgent, lines: AsyncIterable<string>, writeLine: (line: string) => void): Promise<void> {
    this.agent = agent;
    const host = agent as unknown as SkillHost;
    if (!host.skills?.includes(this as unknown as ISkill)) host.addSkill?.(this as unknown as ISkill);
    await (agent as unknown as Lifecycle).initialize?.();
    this.connection = new AcpConnection(writeLine);
    console.error(`[webagents] ${agent.name}: ACP over stdio`);
    try {
      for await (const line of lines) this.onLine(line);
    } finally {
      await this.shutdown();
    }
  }

  private track(task: Promise<void>): void {
    const tracked = task.catch(() => undefined).finally(() => this.tasks.delete(tracked));
    this.tasks.add(tracked);
  }

  private onLine(line: string): void {
    const connection = this.connection as AcpConnection;
    const text = line.trim();
    if (!text) return;
    let message: unknown;
    try {
      message = JSON.parse(text);
    } catch {
      connection.fail(null, new AcpError(P.PARSE_ERROR, 'Parse error'));
      return;
    }
    if (!isRecord(message)) {
      connection.fail(null, new AcpError(P.INVALID_REQUEST, 'Invalid request'));
      return;
    }
    const method = message.method;
    const hasId = 'id' in message;
    const params = isRecord(message.params) ? message.params : {};
    if (typeof method === 'string') {
      if (hasId) this.track(this.answer(message.id, method, params));
      else this.track(this.notification(method, params));
    } else if (hasId && ('result' in message || 'error' in message)) {
      connection.resolve(message);
    } else {
      connection.fail(hasId ? message.id : null, new AcpError(P.INVALID_REQUEST, 'Invalid request'));
    }
  }

  private async answer(requestId: unknown, method: string, params: Record<string, unknown>): Promise<void> {
    const connection = this.connection as AcpConnection;
    try {
      connection.respond(requestId, await this.dispatch(requestId, method, params));
    } catch (err) {
      if (err instanceof AcpError) connection.fail(requestId, err);
      else connection.fail(requestId, new AcpError(P.INTERNAL_ERROR, (err as Error)?.message || String(err)));
    }
  }

  private async dispatch(requestId: unknown, method: string, params: Record<string, unknown>): Promise<unknown> {
    switch (method) {
      case 'initialize':
        return this.initializeMethod(params);
      case 'authenticate':
        return this.authenticate(params);
      case 'session/new':
        return this.sessionNew(params);
      case 'session/prompt':
        return this.sessionPrompt(requestId, params);
      case 'session/load':
        return this.sessionLoad(params);
      case 'session/list':
        return this.sessionList(params);
      default:
        throw new AcpError(P.METHOD_NOT_FOUND, `Method not found: ${method}`);
    }
  }

  private async notification(method: string, params: Record<string, unknown>): Promise<void> {
    if (method === 'session/cancel') {
      this.cancel(params.sessionId, false);
    } else if (method === '$/cancel_request') {
      for (const [sessionId, run] of this.runs) {
        if (run.requestId === params.requestId) this.cancel(sessionId, true);
      }
    }
  }

  private async shutdown(): Promise<void> {
    for (const sessionId of [...this.runs.keys()]) this.cancel(sessionId, false);
    await Promise.allSettled([...this.tasks]);
    for (const skill of this.mcpSkills.splice(0)) {
      try {
        await skill.cleanup?.();
      } catch {
        // shutting down
      }
    }
    this.connection?.close();
  }

  // ---------------------------------------------------------------------
  // Agent methods
  // ---------------------------------------------------------------------

  private async initializeMethod(params: Record<string, unknown>): Promise<Record<string, unknown>> {
    this.clientCapabilities = isRecord(params.clientCapabilities) ? params.clientCapabilities : {};
    return {
      protocolVersion: P.PROTOCOL_VERSION,
      agentCapabilities: P.AGENT_CAPABILITIES,
      agentInfo: { name: P.AGENT_INFO_NAME, title: this.agent?.name ?? '', version: await sdkVersion() },
      authMethods: P.AUTH_METHODS,
    };
  }

  private authenticate(params: Record<string, unknown>): Record<string, unknown> {
    if (!P.AUTH_METHODS.some((method) => method.id === params.methodId)) {
      throw new AcpError(P.INVALID_PARAMS, `Unknown auth method: ${JSON.stringify(params.methodId)}`);
    }
    return {};
  }

  private async sessionNew(params: Record<string, unknown>): Promise<Record<string, unknown>> {
    const cwd = cwdOf(params);
    if (!Array.isArray(params.mcpServers)) throw new AcpError(P.INVALID_PARAMS, 'mcpServers is required: an array, possibly empty');
    const stamp = now();
    const session = new AcpSession(`${P.SESSION_ID_PREFIX}${shortId()}`, cwd, this.agent?.name ?? '', stamp, stamp);
    this.sessions.set(session.sessionId, session);
    await this.attachMcp(session, params.mcpServers);
    await this.store.save(session);
    return { sessionId: session.sessionId };
  }

  private async sessionLoad(params: Record<string, unknown>): Promise<Record<string, unknown>> {
    const sessionId = params.sessionId;
    if (typeof sessionId !== 'string' || !SESSION_ID.test(sessionId)) throw new AcpError(P.INVALID_PARAMS, 'sessionId must be a string');
    const cwd = cwdOf(params);
    if (!Array.isArray(params.mcpServers)) throw new AcpError(P.INVALID_PARAMS, 'mcpServers is required: an array, possibly empty');
    const session = this.sessions.get(sessionId) ?? (await this.store.load(sessionId));
    if (!session) throw new AcpError(P.RESOURCE_NOT_FOUND, `Session not found: ${sessionId}`);
    session.cwd = cwd;
    this.sessions.set(sessionId, session);
    await this.attachMcp(session, params.mcpServers);
    for (const message of session.messages) {
      if (!message.content) continue;
      const sessionUpdate = message.role === 'assistant' ? 'agent_message_chunk' : 'user_message_chunk';
      this.update(session, { sessionUpdate, content: { type: 'text', text: message.content } });
    }
    return {};
  }

  private async sessionList(params: Record<string, unknown>): Promise<Record<string, unknown>> {
    const cwd = typeof params.cwd === 'string' ? params.cwd : null;
    const byId = new Map<string, AcpSession>();
    for (const session of await this.store.listAll()) byId.set(session.sessionId, session);
    for (const [id, session] of this.sessions) byId.set(id, session);
    const sessions = [...byId.values()].filter((s) => cwd === null || s.cwd === cwd);
    sessions.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
    return { sessions: sessions.map((s) => s.listing()) };
  }

  private async sessionPrompt(requestId: unknown, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    const sessionId = params.sessionId;
    const session = typeof sessionId === 'string' ? this.sessions.get(sessionId) : undefined;
    if (!session) throw new AcpError(P.RESOURCE_NOT_FOUND, `Session not found: ${String(sessionId)}`);
    if (this.runs.has(session.sessionId)) throw new AcpError(P.INTERNAL_ERROR, 'A prompt is already running for this session.');
    const text = P.promptText(params.prompt);
    let markCancelled: () => void = () => undefined;
    const whenCancelled = new Promise<void>((resolve) => {
      markCancelled = resolve;
    });
    const run: PromptRun = {
      session,
      requestId,
      controller: new AbortController(),
      cancelled: false,
      cancelIsError: false,
      announced: new Map(),
      answer: '',
      whenCancelled,
      markCancelled,
    };
    this.runs.set(session.sessionId, run);
    session.messages.push({ role: 'user', content: text });
    if (!session.title && text.trim()) session.title = text.trim().split('\n')[0].slice(0, 60);
    try {
      await this.runPrompt(run);
      if (run.cancelled) {
        if (run.cancelIsError) throw new AcpError(P.REQUEST_CANCELLED, 'Request cancelled');
        return { stopReason: P.STOP_CANCELLED };
      }
      if (run.error !== undefined) throw new AcpError(P.INTERNAL_ERROR, run.error);
      // The agent's tool budget ended the turn (2026-09-28): ACP's own
      // reason, and the precise one beside it, as the Python agent answers.
      if (run.finish) return { stopReason: P.STOP_MAX_TURN_REQUESTS, _meta: { webagents_finish: run.finish } };
      return { stopReason: P.STOP_END_TURN };
    } finally {
      this.runs.delete(session.sessionId);
      if (run.answer) session.messages.push({ role: 'assistant', content: run.answer });
      session.updatedAt = now();
      await this.store.save(session);
    }
  }

  // ---------------------------------------------------------------------
  // The turn
  // ---------------------------------------------------------------------

  private async runPrompt(run: PromptRun): Promise<void> {
    const agent = this.agent as IAgent;
    const options: RunOptions = {
      auth: { ...LOCAL_OWNER },
      signal: run.controller.signal,
      sessionData: { acp_session: run.session.sessionId },
    };
    const messages = run.session.messages.map((m) => ({ role: m.role, content: m.content }));
    const iterator = agent.runStreaming(messages, options)[Symbol.asyncIterator]();
    const cancelled = run.whenCancelled.then(() => 'cancelled' as const);
    for (;;) {
      const step = await Promise.race([iterator.next(), cancelled]);
      if (step === 'cancelled' || run.cancelled) {
        // The run is aborted through its signal; whatever it still yields is
        // not forwarded, and the prompt answers after the last update sent.
        void Promise.resolve(iterator.return?.(undefined)).catch(() => undefined);
        return;
      }
      if (step.done) return;
      this.onChunk(run, step.value);
    }
  }

  private onChunk(run: PromptRun, chunk: StreamChunk): void {
    switch (chunk.type) {
      case 'delta':
        if (chunk.delta) {
          run.answer += chunk.delta;
          this.update(run.session, { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: chunk.delta } });
        }
        return;
      case 'thinking':
        if (chunk.thinking?.content) {
          this.update(run.session, { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: chunk.thinking.content } });
        }
        return;
      case 'tool_call':
        if (chunk.tool_call) this.announce(run, chunk.tool_call.id, chunk.tool_call.name, P.parseArguments(chunk.tool_call.arguments));
        return;
      case 'tool_result':
        if (chunk.tool_result) this.finishTool(run, chunk.tool_result.call_id, Boolean(chunk.tool_result.is_error), chunk.tool_result.result);
        return;
      case 'error': {
        // The last, tool-less call brought no answer: the turn still ends
        // by the budget, not as a failure (`core/tool-budget.ts`).
        const ended = agentFinishOf(chunk.error);
        if (ended) {
          run.finish = { ...ended, blocked: false, retried: false };
          return;
        }
        run.error = chunk.error?.message ?? 'The run failed.';
        return;
      }
      case 'done': {
        const finish = chunk.response?.finish;
        if (finish && isAgentFinish(finish.reason)) {
          run.finish = { reason: finish.reason, blocked: false, retried: false, ...(finish.rounds !== undefined ? { rounds: finish.rounds } : {}), ...(finish.tool ? { tool: finish.tool } : {}) };
        }
        return;
      }
      default:
        return;
    }
  }

  private announce(run: PromptRun, callId: string, name: string, rawInput: Record<string, unknown>): void {
    if (run.announced.has(callId)) return;
    run.announced.set(callId, name);
    this.update(run.session, {
      sessionUpdate: 'tool_call',
      toolCallId: callId,
      title: name,
      kind: P.toolKind(name),
      status: 'pending',
      rawInput,
    });
  }

  private finishTool(run: PromptRun, callId: string, isError: boolean, result: unknown): void {
    const text = typeof result === 'string' ? result : JSON.stringify(result ?? '');
    const failed = isError || ERROR_PREFIXES.some((prefix) => text.startsWith(prefix));
    this.update(run.session, {
      sessionUpdate: 'tool_call_update',
      toolCallId: callId,
      status: failed ? 'failed' : 'completed',
      content: [P.textContent(text)],
      rawOutput: text,
    });
    if ((run.announced.get(callId) ?? '').startsWith('todo')) {
      const entries = P.planEntries(this.todoItems());
      if (entries) this.update(run.session, { sessionUpdate: 'plan', entries });
    }
  }

  private todoItems(): unknown {
    for (const skill of (this.agent as unknown as SkillHost)?.skills ?? []) {
      if (skill?.constructor?.name === 'TodoSkill') {
        const items = (skill as unknown as { getItems?: () => unknown }).getItems;
        return typeof items === 'function' ? items.call(skill) : undefined;
      }
    }
    return undefined;
  }

  /**
   * Announce the call, ask the client when its kind needs permission, and
   * abort a refused tool with the sentence the model is told (file comment).
   */
  @hook({ lifecycle: 'before_tool', priority: 1 })
  async acpBeforeTool(data: HookData, context: Context): Promise<HookResult | void> {
    const sessionId = context.get<string>('acp_session');
    const run = sessionId ? this.runs.get(sessionId) : undefined;
    if (!run || !this.connection) return;
    const name = data.tool_name ?? '';
    const call = context.get<{ id?: unknown }>('tool_call');
    const callId = typeof call?.id === 'string' && call.id ? call.id : `call_${shortId().slice(0, 8)}`;
    const rawInput = P.parseArguments(data.tool_params ?? {});
    const kind = P.toolKind(name);
    this.announce(run, callId, name, rawInput);
    if (P.needsPermission(kind)) {
      const asked = this.connection.request('session/request_permission', {
        sessionId: run.session.sessionId,
        toolCall: { toolCallId: callId, title: name, kind, status: 'pending', rawInput },
        options: P.PERMISSION_OPTIONS,
      });
      const outcome = await Promise.race([asked, run.whenCancelled.then(() => ({ outcome: { outcome: 'cancelled' } }))]);
      const decision = decisionOf(outcome);
      if (decision !== 'allow') {
        return { abort: true, abort_reason: decision === 'cancelled' ? P.cancelled(name) : P.rejected(name) };
      }
    }
    this.update(run.session, { sessionUpdate: 'tool_call_update', toolCallId: callId, status: 'in_progress' });
  }

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------

  private update(session: AcpSession, update: Record<string, unknown>): void {
    this.connection?.notify('session/update', { sessionId: session.sessionId, update });
  }

  private cancel(sessionId: unknown, isError: boolean): void {
    const run = typeof sessionId === 'string' ? this.runs.get(sessionId) : undefined;
    if (!run || run.cancelled) return;
    run.cancelled = true;
    run.cancelIsError = isError;
    run.controller.abort();
    run.markCancelled();
  }

  /**
   * The session's MCP servers, as an `mcp` client skill on the agent, started
   * in the session's `cwd`. A server that fails is said on stderr and the
   * session still opens: an editor's optional server must not make the agent
   * unusable.
   */
  private async attachMcp(session: AcpSession, entries: unknown[]): Promise<void> {
    const servers = P.mcpServersConfig(entries);
    if (!Object.keys(servers).length || session.mcpAttached) return;
    for (const server of Object.values(servers)) {
      if (typeof server.command === 'string' && server.cwd === undefined) server.cwd = session.cwd;
    }
    try {
      const { MCPSkill, loadMcpSdk } = await import('../../mcp/skill.js');
      await loadMcpSdk();
      const skill = new MCPSkill({ mcp: servers as never, agentName: this.agent?.name, baseDir: session.cwd });
      // Started BEFORE it is added: `addSkill` snapshots a skill's tools, and
      // the MCP skill only has its servers' tools once it has connected.
      await skill.initialize();
      (this.agent as unknown as SkillHost).addSkill?.(skill as unknown as ISkill);
      this.mcpSkills.push(skill as unknown as ISkill & Lifecycle);
      session.mcpAttached = true;
    } catch (err) {
      console.error(`[webagents] ACP session ${session.sessionId}: MCP servers not attached: ${(err as Error)?.message ?? String(err)}`);
    }
  }
}

function cwdOf(params: Record<string, unknown>): string {
  const cwd = params.cwd;
  if (typeof cwd !== 'string' || !isAbsolutePath(cwd)) throw new AcpError(P.INVALID_PARAMS, 'cwd must be an absolute path');
  return cwd;
}

/** `allow`, `reject` or `cancelled` from a `session/request_permission` result. */
function decisionOf(outcome: unknown): 'allow' | 'reject' | 'cancelled' {
  const selected = isRecord(outcome) && isRecord(outcome.outcome) ? outcome.outcome : null;
  if (!selected) return 'reject';
  if (selected.outcome === 'cancelled') return 'cancelled';
  const option = P.PERMISSION_OPTIONS.find((o) => o.optionId === selected.optionId);
  return option?.kind.startsWith('allow') ? 'allow' : 'reject';
}
