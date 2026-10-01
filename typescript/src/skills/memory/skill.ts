/**
 * The memory skill (gap-closure plan item 2.1, 2026-09-26): one skill, named
 * `memory` in an agent file, the same in both SDKs.
 *
 *     skills:
 *       - memory                                  # notes kept on this machine
 *       - memory: {local: true, portal: true}     # ... and on Robutler, synced
 *       - memory: {portal: true, local: false}    # on Robutler alone
 *
 * WHAT IT GIVES THE MODEL: `memory_search`, `memory_read`, `memory_write`,
 * `memory_forget` and `memory_list` (`definitions.ts`); an index of its notes
 * in the system prompt, one line each (key and description), frozen for the
 * session (`notes.ts`), from which `memory_read` gives a note in full; and the
 * summary of a compacted conversation kept as an episode (`onCompaction`;
 * compaction itself is the agent's, `core/context-compaction.ts`, since
 * 2026-09-29).
 *
 * SCOPED BY CALLER, BY CONSTRUCTION (`namespace.ts`). Every entry lives in a
 * namespace derived from the VERIFIED caller of the turn: `owner`, or
 * `caller:<principal>`. A tool call names no namespace it may not use: a
 * caller reads its own and `shared`, writes its own; the owner reads and
 * writes everything. One caller's `memory_search` therefore never returns
 * another's entries, which `tests/unit/skills/memory/memory-isolation-w2mem.test.ts`
 * pins, as `python/tests/agents/skills/test_memory_isolation_w2mem.py` does
 * for the Python twin (`skills/local/memory/caller_scoped.py`).
 *
 * TWO TIERS. `local` is Markdown files with a full-text index
 * (`local-store.ts`); `portal` is the platform's store with semantic search
 * (`portal-store.ts`). With both, reads are served locally, every local
 * change is pushed to the platform, and what changed on the platform is
 * pulled for the caller's namespaces at the start of a turn (throttled),
 * merged by entry id, last writer wins. The pull cursor is the platform's
 * opaque string, moved only by a pull (S-307, 2026-09-27: a push used to
 * answer a cursor past everything, and the agent never pulled again).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Skill } from '../../core/skill';
import { hook, prompt } from '../../core/decorators';
import type { AuthInfo, Context, HookData, SkillConfig, Tool } from '../../core/types';
import { requestSessionId } from '../session/skill';
import { resolveSkillPlatformUrl } from '../platform-url';
import { MEMORY_TOOL_DEFINITIONS } from './definitions';
import { COMPACTION_RUN } from '../../core/context-compaction';
import { LocalMemoryStore, type EntrySource, type ListOptions, type MemoryEntry, type MemoryLogLine } from './local-store';
import {
  isValidKey,
  keyRefusal,
  namespaceOf,
  OWNER_NAMESPACE,
  readableNamespaces,
  SHARED_NAMESPACE,
  targetNamespace,
} from './namespace';
import { DEFAULT_NOTES_BUDGET, NOTES_TITLES, renderNotes, type NotesSection } from './notes';
import { PortalMemoryStore } from './portal-store';

/** The `- memory: {...}` entry, checked; every sentence is the fixture's. */
export interface ParsedMemoryConfig {
  local: boolean;
  portal: boolean;
  notesBudget: number;
  /**
   * The setting from before compaction was the agent's (2026-09-29): still
   * read, so a file that has it loads, and its `threshold` becomes the
   * agent's `compaction.at` when the agent file has no `compaction:` block.
   */
  compaction: { threshold?: number; keep?: number };
}

const CONFIG_KEYS = new Set(['local', 'portal', 'notes_budget', 'compaction']);

export function parseMemoryConfig(raw: Record<string, unknown> | undefined): ParsedMemoryConfig {
  const config = raw ?? {};
  for (const key of Object.keys(config)) {
    if (!CONFIG_KEYS.has(key)) {
      throw new Error(`memory: unknown key "${key}". It takes local, portal, notes_budget and compaction.`);
    }
  }
  const flag = (name: 'local' | 'portal', fallback: boolean): boolean => {
    const value = config[name];
    if (value === undefined || value === null) return fallback;
    if (typeof value !== 'boolean') throw new Error(`memory: ${name} must be true or false.`);
    return value;
  };
  const local = flag('local', true);
  const portal = flag('portal', false);
  if (!local && !portal) throw new Error('memory: at least one of local and portal must be true.');
  let notesBudget = DEFAULT_NOTES_BUDGET;
  if (config.notes_budget !== undefined && config.notes_budget !== null) {
    if (typeof config.notes_budget !== 'number' || !(config.notes_budget > 0)) {
      throw new Error('memory: notes_budget must be a positive number of characters.');
    }
    notesBudget = Math.floor(config.notes_budget);
  }
  const compaction: { threshold?: number; keep?: number } = {};
  if (config.compaction !== undefined && config.compaction !== null) {
    if (typeof config.compaction !== 'object' || Array.isArray(config.compaction)) {
      throw new Error('memory: compaction must be a mapping of threshold and keep.');
    }
    const block = config.compaction as Record<string, unknown>;
    if (block.threshold !== undefined && block.threshold !== null) {
      if (typeof block.threshold !== 'number' || !(block.threshold > 0)) {
        throw new Error('memory: compaction.threshold must be a positive number of tokens.');
      }
      compaction.threshold = Math.floor(block.threshold);
    }
    if (block.keep !== undefined && block.keep !== null) {
      if (typeof block.keep !== 'number' || block.keep < 0) {
        throw new Error('memory: compaction.keep must be a number of messages.');
      }
      compaction.keep = Math.floor(block.keep);
    }
  }
  return { local, portal, notesBudget, compaction };
}

/** The owner's view of the memory (`MemorySkill.ownerSummary`, the chat's `/memory`). */
export interface MemoryOwnerSummary {
  local: boolean;
  portal: boolean;
  /**
   * Whether the Robutler tier has the agent's own key to reach it with (B5,
   * 2026-09-28): `/memory` said "and on Robutler" for a tier that had none,
   * so nothing ever reached Robutler.
   */
  portalKey: boolean;
  owner: number;
  shared: number;
  /** Distinct callers with notes, and their notes in all. */
  callers: number;
  callerNotes: number;
  /** The owner's newest notes, at most ten, with each note's first line. */
  recent: Array<{ key: string; firstLine: string; updatedAt: string }>;
}

export interface MemorySkillConfig extends SkillConfig {
  local?: boolean;
  portal?: boolean;
  notes_budget?: number;
  compaction?: { threshold?: number; keep?: number };
  /** The agent's folder (the resolver passes it); the local tier lives in its `.webagents/memory`. */
  agentDir?: string;
  /** The agent's name, for entry ids when it has no platform id. */
  agentName?: string;
  /** The agent's platform id: entry ids and the portal tier's store. */
  agentId?: string;
  portalUrl?: string;
  /** The agent's platform key; found (`server/agent-credential.ts`) when unset. */
  apiKey?: string;
  /** For tests: the portal tier's fetch, and the plain index. */
  fetchImpl?: typeof fetch;
  plainIndex?: boolean;
  now?: () => Date;
}

interface SyncState {
  pushedSeq: number;
  /** Per namespace, the platform's cursor as it was received (portal-store.ts `pull`). */
  cursors: Record<string, string>;
}

const PULL_INTERVAL_MS = 60_000;
/** Pages one pull takes per namespace, at most: the platform's key limit (10,000) in its pages of 500. */
const MAX_PULL_PAGES = 20;

/**
 * The cursors of a state file, keeping only the platform's strings. A number
 * is the retired sequence cursor (S-307: `Number.MAX_SAFE_INTEGER` after the
 * first push), which would stop every pull; it is dropped, so that
 * namespace's next pull starts over, which the merge makes harmless.
 */
function readCursors(raw: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
  for (const [namespace, cursor] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof cursor === 'string' && cursor) out[namespace] = cursor;
  }
  return out;
}
const MAX_FROZEN = 500;
/** A run that writes a compaction summary (the agent's, `core/context-compaction.ts`): the notes and the pull stay out of it. */
const COMPACTION_FLAG = COMPACTION_RUN;

const RESTRICTED_DENIED: ReadonlySet<string> = new Set(['memory_write', 'memory_forget']);

export class MemorySkill extends Skill {
  /** Reads are safe from a stranger's turn (S-030); the two writing tools deny themselves below. */
  static override restrictedPostureDefault = 'allow' as const;

  readonly tiers: { local: boolean; portal: boolean };
  readonly notesBudget: number;
  readonly compaction: { threshold?: number; keep?: number };
  readonly agentDir: string;
  private readonly agentName?: string;
  private readonly agentId?: string;
  private readonly portalUrl?: string;
  private readonly apiKey?: string;
  private readonly fetchImpl?: typeof fetch;
  private readonly plainIndex: boolean;
  private readonly now: () => Date;

  private local?: LocalMemoryStore;
  private portal?: PortalMemoryStore;
  private ready?: Promise<void>;
  private _agent?: { name?: string; run?: (messages: unknown[], options: Record<string, unknown>) => Promise<{ content?: string }> };
  private frozen = new Map<string, string>();
  private pulledAt = new Map<string, number>();
  /** Compactions this process ran, for the tests and `doctor`. */
  compactions = 0;

  constructor(config: MemorySkillConfig = {}) {
    super({ ...config, name: config.name || 'memory' });
    const parsed = parseMemoryConfig({
      ...(config.local !== undefined ? { local: config.local } : {}),
      ...(config.portal !== undefined ? { portal: config.portal } : {}),
      ...(config.notes_budget !== undefined ? { notes_budget: config.notes_budget } : {}),
      ...(config.compaction !== undefined ? { compaction: config.compaction } : {}),
    });
    this.tiers = { local: parsed.local, portal: parsed.portal };
    this.notesBudget = parsed.notesBudget;
    this.compaction = parsed.compaction;
    this.agentDir = config.agentDir ?? process.cwd();
    this.agentName = config.agentName;
    this.agentId = config.agentId;
    this.portalUrl = config.portalUrl;
    this.apiKey = config.apiKey;
    this.fetchImpl = config.fetchImpl;
    this.plainIndex = config.plainIndex ?? false;
    this.now = config.now ?? (() => new Date());
    for (const def of MEMORY_TOOL_DEFINITIONS) {
      const handler = {
        memory_search: (params: Record<string, unknown>, context: Context) => this.memorySearch(params, context),
        memory_read: (params: Record<string, unknown>, context: Context) => this.memoryRead(params, context),
        memory_write: (params: Record<string, unknown>, context: Context) => this.memoryWrite(params, context),
        memory_forget: (params: Record<string, unknown>, context: Context) => this.memoryForget(params, context),
        memory_list: (params: Record<string, unknown>, context: Context) => this.memoryList(params, context),
      }[def.function.name];
      if (!handler) continue;
      this.registerTool({
        name: def.function.name,
        description: def.function.description,
        parameters: def.function.parameters,
        scopes: ['all'],
        enabled: true,
        handler,
        ...(RESTRICTED_DENIED.has(def.function.name) ? { restrictedPosture: 'deny' as const } : {}),
      } as Tool);
    }
  }

  setAgent(agent: unknown): void {
    this._agent = agent as MemorySkill['_agent'];
  }

  /** What entry ids are computed over: the platform id, else the name. */
  get storeKey(): string {
    return this.agentId ?? this.agentName ?? this._agent?.name ?? 'agent';
  }

  get memoryRoot(): string {
    return path.join(this.agentDir, '.webagents', 'memory');
  }

  override async initialize(): Promise<void> {
    await this.ensureReady();
  }

  private ensureReady(): Promise<void> {
    if (!this.ready) this.ready = this.openStores();
    return this.ready;
  }

  private async openStores(): Promise<void> {
    if (this.tiers.local) {
      this.local = new LocalMemoryStore({ root: this.memoryRoot, store: this.storeKey, plainIndex: this.plainIndex, now: this.now });
      await this.local.open();
    }
    if (this.tiers.portal) {
      const portalUrl = await resolveSkillPlatformUrl(this.portalUrl);
      this.portal = new PortalMemoryStore({
        portalUrl,
        agentId: this.agentId,
        fetchImpl: this.fetchImpl,
        token: () => this.resolveToken(),
      });
      if (this.local) {
        // Everything this machine knows about, pulled once at start; the
        // caller's namespaces are pulled again per turn (`pullForCaller`).
        await this.pull(await this.local.namespaces()).catch(() => undefined);
        await this.push().catch(() => undefined);
      }
    }
  }

  private async resolveToken(): Promise<string | undefined> {
    if (this.apiKey) return this.apiKey;
    try {
      const { resolveAgentCredential } = await import('../../server/agent-credential.js');
      return (await resolveAgentCredential(this.agentName ?? this._agent?.name, { cwd: this.agentDir }))?.token;
    } catch {
      return undefined;
    }
  }

  override async cleanup(): Promise<void> {
    this.local?.close();
  }

  // -- the caller ------------------------------------------------------------

  private callerNamespace(context: Context | undefined): string | null {
    return namespaceOf(context?.auth as Partial<AuthInfo> | undefined);
  }

  // -- reads through the tiers ------------------------------------------------

  private async listEntries(namespaces: readonly string[] | null, options: ListOptions): Promise<MemoryEntry[]> {
    if (this.local) return this.local.list(namespaces, options);
    return this.portal!.list(namespaces, options);
  }

  private async searchEntries(query: string, namespaces: readonly string[] | null, limit: number): Promise<MemoryEntry[]> {
    const out: MemoryEntry[] = [];
    const seen = new Set<string>();
    const add = (entries: MemoryEntry[]) => {
      for (const e of entries) {
        if (seen.has(e.id) || (namespaces && !namespaces.includes(e.namespace))) continue;
        seen.add(e.id);
        out.push(e);
      }
    };
    if (this.local) add(await this.local.search(query, namespaces, limit));
    if (this.portal && out.length < limit) {
      try {
        add(await this.portal.search(query, namespaces, limit));
      } catch (err) {
        if (!this.local) throw err;
        // The platform's semantic leg is extra when the local index answered.
      }
    }
    return out.slice(0, limit);
  }

  // -- tools ----------------------------------------------------------------------

  async memorySearch(params: Record<string, unknown>, context: Context): Promise<unknown> {
    await this.ensureReady();
    const query = typeof params.query === 'string' ? params.query.trim() : '';
    if (!query) return { error: 'memory: query is required.' };
    const caller = this.callerNamespace(context);
    const readable = readableNamespaces(caller);
    let namespaces = readable;
    if (typeof params.namespace === 'string' && params.namespace.trim()) {
      const one = targetNamespace(caller, params.namespace, 'read');
      if (!one) return { error: 'memory: only the owner may name that namespace.' };
      namespaces = [one];
    }
    const limit = Math.min(Math.max(1, Math.floor(Number(params.limit) || 10)), 50);
    try {
      const entries = await this.searchEntries(query, namespaces, limit);
      return {
        entries: entries.map((e) => ({ key: e.key, namespace: e.namespace, description: e.description ?? '', content: e.content, updated_at: e.updatedAt })),
      };
    } catch (err) {
      return { error: (err as Error).message };
    }
  }

  /** One note in full (2026-09-29): what the memory index in the prompt names. */
  async memoryRead(params: Record<string, unknown>, context: Context): Promise<unknown> {
    await this.ensureReady();
    if (!isValidKey(params.key)) return { error: keyRefusal(params.key) };
    const caller = this.callerNamespace(context);
    if (!caller) return { error: 'memory: nothing is remembered for a caller nothing verified; only shared notes can be read.' };
    let namespaces: readonly string[];
    if (typeof params.namespace === 'string' && params.namespace.trim()) {
      const one = targetNamespace(caller, params.namespace, 'read');
      if (!one) return { error: 'memory: only the owner may name that namespace.' };
      namespaces = [one];
    } else {
      namespaces = readableNamespaces(caller) ?? [OWNER_NAMESPACE, SHARED_NAMESPACE];
    }
    try {
      for (const each of namespaces) {
        const entry = this.local ? await this.local.get(each, params.key) : await this.portal!.get(each, params.key);
        if (entry) {
          return { key: entry.key, namespace: entry.namespace, description: entry.description ?? '', content: entry.content, updated_at: entry.updatedAt };
        }
      }
    } catch (err) {
      return { error: (err as Error).message };
    }
    return { error: `memory: no note called ${params.key}.` };
  }

  async memoryWrite(params: Record<string, unknown>, context: Context): Promise<unknown> {
    await this.ensureReady();
    if (!isValidKey(params.key)) return { error: keyRefusal(params.key) };
    if (typeof params.content !== 'string') return { error: 'memory: content is required.' };
    const caller = this.callerNamespace(context);
    if (!caller) return { error: 'memory: nothing is remembered for a caller nothing verified; only shared notes can be read.' };
    // A non-owner's note goes into its own namespace whatever it asked for
    // (file comment): the parameter is honoured for the owner alone.
    const namespace = caller === OWNER_NAMESPACE ? (targetNamespace(caller, params.namespace, 'write') ?? OWNER_NAMESPACE) : caller;
    if (caller === OWNER_NAMESPACE && typeof params.namespace === 'string' && params.namespace.trim() && namespace !== params.namespace.trim()) {
      return { error: 'memory: namespace must be owner or shared.' };
    }
    try {
      const entry = await this.write(namespace, params.key, params.content, 'tool', typeof params.description === 'string' ? params.description : '');
      return { ok: true, id: entry.id, key: entry.key, namespace: entry.namespace, updated_at: entry.updatedAt };
    } catch (err) {
      return { error: (err as Error).message };
    }
  }

  async memoryForget(params: Record<string, unknown>, context: Context): Promise<unknown> {
    await this.ensureReady();
    if (!isValidKey(params.key)) return { error: keyRefusal(params.key) };
    const caller = this.callerNamespace(context);
    if (!caller) return { error: 'memory: nothing is remembered for a caller nothing verified; only shared notes can be read.' };
    const namespace = targetNamespace(caller, params.namespace, 'write');
    if (!namespace) return { error: 'memory: only the owner may name that namespace.' };
    try {
      const forgotten = await this.forget(namespace, params.key);
      return { ok: true, forgotten: forgotten ? 1 : 0 };
    } catch (err) {
      return { error: (err as Error).message };
    }
  }

  async memoryList(params: Record<string, unknown>, context: Context): Promise<unknown> {
    await this.ensureReady();
    const caller = this.callerNamespace(context);
    let namespaces = readableNamespaces(caller);
    if (typeof params.namespace === 'string' && params.namespace.trim()) {
      const one = targetNamespace(caller, params.namespace, 'read');
      if (!one) return { error: 'memory: only the owner may name that namespace.' };
      namespaces = [one];
    }
    const limit = Math.min(Math.max(1, Math.floor(Number(params.limit) || 50)), 200);
    const prefix = typeof params.prefix === 'string' ? params.prefix : undefined;
    try {
      const entries = await this.listEntries(namespaces, { prefix, limit });
      return { entries: entries.map((e) => ({ key: e.key, namespace: e.namespace, description: e.description ?? '', updated_at: e.updatedAt })) };
    } catch (err) {
      return { error: (err as Error).message };
    }
  }

  // -- writes through the tiers ------------------------------------------------

  /** Write in the tiers that are on: the local file first, then the platform (pushed from the log). */
  async write(namespace: string, key: string, content: string, source: EntrySource, description = ''): Promise<MemoryEntry> {
    await this.ensureReady();
    if (this.local) {
      const entry = await this.local.put(namespace, key, content, source, undefined, description);
      if (this.portal) await this.push().catch(() => undefined);
      return entry;
    }
    return this.portal!.put(namespace, key, content, source, undefined, description.split(/\s+/).filter(Boolean).join(' ').slice(0, 200));
  }

  async forget(namespace: string, key: string): Promise<boolean> {
    await this.ensureReady();
    if (this.local) {
      const removed = await this.local.forget(namespace, key);
      if (this.portal) await this.push().catch(() => undefined);
      return removed;
    }
    return this.portal!.forget(namespace, key);
  }

  // -- the owner's view (the chat's /memory, interactive-mode spec 3.7) --------------

  /**
   * What the agent remembers, for the person who owns it: where the notes are
   * kept, how many are theirs, shared, or a caller's, and their own newest
   * ten with each note's first line. The Python skill answers the same shape
   * (`caller_scoped.py` `owner_summary`).
   */
  async ownerSummary(): Promise<MemoryOwnerSummary> {
    await this.ensureReady();
    const entries = await this.listEntries(null, { limit: 100_000 });
    const callers = new Set<string>();
    let owner = 0;
    let shared = 0;
    let callerNotes = 0;
    for (const e of entries) {
      if (e.namespace === OWNER_NAMESPACE) owner += 1;
      else if (e.namespace === SHARED_NAMESPACE) shared += 1;
      else {
        callerNotes += 1;
        callers.add(e.namespace);
      }
    }
    const recent = entries
      .filter((e) => e.namespace === OWNER_NAMESPACE)
      .slice(0, 10)
      // The note's description when it has one, as the index shows it (2026-09-29).
      .map((e) => ({ key: e.key, firstLine: e.description || (e.content.split('\n').find((l) => l.trim())?.trim() ?? ''), updatedAt: e.updatedAt }));
    const portalKey = Boolean(this.portal) && Boolean(await this.resolveToken());
    return { local: Boolean(this.local), portal: Boolean(this.portal), portalKey, owner, shared, callers: callers.size, callerNotes, recent };
  }

  /** Whether the owner has a note called `key`. */
  async hasOwnNote(key: string): Promise<boolean> {
    await this.ensureReady();
    if (this.local) return (await this.local.get(OWNER_NAMESPACE, key)) !== null;
    return (await this.portal!.list([OWNER_NAMESPACE], { prefix: key, limit: 1000 })).some((e) => e.key === key);
  }

  /** Remove one of the owner's notes (`/memory forget <key>`); false when there is none. */
  async forgetOwn(key: string): Promise<boolean> {
    return this.forget(OWNER_NAMESPACE, key);
  }

  // -- sync ------------------------------------------------------------------------

  private get syncStateFile(): string {
    return path.join(this.memoryRoot, 'sync-state.json');
  }

  private readSyncState(): SyncState {
    try {
      const raw = JSON.parse(fs.readFileSync(this.syncStateFile, 'utf8')) as { pushedSeq?: unknown; cursors?: unknown };
      return { pushedSeq: typeof raw.pushedSeq === 'number' ? raw.pushedSeq : 0, cursors: readCursors(raw.cursors) };
    } catch {
      return { pushedSeq: 0, cursors: {} };
    }
  }

  private writeSyncState(state: SyncState): void {
    fs.mkdirSync(this.memoryRoot, { recursive: true, mode: 0o700 });
    fs.writeFileSync(this.syncStateFile, `${JSON.stringify(state)}\n`, { mode: 0o600 });
  }

  /** Push every local change the platform has not seen, one request per namespace. */
  async push(): Promise<number> {
    if (!this.local || !this.portal) return 0;
    const state = this.readSyncState();
    const lines = this.local.logSince(state.pushedSeq);
    if (!lines.length) return 0;
    const byNamespace = new Map<string, MemoryLogLine[]>();
    for (const line of lines) byNamespace.set(line.namespace, [...(byNamespace.get(line.namespace) ?? []), line]);
    let pushed = 0;
    for (const [namespace, group] of byNamespace) {
      // No cursor comes back and none is kept: our own lines return on the
      // next pull with the times we sent, and the merge keeps what is here.
      await this.portal.push(namespace, group);
      pushed += group.length;
    }
    state.pushedSeq = Math.max(state.pushedSeq, ...lines.map((l) => l.seq));
    this.writeSyncState(state);
    return pushed;
  }

  /** Pull what changed on the platform in `namespaces`, page by page, and apply what is newer here. */
  async pull(namespaces: readonly string[]): Promise<number> {
    if (!this.local || !this.portal) return 0;
    const state = this.readSyncState();
    let applied = 0;
    for (const namespace of namespaces) {
      let since: string | null = state.cursors[namespace] ?? null;
      for (let page = 0; page < MAX_PULL_PAGES; page += 1) {
        const { lines, cursor, more } = await this.portal.pull(namespace, since);
        applied += await this.local.apply(lines);
        since = cursor;
        if (!more) break;
      }
      // The platform's answer replaces what was kept, a null included: a
      // cursor it did not recognise is not kept for the next pull either.
      if (since) state.cursors[namespace] = since;
      else delete state.cursors[namespace];
      this.pulledAt.set(namespace, Date.now());
      this.writeSyncState(state);
    }
    return applied;
  }

  /** At the start of a turn with both tiers: the caller's namespaces, at most once a minute each. */
  @hook({ lifecycle: 'on_connection', priority: 60 })
  async pullForCaller(_data: HookData, context: Context): Promise<void> {
    if (!this.local || !this.portal) return;
    if ((context.metadata as Record<string, unknown> | undefined)?.[COMPACTION_FLAG]) return;
    await this.ensureReady();
    const caller = this.callerNamespace(context);
    const wanted = readableNamespaces(caller) ?? [OWNER_NAMESPACE, SHARED_NAMESPACE];
    const due = wanted.filter((ns) => Date.now() - (this.pulledAt.get(ns) ?? 0) > PULL_INTERVAL_MS);
    if (due.length) await this.pull(due).catch(() => undefined);
  }

  // -- the frozen notes --------------------------------------------------------------

  /** The sections a caller may see, in the order the prompt shows them. */
  async notesSectionsFor(caller: string | null): Promise<NotesSection[]> {
    const options: ListOptions = { excludeSources: ['compaction'], limit: 200 };
    const section = async (title: string, namespace: string): Promise<NotesSection> => ({
      title,
      entries: (await this.listEntries([namespace], options)).map((e) => ({ key: e.key, description: e.description ?? '', content: e.content })),
    });
    if (caller === OWNER_NAMESPACE) {
      return [await section(NOTES_TITLES.owner, OWNER_NAMESPACE), await section(NOTES_TITLES.shared, SHARED_NAMESPACE)];
    }
    const sections = [await section(NOTES_TITLES.shared, SHARED_NAMESPACE)];
    if (caller) sections.push(await section(NOTES_TITLES.caller, caller));
    return sections;
  }

  /**
   * The notes, rendered once per session (`metadata.session_id`, else once
   * per caller for the life of this process) and never again mid-session, so
   * the provider's prompt cache holds (plan principle 6).
   */
  @prompt({ priority: 40 })
  async frozenNotes(context: Context): Promise<string> {
    if ((context?.metadata as Record<string, unknown> | undefined)?.[COMPACTION_FLAG]) return '';
    await this.ensureReady();
    const caller = this.callerNamespace(context);
    const key = `${caller ?? ''}|${requestSessionId(context?.metadata) ?? ''}`;
    const cached = this.frozen.get(key);
    if (cached !== undefined) return cached;
    let text = '';
    try {
      text = renderNotes(await this.notesSectionsFor(caller), this.notesBudget);
    } catch {
      text = '';
    }
    if (this.frozen.size >= MAX_FROZEN) this.frozen.delete(this.frozen.keys().next().value as string);
    this.frozen.set(key, text);
    return text;
  }

  /** For tests and the chat's `/memory` view: forget what was frozen. */
  unfreezeNotes(): void {
    this.frozen.clear();
  }

  // -- compaction ----------------------------------------------------------------------
  //
  // Compaction is the agent's (2026-09-29, `core/context-compaction.ts`): the
  // conversation's owner compacts, once, and tells the skills. The hook that
  // did it here compacted the run's copy while the chat kept the whole
  // history, so every later turn paid for a new summary and saved another
  // episode.

  /**
   * A conversation was compacted: its summary is kept as an episode in the
   * caller's memory, once, so it is still searchable later. Between turns in
   * the chat there is no run, and the caller is the owner.
   */
  async onCompaction(outcome: { summary?: string }, context?: Context): Promise<void> {
    if (!outcome.summary) return;
    const caller = context ? this.callerNamespace(context) : OWNER_NAMESPACE;
    if (!caller) return;
    await this.ensureReady();
    const stamp = this.now().toISOString().replace(/[:.]/g, '-');
    try {
      await this.write(caller, `episode-${stamp}`, outcome.summary, 'compaction');
    } catch {
      return;
    }
    this.compactions += 1;
  }
}
