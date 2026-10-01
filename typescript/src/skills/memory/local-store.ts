/**
 * The local memory tier (gap-closure plan item 2.1, principle 3, 2026-09-26):
 * Markdown files a person can read and edit, indexed for search, with an
 * append-only log the portal sync reads.
 *
 * LAYOUT, under `<agent dir>/.webagents/memory/`:
 *
 *     owner/<key>.md                 the owner's notes
 *     shared/<key>.md                notes the owner shares with every caller
 *     callers/<hash>/<key>.md        one folder per verified caller (`namespace.ts`)
 *     callers/<hash>/caller.json     which caller: {"principal": "user:..."}
 *     index.db                       the full-text index (node:sqlite, FTS5)
 *     log.jsonl                      every local change, in order, for the sync
 *
 * THE FILES ARE THE TRUTH. Each file carries its entry's front matter (id,
 * key, namespace, source, created_at, updated_at) and its content. The index
 * is rebuilt from the files every time the store opens, so a note edited or
 * dropped in by hand is found on the next start, and a lost or corrupt index
 * costs nothing. `node:sqlite` is Node's own SQLite (22.13 and later, no
 * dependency); where the running Node has none, or its SQLite has no FTS5, a
 * plain in-memory index ranks by matched words instead. Embeddings are the
 * portal tier's, never a local dependency.
 *
 * The Python twin is `python/webagents/agents/skills/local/memory/local_memory_store.py`,
 * writing the same files, so an agent folder can move between the CLIs.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { CALLER_PREFIX, entryIdFor, isValidKey, keyRefusal, localDirOf, NAMESPACE_RE } from './namespace';

export type EntrySource = 'tool' | 'compaction' | 'owner' | 'sync';

export interface MemoryEntry {
  id: string;
  namespace: string;
  key: string;
  content: string;
  source: EntrySource;
  createdAt: string;
  updatedAt: string;
  /**
   * One line saying what the note is for, shown in the memory index in the
   * prompt (2026-09-29); empty for a note written without one, whose first
   * line stands in for it (`notes.ts` `noteLine`).
   */
  description?: string;
}

/** What search reads: the description, then the note. */
export function indexedText(entry: MemoryEntry): string {
  return entry.description ? `${entry.description}\n${entry.content}` : entry.content;
}

/** `text` as one line of at most `limit` characters: what a description may be. */
export function oneLine(text: string, limit = 200): string {
  const flat = String(text ?? '').split(/\s+/).filter(Boolean).join(' ');
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1).trimEnd()}…`;
}

/** One change, as the log keeps it and as the sync moves it. */
export interface MemoryLogLine {
  seq: number;
  op: 'put' | 'delete';
  id: string;
  namespace: string;
  key: string;
  content?: string;
  description?: string;
  source?: EntrySource;
  at: string;
}

export interface ListOptions {
  prefix?: string;
  limit?: number;
  excludeSources?: readonly EntrySource[];
}

/** What the store asks of an index: ids ranked for a query, within namespaces. */
export interface MemoryIndex {
  reset(entries: readonly MemoryEntry[]): void;
  upsert(entry: MemoryEntry): void;
  remove(id: string): void;
  search(query: string, namespaces: readonly string[] | null, limit: number): string[];
  close(): void;
}

const SOURCES: ReadonlySet<string> = new Set(['tool', 'compaction', 'owner', 'sync']);

/** The words of a query, lower-cased, without repeats; `[]` for a query with none. */
export function queryTokens(query: string): string[] {
  const found = query.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  return [...new Set(found)];
}

function sourceOf(value: unknown): EntrySource {
  return typeof value === 'string' && SOURCES.has(value) ? (value as EntrySource) : 'tool';
}

// ---------------------------------------------------------------------------
// Front matter
// ---------------------------------------------------------------------------

export function renderEntryFile(entry: MemoryEntry): string {
  return [
    '---',
    `id: ${entry.id}`,
    `key: ${entry.key}`,
    ...(entry.description ? [`description: ${oneLine(entry.description)}`] : []),
    `namespace: ${entry.namespace}`,
    `source: ${entry.source}`,
    `created_at: ${entry.createdAt}`,
    `updated_at: ${entry.updatedAt}`,
    '---',
    entry.content,
    '',
  ].join('\n');
}

/** The entry in a file: its front matter's, or a note written by hand. */
export function parseEntryFile(text: string, fallback: { namespace: string; key: string; store: string }): MemoryEntry | null {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text);
  if (!m) {
    // A NOTE WRITTEN BY HAND (2026-09-27). The docs say the files can be
    // edited by hand, and a plain `.md` with no front matter was ignored
    // without a word. Under a folder whose namespace is known (`owner/`,
    // `shared/`, a caller folder with its caller.json) it is that namespace's
    // note, keyed by its file name and marked `owner`; it gets front matter
    // the next time the store writes it. A file in a caller folder that
    // names no caller stays nobody's, as below. The Python store reads it the
    // same way (`python/tests/fixtures/memory_tool/chat_fixes_hand_written_note.json`).
    if (!NAMESPACE_RE.test(fallback.namespace) || !text.trim()) return null;
    const now = new Date().toISOString();
    return {
      id: entryIdFor(fallback.store, fallback.namespace, fallback.key),
      namespace: fallback.namespace,
      key: fallback.key,
      content: text.replace(/\n+$/, ''),
      source: 'owner',
      createdAt: now,
      updatedAt: now,
    };
  }
  const fields: Record<string, string> = {};
  for (const line of m[1].split(/\r?\n/)) {
    const colon = line.indexOf(':');
    if (colon <= 0) continue;
    fields[line.slice(0, colon).trim()] = line.slice(colon + 1).trim();
  }
  const key = isValidKey(fields.key) ? fields.key : fallback.key;
  const namespace =
    fields.namespace && NAMESPACE_RE.test(fields.namespace)
      ? fields.namespace
      : NAMESPACE_RE.test(fallback.namespace)
        ? fallback.namespace
        : null;
  // A file in a caller folder that names no caller (no caller.json, no
  // namespace line) is nobody's: it is left alone rather than misfiled.
  if (!namespace) return null;
  const content = m[2].replace(/\n$/, '');
  const now = new Date().toISOString();
  return {
    id: entryIdFor(fallback.store, namespace, key),
    namespace,
    key,
    content,
    source: sourceOf(fields.source),
    createdAt: fields.created_at || now,
    updatedAt: fields.updated_at || now,
    ...(fields.description ? { description: fields.description } : {}),
  };
}

// ---------------------------------------------------------------------------
// Indexes
// ---------------------------------------------------------------------------

/** Ranks by matched words: a word in the key counts double. Used where SQLite is not. */
export class PlainMemoryIndex implements MemoryIndex {
  private entries = new Map<string, MemoryEntry>();

  reset(entries: readonly MemoryEntry[]): void {
    this.entries = new Map(entries.map((e) => [e.id, e]));
  }

  upsert(entry: MemoryEntry): void {
    this.entries.set(entry.id, entry);
  }

  remove(id: string): void {
    this.entries.delete(id);
  }

  search(query: string, namespaces: readonly string[] | null, limit: number): string[] {
    const tokens = queryTokens(query);
    if (!tokens.length) return [];
    const scored: Array<{ id: string; score: number; updatedAt: string }> = [];
    for (const e of this.entries.values()) {
      if (namespaces && !namespaces.includes(e.namespace)) continue;
      const key = e.key.toLowerCase();
      const content = indexedText(e).toLowerCase();
      let score = 0;
      for (const t of tokens) {
        if (key.includes(t)) score += 2;
        if (content.includes(t)) score += 1;
      }
      if (score > 0) scored.push({ id: e.id, score, updatedAt: e.updatedAt });
    }
    scored.sort((a, b) => b.score - a.score || (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
    return scored.slice(0, limit).map((s) => s.id);
  }

  close(): void {
    this.entries.clear();
  }
}

interface SqliteStatement {
  run(...params: unknown[]): unknown;
  all(...params: unknown[]): Array<Record<string, unknown>>;
}

interface SqliteDatabase {
  exec(sql: string): void;
  prepare(sql: string): SqliteStatement;
  close(): void;
}

/** FTS5 in Node's own SQLite. `open` answers null where that is not available (file comment). */
export class SqliteMemoryIndex implements MemoryIndex {
  private constructor(private readonly db: SqliteDatabase) {}

  static async open(file: string): Promise<SqliteMemoryIndex | null> {
    const previousEmit = process.emitWarning;
    let db: SqliteDatabase;
    try {
      // Node prints an ExperimentalWarning the first time `node:sqlite`
      // loads. Only that one is kept off the terminal, and only while this
      // loads: a warning about anything else still goes through.
      process.emitWarning = ((warning: unknown, ...rest: unknown[]) => {
        const text = typeof warning === 'string' ? warning : (warning as { message?: string } | undefined)?.message;
        if (typeof text === 'string' && text.includes('SQLite')) return;
        return (previousEmit as (...args: unknown[]) => void).call(process, warning, ...rest);
      }) as typeof process.emitWarning;
      // `process.getBuiltinModule` (Node 22.3 and later) is asked first: a
      // bundler or a test runner's module loader may not know `node:sqlite`
      // as a builtin, and the dynamic import then fails where Node itself
      // would not. The import stays as the fallback for an older Node.
      const builtin = (process as unknown as { getBuiltinModule?: (id: string) => unknown }).getBuiltinModule;
      let mod = (typeof builtin === 'function' ? builtin.call(process, 'node:sqlite') : undefined) as
        | { DatabaseSync?: new (path: string) => SqliteDatabase }
        | undefined;
      if (!mod) {
        mod = (await import(/* @vite-ignore */ 'node:sqlite' as string)) as { DatabaseSync?: new (path: string) => SqliteDatabase };
      }
      if (typeof mod?.DatabaseSync !== 'function') return null;
      db = new mod.DatabaseSync(file);
    } catch {
      return null;
    } finally {
      process.emitWarning = previousEmit;
    }
    try {
      db.exec(
        "CREATE VIRTUAL TABLE IF NOT EXISTS entries USING fts5(id UNINDEXED, namespace UNINDEXED, key, content, tokenize='unicode61')",
      );
    } catch {
      // No FTS5 in this build of SQLite: the plain index serves instead.
      try {
        db.close();
      } catch {
        // already closed
      }
      return null;
    }
    return new SqliteMemoryIndex(db);
  }

  reset(entries: readonly MemoryEntry[]): void {
    this.db.exec('DELETE FROM entries');
    const insert = this.db.prepare('INSERT INTO entries (id, namespace, key, content) VALUES (?, ?, ?, ?)');
    for (const e of entries) insert.run(e.id, e.namespace, e.key, indexedText(e));
  }

  upsert(entry: MemoryEntry): void {
    this.db.prepare('DELETE FROM entries WHERE id = ?').run(entry.id);
    this.db.prepare('INSERT INTO entries (id, namespace, key, content) VALUES (?, ?, ?, ?)').run(entry.id, entry.namespace, entry.key, indexedText(entry));
  }

  remove(id: string): void {
    this.db.prepare('DELETE FROM entries WHERE id = ?').run(id);
  }

  search(query: string, namespaces: readonly string[] | null, limit: number): string[] {
    const tokens = queryTokens(query);
    if (!tokens.length) return [];
    // Each word as a quoted prefix term, any of them: a search is a lookup,
    // not a boolean expression, and quoting keeps FTS5 syntax out of it.
    const match = tokens.map((t) => `"${t.replace(/"/g, '""')}"*`).join(' OR ');
    const params: unknown[] = [match];
    let where = 'entries MATCH ?';
    if (namespaces) {
      if (!namespaces.length) return [];
      where += ` AND namespace IN (${namespaces.map(() => '?').join(', ')})`;
      params.push(...namespaces);
    }
    params.push(limit);
    const rows = this.db.prepare(`SELECT id FROM entries WHERE ${where} ORDER BY bm25(entries) LIMIT ?`).all(...params);
    return rows.map((r) => String(r.id));
  }

  close(): void {
    try {
      this.db.close();
    } catch {
      // already closed
    }
  }
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

function writePrivate(file: string, text: string): void {
  const temp = `${file}.${process.pid}.${Math.random().toString(16).slice(2, 10)}.tmp`;
  fs.writeFileSync(temp, text, { mode: 0o600 });
  fs.renameSync(temp, file);
}

export interface LocalMemoryStoreOptions {
  /** `.webagents/memory` of the agent's folder. */
  root: string;
  /** What entry ids are computed over: the agent's platform id, else its name (`namespace.ts`). */
  store: string;
  /** For tests: force the plain index. */
  plainIndex?: boolean;
  now?: () => Date;
}

export class LocalMemoryStore {
  readonly root: string;
  readonly store: string;
  private readonly now: () => Date;
  private readonly forcePlain: boolean;
  private entries = new Map<string, MemoryEntry>();
  private index: MemoryIndex = new PlainMemoryIndex();
  private lastSeq = 0;
  private opened: Promise<void> | null = null;
  /** Which index serves: `sqlite` or `plain`, for `doctor` and the tests. */
  indexKind: 'sqlite' | 'plain' = 'plain';

  constructor(options: LocalMemoryStoreOptions) {
    this.root = options.root;
    this.store = options.store;
    this.now = options.now ?? (() => new Date());
    this.forcePlain = options.plainIndex ?? false;
  }

  get logFile(): string {
    return path.join(this.root, 'log.jsonl');
  }

  /** Opens once: makes the folders, reads the files, rebuilds the index, finds the log's last seq. */
  open(): Promise<void> {
    if (!this.opened) this.opened = this._open();
    return this.opened;
  }

  private async _open(): Promise<void> {
    fs.mkdirSync(this.root, { recursive: true, mode: 0o700 });
    this.entries = new Map();
    for (const entry of this.readAllFiles()) this.entries.set(entry.id, entry);
    if (!this.forcePlain) {
      const sqlite = await SqliteMemoryIndex.open(path.join(this.root, 'index.db'));
      if (sqlite) {
        this.index = sqlite;
        this.indexKind = 'sqlite';
      }
    }
    this.index.reset([...this.entries.values()]);
    this.lastSeq = 0;
    for (const line of this.readLog()) this.lastSeq = Math.max(this.lastSeq, line.seq);
  }

  close(): void {
    this.index.close();
    this.opened = null;
  }

  // -- files ---------------------------------------------------------------

  private dirFor(namespace: string): string {
    return path.join(this.root, localDirOf(namespace));
  }

  private fileFor(namespace: string, key: string): string {
    return path.join(this.dirFor(namespace), `${key}.md`);
  }

  private *readAllFiles(): Generator<MemoryEntry> {
    const dirs: Array<{ dir: string; namespace: string }> = [];
    for (const name of ['owner', 'shared']) dirs.push({ dir: path.join(this.root, name), namespace: name });
    const callers = path.join(this.root, 'callers');
    if (fs.existsSync(callers)) {
      for (const hash of fs.readdirSync(callers)) {
        const dir = path.join(callers, hash);
        let namespace = '';
        try {
          const meta = JSON.parse(fs.readFileSync(path.join(dir, 'caller.json'), 'utf8')) as { principal?: unknown };
          if (typeof meta.principal === 'string') namespace = `${CALLER_PREFIX}${meta.principal}`;
        } catch {
          // no caller.json: the files' own front matter names the namespace
        }
        dirs.push({ dir, namespace });
      }
    }
    for (const { dir, namespace } of dirs) {
      if (!fs.existsSync(dir)) continue;
      for (const file of fs.readdirSync(dir)) {
        if (!file.endsWith('.md')) continue;
        const key = file.slice(0, -3);
        if (!isValidKey(key)) continue;
        let text: string;
        try {
          text = fs.readFileSync(path.join(dir, file), 'utf8');
        } catch {
          continue;
        }
        const entry = parseEntryFile(text, { namespace, key, store: this.store });
        if (entry && (namespace ? entry.namespace === namespace : true)) yield entry;
      }
    }
  }

  private writeFile(entry: MemoryEntry): void {
    const dir = this.dirFor(entry.namespace);
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    if (entry.namespace.startsWith(CALLER_PREFIX)) {
      const meta = path.join(dir, 'caller.json');
      if (!fs.existsSync(meta)) writePrivate(meta, `${JSON.stringify({ principal: entry.namespace.slice(CALLER_PREFIX.length) })}\n`);
    }
    writePrivate(this.fileFor(entry.namespace, entry.key), renderEntryFile(entry));
  }

  // -- log -----------------------------------------------------------------

  private readLog(): MemoryLogLine[] {
    let text: string;
    try {
      text = fs.readFileSync(this.logFile, 'utf8');
    } catch {
      return [];
    }
    const lines: MemoryLogLine[] = [];
    for (const raw of text.split('\n')) {
      if (!raw.trim()) continue;
      try {
        const line = JSON.parse(raw) as MemoryLogLine;
        if (typeof line.seq === 'number' && (line.op === 'put' || line.op === 'delete')) lines.push(line);
      } catch {
        // a torn last line: the next append starts a clean one
      }
    }
    return lines;
  }

  private appendLog(line: Omit<MemoryLogLine, 'seq'>): MemoryLogLine {
    const full: MemoryLogLine = { seq: this.lastSeq + 1, ...line };
    fs.appendFileSync(this.logFile, `${JSON.stringify(full)}\n`, { mode: 0o600 });
    this.lastSeq = full.seq;
    return full;
  }

  /** The local changes after `seq`, oldest first, for one namespace or all. */
  logSince(seq: number, namespace?: string): MemoryLogLine[] {
    return this.readLog().filter((l) => l.seq > seq && (!namespace || l.namespace === namespace));
  }

  get lastLogSeq(): number {
    return this.lastSeq;
  }

  // -- reads ---------------------------------------------------------------

  async get(namespace: string, key: string): Promise<MemoryEntry | null> {
    await this.open();
    return this.entries.get(entryIdFor(this.store, namespace, key)) ?? null;
  }

  /** Entries in `namespaces` (all when null), newest first. */
  async list(namespaces: readonly string[] | null, options: ListOptions = {}): Promise<MemoryEntry[]> {
    await this.open();
    const out: MemoryEntry[] = [];
    for (const e of this.entries.values()) {
      if (namespaces && !namespaces.includes(e.namespace)) continue;
      if (options.prefix && !e.key.startsWith(options.prefix)) continue;
      if (options.excludeSources?.includes(e.source)) continue;
      out.push(e);
    }
    out.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : a.key.localeCompare(b.key)));
    return out.slice(0, options.limit ?? 50);
  }

  async search(query: string, namespaces: readonly string[] | null, limit = 10): Promise<MemoryEntry[]> {
    await this.open();
    const ids = this.index.search(query, namespaces, limit);
    const out: MemoryEntry[] = [];
    for (const id of ids) {
      const e = this.entries.get(id);
      // The index is checked against the namespaces again here: an index is
      // a ranking, never the access decision.
      if (e && (!namespaces || namespaces.includes(e.namespace))) out.push(e);
    }
    return out;
  }

  /** Every namespace with at least one entry. */
  async namespaces(): Promise<string[]> {
    await this.open();
    return [...new Set([...this.entries.values()].map((e) => e.namespace))].sort();
  }

  // -- writes --------------------------------------------------------------

  async put(namespace: string, key: string, content: string, source: EntrySource = 'tool', at?: string, description = ''): Promise<MemoryEntry> {
    await this.open();
    if (!isValidKey(key)) throw new Error(keyRefusal(key));
    if (!NAMESPACE_RE.test(namespace)) throw new Error(`memory: not a namespace: ${JSON.stringify(namespace)}`);
    const flat = oneLine(description);
    const entry = this.putQuietly(namespace, key, content, source, at ?? this.now().toISOString(), flat);
    this.appendLog({ op: 'put', id: entry.id, namespace, key, content, ...(flat ? { description: flat } : {}), source, at: entry.updatedAt });
    return entry;
  }

  private putQuietly(namespace: string, key: string, content: string, source: EntrySource, at: string, description = ''): MemoryEntry {
    const id = entryIdFor(this.store, namespace, key);
    const existing = this.entries.get(id);
    const entry: MemoryEntry = {
      id,
      namespace,
      key,
      content,
      source,
      createdAt: existing?.createdAt ?? at,
      updatedAt: at,
      ...(description ? { description } : {}),
    };
    this.writeFile(entry);
    this.entries.set(id, entry);
    this.index.upsert(entry);
    return entry;
  }

  async forget(namespace: string, key: string): Promise<boolean> {
    await this.open();
    const removed = this.forgetQuietly(namespace, key);
    if (removed) {
      this.appendLog({ op: 'delete', id: removed.id, namespace, key, at: this.now().toISOString() });
    }
    return removed !== null;
  }

  private forgetQuietly(namespace: string, key: string): MemoryEntry | null {
    const id = entryIdFor(this.store, namespace, key);
    const existing = this.entries.get(id);
    if (!existing) return null;
    try {
      fs.unlinkSync(this.fileFor(namespace, key));
    } catch {
      // already gone
    }
    this.entries.delete(id);
    this.index.remove(id);
    return existing;
  }

  /**
   * Changes from the other tier, applied without logging (they are not this
   * machine's to push back). Last writer wins by time; the count is what was
   * newer than what was here.
   */
  async apply(lines: readonly MemoryLogLine[]): Promise<number> {
    await this.open();
    let applied = 0;
    for (const line of lines) {
      if (!isValidKey(line.key) || !NAMESPACE_RE.test(line.namespace) || typeof line.at !== 'string') continue;
      const id = entryIdFor(this.store, line.namespace, line.key);
      const existing = this.entries.get(id);
      if (existing && existing.updatedAt >= line.at) continue;
      if (line.op === 'put') {
        this.putQuietly(
          line.namespace,
          line.key,
          typeof line.content === 'string' ? line.content : '',
          sourceOf(line.source) === 'tool' ? 'sync' : sourceOf(line.source),
          line.at,
          typeof line.description === 'string' ? oneLine(line.description) : '',
        );
        applied += 1;
      } else if (existing) {
        this.forgetQuietly(line.namespace, line.key);
        applied += 1;
      }
    }
    return applied;
  }
}
