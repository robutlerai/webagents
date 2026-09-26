/**
 * Agent discovery for `webagents daemon` (2026-09-25): the agents under a
 * folder, kept current as their files change.
 *
 * THE PYTHON DAEMON'S RULES (`python/webagents/cli/daemon/registry.py`,
 * `is_discoverable`, and `cli/daemon/watcher.py`), ported here. This watched
 * the top level only, took any `AGENT*.md` in any case (`agent.md`,
 * `AGENT_notes.md`), and never registered the agents that were there when the
 * daemon started: the first scan filled a map and emitted nothing, while the
 * daemon registers only on events, so a daemon started beside an AGENT.md
 * served nothing until the file was touched. A deleted file stayed served.
 * Now:
 *
 *  - an agent file is `AGENT.md` or `AGENT-<name>.md`, by exact name, as the
 *    CLI reads a folder (`cli/agent-files.ts`) and the Python daemon does;
 *    `AGENTS.md`, another tool's file, is not one;
 *  - anywhere under the folder, except inside a tool's own directory
 *    (`IGNORED_DIRS`, the Python set: vendored and generated trees routinely
 *    hold files named like agents, and a copy of the project in one would
 *    replace the real agent under its own name); symlinked directories are
 *    not followed;
 *  - every file found at start is `agent:added`. A change re-reads the tree
 *    and reports what was added, updated and removed by comparing it with the
 *    last reading, rather than trusting `fs.watch` event names, which differ
 *    by platform and arrive twice. Where recursive watching is not available,
 *    the tree is re-read every `POLL_MS`.
 *
 * `WEBAGENTS.md` is not an agent and does not trigger a reload here: this
 * loader reads the agent file alone, where the Python one merges the context
 * file into the agents below it (`docs/cli/index.md`, Differences).
 */

import { EventEmitter } from 'events';

import { parseAgentMarkdown } from '../agents/index';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Agent definition from markdown file
 */
export interface AgentDefinition {
  /** Agent name from frontmatter */
  name: string;
  /** Agent description */
  description?: string;
  /** System instructions */
  instructions?: string;
  /** Skills to load */
  skills?: string[];
  /** The skills as written, config included (`- rest: {sign: always}`). */
  skillEntries?: Array<string | Record<string, unknown>>;
  /** The `access:` block as written (ADR-0045), when the file has one. */
  access?: unknown;
  /** Model to use */
  model?: string;
  /** Source file path */
  filePath: string;
  /** Raw markdown content */
  content: string;
}

/**
 * Watcher events. An update and a removal carry the definition they replace,
 * so the daemon can let go of the name it served (a file can rename its agent).
 */
export interface WatcherEvents {
  'agent:added': (definition: AgentDefinition) => void;
  'agent:updated': (definition: AgentDefinition, previous: AgentDefinition) => void;
  'agent:removed': (filePath: string, previous: AgentDefinition) => void;
  'error': (error: Error) => void;
}

/** Directories never searched: tools' own, never a person's work (the Python `IGNORED_DIRS`). */
export const IGNORED_DIRS: ReadonlySet<string> = new Set([
  '.webagents',
  '.git',
  '.hg',
  '.svn',
  'node_modules',
  '.venv',
  'venv',
  '__pycache__',
  '.tox',
  '.mypy_cache',
  '.pytest_cache',
]);

/** Re-read interval where recursive watching is not available. */
export const POLL_MS = 2000;
/** A burst of change events settles into one re-read. */
const SETTLE_MS = 100;

/** Whether `name` is an agent file's name: `AGENT.md` or `AGENT-<name>.md` (the Python `is_discoverable`). */
export function isAgentFileName(name: string): boolean {
  return name === 'AGENT.md' || (name.startsWith('AGENT-') && name.endsWith('.md'));
}

/**
 * Every agent file under `root`, with a stamp that changes when the file
 * does (modification time and size), skipping `IGNORED_DIRS` at any depth and
 * never following a symlinked directory.
 */
export function findAgentFiles(root: string): Map<string, string> {
  const found = new Map<string, string>();
  const walk = (dir: string): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!IGNORED_DIRS.has(entry.name)) walk(full);
        continue;
      }
      if (!isAgentFileName(entry.name)) continue;
      try {
        const stat = fs.statSync(full);
        if (stat.isFile()) found.set(full, `${stat.mtimeMs}:${stat.size}`);
      } catch {
        // A dangling link is not an agent.
      }
    }
  };
  walk(root);
  return found;
}

/**
 * The agents under a folder (file comment): `start()` reads the tree, emits
 * `agent:added` for each, then keeps it current.
 */
export class AgentWatcher extends EventEmitter {
  private watchDir: string;
  private watcher: fs.FSWatcher | null = null;
  private poll: NodeJS.Timeout | null = null;
  private settle: NodeJS.Timeout | null = null;
  private known: Map<string, { stamp: string; definition: AgentDefinition }> = new Map();

  constructor(watchDir: string) {
    super();
    this.watchDir = watchDir;
  }

  /** The folder this watches. */
  get folder(): string {
    return this.watchDir;
  }

  /** Read the tree (every agent found is `agent:added`), then watch it. */
  start(): void {
    if (this.watcher || this.poll) return;
    this.rescan();
    try {
      this.watcher = fs.watch(this.watchDir, { recursive: true }, () => this.scheduleRescan());
      this.watcher.on('error', (error) => this.emit('error', error));
    } catch {
      this.poll = setInterval(() => this.rescan(), POLL_MS);
      this.poll.unref?.();
    }
  }

  /** Stop watching. */
  stop(): void {
    this.watcher?.close();
    this.watcher = null;
    if (this.poll) clearInterval(this.poll);
    this.poll = null;
    if (this.settle) clearTimeout(this.settle);
    this.settle = null;
  }

  /** Every agent found, by file. */
  getAgents(): AgentDefinition[] {
    return [...this.known.values()].map((entry) => entry.definition);
  }

  private scheduleRescan(): void {
    if (this.settle) clearTimeout(this.settle);
    this.settle = setTimeout(() => {
      this.settle = null;
      this.rescan();
    }, SETTLE_MS);
    this.settle.unref?.();
  }

  /**
   * Read the tree again and report the difference from the last reading:
   * files added, changed (by modification time or size) and gone.
   */
  rescan(): void {
    const found = findAgentFiles(this.watchDir);
    for (const [filePath, stamp] of found) {
      const before = this.known.get(filePath);
      if (before && before.stamp === stamp) continue;
      const definition = this.load(filePath);
      if (!definition) continue;
      this.known.set(filePath, { stamp, definition });
      if (before) this.emit('agent:updated', definition, before.definition);
      else this.emit('agent:added', definition);
    }
    for (const [filePath, before] of [...this.known]) {
      if (found.has(filePath)) continue;
      this.known.delete(filePath);
      this.emit('agent:removed', filePath, before.definition);
    }
  }

  private load(filePath: string): AgentDefinition | null {
    try {
      return this.toAgentDefinition(fs.readFileSync(filePath, 'utf-8'), filePath);
    } catch (error) {
      this.emit('error', new Error(`Failed to load ${filePath}: ${(error as Error).message}`));
      return null;
    }
  }

  /**
   * Parse an agent markdown file.
   *
   * Delegates to the package's one loader (2026-09-23). This used to be a
   * hand-rolled line scanner matching `^(\w+):\s*(.*)$`, which meant the
   * documented block form
   *
   *     skills:
   *       - memory
   *       - mcp
   *
   * matched the `skills:` line with an EMPTY value and produced `['']`: one
   * unnamed skill, and the real two silently gone. It also could not see
   * `namespace`, `intents`, `cron` or anything else outside its four cases.
   */
  private toAgentDefinition(content: string, filePath: string): AgentDefinition | null {
    const parsed = parseAgentMarkdown(content, filePath);

    return {
      name: parsed.name,
      description: parsed.description || undefined,
      instructions: parsed.instructions || content,
      skills: parsed.skills,
      skillEntries: parsed.skillEntries,
      ...(parsed.access !== undefined ? { access: parsed.access } : {}),
      model: parsed.model,
      filePath,
      content,
    };
  }
}
