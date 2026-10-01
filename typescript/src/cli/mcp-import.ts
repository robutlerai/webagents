/**
 * `webagents mcp list` and `webagents mcp add`, and the chat's `/mcp list` and
 * `/mcp add` (2026-09-29, the owner: "can we list available local mcps?").
 *
 * WHAT IS LISTED. The MCP servers other apps on this machine already use, read
 * from their own settings files and never written to: Claude Desktop
 * (`claude_desktop_config.json`), Claude Code (`~/.claude.json`, its servers and
 * this folder's, and the folder's `.mcp.json`), Cursor (`~/.cursor/mcp.json` and
 * the folder's), VS Code (the user folder's `mcp.json`, `servers:`, and the
 * folder's `.vscode/mcp.json`) and Windsurf (`~/.codeium/windsurf/mcp_config.json`).
 * VS Code allows comments and trailing commas; they are dropped before reading.
 * Each entry is read into this SDK's shape (`command`/`args`/`env`/`cwd`, or
 * `url`/`transport`/`headers`); one with neither a command nor an address is
 * skipped. A listing names environment variables and headers, never values.
 *
 * WHAT ADD DOES. It copies one of those entries into an agent. A value that
 * looks like a key (`looksLikeSecret`, or a variable named like one) is stored
 * in this profile's secrets, the store `webagents secrets set` writes, and the
 * entry reads `${secret:NAME}`; an `Authorization: Bearer <token>` header keeps
 * its `Bearer `. VS Code's `${workspaceFolder}` becomes the folder; its
 * `${input:NAME}` becomes `${secret:NAME}` for the person to set. A key in the
 * address's query is stored the same way. A key on the command line (one that
 * looks like a key, or follows a flag named like one), where a reference is
 * refused (S-292), is refused here too, and so is any other `${...}` only the
 * other app fills in. A listing shows every such key as ***. The entry goes where
 * the agent reads its servers: the agent file's own `- mcp:` block when it has
 * one (edited line by line, read back as YAML and compared before it is
 * written: the `skills-edit.ts` rule; a layout that cannot be edited safely is
 * refused with the lines to paste), else `mcp.json` next to the agent file,
 * with a bare `- mcp` added to its skills when it has none.
 *
 * The Python CLI is `webagents/cli/mcp_import.py`, rule for rule;
 * `python/tests/fixtures/mcp_tool/import.json` holds the words and the cases
 * both suites run.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import { looksLikeSecret, REFERENCE_NAME, suggestedSecretName } from '../skills/secrets/references';
import { editSkillList, isTriviaLine, parseFrontMatter, SkillListError, skillEntriesOf, writeByRename, yamlScalar } from './skills-edit';

export const WORDS = {
  heading: 'MCP servers other apps on this machine use',
  group: '{app}  {file}',
  server: '  {name}  {what}',
  unreadable: '{app}  {file}: could not read it ({reason})',
  none: 'No MCP servers in the settings of Claude Desktop, Claude Code, Cursor, VS Code or Windsurf.',
  addHint: 'Add one to an agent: {command}, or /mcp add <name> in the chat.',
  added: 'Added {name} to {agent} ({where}).',
  stored: "Stored {names} in this profile's secrets; the entry reads them as ${secret:NAME}.",
  toSet: 'Set {names} before it connects: {hints}.',
  reload: '/reload connects it.',
  restart: 'The agent connects it the next time it starts.',
  notFound: "No MCP server named {name} in other apps' settings; {command} shows them.",
  ambiguous: "{name} is in more than one app's settings ({apps}); say which with --from <app>.",
  unknownApp: 'No app named {app}; use one of {apps}.',
  exists: '{agent} already has an MCP server named {name}.',
  keyOnCommandLine: '{name} carries what looks like a key in its {field}; move it to env in {app}, then add it again.',
  otherAppReference: '{name} uses {reference}, which only {app} fills in; add it by hand with the value in place.',
  cannotEdit: '{file} lists its MCP servers in a layout this cannot edit safely. Add these lines under its mcp entry:',
  noAgent: 'No agent file in {folder}; {command} makes one.',
  manyAgents: 'More than one agent file in {folder}; name one: {files}.',
  removed: 'Removed {name} from {agent} ({where}).',
  notThere: '{agent} has no MCP server named {name}.',
  secretsStay: "It read {names} from this profile's secrets; they stay stored (`{command}` removes one).",
  cannotRemove: '{file} lists its MCP servers in a layout this cannot edit safely; take {name} out of it by hand.',
  reloadRemoved: '/reload stops it in this chat.',
  restartRemoved: 'The agent stops using it the next time it starts.',
} as const;

/** The apps read, in the order a listing shows them: id, name. */
export const APPS: ReadonlyArray<readonly [string, string]> = [
  ['claude-desktop', 'Claude Desktop'],
  ['claude-code', 'Claude Code'],
  ['cursor', 'Cursor'],
  ['vscode', 'VS Code'],
  ['windsurf', 'Windsurf'],
];

/** An environment variable named like a credential: its value is stored as a secret whatever it looks like. */
const KEY_NAME = /(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIALS?|_PAT)$/i;
/** A header or an address's query parameter named like a credential. */
const KEY_FIELD = /(api[-_]?key|token|secret|auth|password|passwd|^key$)/i;
/** A command-line flag named like a credential: the argument after it, or after its `=`, is a key. */
const KEY_FLAG = /^-{1,2}(?!-)(?!no-)[A-Za-z0-9_-]*?(api[-_]?key|token|secret|password|passwd)$/i;
/** A query parameter in an address: the separator, the name, the value. */
const QUERY = /([?&])([^=&#]+)=([^&#]*)/g;

export type McpEntry = {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  url?: string;
  transport?: 'http' | 'sse';
  headers?: Record<string, string>;
};

export interface FoundServer {
  appId: string;
  app: string;
  file: string;
  name: string;
  entry: McpEntry;
}

export interface Discovery {
  found: FoundServer[];
  /** `[app, file, reason]` for a settings file that is there and could not be read. */
  unreadable: Array<[string, string, string]>;
}

/** `[app id, app, file, shape]` for every settings file read, in listing order. */
export function settingsFiles(
  home: string,
  folder: string,
  system: string = process.platform,
  appdata: string | undefined = process.env.APPDATA,
): Array<[string, string, string, 'mcpServers' | 'servers' | 'claude-json']> {
  const join = system.startsWith('win') ? path.win32.join : path.posix.join;
  const support =
    system === 'darwin' ? join(home, 'Library', 'Application Support') : system.startsWith('win') ? appdata || join(home, 'AppData', 'Roaming') : join(home, '.config');
  return [
    ['claude-desktop', 'Claude Desktop', join(support, 'Claude', 'claude_desktop_config.json'), 'mcpServers'],
    ['claude-code', 'Claude Code', join(home, '.claude.json'), 'claude-json'],
    ['claude-code', 'Claude Code', join(folder, '.mcp.json'), 'mcpServers'],
    ['cursor', 'Cursor', join(home, '.cursor', 'mcp.json'), 'mcpServers'],
    ['cursor', 'Cursor', join(folder, '.cursor', 'mcp.json'), 'mcpServers'],
    ['vscode', 'VS Code', join(support, 'Code', 'User', 'mcp.json'), 'servers'],
    ['vscode', 'VS Code', join(folder, '.vscode', 'mcp.json'), 'servers'],
    ['windsurf', 'Windsurf', join(home, '.codeium', 'windsurf', 'mcp_config.json'), 'mcpServers'],
  ];
}

/**
 * `text` with `step(text, i)` deciding, at every character outside a JSON
 * string, what to write and where to go on; strings are copied as they are.
 */
function outsideStrings(text: string, step: (text: string, i: number) => [string, number]): string {
  let out = '';
  let inString = false;
  for (let i = 0; i < text.length; ) {
    const ch = text[i];
    if (inString) {
      out += ch === '\\' ? text.slice(i, i + 2) : ch;
      inString = ch !== '"';
      i += ch === '\\' ? 2 : 1;
    } else if (ch === '"') {
      out += ch;
      inString = true;
      i += 1;
    } else {
      const [written, next] = step(text, i);
      out += written;
      i = next;
    }
  }
  return out;
}

function comment(text: string, i: number): [string, number] {
  if (text.startsWith('//', i)) {
    const end = text.indexOf('\n', i);
    return ['', end < 0 ? text.length : end];
  }
  if (text.startsWith('/*', i)) {
    const end = text.indexOf('*/', i + 2);
    return ['', end < 0 ? text.length : end + 2];
  }
  return [text[i], i + 1];
}

function trailingComma(text: string, i: number): [string, number] {
  if (text[i] === ',') {
    let j = i + 1;
    while (j < text.length && ' \t\r\n'.includes(text[j])) j += 1;
    if (text[j] === '}' || text[j] === ']') return ['', i + 1];
  }
  return [text[i], i + 1];
}

/** `text` without `//` and `/* *\/` comments and trailing commas; a string is never touched, whatever it holds. */
export function stripJsonc(text: string): string {
  return outsideStrings(outsideStrings(text, comment), trailingComma);
}

/** A scalar from another app's JSON as text; null for null, a list or a mapping, which are dropped (the Python `_text`). */
function text(value: unknown): string | null {
  if (value === null || value === undefined || typeof value === 'object') return null;
  return String(value);
}

function strings(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    const t = text(v);
    if (t !== null) out[k] = t;
  }
  return out;
}

function isFile(file: string): boolean {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

/**
 * `text` split into words the way Python's `shlex.split` does: space, tab and
 * line breaks between words, single quotes literal, a backslash escaping the
 * next character, and inside double quotes only `"` and `\\`. Null when a
 * quote is left open or the text ends in a lone backslash, where `shlex` raises.
 */
export function shellWords(text: string): string[] | null {
  const words: string[] = [];
  let word = '';
  let started = false;
  let quote: "'" | '"' | null = null;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote === "'") {
      if (ch === "'") quote = null;
      else word += ch;
    } else if (quote === '"') {
      if (ch === '"') quote = null;
      else if (ch === '\\' && i + 1 < text.length && '"\\'.includes(text[i + 1])) word += text[++i];
      else word += ch;
    } else if (' \t\r\n'.includes(ch)) {
      if (started) words.push(word);
      word = '';
      started = false;
    } else {
      started = true;
      if (ch === "'" || ch === '"') quote = ch;
      else if (ch === '\\' && i + 1 >= text.length) return null;
      else if (ch === '\\') word += text[++i];
      else word += ch;
    }
  }
  if (quote) return null;
  if (started) words.push(word);
  return words;
}

/** Another app's entry in this SDK's shape; null when it names neither a command nor an address. */
export function normalizeEntry(raw: unknown): McpEntry | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.command === 'string' && r.command) {
    let command = r.command;
    let args = (Array.isArray(r.args) ? r.args : []).map(text).filter((a): a is string => a !== null);
    // A whole command line in `command` with no `args` (Cursor writes
    // `"command": "npx -y chrome-devtools-mcp@latest"`) is split the way a
    // shell would, unless it is the path of a file that is there.
    if (!args.length && /\s/.test(command.trim()) && !isFile(command)) {
      const words = shellWords(command);
      if (words && words.length) [command, ...args] = words;
    }
    const entry: McpEntry = { command, args };
    const env = strings(r.env);
    if (Object.keys(env).length) entry.env = env;
    if (typeof r.cwd === 'string' && r.cwd) entry.cwd = r.cwd;
    return entry;
  }
  const url = (['url', 'serverUrl', 'httpUrl'] as const).map((k) => r[k]).find((v) => typeof v === 'string' && v) as string | undefined;
  if (!url) return null;
  const entry: McpEntry = { url };
  const kind = String(r.type || r.transport || '').toLowerCase().replace(/_/g, '-');
  if (kind === 'sse') entry.transport = 'sse';
  else if (kind === 'http' || kind === 'streamable-http' || kind === 'streamablehttp') entry.transport = 'http';
  const headers = strings(r.headers);
  if (Object.keys(headers).length) entry.headers = headers;
  return entry;
}

function serversIn(data: unknown, shape: string, folder: string): Array<[string, unknown]> {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return [];
  const d = data as Record<string, unknown>;
  const mapOf = (value: unknown): Array<[string, unknown]> =>
    value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value as Record<string, unknown>) : [];
  if (shape === 'claude-json') {
    const projects = d.projects && typeof d.projects === 'object' ? (d.projects as Record<string, unknown>) : {};
    const project = projects[folder] as Record<string, unknown> | undefined;
    return [...mapOf(d.mcpServers), ...mapOf(project?.mcpServers)];
  }
  return mapOf(shape === 'servers' ? d.servers : d.mcpServers);
}

/**
 * Settings files as read, by path, kept while their time stamp and size stay
 * the same: the chat reads them before every prompt (for completion), and
 * Claude Code's `~/.claude.json` can run to megabytes.
 */
const READ = new Map<string, { key: string; ok: boolean; data: unknown }>();

/**
 * `{ ok: true, data }` for a settings file, `{ ok: false }` for one that is
 * not valid JSON, null for one that is not there or cannot be opened. Plain
 * JSON is read as it is; only a file that fails is read again without
 * comments and trailing commas. A byte-order mark is dropped (the Python
 * `_read_settings`).
 */
function readSettings(file: string): { ok: boolean; data: unknown } | null {
  let key: string;
  let text: string;
  try {
    const stat = fs.statSync(file, { bigint: true });
    if (!stat.isFile()) return null;
    key = `${stat.mtimeNs}:${stat.size}`;
    const cached = READ.get(file);
    if (cached && cached.key === key) return cached;
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return null;
  }
  if (text.startsWith('\ufeff')) text = text.slice(1);
  let read: { key: string; ok: boolean; data: unknown };
  try {
    read = { key, ok: true, data: JSON.parse(text) };
  } catch {
    try {
      read = { key, ok: true, data: JSON.parse(stripJsonc(text)) };
    } catch {
      read = { key, ok: false, data: null };
    }
  }
  READ.set(file, read);
  return read;
}

/** Every server the apps' settings files name, in listing order. */
export function discover(home: string = os.homedir(), folder: string = process.cwd(), system: string = process.platform, appdata?: string): Discovery {
  let real = folder;
  try {
    real = fs.realpathSync(folder);
  } catch {
    real = path.resolve(folder);
  }
  const result: Discovery = { found: [], unreadable: [] };
  for (const [appId, app, file, shape] of settingsFiles(home, real, system, appdata ?? process.env.APPDATA)) {
    const read = readSettings(file);
    if (!read) continue;
    if (!read.ok) {
      result.unreadable.push([app, file, 'not valid JSON']);
      continue;
    }
    const data = read.data;
    for (const [name, raw] of serversIn(data, shape, real)) {
      const entry = normalizeEntry(raw);
      if (entry) result.found.push({ appId, app, file, name, entry });
    }
  }
  return result;
}

function shownPath(file: string, home: string): string {
  return home && (file === home || file.startsWith(`${home}${path.sep}`)) ? `~${file.slice(home.length)}` : file;
}

/**
 * `args` with every key in them shown as ***, and whether there was one: an
 * argument that looks like a key, the value of `--api-key=VALUE`, and the
 * argument after a bare `--api-key` (a flag named like a credential).
 */
export function maskedArgs(args: readonly string[]): [string[], boolean] {
  const out: string[] = [];
  let carried = false;
  let afterFlag = false;
  for (const arg of args) {
    const eq = arg.indexOf('=');
    const flag = eq >= 0 ? arg.slice(0, eq) : arg;
    const value = eq >= 0 ? arg.slice(eq + 1) : '';
    if ((afterFlag && !arg.startsWith('-')) || looksLikeSecret(arg)) {
      out.push('***');
      carried = true;
    } else if (eq >= 0 && value && KEY_FLAG.test(flag)) {
      out.push(`${flag}=***`);
      carried = true;
    } else {
      out.push(arg);
    }
    afterFlag = eq < 0 && KEY_FLAG.test(arg);
  }
  return [out, carried];
}

/** Whether an address's query value is a key: it looks like one, or its parameter is named like one. */
function queryKey(name: string, value: string): boolean {
  return !!value && !value.includes('${') && (looksLikeSecret(value) || KEY_FIELD.test(name));
}

/**
 * One line for an entry: the command and its arguments (`maskedArgs`), or the
 * address (a key in its query as ***) and its transport; variable and header
 * names, never their values.
 */
export function describe(entry: McpEntry): string {
  if (entry.command !== undefined) {
    let text = [looksLikeSecret(entry.command) ? '***' : entry.command, ...maskedArgs(entry.args ?? [])[0]].join(' ');
    if (entry.env && Object.keys(entry.env).length) text += `  env ${Object.keys(entry.env).join(', ')}`;
    return text;
  }
  const url = (entry.url ?? '').replace(QUERY, (_m, separator: string, name: string, value: string) => `${separator}${name}=${queryKey(name, value) ? '***' : value}`);
  let text = url + (entry.transport ? ` (${entry.transport})` : '');
  if (entry.headers && Object.keys(entry.headers).length) text += `  headers ${Object.keys(entry.headers).join(', ')}`;
  return text;
}

/** What `mcp list` prints. */
export function listLines(discovery: Discovery, home: string = os.homedir(), addCommand = 'webagents mcp add <name>'): string[] {
  if (!discovery.found.length && !discovery.unreadable.length) return [WORDS.none];
  const lines: string[] = [WORDS.heading];
  const groups = new Map<string, FoundServer[]>();
  for (const found of discovery.found) {
    const key = `${found.app}\u0000${found.file}`;
    groups.set(key, [...(groups.get(key) ?? []), found]);
  }
  for (const servers of groups.values()) {
    lines.push(WORDS.group.replace('{app}', servers[0].app).replace('{file}', shownPath(servers[0].file, home)));
    for (const found of servers) lines.push(WORDS.server.replace('{name}', found.name).replace('{what}', describe(found.entry)));
  }
  for (const [app, file, reason] of discovery.unreadable) {
    lines.push(WORDS.unreadable.replace('{app}', app).replace('{file}', shownPath(file, home)).replace('{reason}', reason));
  }
  if (discovery.found.length) lines.push(WORDS.addHint.replace('{command}', addCommand));
  return lines;
}

/** `mcp add` would not copy the server; the message says why. */
export class AddRefused extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AddRefused';
  }
}

/** `value` with every mapping's keys in order, so two can be compared whatever order they were built in. */
function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([k, v]) => [k, canonical(v)]));
  }
  return value;
}

/** What the conversion needs of a secret store. */
export interface SecretStoreLike {
  get(name: string): Promise<string | null>;
  set(name: string, value: string): Promise<unknown>;
}

async function secretName(store: SecretStoreLike, preferred: string, fallback: string, value: string): Promise<string> {
  const candidates = REFERENCE_NAME.test(preferred) ? [preferred, fallback] : [fallback];
  for (const name of candidates) {
    const held = await store.get(name);
    if (held === null || held === undefined || held === value) return name;
  }
  for (let n = 2; ; n += 1) {
    const name = `${fallback.slice(0, 120)}_${n}`;
    const held = await store.get(name);
    if (held === null || held === undefined || held === value) return name;
  }
}

export interface Converted {
  entry: McpEntry;
  /** Secret names stored with a value from the other app's settings. */
  stored: string[];
  /** Secret names the entry reads that have no value yet (VS Code's inputs). */
  toSet: string[];
}

/**
 * `found.entry` as the agent will hold it (file comment): keys moved to this
 * profile's secrets, VS Code's variables filled in or turned into references.
 * Throws `AddRefused` for what cannot be copied safely.
 */
export async function convertEntry(found: FoundServer, folder: string, store: SecretStoreLike): Promise<Converted> {
  const entry: McpEntry = JSON.parse(JSON.stringify(found.entry));
  const result: Converted = { entry, stored: [], toSet: [] };
  const refuseReference = (reference: string): never => {
    throw new AddRefused(WORDS.otherAppReference.replace('{name}', found.name).replace('{reference}', reference).replace('{app}', found.app));
  };
  const plain = (value: string, where: string): string => {
    const filled = value.split('${workspaceFolder}').join(folder);
    const reference = /\$\{[^}]*\}/.exec(filled);
    if (reference) refuseReference(reference[0]);
    if (looksLikeSecret(filled)) {
      throw new AddRefused(WORDS.keyOnCommandLine.replace('{name}', found.name).replace('{field}', where).replace('{app}', found.app));
    }
    return filled;
  };
  const inputs = (value: string): string => {
    const swapped = value.split('${workspaceFolder}').join(folder).replace(/\$\{input:([^}]+)\}/g, (_m, raw: string) => {
      // Folded to the reference grammar (`REFERENCE_NAME`), which is also what
      // `webagents secrets set` accepts: `api-key` becomes `API_KEY`.
      const folded = raw.toUpperCase().replace(/[^A-Z0-9_]/g, '_');
      const name = (/^[0-9]/.test(folded) ? `_${folded}` : folded).slice(0, 128);
      if (!result.toSet.includes(name)) result.toSet.push(name);
      return `\${secret:${name}}`;
    });
    const other = /\$\{(?!env:|secret:)[^}]*\}/.exec(swapped);
    if (other) refuseReference(other[0]);
    return swapped;
  };
  const stored = async (preferred: string, fallback: string, value: string): Promise<string> => {
    const name = await secretName(store, preferred, fallback, value);
    if ((await store.get(name)) !== value) await store.set(name, value);
    if (!result.stored.includes(name)) result.stored.push(name);
    return name;
  };

  if (entry.command !== undefined) {
    entry.command = plain(entry.command, 'command');
    entry.args = (entry.args ?? []).map((a) => plain(a, 'arguments'));
    if (maskedArgs(entry.args)[1]) {
      throw new AddRefused(WORDS.keyOnCommandLine.replace('{name}', found.name).replace('{field}', 'arguments').replace('{app}', found.app));
    }
    if (entry.cwd !== undefined) entry.cwd = plain(entry.cwd, 'cwd');
    // Every value is checked before any key is stored, so a refusal leaves the store as it was.
    const checked = Object.entries(entry.env ?? {}).map(([key, raw]) => [key, inputs(raw)] as const);
    const env: Record<string, string> = {};
    for (const [key, filled] of checked) {
      let value = filled;
      if (!value.includes('${') && value && (looksLikeSecret(value) || KEY_NAME.test(key))) {
        value = `\${secret:${await stored(key, suggestedSecretName(found.name, key), value)}}`;
      }
      env[key] = value;
    }
    if (Object.keys(env).length) entry.env = env;
    return result;
  }
  // A key in the address's query is stored and read back by reference: an
  // address is not on the process list, and references resolve in it.
  const url = inputs(entry.url ?? '');
  const checkedHeaders = Object.entries(entry.headers ?? {}).map(([key, raw]) => [key, inputs(raw)] as const);
  const parts: string[] = [];
  let last = 0;
  for (const match of url.matchAll(QUERY)) {
    const [whole, separator, param, value] = match;
    if (!queryKey(param, value)) continue;
    const name = suggestedSecretName(found.name, param);
    parts.push(url.slice(last, match.index), `${separator}${param}=\${secret:${await stored(name, name, value)}}`);
    last = (match.index ?? 0) + whole.length;
  }
  entry.url = parts.join('') + url.slice(last);
  const headers: Record<string, string> = {};
  for (const [key, filled] of checkedHeaders) {
    let value = filled;
    const bearer = /^(Bearer\s+)(\S.*)$/.exec(value);
    if (bearer && !bearer[2].includes('${')) {
      const name = suggestedSecretName(found.name, 'token');
      value = `${bearer[1]}\${secret:${await stored(name, name, bearer[2])}}`;
    } else if (!value.includes('${') && value && (looksLikeSecret(value) || KEY_FIELD.test(key))) {
      const name = suggestedSecretName(found.name, key);
      value = `\${secret:${await stored(name, name, value)}}`;
    }
    headers[key] = value;
  }
  if (Object.keys(headers).length) entry.headers = headers;
  return result;
}

/** `name: {entry}` as block YAML at `indent`, fields `step` deeper. */
export function entryLines(name: string, entry: McpEntry, indent: number, step: number): string[] {
  const pad = ' '.repeat(indent);
  const inner = ' '.repeat(indent + step);
  const deeper = ' '.repeat(indent + 2 * step);
  const lines = [`${pad}${yamlScalar(name)}:`];
  for (const key of ['command', 'args', 'cwd', 'url', 'transport'] as const) {
    const value = entry[key];
    if (value === undefined) continue;
    lines.push(`${inner}${key}: ${Array.isArray(value) ? pyJson(value) : yamlScalar(String(value))}`);
  }
  for (const key of ['env', 'headers'] as const) {
    const map = entry[key];
    if (map && Object.keys(map).length) {
      lines.push(`${inner}${key}:`);
      for (const [k, v] of Object.entries(map)) lines.push(`${deeper}${yamlScalar(k)}: ${yamlScalar(v)}`);
    }
  }
  return lines;
}

/** A list as Python's `json.dumps` writes it (`["a", "b"]`), so both CLIs write the same line. */
function pyJson(values: string[]): string {
  return `[${values.map((v) => JSON.stringify(v)).join(', ')}]`;
}

/** The agent file's `mcp` block is in a layout this does not edit. */
export class CannotEdit extends Error {}

/**
 * The agent file `text` with `name` added to its `- mcp:` block (file
 * comment). Throws `CannotEdit` for a layout it will not touch.
 */
export function insertIntoAgentFile(text: string, name: string, entry: McpEntry, file: string): string {
  let parsed: ReturnType<typeof parseFrontMatter>;
  let entries: unknown[];
  try {
    parsed = parseFrontMatter(text, file);
    entries = skillEntriesOf(parsed.data, file);
  } catch (error) {
    if (error instanceof SkillListError) throw new CannotEdit(file);
    throw error;
  }
  const isMap = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
  const holder = entries.find((e) => isMap(e) && isMap(e.mcp)) as Record<string, Record<string, unknown>> | undefined;
  if (parsed.close < 0 || !holder) throw new CannotEdit(file);
  const block = holder.mcp;
  const wrapper = isMap(block.mcpServers);
  const lines = parsed.lines;
  const indentOf = (line: string): number => line.length - line.replace(/^ +/, '').length;
  let start = lines.findIndex((line, i) => i > 0 && i < parsed.close && /^\s*-\s+mcp:\s*(#.*)?$/.test(line));
  if (start < 0) throw new CannotEdit(file);
  let parent = indentOf(lines[start]);
  if (wrapper) {
    const inner = lines.findIndex((line, i) => i > start && i < parsed.close && /^\s*mcpServers:\s*(#.*)?$/.test(line) && indentOf(line) > parent);
    if (inner < 0) throw new CannotEdit(file);
    start = inner;
    parent = indentOf(lines[start]);
  }
  const first = lines.findIndex((line, i) => i > start && i < parsed.close && !isTriviaLine(line));
  if (first < 0 || indentOf(lines[first]) <= parent || lines[first].slice(0, indentOf(lines[first]) + 1).includes('\t')) throw new CannotEdit(file);
  const serverIndent = indentOf(lines[first]);
  const deeper = lines.findIndex((line, i) => i > first && i < parsed.close && !isTriviaLine(line) && indentOf(line) > serverIndent);
  const step = deeper >= 0 ? indentOf(lines[deeper]) - serverIndent : 2;
  let end = first;
  for (let i = first + 1; i < parsed.close; i += 1) {
    if (isTriviaLine(lines[i])) continue;
    if (indentOf(lines[i]) < serverIndent) break;
    end = i;
  }
  const added = [...lines.slice(0, end + 1), ...entryLines(name, entry, serverIndent, step), ...lines.slice(end + 1)];
  const newText = parsed.bom + added.join(parsed.nl);
  const expected = JSON.parse(JSON.stringify(parsed.data)) as Record<string, unknown>;
  const target = (skillEntriesOf(expected, file).find((e) => isMap(e) && isMap((e as Record<string, unknown>).mcp)) as Record<string, Record<string, unknown>>).mcp;
  (wrapper ? (target.mcpServers as Record<string, unknown>) : target)[name] = entry;
  let check: Record<string, unknown>;
  try {
    check = parseFrontMatter(newText, file).data;
  } catch {
    throw new CannotEdit(file);
  }
  if (JSON.stringify(canonical(check)) !== JSON.stringify(canonical(expected))) throw new CannotEdit(file);
  return newText;
}

export interface AddPlan {
  /** `[file, new text]` for every file `add` writes. */
  writes: Array<[string, string]>;
  where: string;
  converted: Converted;
}

type Json = Record<string, unknown>;

/** Where the agent reads its servers, and their names (the Python `_agent_servers`). */
function agentServers(data: Json, mcpJson: Json | null): ['inline' | 'json' | 'none', Json] {
  const raw = mcpJson ?? {};
  const isMap = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v);
  const jsonServers = isMap(raw.mcpServers) ? raw.mcpServers : raw;
  const entries = Array.isArray(data.skills) ? data.skills : [];
  for (const entry of entries) {
    if (entry === 'mcp') return ['json', jsonServers];
    if (isMap(entry) && 'mcp' in entry) {
      const block = entry.mcp;
      if (isMap(block) && Object.keys(block).length) return ['inline', isMap(block.mcpServers) ? block.mcpServers : block];
      return ['json', jsonServers];
    }
  }
  return ['none', jsonServers];
}

/** The one server `name` (from `app` when given) names; throws `AddRefused` otherwise. */
export function choose(discovery: Discovery, name: string, app: string | undefined, listCommand: string): FoundServer {
  const ids = APPS.map(([id]) => id);
  if (app !== undefined && !ids.includes(app)) throw new AddRefused(WORDS.unknownApp.replace('{app}', app).replace('{apps}', ids.join(', ')));
  const matches = discovery.found.filter((f) => f.name === name && (app === undefined || f.appId === app));
  if (!matches.length) throw new AddRefused(WORDS.notFound.replace('{name}', name).replace('{command}', listCommand));
  if (new Set(matches.map((m) => JSON.stringify(canonical(m.entry)))).size > 1) {
    throw new AddRefused(WORDS.ambiguous.replace('{name}', name).replace('{apps}', [...new Set(matches.map((m) => m.appId))].join(', ')));
  }
  return matches[0];
}

/** The files `add` writes for `found`, nothing written yet (secrets aside). */
export async function planAdd(found: FoundServer, agentFile: string, store: SecretStoreLike, agentName: string): Promise<AddPlan> {
  const text = fs.readFileSync(agentFile, 'utf8');
  const parsed = parseFrontMatter(text, path.basename(agentFile));
  const mcpJsonFile = path.join(path.dirname(agentFile), 'mcp.json');
  let mcpJson: Json | null = null;
  if (fs.existsSync(mcpJsonFile)) {
    const loaded = JSON.parse(fs.readFileSync(mcpJsonFile, 'utf8')) as unknown;
    mcpJson = loaded && typeof loaded === 'object' && !Array.isArray(loaded) ? (loaded as Json) : {};
  }
  const [kind, servers] = agentServers(parsed.data, mcpJson);
  if (found.name in servers) throw new AddRefused(WORDS.exists.replace('{agent}', agentName).replace('{name}', found.name));
  let folder = path.dirname(agentFile);
  try {
    folder = fs.realpathSync(folder);
  } catch {
    folder = path.resolve(folder);
  }
  const converted = await convertEntry(found, folder, store);
  if (kind === 'inline') {
    try {
      return { writes: [[agentFile, insertIntoAgentFile(text, found.name, converted.entry, path.basename(agentFile))]], where: path.basename(agentFile), converted };
    } catch (error) {
      if (!(error instanceof CannotEdit)) throw error;
      // The lines to paste read the keys by reference; say where they went.
      const snippet = entryLines(found.name, converted.entry, 2, 2).join('\n');
      const said = converted.stored.length ? `\n${WORDS.stored.replace('{names}', converted.stored.join(', '))}` : '';
      throw new AddRefused(`${WORDS.cannotEdit.replace('{file}', path.basename(agentFile))}\n${snippet}${said}`);
    }
  }
  const data: Json = mcpJson ?? { mcpServers: {} };
  const isMap = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v);
  const target = isMap(data.mcpServers) || mcpJson === null ? (data.mcpServers as Json) : data;
  target[found.name] = converted.entry;
  const writes: Array<[string, string]> = [[mcpJsonFile, `${JSON.stringify(data, null, 2)}\n`]];
  if (kind === 'none') {
    const edit = editSkillList(text, 'add', ['mcp'], path.basename(agentFile));
    if (edit.changed) writes.push([agentFile, edit.text]);
  }
  return { writes, where: 'mcp.json', converted };
}

/** What `add` says after writing. */
export function resultLines(plan: AddPlan, name: string, agentName: string, inChat: boolean, cliCommand: (args: string) => string): string[] {
  const lines = [WORDS.added.replace('{name}', name).replace('{agent}', agentName).replace('{where}', plan.where)];
  if (plan.converted.stored.length) lines.push(WORDS.stored.replace('{names}', plan.converted.stored.join(', ')));
  if (plan.converted.toSet.length) {
    const hints = plan.converted.toSet.map((n) => `\`${cliCommand(`secrets set ${n}`)}\``).join(', ');
    lines.push(WORDS.toSet.replace('{names}', plan.converted.toSet.join(', ')).replace('{hints}', hints));
  }
  lines.push(inChat ? WORDS.reload : WORDS.restart);
  return lines;
}

/** Write every file of `plan`, each by renaming a temporary file. */
export function writePlan(plan: AddPlan): void {
  for (const [file, text] of plan.writes) writeByRename(file, text);
}

/** The store `${secret:NAME}` reads (`ownerReferenceSources`). */
export async function secretStore(): Promise<SecretStoreLike> {
  const { providerKeyStore } = await import('./provider-keys.js');
  return providerKeyStore();
}

// -- remove ----------------------------------------------------------------------------------
//
// `mcp remove <name>` and `/mcp remove <name>` (2026-09-29, the subcommand
// rule: what `add` puts in, `remove` takes out). The entry goes from where
// the agent reads it: its line block in the agent file's own `- mcp:` block
// (every other byte as it was, read back as YAML and compared before it is
// written), or its key in `mcp.json`. The last server of an agent file's own
// block takes the whole `- mcp` entry with it: a bare `- mcp` left behind
// would read `mcp.json`, and servers the person never chose would start. The
// secrets the entry read stay stored, since another entry may read them too;
// the sentence names them. The Python `remove_from_agent_file`, rule for rule.

/** `mcp remove` would not take the server out; the message says why. */
export class RemoveRefused extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RemoveRefused';
  }
}

const KEY_LINE = /^(\s*)(?:"((?:[^"\\]|\\.)*)"|'((?:[^']|'')*)'|([^\s:#'"][^:#]*?))\s*:(?:\s|$)/;

/** The mapping key a block line opens, unquoted; null for any other line. */
function lineKey(line: string): string | null {
  const match = KEY_LINE.exec(line);
  if (!match) return null;
  if (match[2] !== undefined) return JSON.parse(`"${match[2]}"`) as string;
  if (match[3] !== undefined) return match[3].replace(/''/g, "'");
  return match[4].trim();
}

/** The agent file `text` with the server `name` taken out of its `- mcp:` block; throws `CannotEdit` for a layout it will not touch. */
export function removeFromAgentFile(text: string, name: string, file: string): string {
  let parsed: ReturnType<typeof parseFrontMatter>;
  let entries: unknown[];
  try {
    parsed = parseFrontMatter(text, file);
    entries = skillEntriesOf(parsed.data, file);
  } catch (error) {
    if (error instanceof SkillListError) throw new CannotEdit(file);
    throw error;
  }
  const isMap = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
  const holder = entries.find((e) => isMap(e) && isMap(e.mcp)) as Record<string, Record<string, unknown>> | undefined;
  if (parsed.close < 0 || !holder) throw new CannotEdit(file);
  const block = holder.mcp;
  const wrapper = isMap(block.mcpServers);
  const servers = (wrapper ? block.mcpServers : block) as Record<string, unknown>;
  if (!(name in servers)) throw new CannotEdit(file);
  const lines = parsed.lines;
  const indentOf = (line: string): number => line.length - line.replace(/^ +/, '').length;
  const endOf = (start: number, parent: number): number => {
    let end = start;
    for (let i = start + 1; i < parsed.close; i += 1) {
      if (isTriviaLine(lines[i])) continue;
      if (indentOf(lines[i]) <= parent) break;
      end = i;
    }
    return end;
  };
  const entryLine = lines.findIndex((line, i) => i > 0 && i < parsed.close && /^\s*-\s+mcp:\s*(#.*)?$/.test(line));
  if (entryLine < 0) throw new CannotEdit(file);
  const parentIndent = indentOf(lines[entryLine]);
  let top = entryLine;
  if (wrapper) {
    top = lines.findIndex((line, i) => i > entryLine && i < parsed.close && /^\s*mcpServers:\s*(#.*)?$/.test(line) && indentOf(line) > parentIndent);
    if (top < 0) throw new CannotEdit(file);
  }
  const first = lines.findIndex((line, i) => i > top && i < parsed.close && !isTriviaLine(line));
  if (first < 0) throw new CannotEdit(file);
  const serverIndent = indentOf(lines[first]);
  const at = lines.findIndex((line, i) => i >= first && i < parsed.close && !isTriviaLine(line) && indentOf(line) === serverIndent && lineKey(line) === name);
  if (at < 0) throw new CannotEdit(file);
  const expected = JSON.parse(JSON.stringify(parsed.data)) as Record<string, unknown>;
  const targetEntries = skillEntriesOf(expected, file);
  const target = targetEntries.find((e) => isMap(e) && isMap((e as Record<string, unknown>).mcp)) as Record<string, Record<string, unknown>>;
  delete (wrapper ? (target.mcp.mcpServers as Record<string, unknown>) : target.mcp)[name];
  let start: number;
  let end: number;
  if (Object.keys(servers).length === 1 && (!wrapper || Object.keys(block).every((k) => k === 'mcpServers'))) {
    // The last server: the whole `- mcp` entry goes (the note above).
    targetEntries.splice(targetEntries.indexOf(target), 1);
    start = entryLine;
    end = endOf(entryLine, parentIndent);
  } else {
    start = at;
    end = endOf(at, serverIndent);
    // The comment lines right above a server, at its indent, are about it.
    while (start > first && lines[start - 1].trim().startsWith('#') && indentOf(lines[start - 1]) === serverIndent) start -= 1;
  }
  const newText = parsed.bom + [...lines.slice(0, start), ...lines.slice(end + 1)].join(parsed.nl);
  let check: Record<string, unknown>;
  try {
    check = parseFrontMatter(newText, file).data;
  } catch {
    throw new CannotEdit(file);
  }
  if (JSON.stringify(canonical(check)) !== JSON.stringify(canonical(expected))) throw new CannotEdit(file);
  return newText;
}

/** The `${secret:NAME}` references in an entry, in order, once each. */
function secretNames(entry: unknown): string[] {
  return [...new Set([...JSON.stringify(entry).matchAll(/\$\{secret:([A-Za-z_][A-Za-z0-9_]*)\}/g)].map((m) => m[1]))];
}

/** The files `remove` writes, where the entry was, and the secrets it read. */
export function planRemove(name: string, agentFile: string, agentName: string): { writes: Array<[string, string]>; where: string; secrets: string[] } {
  const text = fs.readFileSync(agentFile, 'utf8');
  const parsed = parseFrontMatter(text, path.basename(agentFile));
  const mcpJsonFile = path.join(path.dirname(agentFile), 'mcp.json');
  let mcpJson: Json | null = null;
  if (fs.existsSync(mcpJsonFile)) {
    const loaded = JSON.parse(fs.readFileSync(mcpJsonFile, 'utf8')) as unknown;
    mcpJson = loaded && typeof loaded === 'object' && !Array.isArray(loaded) ? (loaded as Json) : {};
  }
  const [kind, servers] = agentServers(parsed.data, mcpJson);
  if (kind === 'none' || !(name in servers)) throw new RemoveRefused(WORDS.notThere.replace('{agent}', agentName).replace('{name}', name));
  const secrets = secretNames(servers[name]);
  if (kind === 'inline') {
    try {
      return { writes: [[agentFile, removeFromAgentFile(text, name, path.basename(agentFile))]], where: path.basename(agentFile), secrets };
    } catch (error) {
      if (!(error instanceof CannotEdit)) throw error;
      throw new RemoveRefused(WORDS.cannotRemove.replace('{file}', path.basename(agentFile)).replace('{name}', name));
    }
  }
  const data: Json = mcpJson ?? {};
  const isMap = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v);
  delete (isMap(data.mcpServers) ? data.mcpServers : data)[name];
  return { writes: [[mcpJsonFile, `${JSON.stringify(data, null, 2)}\n`]], where: 'mcp.json', secrets };
}

/**
 * `mcp remove` and `/mcp remove`: take the server `name` out of the agent
 * `target` names, and say what was done. Throws `RemoveRefused` with the reason.
 */
export function removeFromAgent(name: string, target: string, options: { inChat: boolean; cliCommand: (args: string) => string }): string[] {
  let agentFile: string;
  try {
    agentFile = agentFileFor(target, options.cliCommand);
  } catch (error) {
    if (error instanceof AddRefused) throw new RemoveRefused(error.message);
    throw error;
  }
  const agentName = agentNameOf(agentFile, fs.readFileSync(agentFile, 'utf8'));
  const { writes, where, secrets } = planRemove(name, agentFile, agentName);
  for (const [file, text] of writes) writeByRename(file, text);
  const lines = [WORDS.removed.replace('{name}', name).replace('{agent}', agentName).replace('{where}', where)];
  if (secrets.length) lines.push(WORDS.secretsStay.replace('{names}', secrets.join(', ')).replace('{command}', options.cliCommand('secrets remove <NAME>')));
  lines.push(options.inChat ? WORDS.reloadRemoved : WORDS.restartRemoved);
  return lines;
}

/**
 * The agent file `target` names: the file itself, or a folder's `AGENT.md`,
 * or its one `AGENT-<name>.md`; throws `AddRefused` otherwise.
 */
export function agentFileFor(target: string, cliCommand: (args: string) => string): string {
  const isFile = (p: string): boolean => {
    try {
      return fs.statSync(p).isFile();
    } catch {
      return false;
    }
  };
  if (isFile(target)) return target;
  let folder = target;
  try {
    if (!fs.statSync(target).isDirectory()) folder = path.dirname(target);
  } catch {
    folder = path.dirname(target);
  }
  if (isFile(path.join(folder, 'AGENT.md'))) return path.join(folder, 'AGENT.md');
  let named: string[] = [];
  try {
    named = fs.readdirSync(folder).filter((n) => /^AGENT-.+\.md$/.test(n)).sort();
  } catch {
    named = [];
  }
  if (named.length === 1) return path.join(folder, named[0]);
  if (named.length) throw new AddRefused(WORDS.manyAgents.replace('{folder}', folder).replace('{files}', named.join(', ')));
  throw new AddRefused(WORDS.noAgent.replace('{folder}', folder).replace('{command}', cliCommand('init')));
}

/** The agent's name as its file gives it: `name:`, else as the loaders name the file. */
export function agentNameOf(agentFile: string, text: string): string {
  let name: unknown;
  try {
    name = parseFrontMatter(text, path.basename(agentFile)).data.name;
  } catch {
    name = undefined;
  }
  if (typeof name === 'string' && name.trim()) return name.trim();
  const base = path.basename(agentFile);
  const stem = base.toLowerCase().endsWith('.md') ? base.slice(0, -3) : base;
  return stem === 'AGENT' ? 'default' : stem.startsWith('AGENT-') ? stem.slice('AGENT-'.length) : stem;
}

/**
 * `mcp add` and `/mcp add`: copy the server `name` into the agent `target`
 * names, and say what was done. Throws `AddRefused` with the reason.
 */
export async function addToAgent(
  name: string,
  target: string,
  source: string | undefined,
  options: { inChat: boolean; listCommand: string; cliCommand: (args: string) => string; store?: SecretStoreLike; discovery?: Discovery },
): Promise<string[]> {
  const agentFile = agentFileFor(target, options.cliCommand);
  const found = choose(options.discovery ?? discover(os.homedir(), path.dirname(agentFile)), name, source, options.listCommand);
  const agentName = agentNameOf(agentFile, fs.readFileSync(agentFile, 'utf8'));
  const plan = await planAdd(found, agentFile, options.store ?? (await secretStore()), agentName);
  writePlan(plan);
  return resultLines(plan, found.name, agentName, options.inChat, options.cliCommand);
}
