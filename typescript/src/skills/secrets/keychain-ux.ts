/**
 * Keychain dialogs on macOS: the item names each SDK files its secrets under,
 * the four lines said before macOS can ask, and the rule that nothing waits on
 * a question nobody can answer.
 *
 * WHY (the owner's request, 2026-09-27). macOS asks before a program reads a
 * keychain item it did not create, in a dialog that names the program ("node",
 * "Python") and the item. People met that dialog with no idea what it was:
 *
 *   - Both SDKs filed their items under the SAME names (`webagents:<ns>`), so
 *     signing in with one and running the other raised a dialog naming the
 *     other's interpreter.
 *   - The creator macOS trusts is the INTERPRETER, not webagents. Homebrew's
 *     node and Python are ad-hoc signed, so the keychain knows each by a hash
 *     of its binary, and after an upgrade even the creator is asked once more.
 *   - A process with nobody to answer waits forever. This CLI hung on exactly
 *     that in a headless shell on 2026-09-23.
 *
 * What this module does about it, identically to the Python twin
 * (`python/webagents/agents/skills/local/secrets/keychain_ux.py`), with every
 * name and sentence pinned by `python/tests/fixtures/keychain_ux/keychain_ux.json`:
 *
 *   1. SERVICE NAMES PER SDK: `webagents (TypeScript) <ns>` here,
 *      `webagents (Python) <ns>` there, readable in the dialog. Each SDK reads
 *      only its own. An item under the old shared name is copied the first
 *      time this SDK finds none of its own, and left in place.
 *   2. A RECORD beside the items (`keychain.json` in the profile's folder, 0600,
 *      no secret in it) of which program last used each one.
 *   3. THE EXPLANATION, four lines, at most once per run, in a terminal, before
 *      a read that may raise a dialog.
 *   4. NEVER BLOCK. The Python SDK switches macOS's dialogs off around each
 *      call (`SecKeychainSetUserInteractionAllowed`, proved on macOS 26.2 on
 *      2026-09-27). This one cannot: `@napi-rs/keyring` has no such switch and
 *      node has no FFI. So it decides BEFORE reading: the record says which
 *      program last used an item, and `security find-generic-password` without
 *      `-w` says whether it exists from its attributes alone, which cannot ask
 *      (measured the same day). An item another program may have made is read
 *      only in a terminal, after the explanation; `serve`, the daemon and a
 *      pipe get one sentence naming `webagents whoami` instead. And a read the
 *      record says is safe is still bounded by a timeout when nobody can answer
 *      (`AsyncEntry` with an abort signal), because a person who chose Allow
 *      rather than Always Allow left an item that asks again.
 *   5. NO DEFAULT KEYCHAIN MEANS NO KEYCHAIN. With HOME pointed elsewhere (a
 *      test, a CI runner) macOS finds none, and the first add shows its
 *      "keychain cannot be found" prompt and waits: a test suite hung on
 *      exactly that on 2026-09-27. `defaultKeychainAvailable` is checked
 *      before the store picks the keychain.
 *
 * Node built-ins are imported dynamically, as in `store.ts`: this SDK also
 * ships browser agents, and a static `node:` import would break their bundles.
 *
 * Names only, never a value, in the record, the sentences and the logs.
 */

export const RUNTIME = 'typescript' as const;
export const OTHER_RUNTIME = 'python' as const;
export type Runtime = 'python' | 'typescript';

export const RUNTIME_LABELS: Record<Runtime, string> = { python: 'Python', typescript: 'TypeScript' };
export const INTERPRETER_NAMES: Record<Runtime, string> = { python: 'Python', typescript: 'Node' };
export const CLI_NAMES: Record<Runtime, string> = { python: 'Python CLI', typescript: 'TypeScript CLI' };

export const SERVICE_TEMPLATE = 'webagents ({runtime}) {namespace}';
export const LEGACY_SERVICE_TEMPLATE = 'webagents:{namespace}';

export const RECORD_FILE = 'keychain.json';
export const RECORD_FILE_ELSEWHERE = 'keychain.record.json';
export const RECORD_ABOUT =
  'Which program last used each webagents keychain item on this machine. macOS asks before ' +
  'a different program reads an item, and webagents reads this file to say so first. It holds no secret.';

export const EXPLANATION = [
  'macOS is about to ask whether {program} may use "{item}" in your keychain: {program} is what webagents runs on.',
  'Choose Always Allow so it does not ask again.',
  'If it asks for your Mac password, the password goes to macOS, never to webagents.',
  'webagents only ever asks for items whose name starts with "webagents", so choose Deny for anything else.',
] as const;
export const BLOCKED =
  'macOS may ask before {program} can use "{item}" in your keychain, and nothing here can answer it: ' +
  'run `{command}` once in a terminal, then try again.';
export const BLOCKED_COMMAND = 'whoami';
export const LEFT_BEHIND =
  '"{item}" ({account}) from an earlier version of webagents is still in your keychain, because removing it ' +
  'would make macOS ask. To remove it, open Keychain Access, search for "{item}" and delete it.';
export const DENIED = 'macOS did not allow {program} to use "{item}", so webagents carries on without it.';
export const NO_DEFAULT_KEYCHAIN =
  'macOS has no default keychain for this user here (HOME is {home}), and making one would raise a dialog';
export const OTHER_SIGNED_IN = 'The {other} is signed in on this machine, but each CLI keeps its own sign-in in the keychain.';
export const OTHER_KEYS =
  'The {other} has keys stored on this machine, but each CLI keeps its own: store them for this one with `{command}`.';
export const OTHER_KEYS_COMMAND = 'secrets set NAME';

/** The `keychain` line of `doctor` and the `Keychain` row of the chat's /status. */
export const DOCTOR_WORDS = {
  name: 'keychain',
  file: 'not used: the sign-in and keys are in owner-only files in {dir}',
  keystore: 'the system keystore, which asks no per-program questions',
  keychain: 'macOS keychain: {parts}',
  items: '{count} {noun} named "webagents ({runtime}) ...", last used by {interpreter} {version}',
  none: 'nothing stored by the {cli} yet',
  quiet: 'the next use will not ask',
  upgraded: '{interpreter} was upgraded since the last use: macOS will ask once',
  changed: '{interpreter} changed since the last use: macOS will ask once',
  unknown: 'no record says which program made them: macOS may ask once',
  legacy: 'items from an earlier version are here: the next use in a terminal copies them, and macOS may ask once',
  pending: '{count} {noun} could not be read by a run with nobody to answer macOS',
  other: 'the {other} keeps its own items here',
  env_token: 'WEBAGENTS_TOKEN is set, so the sign-in is not read from here',
  fix: '`{command}` once in a terminal, then choose Always Allow',
  noun: { one: 'item', many: 'items' },
} as const;
export const STATUS_ROW = 'Keychain';

/**
 * The OSStatus values the Python SDK reads as "macOS would have asked" with
 * its dialogs off. Kept here for the shared fixture: this SDK has no such
 * switch, so it never sees them.
 */
export const DIALOG_STATUSES: Record<string, string> = {
  '-25293': 'errSecAuthFailed',
  '-25308': 'errSecInteractionNotAllowed',
  '-25244': 'errSecInvalidOwnerEdit',
  '-128': 'errSecUserCanceled',
};
export const ITEM_NOT_FOUND = -25300;
/** How long a keychain call with nobody to answer may take before it counts as a dialog. */
export const READ_TIMEOUT_MS = 10_000;
export const SECURITY_TOOL = '/usr/bin/security';

function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (whole, key: string) => (key in values ? String(values[key]) : whole));
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

/** The keychain service this SDK files `namespace` under. */
export function serviceName(namespace: string, runtime: Runtime = RUNTIME): string {
  return fill(SERVICE_TEMPLATE, { runtime: RUNTIME_LABELS[runtime], namespace });
}

/** The name both SDKs shared before 2026-09-27. Read once, to copy, and left in place. */
export function legacyServiceName(namespace: string): string {
  return fill(LEGACY_SERVICE_TEMPLATE, { namespace });
}

/**
 * What macOS calls the running program in its dialog: the bundle's name when
 * it runs inside `NAME.app/Contents/MacOS/`, else the file's name (`node`).
 */
export function programName(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  for (let i = 0; i + 2 < parts.length; i += 1) {
    if (parts[i].endsWith('.app') && parts[i + 1] === 'Contents' && parts[i + 2] === 'MacOS') {
      return parts[i].slice(0, -'.app'.length);
    }
  }
  return parts[parts.length - 1] ?? path;
}

export interface ProgramIdentity {
  runtime: Runtime;
  program: string;
  path: string;
  version: string;
  stamp: string;
}

let programCache: ProgramIdentity | null = null;

/**
 * This interpreter as the record keeps it: the program name macOS shows, the
 * real path, the version, and a stamp (size and modification time) that
 * changes when the binary is reinstalled at the same path and version.
 */
export async function currentProgram(): Promise<ProgramIdentity> {
  if (programCache) return { ...programCache };
  const fs = await import('node:fs');
  let path = process.execPath;
  try {
    path = fs.realpathSync(process.execPath);
  } catch {
    // The path as node reports it.
  }
  let stamp = '';
  try {
    const info = fs.statSync(path);
    stamp = `${info.size}:${Math.floor(info.mtimeMs / 1000)}`;
  } catch {
    stamp = '';
  }
  programCache = { runtime: RUNTIME, program: programName(path), path, version: process.versions.node, stamp };
  return { ...programCache };
}

export type Prediction = 'quiet' | 'upgraded' | 'changed' | 'unknown';

/** What the recorded program and this one mean for the next read of an item that exists. */
export function prediction(
  recorded: { path?: string; version?: string; stamp?: string } | null | undefined,
  current: { path?: string; version?: string; stamp?: string },
): Prediction {
  if (!recorded) return 'unknown';
  const sameStamp = !recorded.stamp || !current.stamp || recorded.stamp === current.stamp;
  if (recorded.path === current.path && recorded.version === current.version && sameStamp) return 'quiet';
  if (recorded.version !== current.version) return 'upgraded';
  return 'changed';
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

function splitPath(path: string): { parent: string; base: string; sep: string } {
  const trimmed = path.replace(/[\\/]+$/, '');
  const cut = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  const sep = cut >= 0 ? trimmed[cut] : '/';
  return { parent: cut >= 0 ? trimmed.slice(0, cut) || sep : '.', base: trimmed.slice(cut + 1), sep };
}

/**
 * Where the record for a store with this secrets folder lives: in its parent
 * when the folder is called `secrets` (the profile's folder), else inside it,
 * so a test's temporary folder never writes above itself.
 */
export function recordPath(secretsDir: string): string {
  const { parent, base, sep } = splitPath(secretsDir);
  const folder = secretsDir.replace(/[\\/]+$/, '');
  return base === 'secrets' ? `${parent}${parent.endsWith(sep) ? '' : sep}${RECORD_FILE}` : `${folder}${sep}${RECORD_FILE_ELSEWHERE}`;
}

function now(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

type Json = Record<string, unknown>;

export interface RecordEntry {
  program?: string;
  path?: string;
  version?: string;
  stamp?: string;
  at?: string;
}

export interface PendingEntry {
  service: string;
  account: string;
  namespace: string;
  secrets_dir: string;
  legacy: boolean;
  at?: string;
}

function isObject(value: unknown): value is Json {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (!isObject(value)) return value;
  const out: Json = {};
  for (const key of Object.keys(value).sort()) out[key] = sortKeys(value[key]);
  return out;
}

/** One writer at a time per record in this process: a lost update costs an explanation, never data. */
const recordLocks = new Map<string, Promise<void>>();

/**
 * Which program last used each item, what was copied from an old name, and
 * what a run with nobody to answer could not read. Names only. The file is
 * shared with the Python SDK, which keeps its own section.
 */
export class KeychainRecord {
  constructor(
    readonly path: string,
    readonly runtime: Runtime = RUNTIME,
  ) {}

  async read(): Promise<Json> {
    const fs = await import('node:fs/promises');
    let raw: string;
    try {
      raw = await fs.readFile(this.path, 'utf-8');
    } catch {
      return {};
    }
    try {
      const info = await fs.stat(this.path);
      if (info.mode & 0o077) await fs.chmod(this.path, 0o600);
    } catch {
      // 0600 when it can be; the record is a convenience.
    }
    try {
      const parsed = JSON.parse(raw) as unknown;
      return isObject(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  private section(data: Json, runtime?: Runtime, create = false): Json {
    const name = runtime ?? this.runtime;
    let runtimes = data.runtimes;
    if (!isObject(runtimes)) {
      if (!create) return {};
      runtimes = data.runtimes = {};
    }
    const all = runtimes as Json;
    let section = all[name];
    if (!isObject(section)) {
      if (!create) return {};
      section = all[name] = {};
    }
    const sec = section as Json;
    if (create) for (const key of ['items', 'legacy_done', 'pending']) if (!isObject(sec[key])) sec[key] = {};
    return sec;
  }

  private async update(change: (data: Json) => boolean): Promise<void> {
    const previous = recordLocks.get(this.path) ?? Promise.resolve();
    const next = previous.then(async () => {
      const data = await this.read();
      if (!change(data)) return;
      data.about = RECORD_ABOUT;
      await this.write(data);
    });
    recordLocks.set(this.path, next.catch(() => {}));
    await next.catch(() => {});
  }

  /** Atomically, 0600 from the first byte. A record that cannot be written costs an explanation later. */
  private async write(data: Json): Promise<void> {
    const fs = await import('node:fs/promises');
    const { parent, sep } = splitPath(this.path);
    const temp = `${parent}${sep}.keychain-${process.pid}-${Math.random().toString(36).slice(2)}.tmp`;
    try {
      await fs.mkdir(parent, { recursive: true, mode: 0o700 });
      await fs.writeFile(temp, `${JSON.stringify(sortKeys(data), null, 2)}\n`, { mode: 0o600 });
      await fs.rename(temp, this.path);
      await fs.chmod(this.path, 0o600).catch(() => {});
    } catch {
      await fs.unlink(temp).catch(() => {});
    }
  }

  async items(runtime?: Runtime): Promise<Record<string, Record<string, RecordEntry>>> {
    const found = this.section(await this.read(), runtime).items;
    const out: Record<string, Record<string, RecordEntry>> = {};
    if (!isObject(found)) return out;
    for (const [service, accounts] of Object.entries(found)) {
      if (!isObject(accounts)) continue;
      const entries: Record<string, RecordEntry> = {};
      for (const [account, entry] of Object.entries(accounts)) if (isObject(entry)) entries[account] = entry as RecordEntry;
      if (Object.keys(entries).length) out[service] = entries;
    }
    return out;
  }

  async item(service: string, account: string, runtime?: Runtime): Promise<RecordEntry | undefined> {
    return (await this.items(runtime))[service]?.[account];
  }

  /** This program read or wrote the item: record it (only when it changed), and forget any pending entry for it. */
  async noteUsed(service: string, account: string): Promise<void> {
    const me = await currentProgram();
    const entry = { program: me.program, path: me.path, version: me.version, stamp: me.stamp };
    await this.update((data) => {
      const sec = this.section(data, undefined, true);
      const items = sec.items as Json;
      const accounts = (isObject(items[service]) ? items[service] : (items[service] = {})) as Json;
      const old = accounts[account];
      let changed = false;
      if (!isObject(old) || Object.entries(entry).some(([key, value]) => old[key] !== value)) {
        accounts[account] = { ...entry, at: now() };
        changed = true;
      }
      return KeychainRecord.dropPending(sec, service, account) || changed;
    });
  }

  async forget(service: string, account: string): Promise<void> {
    await this.update((data) => {
      const sec = this.section(data);
      if (!Object.keys(sec).length) return false;
      let changed = false;
      const items = sec.items;
      if (isObject(items) && isObject(items[service]) && account in (items[service] as Json)) {
        delete (items[service] as Json)[account];
        if (!Object.keys(items[service] as Json).length) delete items[service];
        changed = true;
      }
      return KeychainRecord.dropPending(sec, service, account) || changed;
    });
  }

  /** True once this SDK copied, removed or replaced the old item: it is never read again. */
  async legacyDone(legacyService: string, account: string): Promise<boolean> {
    const done = this.section(await this.read()).legacy_done;
    const list = isObject(done) ? done[legacyService] : undefined;
    return Array.isArray(list) && list.includes(account);
  }

  async markLegacyDone(legacyService: string, account: string): Promise<void> {
    await this.update((data) => {
      const sec = this.section(data, undefined, true);
      const done = sec.legacy_done as Json;
      const list = Array.isArray(done[legacyService]) ? (done[legacyService] as string[]) : (done[legacyService] = []);
      if ((list as string[]).includes(account)) return false;
      (list as string[]).push(account);
      (list as string[]).sort();
      return true;
    });
  }

  private static dropPending(sec: Json, service: string, account: string): boolean {
    const pending = sec.pending;
    if (!isObject(pending) || !isObject(pending[service]) || !(account in (pending[service] as Json))) return false;
    delete (pending[service] as Json)[account];
    if (!Object.keys(pending[service] as Json).length) delete pending[service];
    return true;
  }

  async notePending(service: string, account: string, namespace: string, secretsDir: string, legacy: boolean): Promise<void> {
    await this.update((data) => {
      const sec = this.section(data, undefined, true);
      const pending = sec.pending as Json;
      const entries = (isObject(pending[service]) ? pending[service] : (pending[service] = {})) as Json;
      if (account in entries) return false;
      entries[account] = { namespace, secrets_dir: secretsDir, legacy, at: now() };
      return true;
    });
  }

  async clearPending(service: string, account: string): Promise<void> {
    await this.update((data) => {
      const sec = this.section(data);
      return Object.keys(sec).length ? KeychainRecord.dropPending(sec, service, account) : false;
    });
  }

  async pending(runtime?: Runtime): Promise<PendingEntry[]> {
    const pending = this.section(await this.read(), runtime).pending;
    const out: PendingEntry[] = [];
    if (!isObject(pending)) return out;
    for (const [service, accounts] of Object.entries(pending)) {
      if (!isObject(accounts)) continue;
      for (const [account, entry] of Object.entries(accounts)) {
        if (isObject(entry)) out.push({ service, account, ...(entry as Omit<PendingEntry, 'service' | 'account'>) });
      }
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Who may be asked
// ---------------------------------------------------------------------------

let forbidden: string | null = null;
let interactiveOverride: boolean | null = null;
let explained = false;
let saidBlocked = false;
const blockedSeen: Array<[string, string]> = [];
let writer: ((text: string) => void) | null = null;

/**
 * This process never makes a keychain call that may raise a dialog. Called by
 * `serve`, the daemon and every server entry point: a dialog there would
 * appear on someone's screen at a random later moment, or on none.
 */
export function forbidKeychainDialogs(reason = 'serve'): void {
  forbidden = reason || 'serve';
}

/** A person is at a terminal to answer: stdin and stderr are terminals and nothing forbade it. */
export function dialogsAllowed(): boolean {
  if (forbidden) return false;
  if (interactiveOverride !== null) return interactiveOverride;
  if (typeof process === 'undefined') return false;
  return Boolean(process.stdin?.isTTY && process.stderr?.isTTY);
}

/** Straight to stderr, never through `console`: `doctor` holds the console while an agent starts. */
function say(text: string): void {
  if (writer) {
    writer(text);
    return;
  }
  try {
    process.stderr.write(`${text}\n`);
  } catch {
    // Nowhere to say it.
  }
}

/** A `webagents` command as the person should type it: with `--profile` while one is active (`cliCommand`). */
export function cliCommandFor(rest: string): string {
  const active = typeof process !== 'undefined' ? process.env?.WEBAGENTS_PROFILE : undefined;
  let base = 'webagents';
  if (active) {
    const word = /^[A-Za-z0-9._-]+$/.test(active) ? active : `'${active.replace(/'/g, `'"'"'`)}'`;
    base = `webagents --profile ${word}`;
  }
  return rest ? `${base} ${rest}` : base;
}

export function explanationLines(item: string, program: string): string[] {
  return EXPLANATION.map((line) => fill(line, { program, item }));
}

/** The four lines, at most once per run. True when they were printed now. */
export async function explainOnce(item: string): Promise<boolean> {
  if (explained) return false;
  explained = true;
  say(explanationLines(item, (await currentProgram()).program).join('\n'));
  return true;
}

export async function blockedSentence(item: string): Promise<string> {
  return fill(BLOCKED, { program: (await currentProgram()).program, item, command: cliCommandFor(BLOCKED_COMMAND) });
}

/** A read was refused because nobody can answer: said once per run. */
export async function noteBlocked(item: string, account: string): Promise<string> {
  blockedSeen.push([item, account]);
  const sentence = await blockedSentence(item);
  if (!saidBlocked) {
    saidBlocked = true;
    say(sentence);
  }
  return sentence;
}

export function wasBlocked(item: string, account?: string): boolean {
  return blockedSeen.some(([seen, name]) => seen === item && (account === undefined || name === account));
}

export function leftBehindSentence(item: string, account: string): string {
  return fill(LEFT_BEHIND, { item, account });
}

/** Forget what this run said and forbade. Tests only. */
export function resetKeychainForTests(options: { interactive?: boolean | null; writer?: ((text: string) => void) | null } = {}): void {
  forbidden = null;
  interactiveOverride = options.interactive ?? null;
  explained = false;
  saidBlocked = false;
  blockedSeen.length = 0;
  writer = options.writer ?? null;
  programCache = null;
  defaultKeychainCache = undefined;
}

// ---------------------------------------------------------------------------
// The macOS calls
// ---------------------------------------------------------------------------

async function runSecurity(args: string[]): Promise<number | null> {
  const { execFile } = await import('node:child_process');
  return new Promise((resolve) => {
    try {
      execFile(SECURITY_TOOL, args, { timeout: 5000, windowsHide: true }, (error) => {
        if (!error) return resolve(0);
        const code = (error as { code?: unknown }).code;
        resolve(typeof code === 'number' ? code : null);
      });
    } catch {
      resolve(null);
    }
  });
}

let defaultKeychainCache: boolean | null | undefined;

/**
 * Whether macOS has a default keychain for this user under this HOME.
 * `security default-keychain` only reads the search list, never an item.
 */
export async function defaultKeychainAvailable(): Promise<boolean | null> {
  if (process.platform !== 'darwin') return null;
  if (defaultKeychainCache === undefined) {
    const code = await runSecurity(['default-keychain', '-d', 'user']);
    defaultKeychainCache = code === null ? null : code === 0;
  }
  return defaultKeychainCache;
}

/** The macOS-only calls, behind one object so tests can replace it. */
export class MacKeychain {
  private readonly cache = new Map<string, boolean | null>();

  /**
   * Whether an item exists, from its attributes alone: no secret is requested,
   * so no dialog is possible (measured 2026-09-27 on an item another program
   * made). `null` when it cannot be told.
   */
  async exists(service: string, account?: string, cache = false): Promise<boolean | null> {
    const key = `${service}\u0000${account ?? ''}`;
    if (cache && this.cache.has(key)) return this.cache.get(key)!;
    const code = await runSecurity(['find-generic-password', '-s', service, ...(account ? ['-a', account] : [])]);
    const found = code === 0 ? true : code === 44 ? false : null;
    if (cache) this.cache.set(key, found);
    return found;
  }

  /** Forget what was remembered about `service`: this process just wrote under it. */
  invalidate(service: string): void {
    for (const key of [...this.cache.keys()]) if (key.startsWith(`${service}\u0000`)) this.cache.delete(key);
  }
}

/** The macOS calls when the keychain behind `@napi-rs/keyring` is the macOS one. */
export function macKeychainFor(platform: string = typeof process !== 'undefined' ? process.platform : ''): MacKeychain | null {
  return platform === 'darwin' ? new MacKeychain() : null;
}

/** A write or delete that needs macOS to ask, in a run where nobody can answer. The message is the one sentence. */
export class KeychainDialogBlocked extends Error {
  constructor(
    sentence: string,
    readonly item: string,
    readonly account: string,
  ) {
    super(sentence);
    this.name = 'KeychainDialogBlocked';
  }
}

// ---------------------------------------------------------------------------
// The store's keychain calls
// ---------------------------------------------------------------------------

/** The slice of `@napi-rs/keyring` this uses. `AsyncEntry` bounds a call with nobody to answer. */
export interface KeyringEntryLike {
  setPassword(password: string): void;
  getPassword(): string | null;
  deletePassword(): boolean;
}
export interface AsyncKeyringEntryLike {
  setPassword(password: string, signal?: AbortSignal | null): Promise<void>;
  getPassword(signal?: AbortSignal | null): Promise<string | undefined | null>;
  deletePassword(signal?: AbortSignal | null): Promise<boolean>;
}
export interface KeyringModuleLike {
  Entry: new (service: string, username: string) => KeyringEntryLike;
  AsyncEntry?: new (service: string, username: string) => AsyncKeyringEntryLike;
}

export type KeychainOutcome = 'found' | 'adopted' | 'absent' | 'denied' | 'blocked';
export interface KeychainRead {
  value: string | null;
  outcome: KeychainOutcome;
}

const ABSENT: KeychainRead = { value: null, outcome: 'absent' };

function aborted(error: unknown): boolean {
  const e = error as { name?: string; message?: string } | null;
  return Boolean(e && (e.name === 'AbortError' || e.name === 'TimeoutError' || /abort|timed? ?out/i.test(e.message ?? '')));
}

function parentOf(file: string): string {
  return splitPath(file).parent;
}

/**
 * Every keychain call a `SecretStore` makes, under this SDK's own names, with
 * the old shared name copied once and the dialogs said before and never
 * waited on.
 */
export class KeychainAccess {
  readonly service: string;
  readonly legacyService: string;
  readonly mac: MacKeychain | null;
  readonly record: KeychainRecord;
  /** Old items `delete` did not remove (this SDK never removes one: it cannot prove no dialog), as `{item, account}`. */
  leftBehind: Array<{ item: string; account: string }> = [];
  /** The item and whether it was the old one, for the last read that was refused. */
  lastBlocked: { item: string; legacy: boolean } | null = null;
  private readonly timeoutMs: number;

  constructor(
    private readonly keyring: KeyringModuleLike,
    readonly namespace: string,
    readonly secretsDir: string,
    options: { mac?: MacKeychain | null; record?: KeychainRecord; timeoutMs?: number } = {},
  ) {
    this.service = serviceName(namespace);
    this.legacyService = legacyServiceName(namespace);
    this.mac = options.mac === undefined ? macKeychainFor() : options.mac;
    this.record = options.record ?? new KeychainRecord(recordPath(secretsDir));
    this.timeoutMs = options.timeoutMs ?? READ_TIMEOUT_MS;
  }

  /**
   * The access for a store whose fallback file is `filePath`. `macOSKeychain`
   * says the module is the real `@napi-rs/keyring`, which on macOS is the
   * keychain and gets the guard; any other module asks nobody anything.
   */
  static forFile(keyring: KeyringModuleLike, namespace: string, filePath: string, macOSKeychain = true): KeychainAccess {
    return new KeychainAccess(keyring, namespace, parentOf(filePath), macOSKeychain ? {} : { mac: null });
  }

  // -- reads -----------------------------------------------------------------

  /** `found`, `adopted` (copied from the old name now), `absent`, `denied` or `blocked` (nobody can answer). */
  async get(name: string): Promise<KeychainRead> {
    this.lastBlocked = null;
    const own = await this.read(this.service, name, false);
    if (own.outcome !== 'absent') return own;
    if (await this.record.legacyDone(this.legacyService, name)) return ABSENT;
    const old = await this.read(this.legacyService, name, true);
    if (old.outcome === 'found' && old.value !== null) {
      await this.adopt(name, old.value);
      return { value: old.value, outcome: 'adopted' };
    }
    return old;
  }

  private plainGet(service: string, name: string): string | null {
    return new this.keyring.Entry(service, name).getPassword() ?? null;
  }

  private async read(service: string, name: string, legacy: boolean): Promise<KeychainRead> {
    if (!this.mac) {
      // No per-program dialog here (Linux, Windows): read in any run.
      let value: string | null;
      try {
        value = this.plainGet(service, name);
      } catch {
        // The Rust crate raises for "no entry" on some backends. Absent.
        return ABSENT;
      }
      if (value === null) return ABSENT;
      if (!legacy) await this.record.noteUsed(service, name);
      return { value, outcome: 'found' };
    }
    if (legacy) {
      // One lookup for the whole old name first: with no old item at all
      // (a fresh machine, or everything copied) nothing else runs.
      if ((await this.mac.exists(service, undefined, true)) === false) return ABSENT;
      const exists = await this.mac.exists(service, name, true);
      if (exists === false) return ABSENT;
      // An old item is never read in a run with nobody to answer: its
      // existence is enough to say what to do. In a terminal it is read
      // after the explanation, since an earlier version, perhaps the other
      // SDK's, made it.
      if (!dialogsAllowed()) return this.blocked(service, true);
      return this.readAfterExplaining(service, name, true);
    }
    const recorded = await this.record.item(service, name);
    if (recorded && prediction(recorded, await currentProgram()) === 'quiet') return this.readBounded(service, name);
    // One lookup for the whole name first, remembered for the run: with no
    // item under it (a fresh machine, or keys never stored) no per-key lookup
    // runs. This store's own writes forget it (`set`).
    const any = await this.mac.exists(service, undefined, true);
    if (any === false) {
      if (recorded) await this.record.forget(service, name);
      return ABSENT;
    }
    const exists = await this.mac.exists(service, name);
    if (exists === false) {
      if (recorded) await this.record.forget(service, name);
      return ABSENT;
    }
    // It exists (or cannot be told) and another program may have made it.
    if (!dialogsAllowed()) return this.blocked(service, false);
    return this.readAfterExplaining(service, name, false);
  }

  /**
   * A read the record says will not ask. With nobody to answer it is still
   * bounded: an item the person chose Allow (not Always Allow) for asks again.
   */
  private async readBounded(service: string, name: string): Promise<KeychainRead> {
    let value: string | null;
    if (dialogsAllowed() || !this.keyring.AsyncEntry) {
      try {
        value = this.plainGet(service, name);
      } catch {
        return ABSENT;
      }
    } else {
      try {
        value = (await new this.keyring.AsyncEntry(service, name).getPassword(AbortSignal.timeout(this.timeoutMs))) ?? null;
      } catch (error) {
        if (!aborted(error)) return ABSENT;
        // It asked after all: the record was wrong, so drop it.
        await this.record.forget(service, name);
        return this.blocked(service, false);
      }
    }
    if (value === null) return ABSENT;
    await this.record.noteUsed(service, name);
    return { value, outcome: 'found' };
  }

  private async readAfterExplaining(service: string, name: string, legacy: boolean): Promise<KeychainRead> {
    await explainOnce(service);
    let value: string | null;
    try {
      value = this.plainGet(service, name);
    } catch {
      say(fill(DENIED, { program: (await currentProgram()).program, item: service }));
      return { value: null, outcome: 'denied' };
    }
    if (value === null) return ABSENT;
    if (!legacy) await this.record.noteUsed(service, name);
    return { value, outcome: 'found' };
  }

  private blocked(service: string, legacy: boolean): KeychainRead {
    this.lastBlocked = { item: service, legacy };
    return { value: null, outcome: 'blocked' };
  }

  /**
   * Say the sentence (once per run) and remember the item for the next
   * `webagents whoami` in a terminal. The store calls this when no file copy
   * could stand in.
   */
  async reportBlocked(name: string): Promise<string> {
    const { item, legacy } = this.lastBlocked ?? { item: this.service, legacy: false };
    await this.record.notePending(this.service, name, this.namespace, this.secretsDir, legacy);
    return noteBlocked(item, name);
  }

  /** Copy an old item to this SDK's name. The old one stays. */
  private async adopt(name: string, value: string): Promise<void> {
    try {
      new this.keyring.Entry(this.service, name).setPassword(value);
    } catch {
      return;
    }
    await this.record.noteUsed(this.service, name);
    await this.record.markLegacyDone(this.legacyService, name);
  }

  // -- writes ----------------------------------------------------------------

  /**
   * Whether replacing or removing this SDK's item may ask: the Rust keyring
   * reads an item before it changes it, and macOS asks a program that did not
   * make it. Throws the one sentence when nobody can answer.
   */
  private async guardChange(name: string): Promise<'quiet' | 'asks' | 'absent'> {
    if (!this.mac) return 'quiet';
    const recorded = await this.record.item(this.service, name);
    if (recorded && prediction(recorded, await currentProgram()) === 'quiet') return 'quiet';
    if ((await this.mac.exists(this.service, name)) === false) return 'absent';
    if (!dialogsAllowed()) throw new KeychainDialogBlocked(await blockedSentence(this.service), this.service, name);
    await explainOnce(this.service);
    return 'asks';
  }

  async set(name: string, value: string): Promise<void> {
    const guard = await this.guardChange(name);
    if (guard === 'quiet' && this.mac && !dialogsAllowed() && this.keyring.AsyncEntry) {
      try {
        await new this.keyring.AsyncEntry(this.service, name).setPassword(value, AbortSignal.timeout(this.timeoutMs));
      } catch (error) {
        if (!aborted(error)) throw error;
        await this.record.forget(this.service, name);
        throw new KeychainDialogBlocked(await blockedSentence(this.service), this.service, name);
      }
    } else {
      new this.keyring.Entry(this.service, name).setPassword(value);
    }
    this.mac?.invalidate(this.service);
    await this.record.noteUsed(this.service, name);
    // A new value replaces whatever the old shared name held.
    await this.record.markLegacyDone(this.legacyService, name);
  }

  /**
   * Remove this SDK's item, and retire the old one: this SDK never removes an
   * old item (it cannot prove that removing it asks nothing), so an old item
   * that exists is named in `leftBehind`. Either way it is never read again,
   * so a removed secret does not come back.
   */
  async delete(name: string): Promise<boolean> {
    this.leftBehind = [];
    let removed = await this.deleteOwn(name);
    removed = (await this.retireLegacy(name)) || removed;
    await this.record.forget(this.service, name);
    return removed;
  }

  private async deleteOwn(name: string): Promise<boolean> {
    let guard: 'quiet' | 'asks' | 'absent';
    guard = await this.guardChange(name);
    if (guard === 'absent') return false;
    if (guard === 'quiet' && this.mac && !dialogsAllowed() && this.keyring.AsyncEntry) {
      try {
        return await new this.keyring.AsyncEntry(this.service, name).deletePassword(AbortSignal.timeout(this.timeoutMs));
      } catch (error) {
        if (!aborted(error)) return false;
        await this.record.forget(this.service, name);
        throw new KeychainDialogBlocked(await blockedSentence(this.service), this.service, name);
      }
    }
    try {
      return new this.keyring.Entry(this.service, name).deletePassword();
    } catch {
      return false;
    }
  }

  private async retireLegacy(name: string): Promise<boolean> {
    let removed = false;
    try {
      if (!this.mac) {
        try {
          removed = new this.keyring.Entry(this.legacyService, name).deletePassword();
        } catch {
          removed = false;
        }
      } else if ((await this.mac.exists(this.legacyService, undefined, true)) !== false && (await this.mac.exists(this.legacyService, name))) {
        this.leftBehind.push({ item: this.legacyService, account: name });
        removed = true;
      }
    } finally {
      await this.record.markLegacyDone(this.legacyService, name);
    }
    return removed;
  }

  // -- names -----------------------------------------------------------------

  /** The old index's names this SDK has not copied or retired yet: still readable through the copy. */
  async legacyNames(legacyIndex: string[]): Promise<string[]> {
    const out: string[] = [];
    for (const name of legacyIndex) if (!(await this.record.legacyDone(this.legacyService, name))) out.push(name);
    return out;
  }
}

// ---------------------------------------------------------------------------
// `webagents whoami`, the other SDK, and `doctor`
// ---------------------------------------------------------------------------

/**
 * `webagents whoami` in a terminal: read every item a run with nobody to
 * answer could not read (macOS asks now, after the explanation), so that run
 * can read it next time. Values are read and dropped. Returns how many were read.
 */
export async function settlePending(
  recordPaths: string[],
  openStore: (namespace: string, secretsDir: string) => Promise<{ get(name: string): Promise<string | null> }>,
): Promise<number> {
  if (!dialogsAllowed()) return 0;
  let settled = 0;
  const seen = new Set<string>();
  for (const path of [...new Set(recordPaths)]) {
    const record = new KeychainRecord(path);
    for (const entry of await record.pending()) {
      const key = `${entry.namespace}\u0000${entry.secrets_dir}\u0000${entry.account}`;
      if (seen.has(key) || !entry.namespace || !entry.secrets_dir || !entry.account) continue;
      seen.add(key);
      let value: string | null;
      try {
        value = await (await openStore(entry.namespace, entry.secrets_dir)).get(entry.account);
      } catch {
        continue;
      }
      if (value !== null) settled += 1;
      // Gone, or the person said no: in a terminal nothing is left to settle.
      else await record.clearPending(entry.service, entry.account);
    }
  }
  return settled;
}

/** The names the other SDK recorded under `namespace`, never values. */
export async function otherRuntimeItems(record: KeychainRecord, namespace: string): Promise<string[]> {
  return Object.keys((await record.items(OTHER_RUNTIME))[serviceName(namespace, OTHER_RUNTIME)] ?? {}).sort();
}

export function otherSignedInSentence(): string {
  return fill(OTHER_SIGNED_IN, { other: CLI_NAMES[OTHER_RUNTIME] });
}

export function otherKeysSentence(): string {
  return fill(OTHER_KEYS, { other: CLI_NAMES[OTHER_RUNTIME], command: cliCommandFor(OTHER_KEYS_COMMAND) });
}

export function shortPath(path: string, home: string): string {
  return path === home || path.startsWith(`${home}/`) ? `~${path.slice(home.length)}` : path;
}

export interface KeychainFacts {
  backend: 'keychain' | 'keystore' | 'file';
  runtime: Runtime;
  dir?: string;
  count?: number;
  interpreter_version?: string;
  prediction?: Prediction | 'none';
  legacy?: boolean;
  pending?: number;
  other?: boolean;
  env_token?: boolean;
  command?: string;
}

export interface KeychainLine {
  name: string;
  status: 'ok' | 'warn';
  detail: string;
  fix?: string;
}

/** The `keychain` line from what `doctorFacts` found. Pure, so both SDKs are held to the fixture's cases. */
export function doctorLine(facts: KeychainFacts): KeychainLine {
  const words = DOCTOR_WORDS;
  if (facts.backend === 'file') return { name: words.name, status: 'ok', detail: fill(words.file, { dir: facts.dir ?? '' }) };
  if (facts.backend !== 'keychain') return { name: words.name, status: 'ok', detail: words.keystore };
  const runtime = facts.runtime ?? RUNTIME;
  const other: Runtime = runtime === 'python' ? 'typescript' : 'python';
  const interpreter = INTERPRETER_NAMES[runtime];
  const noun = (count: number) => (count === 1 ? words.noun.one : words.noun.many);
  const count = Number(facts.count ?? 0);
  const parts: string[] = [];
  let warn = false;
  if (count) {
    parts.push(fill(words.items, { count, noun: noun(count), runtime: RUNTIME_LABELS[runtime], interpreter, version: facts.interpreter_version ?? '' }));
    const kind = facts.prediction ?? 'quiet';
    parts.push(kind === 'upgraded' || kind === 'changed' || kind === 'unknown' ? fill(words[kind], { interpreter }) : words.quiet);
    warn = kind === 'upgraded' || kind === 'changed' || kind === 'unknown';
  } else {
    parts.push(fill(words.none, { cli: CLI_NAMES[runtime] }));
  }
  if (facts.legacy) {
    parts.push(words.legacy);
    warn = true;
  }
  const pending = Number(facts.pending ?? 0);
  if (pending) {
    parts.push(fill(words.pending, { count: pending, noun: noun(pending) }));
    warn = true;
  }
  if (!count && facts.other) parts.push(fill(words.other, { other: CLI_NAMES[other] }));
  if (facts.env_token) parts.push(words.env_token);
  const line: KeychainLine = { name: words.name, status: warn ? 'warn' : 'ok', detail: fill(words.keychain, { parts: parts.join('; ') }) };
  if (warn) line.fix = fill(words.fix, { command: facts.command ?? cliCommandFor(BLOCKED_COMMAND) });
  return line;
}

/** What a store tells `doctorFacts`. */
export interface KeychainStoreLike {
  readonly namespace: string;
  status(): { backend: string; path?: string };
  ownIndexNames(): Promise<string[]>;
}

/**
 * What the `keychain` line says, gathered WITHOUT reading a value: the record,
 * this program, and attribute-only lookups. `stores` are the CLI's open stores
 * (the sign-in's and the keys'); `legacyAccounts` maps an old service name to
 * the accounts worth looking for under it.
 */
export async function doctorFacts(
  stores: KeychainStoreLike[],
  record: KeychainRecord,
  legacyAccounts: Record<string, string[]>,
  options: { envToken?: boolean; mac?: MacKeychain | null; home?: string } = {},
): Promise<KeychainFacts> {
  const first = stores[0];
  const status = first ? first.status() : { backend: 'file', path: '' };
  if (status.backend !== 'keystore') {
    const home = options.home ?? (await import('node:os')).homedir();
    return { backend: 'file', runtime: RUNTIME, dir: shortPath(parentOf(status.path ?? ''), home) };
  }
  const probe = options.mac === undefined ? macKeychainFor() : options.mac;
  if (!probe) return { backend: 'keystore', runtime: RUNTIME };
  const me = await currentProgram();
  const services = new Set(stores.map((store) => serviceName(store.namespace)));
  const recorded = await record.items();
  const mine: Record<string, Record<string, RecordEntry | undefined>> = {};
  for (const [service, accounts] of Object.entries(recorded)) if (services.has(service)) mine[service] = { ...accounts };
  for (const store of stores) {
    const service = serviceName(store.namespace);
    for (const account of await store.ownIndexNames().catch(() => [] as string[])) {
      mine[service] ??= {};
      if (!(account in mine[service])) mine[service][account] = undefined;
    }
  }
  const order: Prediction[] = ['quiet', 'unknown', 'changed', 'upgraded'];
  let count = 0;
  let worst: Prediction = 'quiet';
  let latest: RecordEntry | undefined;
  for (const [service, accounts] of Object.entries(mine)) {
    for (const [account, entry] of Object.entries(accounts)) {
      if ((await probe.exists(service, account)) !== true) continue;
      count += 1;
      const kind = prediction(entry, me);
      if (order.indexOf(kind) > order.indexOf(worst)) worst = kind;
      if (entry && (!latest || String(entry.at ?? '') > String(latest.at ?? ''))) latest = entry;
    }
  }
  let legacy = false;
  for (const [legacyService, accounts] of Object.entries(legacyAccounts)) {
    if ((await probe.exists(legacyService)) !== true) continue;
    for (const account of accounts) {
      if (!(await record.legacyDone(legacyService, account)) && (await probe.exists(legacyService, account))) {
        legacy = true;
        break;
      }
    }
    if (legacy) break;
  }
  const others = await record.items(OTHER_RUNTIME);
  return {
    backend: 'keychain',
    runtime: RUNTIME,
    count,
    interpreter_version: latest?.version || me.version,
    prediction: count ? worst : 'none',
    legacy,
    pending: (await record.pending()).length,
    other: stores.some((store) => serviceName(store.namespace, OTHER_RUNTIME) in others),
    env_token: Boolean(options.envToken),
  };
}
