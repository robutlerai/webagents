/**
 * What `robutler app` needs beneath its commands: the HTTP client, the rule
 * for printing what a server sent, the kind of credential in hand, and the
 * file that remembers which app a folder is.
 *
 * THREE RULES LIVE HERE, so no command can forget them:
 *
 *  - TEXT FROM THE SERVER IS DATA. Every string a response carries goes
 *    through `clean` before it reaches the terminal: escape sequences,
 *    control characters and direction overrides are removed, so a name or an
 *    error message cannot move the cursor, recolour the screen or hide text.
 *  - AN ERROR NEVER CARRIES WHAT WAS SENT. `AppApiError` is built from the
 *    status and the response's own code and message. The request body is
 *    never part of it, and a caller can name values (`redact`) that must not
 *    appear even if the server echoes them back.
 *  - THE CREDENTIAL IS SENT TO ONE ORIGIN: the portal this client was made
 *    for. A redirect is not followed.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

/** Longest single string printed from a response. */
const CLEAN_MAX = 2000;
/** Largest response body read. */
const BODY_MAX = 2 * 1024 * 1024;
const TIMEOUT_MS = 60_000;

// ESC-introduced sequences: CSI (ESC [ ... final), OSC (ESC ] ... BEL or ESC \), and two-character escapes.
const ESCAPES = /\u001b\[[0-?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)?|\u001b[@-Z\\-_]/g;
// C0 and C1 controls, DEL, and the Unicode direction and invisible-format characters.
const CONTROLS = /[\u0000-\u001f\u007f-\u009f؜​-‏‪-‮⁠-⁩﻿]/g;

/**
 * A server string, made safe to print on one line. Anything that is not a
 * string prints as its JSON, cleaned the same way.
 */
export function clean(value: unknown, max = CLEAN_MAX): string {
  let text: string;
  if (typeof value === 'string') text = value;
  else if (value === null || value === undefined) text = '';
  else if (typeof value === 'number' || typeof value === 'boolean') text = String(value);
  else {
    try {
      text = JSON.stringify(value) ?? '';
    } catch {
      text = '';
    }
  }
  text = text.replace(ESCAPES, '').replace(/[\r\n\t]+/g, ' ').replace(CONTROLS, '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/** `clean`, with every occurrence of the named values replaced. */
export function cleanWithout(value: unknown, redact: readonly string[]): string {
  let text = typeof value === 'string' ? value : clean(value);
  for (const secret of redact) {
    if (secret.length >= 1) text = text.split(secret).join('[redacted]');
  }
  return clean(text);
}

/** What a command failed with: a stable code, one sentence, and what to do next when there is something to do. */
export class AppCliError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly fix = '',
    /** The process exit code: 1 failed, 2 usage, 3 the security change was not accepted, 4 continue in the browser. */
    readonly exitCode = 1,
  ) {
    super(message);
    this.name = 'AppCliError';
  }
}

/** A response that was not a success. `code` and `message` are the server's own, cleaned. */
export class AppApiError extends AppCliError {
  constructor(
    readonly status: number,
    code: string,
    message: string,
    readonly body: Record<string, unknown> | null = null,
    fix = '',
  ) {
    super(code, message, fix);
    this.name = 'AppApiError';
  }
}

// ---------------------------------------------------------------------------
// The credential
// ---------------------------------------------------------------------------

export type CredentialKind =
  /** A signed sign-in token with nothing that marks it as an agent's or a release token. */
  | 'sign-in'
  /** A raw API key: it can be held by anything, so it is not a person's act. */
  | 'api-key'
  /** A token that names an agent. */
  | 'agent-key'
  /** A release token for CI. */
  | 'release-token'
  | 'none';

function jwtPayload(token: string): Record<string, unknown> | null {
  const parts = token.split('.');
  if (parts.length !== 3 || parts.some((p) => !/^[A-Za-z0-9_-]+$/.test(p))) return null;
  try {
    const parsed: unknown = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf8'));
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/**
 * What kind of credential a token is, read from the token alone. Nothing is
 * verified here (the server does that); this only decides what the CLI is
 * willing to attempt with it.
 */
export function credentialKind(token: string | null | undefined): CredentialKind {
  if (!token) return 'none';
  const payload = jwtPayload(token.trim());
  if (!payload) return 'api-key';
  const scopes = Array.isArray(payload.scopes) ? payload.scopes.filter((s): s is string => typeof s === 'string') : [];
  if (scopes.includes('apps:release')) return 'release-token';
  if (typeof payload.agent_id === 'string' && payload.agent_id) return 'agent-key';
  if (payload.daemon && typeof payload.daemon === 'object') return 'agent-key';
  return 'sign-in';
}

const NOT_A_PERSON: Record<Exclude<CredentialKind, 'sign-in'>, string> = {
  none: 'You are not signed in.',
  'api-key': 'This run uses an API key. An API key can be held by a program, so it cannot do this.',
  'agent-key': "This run uses an agent's key. An agent cannot do this.",
  'release-token': 'This run uses a release token. A release token releases from CI and does nothing else.',
};

// ---------------------------------------------------------------------------
// The HTTP client
// ---------------------------------------------------------------------------

export interface ApiAnswer {
  status: number;
  body: Record<string, unknown>;
}

export interface AppApiOptions {
  /** The portal's origin, with no trailing slash. */
  baseUrl: string;
  /** The signed-in credential, or null. */
  token: string | null;
  fetch?: typeof fetch;
}

export interface RequestOptions {
  body?: unknown;
  /** Send this bearer instead of the client's own (the CI release). `null` sends none. */
  bearer?: string | null;
  /** Statuses returned to the caller instead of thrown. 2xx always is. */
  accept?: readonly number[];
  /** Values that must never be printed, even when the server sends them back. */
  redact?: readonly string[];
}

export class AppApi {
  readonly baseUrl: string;
  readonly token: string | null;
  private readonly fetchImpl: typeof fetch;

  constructor(options: AppApiOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.token = options.token && options.token.trim() ? options.token.trim() : null;
    this.fetchImpl = options.fetch ?? fetch;
  }

  /** The portal's host, for messages. */
  get portalHost(): string {
    return this.baseUrl.replace(/^https?:\/\//, '');
  }

  async request(method: string, pathname: string, options: RequestOptions = {}): Promise<ApiAnswer> {
    const bearer = options.bearer === undefined ? this.token : options.bearer;
    const headers: Record<string, string> = { accept: 'application/json' };
    if (bearer) headers.authorization = `Bearer ${bearer}`;
    if (options.body !== undefined) headers['content-type'] = 'application/json';
    const redact = [...(options.redact ?? []), ...(bearer ? [bearer] : [])];

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${pathname}`, {
        method,
        headers,
        ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {}),
        // The credential goes to the portal and nowhere a redirect points.
        redirect: 'manual',
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch (error) {
      const err = error as Error;
      throw new AppCliError(
        'unreachable',
        err.name === 'TimeoutError' ? `No answer from ${this.portalHost} within ${TIMEOUT_MS / 1000} seconds.` : `Could not reach ${this.portalHost}.`,
        'Check the network and the portal address (`webagents config get platform.url`).',
      );
    }

    const text = await readCapped(response);
    let body: Record<string, unknown> = {};
    if (text) {
      try {
        const parsed: unknown = JSON.parse(text);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) body = parsed as Record<string, unknown>;
      } catch {
        // Not JSON: the status alone speaks.
      }
    }
    if ((response.status >= 200 && response.status < 300) || options.accept?.includes(response.status)) {
      return { status: response.status, body };
    }
    throw apiError(response.status, body, this.portalHost, redact);
  }
}

async function readCapped(response: Response): Promise<string> {
  try {
    const text = await response.text();
    return text.length > BODY_MAX ? '' : text;
  } catch {
    return '';
  }
}

/** The portal's two error shapes, `{error: {code, message}}` and `{error: "..."}`, as one. */
function apiError(status: number, body: Record<string, unknown>, host: string, redact: readonly string[]): AppApiError {
  const raw = body.error;
  let code = '';
  let message = '';
  if (raw && typeof raw === 'object') {
    const e = raw as Record<string, unknown>;
    if (typeof e.code === 'string') code = e.code;
    if (typeof e.message === 'string') message = e.message;
  } else if (typeof raw === 'string') {
    message = raw;
  }
  if (!code && typeof body.code === 'string') code = body.code;
  if (!message && typeof body.message === 'string') message = body.message;
  code = clean(code, 80).replace(/[^A-Za-z0-9:_.-]/g, '_');
  if (!code) {
    code =
      status === 401 ? 'unauthorized' : status === 403 ? 'forbidden' : status === 404 ? 'not_found' : status === 429 ? 'rate_limited' : `http_${status}`;
  }
  const shown = cleanWithout(message, redact) || `${host} answered ${status}.`;
  return new AppApiError(status, code, shown, body, status === 401 ? 'Sign in again with `robutler login`.' : '');
}

// ---------------------------------------------------------------------------
// A person, or not
// ---------------------------------------------------------------------------

/**
 * Refuse, by name, anything but a person's own sign-in. Used before a release
 * and before a release token is made: both decide what code runs for other
 * people, so neither is something a key or an agent does.
 *
 * Two checks: the token itself (a raw key, an agent's token and a release
 * token are refused before any request), then the account the portal says it
 * belongs to. The portal applies its own rule as well; this one exists so the
 * refusal comes first and says why.
 */
export async function assertPerson(api: AppApi, what: string): Promise<{ username: string }> {
  const kind = credentialKind(api.token);
  if (kind !== 'sign-in') {
    throw new AppCliError(
      'not_a_person',
      `${NOT_A_PERSON[kind]} Only a person, signed in as themselves, can ${what}.`,
      'Sign in with `robutler login` (the browser sign-in), with no --token and no WEBAGENTS_TOKEN set.',
    );
  }
  const { body } = await api.request('GET', '/api/users/me');
  const user = (body.user && typeof body.user === 'object' ? body.user : body) as Record<string, unknown>;
  const accountType = typeof user.accountType === 'string' ? user.accountType : typeof user.account_type === 'string' ? user.account_type : '';
  // The portal's word for a person's account is `human`. Anything else, an unknown value included, is refused.
  if (accountType !== 'human') {
    throw new AppCliError(
      'not_a_person',
      `This sign-in is not a person's account. Only a person, signed in as themselves, can ${what}.`,
      'Sign in with `robutler login` as the person who owns the app.',
    );
  }
  return { username: clean(user.username, 80) };
}

// ---------------------------------------------------------------------------
// Which app a folder is
// ---------------------------------------------------------------------------

export const APP_STATE_DIR = '.robutler';
export const APP_STATE_FILE = 'app.json';
export const APP_REPO_FILE = 'robutler.json';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** An app's id is the id of its source folder on the portal. */
export function isAppId(value: unknown): value is string {
  return typeof value === 'string' && UUID.test(value);
}

export interface AppBinding {
  id: string;
  name?: string;
  /** The row that opens the app. */
  widgetId?: string;
  /** The agent that runs the app's functions, when it has one. */
  agentId?: string;
}

interface AppStateFile {
  v: 1;
  /** One entry per portal: the same folder is a different app on each. */
  apps: Record<string, AppBinding>;
}

/** The app's root: the nearest folder, from `cwd` upwards, that holds `.robutler/app.json` or `robutler.json`. */
export function appRoot(cwd: string): string {
  let dir = path.resolve(cwd);
  for (;;) {
    if (fs.existsSync(path.join(dir, APP_STATE_DIR, APP_STATE_FILE)) || fs.existsSync(path.join(dir, APP_REPO_FILE))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) return path.resolve(cwd);
    dir = parent;
  }
}

function readState(root: string): AppStateFile {
  try {
    const parsed = JSON.parse(fs.readFileSync(path.join(root, APP_STATE_DIR, APP_STATE_FILE), 'utf8')) as Partial<AppStateFile>;
    if (parsed && typeof parsed === 'object' && parsed.apps && typeof parsed.apps === 'object') return { v: 1, apps: parsed.apps };
  } catch {
    // Absent or unreadable: no app is remembered.
  }
  return { v: 1, apps: {} };
}

/** The app this folder is on `baseUrl`, or null. */
export function readBinding(root: string, baseUrl: string): AppBinding | null {
  const entry = readState(root).apps[baseUrl];
  if (!entry || !isAppId(entry.id)) return null;
  return {
    id: entry.id,
    ...(typeof entry.name === 'string' ? { name: clean(entry.name, 120) } : {}),
    ...(isAppId(entry.widgetId) ? { widgetId: entry.widgetId } : {}),
    ...(isAppId(entry.agentId) ? { agentId: entry.agentId } : {}),
  };
}

/**
 * Remember which app this folder is on `baseUrl`. The file holds ids and a
 * name, never a credential, so it can be committed with the app.
 */
export function writeBinding(root: string, baseUrl: string, binding: AppBinding): string {
  const state = readState(root);
  state.apps[baseUrl] = binding;
  const dir = path.join(root, APP_STATE_DIR);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, APP_STATE_FILE);
  fs.writeFileSync(file, `${JSON.stringify(state, null, 2)}\n`);
  return file;
}

/** `sourcePath` from the folder's `robutler.json`: where the served files are, relative to the root. */
export function repoSourcePath(root: string): string | null {
  try {
    const parsed = JSON.parse(fs.readFileSync(path.join(root, APP_REPO_FILE), 'utf8')) as { sourcePath?: unknown };
    return typeof parsed.sourcePath === 'string' && parsed.sourcePath ? parsed.sourcePath : null;
  } catch {
    return null;
  }
}
