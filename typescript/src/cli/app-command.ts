/**
 * `robutler app`: an app's life from a terminal.
 *
 *   app create <name> [--template <t>]        make a new private app and bind this folder to it
 *   app use <app id>                          bind this folder to an app that already exists
 *   app link <owner/repo> [--branch <b>] [--path <dir>] [--no-follow]
 *                                             connect the app's source repository (finished in the browser)
 *   app push [dir] [--dry-run] [--keep-extra] copy the served files into the app's draft
 *   app status                                the draft, the release and the repository link
 *   app release [--ref <tag>] [--expect-commit <sha>] [--accept-security <hash>] [--dry-run]
 *   app release --ci --ref <tag> --expect-commit <sha>
 *   app hooks run <up|down|post> | app hooks status
 *   app secrets set|unset <NAME> --function <fn> | app secrets list --function <fn>
 *   app logs --function <fn> [--limit <n>]
 *   app token create [--name <n>] [--all-apps] [--days <n>] | app token list | app token revoke <id>
 *
 * Every command is one or two requests to the portal's HTTP API under the
 * CLI's sign-in. What the commands will not do is part of the design:
 *
 *  - A RELEASE IS A PERSON'S ACT. `release` and `token create` refuse an API
 *    key, an agent's key and an account that is not a person's, before any
 *    request that changes something (`assertPerson`). The portal applies its
 *    own rule too; an answer of 202 means the owner was asked in the browser
 *    and nothing is released yet.
 *  - A RELEASE THAT WIDENS WHAT THE APP MAY REACH IS NEVER ACCEPTED FOR YOU.
 *    The change is printed with its hash, and the release goes ahead only
 *    with `--accept-security <that hash>`. A first release is treated the
 *    same way: it makes the app public.
 *  - A SECRET'S VALUE COMES FROM STANDARD INPUT ONLY. A value on the command
 *    line is refused (it would be in the shell history and the process
 *    list). The value is never printed, and never part of an error.
 *  - A RELEASE TOKEN IS SHOWN ONCE. It is written to standard output and
 *    nowhere else; this command keeps no copy.
 *  - LINKING HAPPENS IN THE BROWSER. `link` opens the portal's own page for
 *    it; the CLI's credential is not part of that address.
 *
 * `runApp` takes everything it touches as `AppDeps`, so the tests drive every
 * command against a local server with no network and no keystore.
 */

import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  AppApi,
  AppApiError,
  AppCliError,
  type AppBinding,
  appRoot,
  assertPerson,
  clean,
  cleanWithout,
  isAppId,
  readBinding,
  repoSourcePath,
  writeBinding,
} from './app-api';

export const APP_USAGE = `robutler app - your apps on Robutler

Usage:
  robutler app <command> [options]

Commands:
  create <name>                 Make a new private app and bind this folder to it
  use <app id>                  Bind this folder to an app that already exists
  link <owner/repo>             Connect the app's source repository (finished in your browser)
  push [dir]                    Copy the served files into the app's draft
  status                        Show the draft, the release and the repository link
  release                       Release the draft, or a tag of the linked repository
  hooks run <up|down|post>      Run one of the app's release hooks now
  hooks status                  Show what the release hooks last did
  secrets set <NAME>            Set a function's secret (the value is read from standard input)
  secrets unset <NAME>          Remove a function's secret
  secrets list                  List a function's secret names
  logs                          Show recent runs of one of the app's functions
  token create                  Make a release token for CI (shown once)
  token list                    List your release tokens
  token revoke <id>             Revoke a release token

Options:
      --app <id>                Act on this app instead of the one this folder is bound to
      --json                    One JSON document on stdout, diagnostics on stderr
  -h, --help                    Show this help

  create:   --template <name>
  link:     --branch <name>  --path <dir>  --no-follow
  push:     --dry-run  --keep-extra
  release:  --ref <tag>  --expect-commit <sha>  --accept-security <hash>  --dry-run
            --ci (uses ROBUTLER_RELEASE_TOKEN; needs --ref and --expect-commit)
  secrets:  --function <name>  --agent <id>
  logs:     --function <name>  --agent <id>  --limit <n>
  token:    --name <text>  --all-apps  --days <n>
`;

/** What `push` needs of the app's draft. The SDK's project handle is one. */
export interface DraftHandle {
  head: string | null;
  list(path?: string): Promise<Array<{ name: string; type: string; size?: number; sha256?: string }>>;
  write(path: string, data: Uint8Array): Promise<unknown>;
  delete(path: string): Promise<void>;
  commit(): Promise<string>;
}

export interface AppDeps {
  cwd: string;
  env: Record<string, string | undefined>;
  /** The portal's origin. */
  baseUrl: string;
  /** The CLI's sign-in, or null. */
  token: string | null;
  /** Standard output: results. */
  out(line: string): void;
  /** Standard error: everything a pipeline should not capture. */
  err(line: string): void;
  fetch?: typeof fetch;
  openBrowser(url: string): void;
  /** A secret's value: typed with echo off at a terminal, or everything piped in. Never an argument. */
  readSecret(prompt: string): Promise<string>;
  openDraft(appId: string): Promise<DraftHandle>;
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

const VALUE_FLAGS = new Set([
  '--app', '--template', '--branch', '--path', '--ref', '--expect-commit', '--accept-security', '--function', '--agent', '--limit', '--name', '--days',
]);
const BOOL_FLAGS = new Set(['--json', '--dry-run', '--keep-extra', '--no-follow', '--ci', '--all-apps']);

interface Parsed {
  words: string[];
  flags: Map<string, string | true>;
  help: boolean;
}

function usage(message: string): AppCliError {
  return new AppCliError('usage', message, 'Run `robutler app --help`.', 2);
}

export function parseAppArgs(argv: string[]): Parsed {
  const words: string[] = [];
  const flags = new Map<string, string | true>();
  let help = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') {
      help = true;
    } else if (arg === '--value' || arg.startsWith('--value=')) {
      throw new AppCliError('value_on_argv', 'A secret value is never taken from the command line.', 'Pipe it in: printf %s "$VALUE" | robutler app secrets set NAME --function <fn>', 2);
    } else if (VALUE_FLAGS.has(arg)) {
      if (i + 1 >= argv.length) throw usage(`Missing value for ${arg}`);
      flags.set(arg, argv[++i]);
    } else if (BOOL_FLAGS.has(arg)) {
      flags.set(arg, true);
    } else if (arg.startsWith('-') && arg !== '-') {
      throw usage(`Unknown option: ${clean(arg, 60)}`);
    } else {
      words.push(arg);
    }
  }
  return { words, flags, help };
}

const str = (flags: Parsed['flags'], name: string): string | undefined => {
  const v = flags.get(name);
  return typeof v === 'string' ? v : undefined;
};

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

interface Ctx {
  deps: AppDeps;
  api: AppApi;
  root: string;
  flags: Parsed['flags'];
  json: boolean;
  /** A line for a person. Silent under `--json`. */
  say(line: string): void;
}

/** Run `robutler app <argv>`. Returns the process exit code. */
export async function runApp(argv: string[], deps: AppDeps): Promise<number> {
  let json = argv.includes('--json');
  try {
    const parsed = parseAppArgs(argv);
    json = parsed.flags.get('--json') === true;
    if (parsed.help || parsed.words.length === 0) {
      deps.out(APP_USAGE);
      return parsed.help ? 0 : 2;
    }
    const ctx: Ctx = {
      deps,
      api: new AppApi({ baseUrl: deps.baseUrl, token: deps.token, fetch: deps.fetch }),
      root: appRoot(deps.cwd),
      flags: parsed.flags,
      json,
      say: (line) => {
        if (!json) deps.out(line);
      },
    };
    const data = await dispatch(ctx, parsed.words);
    if (json) deps.out(JSON.stringify({ ok: true, data }, null, 2));
    return 0;
  } catch (error) {
    const e =
      error instanceof AppCliError
        ? error
        : new AppCliError(clean((error as { code?: unknown })?.code, 80) || 'failed', clean((error as Error)?.message) || 'The command failed.');
    if (json) {
      deps.out(JSON.stringify({ ok: false, error: { code: e.code, message: e.message, ...(e.fix ? { fix: e.fix } : {}) } }, null, 2));
    } else {
      deps.err(e.message);
      if (e.fix) deps.err(e.fix);
    }
    return e.exitCode;
  }
}

async function dispatch(ctx: Ctx, words: string[]): Promise<unknown> {
  const [command, ...rest] = words;
  switch (command) {
    case 'create':
      return create(ctx, rest);
    case 'use':
      return use(ctx, rest);
    case 'link':
      return link(ctx, rest);
    case 'push':
      return push(ctx, rest);
    case 'status':
      return status(ctx);
    case 'release':
      return release(ctx);
    case 'hooks':
      return hooks(ctx, rest);
    case 'secrets':
      return secrets(ctx, rest);
    case 'logs':
      return logs(ctx);
    case 'token':
      return token(ctx, rest);
    default:
      throw usage(`Unknown command: app ${clean(command, 60)}`);
  }
}

function requireSignIn(ctx: Ctx): void {
  if (!ctx.api.token) throw new AppCliError('not_signed_in', `You are not signed in to ${ctx.api.portalHost}.`, 'Run `robutler login`.');
}

/** The app to act on: `--app`, then `ROBUTLER_APP_ID`, then what this folder is bound to. */
function binding(ctx: Ctx): AppBinding {
  const explicit = str(ctx.flags, '--app') ?? ctx.deps.env.ROBUTLER_APP_ID;
  const bound = readBinding(ctx.root, ctx.api.baseUrl);
  if (explicit) {
    const id = explicit.trim().toLowerCase();
    if (!isAppId(id)) throw usage('An app id looks like 00000000-0000-4000-8000-000000000000.');
    return bound && bound.id === id ? bound : { id };
  }
  if (!bound) {
    throw new AppCliError(
      'no_app',
      `This folder is not bound to an app on ${ctx.api.portalHost}.`,
      'Run `robutler app create <name>` for a new app, or `robutler app use <app id>` for one that exists.',
    );
  }
  return bound;
}

const text = (v: unknown, max = 200): string => (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean' ? clean(v, max) : '');
const record = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** A route this portal does not have: one sentence, and the way around when there is one. */
function notOffered(ctx: Ctx, what: string, fix = ''): AppCliError {
  return new AppCliError('not_available', `${ctx.api.portalHost} does not offer ${what} to the command line.`, fix);
}

// ---------------------------------------------------------------------------
// create, use
// ---------------------------------------------------------------------------

async function create(ctx: Ctx, words: string[]): Promise<unknown> {
  if (words.length !== 1) throw usage('Usage: robutler app create <name>');
  const name = words[0].trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9 _-]{0,62}$/.test(name)) throw usage('A name is 1 to 63 letters, digits, spaces, "-" or "_", starting with a letter or digit.');
  requireSignIn(ctx);
  const bound = readBinding(ctx.root, ctx.api.baseUrl);
  if (bound) {
    throw new AppCliError(
      'already_bound',
      `This folder is already the app ${bound.name ?? bound.id} on ${ctx.api.portalHost}.`,
      'Use another folder, or `robutler app use <app id>` to point this one at a different app.',
    );
  }
  const template = str(ctx.flags, '--template');
  const answer = await ctx.api.request('POST', '/api/apps', { body: { name, ...(template ? { template } : {}) }, accept: [404, 405] });
  if (answer.status !== 200 && answer.status !== 201) {
    throw notOffered(ctx, 'creating an app', 'Create the app in the portal, then run `robutler app use <app id>` in this folder.');
  }
  const app = record(answer.body.app);
  if (!isAppId(app.id)) throw new AppCliError('bad_answer', `${ctx.api.portalHost} answered without an app id.`);
  const made: AppBinding = {
    id: app.id,
    name: text(app.name, 120) || name,
    ...(isAppId(app.widgetId) ? { widgetId: app.widgetId } : {}),
    ...(isAppId(app.agentId) ? { agentId: app.agentId } : {}),
  };
  const file = writeBinding(ctx.root, ctx.api.baseUrl, made);
  ctx.say(`Created ${made.name} (${made.id}) on ${ctx.api.portalHost}. It is private until you release it.`);
  if (text(app.openUrl)) ctx.say(`Open it: ${text(app.openUrl, 400)}`);
  ctx.say(`This folder is now bound to it (${path.relative(ctx.deps.cwd, file) || file}).`);
  ctx.say('Next: `robutler app push` copies your files into its draft.');
  return { app: made, ...(text(app.openUrl) ? { openUrl: text(app.openUrl, 400) } : {}) };
}

async function use(ctx: Ctx, words: string[]): Promise<unknown> {
  if (words.length !== 1) throw usage('Usage: robutler app use <app id>');
  const id = words[0].trim().toLowerCase();
  if (!isAppId(id)) throw usage('An app id looks like 00000000-0000-4000-8000-000000000000.');
  requireSignIn(ctx);
  const seen = await describe(ctx, { id });
  const bound: AppBinding = { id, ...(seen.name ? { name: seen.name } : {}), ...(seen.widgetId ? { widgetId: seen.widgetId } : {}), ...(seen.agentId ? { agentId: seen.agentId } : {}) };
  writeBinding(ctx.root, ctx.api.baseUrl, bound);
  ctx.say(`This folder is now bound to ${bound.name ?? id} on ${ctx.api.portalHost}.`);
  return { app: bound };
}

// ---------------------------------------------------------------------------
// link
// ---------------------------------------------------------------------------

async function link(ctx: Ctx, words: string[]): Promise<unknown> {
  if (words.length !== 1) throw usage('Usage: robutler app link <owner/repo> [--branch <name>] [--path <dir>]');
  const repo = words[0].trim();
  if (!/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})\/[A-Za-z0-9._-]{1,100}$/.test(repo)) throw usage('A repository is <owner>/<name>, as on GitHub. Not a URL.');
  const branch = str(ctx.flags, '--branch') ?? 'main';
  if (!/^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$/.test(branch) || branch.includes('..')) throw usage('That is not a branch name.');
  const folder = (str(ctx.flags, '--path') ?? repoSourcePath(ctx.root) ?? '').replace(/^\.\/+/, '').replace(/\/+$/, '');
  if (folder && (!/^[A-Za-z0-9._/-]{1,200}$/.test(folder) || folder.startsWith('/') || folder.split('/').includes('..'))) {
    throw usage('--path is a folder inside the repository, such as app.');
  }
  const app = binding(ctx);
  // `direction` first: an address cut short at its first `&` then names no repository, and the portal refuses it.
  const query = new URLSearchParams({ direction: 'source', repo, branch });
  if (folder) query.set('folder', folder);
  if (ctx.flags.get('--no-follow') !== true) query.set('follow', '1');
  // The portal's own page. It carries no credential: the browser's sign-in decides who is linking.
  const url = `${ctx.api.baseUrl}/api/casfs/projects/${app.id}/git/github?${query.toString()}`;
  ctx.say(`Linking ${repo} (${branch}${folder ? `, folder ${folder}` : ''}) to ${app.name ?? app.id} is finished in your browser, signed in as the app's owner.`);
  ctx.say(`If the browser does not open, visit: ${url}`);
  ctx.deps.openBrowser(url);
  ctx.say('Then run `robutler app status` to see the link.');
  return { url, repo, branch, ...(folder ? { folder } : {}) };
}

// ---------------------------------------------------------------------------
// push
// ---------------------------------------------------------------------------

const PUSH_SKIP = new Set(['.git', '.robutler', 'node_modules', '.DS_Store']);
const PUSH_MAX_FILES = 5000;
const PUSH_MAX_BYTES = 256 * 1024 * 1024;

/** Every file under `dir`, as forward-slash paths. A symbolic link is refused by name. */
export function walkLocal(dir: string): Map<string, string> {
  const files = new Map<string, string>();
  const visit = (abs: string, rel: string): void => {
    for (const entry of fs.readdirSync(abs, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      if (PUSH_SKIP.has(entry.name)) continue;
      const childAbs = path.join(abs, entry.name);
      const childRel = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isSymbolicLink()) throw new AppCliError('symlink', `${childRel} is a symbolic link. An app's files are plain files.`);
      if (entry.isDirectory()) visit(childAbs, childRel);
      else if (entry.isFile()) {
        files.set(childRel, childAbs);
        if (files.size > PUSH_MAX_FILES) throw new AppCliError('too_many_files', `More than ${PUSH_MAX_FILES} files. Is this the app's served folder?`);
      }
    }
  };
  visit(dir, '');
  return files;
}

async function walkDraft(draft: DraftHandle, at = '', found = new Map<string, { size?: number; sha256?: string }>()): Promise<Map<string, { size?: number; sha256?: string }>> {
  for (const entry of await draft.list(at)) {
    const rel = at ? `${at}/${entry.name}` : entry.name;
    if (entry.type === 'dir') await walkDraft(draft, rel, found);
    // A mount is another project's files, placed here; a push neither rewrites nor removes it.
    else if (entry.type === 'file') found.set(rel, { size: entry.size, sha256: entry.sha256 });
  }
  return found;
}

async function push(ctx: Ctx, words: string[]): Promise<unknown> {
  if (words.length > 1) throw usage('Usage: robutler app push [dir]');
  const dir = words[0] ? path.resolve(ctx.deps.cwd, words[0]) : path.resolve(ctx.root, repoSourcePath(ctx.root) ?? '.');
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) throw new AppCliError('no_folder', `${dir} is not a folder.`);
  if (!fs.existsSync(path.join(dir, 'widget.json'))) {
    throw new AppCliError('not_an_app_folder', `${dir} has no widget.json, so it is not an app's served folder.`, 'Pass the folder that holds widget.json, or name it as sourcePath in robutler.json.');
  }
  requireSignIn(ctx);
  const app = binding(ctx);
  const local = walkLocal(dir);
  let total = 0;
  for (const abs of local.values()) total += fs.statSync(abs).size;
  if (total > PUSH_MAX_BYTES) throw new AppCliError('too_large', `The folder holds ${Math.round(total / 1048576)} MB; a push takes at most ${PUSH_MAX_BYTES / 1048576} MB.`);

  const dryRun = ctx.flags.get('--dry-run') === true;
  let draft: DraftHandle;
  try {
    draft = await ctx.deps.openDraft(app.id);
  } catch (error) {
    throw draftError(error, ctx);
  }
  try {
    const remote = await walkDraft(draft);
    const written: string[] = [];
    let unchanged = 0;
    for (const [rel, abs] of local) {
      const bytes = fs.readFileSync(abs);
      const there = remote.get(rel);
      if (there && there.size === bytes.length && there.sha256 === createHash('sha256').update(bytes).digest('hex')) {
        unchanged += 1;
        continue;
      }
      written.push(rel);
      if (!dryRun) await draft.write(rel, new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    }
    const removed = ctx.flags.get('--keep-extra') === true ? [] : [...remote.keys()].filter((rel) => !local.has(rel));
    if (!dryRun) for (const rel of removed) await draft.delete(rel);
    const changed = written.length + removed.length > 0;
    const head = dryRun || !changed ? draft.head : await draft.commit();
    const name = app.name ?? app.id;
    if (dryRun) {
      for (const rel of written) ctx.say(`  write   ${clean(rel, 300)}`);
      for (const rel of removed) ctx.say(`  remove  ${clean(rel, 300)}`);
      ctx.say(`Would write ${written.length} and remove ${removed.length} (${unchanged} unchanged) in the draft of ${name}. Nothing was sent.`);
    } else if (!changed) {
      ctx.say(`The draft of ${name} already holds these ${unchanged} files. Nothing was sent.`);
    } else {
      ctx.say(`Pushed to the draft of ${name}: ${written.length} written, ${removed.length} removed, ${unchanged} unchanged.`);
      ctx.say(`Draft: ${text(head, 64)}. Nothing is released; \`robutler app release\` does that.`);
    }
    return { app: app.id, dryRun, written, removed, unchanged, draft: text(head, 64) || null };
  } catch (error) {
    throw draftError(error, ctx);
  }
}

function draftError(error: unknown, ctx: Ctx): AppCliError {
  if (error instanceof AppCliError) return error;
  const code = clean((error as { code?: unknown })?.code, 80) || 'push_failed';
  const fix =
    code === 'Cas:NotFound'
      ? `Either the app is not yours on ${ctx.api.portalHost}, or it does not keep its files as a project there.`
      : code === 'Cas:Unauthenticated'
        ? 'Sign in again with `robutler login`.'
        : '';
  return new AppCliError(code, cleanWithout((error as Error)?.message, ctx.api.token ? [ctx.api.token] : []) || 'The push failed.', fix);
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

interface Described {
  id: string;
  name?: string;
  widgetId?: string;
  agentId?: string;
  public?: boolean;
  draft?: string;
  release?: Record<string, string>;
  link?: Record<string, string>;
  releases?: number;
  notes: string[];
}

/** One line per known key, in a fixed order, values cleaned. */
function pick(source: Record<string, unknown>, keys: readonly string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of keys) {
    const value = source[key];
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') out[key] = clean(value, 200);
    else if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [k, v] of Object.entries(value)) {
        if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') out[`${key}.${clean(k, 40)}`] = clean(v, 200);
      }
    }
  }
  return out;
}

const LINK_KEYS = ['provider', 'repo', 'branch', 'folder', 'followMain', 'status', 'lastPullAt', 'lastError', 'ciRelease', 'access'] as const;
const RELEASE_KEYS = ['version', 'ref', 'sha', 'root', 'at', 'releasedAt'] as const;

/**
 * What the portal says about the app. The app's own route when the portal has
 * it; otherwise the project, its repository link and its releases, each asked
 * on its own and each allowed to be missing.
 */
async function describe(ctx: Ctx, app: AppBinding): Promise<Described> {
  const seen: Described = { id: app.id, ...(app.name ? { name: app.name } : {}), notes: [] };
  const own = await ctx.api.request('GET', `/api/apps/${app.id}`, { accept: [404, 405] });
  if (own.status === 200 && isAppId(record(own.body.app).id)) {
    const a = record(own.body.app);
    if (text(a.name)) seen.name = text(a.name, 120);
    if (isAppId(a.widgetId)) seen.widgetId = a.widgetId;
    if (isAppId(a.agentId)) seen.agentId = a.agentId;
    if (typeof a.public === 'boolean') seen.public = a.public;
    if (text(a.draft)) seen.draft = text(a.draft, 64);
    if (a.release && typeof a.release === 'object') seen.release = pick(record(a.release), RELEASE_KEYS);
    if (a.link && typeof a.link === 'object') seen.link = pick(record(a.link), LINK_KEYS);
    return seen;
  }
  const project = await ctx.api.request('GET', `/api/casfs/projects/${app.id}`, { accept: [404] });
  if (project.status === 404) {
    throw new AppCliError('app_not_found', `No app ${app.id} that you can reach on ${ctx.api.portalHost}.`, 'Check the id, and that you are signed in as its owner (`webagents whoami`).');
  }
  if (text(project.body.head)) seen.draft = text(project.body.head, 64);
  const git = await ctx.api.request('GET', `/api/casfs/projects/${app.id}/git`, { accept: [401, 403, 404] });
  if (git.status === 200 && git.body.link && typeof git.body.link === 'object') seen.link = pick(record(git.body.link), LINK_KEYS);
  else if (git.status === 200) seen.notes.push('No repository is linked. `robutler app link <owner/repo>` starts one.');
  const releases = await ctx.api.request('GET', `/api/folders/${app.id}/releases`, { accept: [401, 403, 404, 409] });
  if (releases.status === 200) {
    const all = list(releases.body.releases);
    seen.releases = all.length;
    const live = text(releases.body.live, 64);
    const current = all.map(record).find((r) => live && text(r.root, 64) === live) ?? (all.length ? record(all[all.length - 1]) : null);
    if (current) seen.release = pick(current, RELEASE_KEYS);
  } else {
    seen.notes.push('The releases are shown in the portal; this sign-in cannot list them.');
  }
  return seen;
}

async function status(ctx: Ctx): Promise<unknown> {
  requireSignIn(ctx);
  const app = binding(ctx);
  const seen = await describe(ctx, app);
  // Keep what was learned, so the commands that need the app's agent find it.
  const bound = readBinding(ctx.root, ctx.api.baseUrl);
  if (bound && bound.id === seen.id && (seen.agentId || seen.widgetId || seen.name)) {
    writeBinding(ctx.root, ctx.api.baseUrl, {
      ...bound,
      ...(seen.name ? { name: seen.name } : {}),
      ...(seen.widgetId ? { widgetId: seen.widgetId } : {}),
      ...(seen.agentId ? { agentId: seen.agentId } : {}),
    });
  }
  ctx.say(`${seen.name ?? 'App'} (${seen.id}) on ${ctx.api.portalHost}`);
  if (seen.public !== undefined) ctx.say(`  public:   ${seen.public ? 'yes' : 'no'}`);
  ctx.say(`  draft:    ${seen.draft ?? 'empty'}`);
  if (seen.release && Object.keys(seen.release).length) {
    ctx.say('  release:');
    for (const [k, v] of Object.entries(seen.release)) ctx.say(`    ${k}: ${v}`);
  } else if (seen.releases === 0) {
    ctx.say('  release:  none yet');
  }
  if (seen.link && Object.keys(seen.link).length) {
    ctx.say('  repository:');
    for (const [k, v] of Object.entries(seen.link)) ctx.say(`    ${k}: ${v}`);
  }
  for (const note of seen.notes) ctx.say(`  ${note}`);
  return seen;
}

// ---------------------------------------------------------------------------
// release
// ---------------------------------------------------------------------------

const SHA40 = /^[0-9a-f]{40}$/;
const HASH64 = /^[0-9a-f]{64}$/;
const TAG = /^v[0-9][0-9a-z._-]{0,62}$/;

interface SecurityNote {
  hash: string;
  widenedCount: number;
  widened: Array<{ field: string; item: string; message: string }>;
}

/** The security note of an answer, trusting nothing but its shape. */
export function securityNote(value: unknown): SecurityNote | null {
  const s = record(value);
  if (typeof s.hash !== 'string' || !HASH64.test(s.hash)) return null;
  const widened = list(s.widened)
    .map(record)
    .map((e) => ({ field: text(e.field, 120), item: text(e.item, 160), message: text(e.message, 300) }));
  const count = typeof s.widenedCount === 'number' && Number.isInteger(s.widenedCount) && s.widenedCount >= 0 ? s.widenedCount : widened.length;
  // The larger of the two: a count of zero beside a list of changes is still a widening.
  return { hash: s.hash, widenedCount: Math.max(count, widened.length), widened };
}

async function release(ctx: Ctx): Promise<unknown> {
  const ref = str(ctx.flags, '--ref');
  if (ref !== undefined && !TAG.test(ref)) throw usage('A release tag is v, a digit, then lower case letters, digits, ".", "_" or "-": v1, v1.2.0.');
  const expect = str(ctx.flags, '--expect-commit')?.toLowerCase();
  if (expect !== undefined && !SHA40.test(expect)) throw usage('--expect-commit takes the full commit id: 40 hexadecimal characters.');
  const accepted = str(ctx.flags, '--accept-security')?.toLowerCase();
  if (accepted !== undefined && !HASH64.test(accepted)) throw usage('--accept-security takes the hash this command printed: 64 hexadecimal characters.');
  const app = binding(ctx);
  if (ctx.flags.get('--ci') === true) return releaseFromCi(ctx, app, ref, expect, accepted);

  requireSignIn(ctx);
  const who = await assertPerson(ctx.api, 'release an app');
  const dryRun = ctx.flags.get('--dry-run') === true;
  const route = `/api/apps/${app.id}/release`;
  const ask = { ...(ref ? { ref } : {}), ...(expect ? { sha: expect } : {}) };

  // First, what the release would be. Nothing moves.
  const preview = await ctx.api.request('POST', route, { body: { ...ask, dryRun: true }, accept: [404, 405] });
  if (preview.status === 404 || preview.status === 405) {
    throw notOffered(ctx, 'releasing an app', 'Release it from the app in the portal, or from a tag with the release workflow.');
  }
  const sha = text(preview.body.sha, 64).toLowerCase();
  const note = securityNote(preview.body.security);
  const first = preview.body.firstRelease === true;
  const name = app.name ?? app.id;
  ctx.say(`Release of ${name} on ${ctx.api.portalHost}, by @${who.username}`);
  if (ref) ctx.say(`  tag:     ${ref}${text(preview.body.repo) ? ` of ${text(preview.body.repo, 160)}` : ''}`);
  if (sha) ctx.say(`  commit:  ${sha}`);
  if (text(preview.body.draft)) ctx.say(`  draft:   ${text(preview.body.draft, 64)}`);

  if (ref && !SHA40.test(sha)) throw new AppCliError('bad_answer', `${ctx.api.portalHost} did not say which commit ${ref} is.`);
  if (expect && sha !== expect) {
    throw new AppCliError('commit_mismatch', `${ref ?? 'The release'} is at ${sha || 'an unknown commit'}, not the ${expect} you expected. Nothing was released.`);
  }
  if (!note) throw new AppCliError('bad_answer', `${ctx.api.portalHost} did not say what this release changes, so it was not released.`);

  if (first) ctx.say('  This is the first release: it makes the app public.');
  if (note.widenedCount > 0) {
    ctx.say(`  This version lets the app reach more than the current release (${note.widenedCount}):`);
    for (const entry of note.widened) ctx.say(`    - ${entry.message || `${entry.field} ${entry.item}`.trim()}`);
    if (note.widenedCount > note.widened.length) ctx.say(`    and ${note.widenedCount - note.widened.length} more: read the app's widget.json.`);
  } else {
    ctx.say('  It does not let the app reach more than the current release.');
  }
  ctx.say(`  security hash: ${note.hash}`);

  const needsAccept = first || note.widenedCount > 0;
  const again = `robutler app release${ref ? ` --ref ${ref}` : ''}${sha ? ` --expect-commit ${sha}` : ''} --accept-security ${note.hash}`;
  if (accepted !== undefined && accepted !== note.hash) {
    throw new AppCliError('security_hash_mismatch', 'The hash given with --accept-security is not the hash of this release. Nothing was released.', `Read the list above. To accept it: ${again}`, 3);
  }
  if (needsAccept && accepted === undefined) {
    throw new AppCliError(
      'security_acceptance_required',
      first && note.widenedCount === 0 ? 'A first release needs your acceptance. Nothing was released.' : 'This release widens what the app may reach. Nothing was released.',
      `Read the list above. To accept it: ${again}`,
      3,
    );
  }
  const data = { app: app.id, ...(ref ? { ref } : {}), ...(sha ? { sha } : {}), firstRelease: first, security: note };
  if (dryRun) {
    ctx.say('Dry run: nothing was released.');
    return { ...data, outcome: 'preview' };
  }

  // The hash that was shown travels with the request, so the portal releases exactly what was read here or refuses.
  const done = await ctx.api.request('POST', route, { body: { ...(ref ? { ref } : {}), ...(sha ? { sha } : {}), acceptSecurity: note.hash } });
  return reportRelease(ctx, done, data);
}

/** Say what a release answer means: released, unchanged, or the owner was asked. */
function reportRelease(ctx: Ctx, answer: { status: number; body: Record<string, unknown> }, data: Record<string, unknown>): unknown {
  const outcome = text(answer.body.outcome, 40);
  if (answer.status === 202 || outcome === 'approval_requested') {
    const url = text(answer.body.approvalUrl, 500);
    ctx.say('Nothing is released yet. The app\'s owner has been asked to approve this release.');
    if (url) ctx.say(`Approve it in your browser: ${url}`);
    for (const entry of securityNote(answer.body.security)?.widened ?? []) ctx.say(`  - ${entry.message}`);
    return { ...data, outcome: 'approval_requested', ...(url ? { approvalUrl: url } : {}) };
  }
  if (outcome === 'unchanged') {
    ctx.say('Unchanged: this is already the release.');
    return { ...data, outcome };
  }
  if (outcome !== 'released') throw new AppCliError('bad_answer', `${ctx.api.portalHost} did not say whether the app was released. Check \`robutler app status\`.`);
  ctx.say(`Released${text(answer.body.version) ? ` as version ${text(answer.body.version, 40)}` : ''}.`);
  if (text(answer.body.url)) ctx.say(`Open it: ${text(answer.body.url, 400)}`);
  return { ...data, outcome, ...(text(answer.body.version) ? { version: text(answer.body.version, 40) } : {}) };
}

/**
 * The release a workflow makes: a tag, the commit the workflow ran on, and a
 * release token from the environment. The CLI's own sign-in is not sent.
 * The portal never widens on this path: it answers 202 and asks the owner.
 */
async function releaseFromCi(ctx: Ctx, app: AppBinding, ref: string | undefined, sha: string | undefined, accepted: string | undefined): Promise<unknown> {
  if (accepted !== undefined) throw usage('--accept-security is not used with --ci: a release from CI never widens what the app may reach.');
  if (!ref || !sha) throw usage('A release from CI needs --ref <tag> and --expect-commit <sha>.');
  const releaseToken = (ctx.deps.env.ROBUTLER_RELEASE_TOKEN ?? '').trim();
  if (!releaseToken) throw new AppCliError('no_release_token', 'ROBUTLER_RELEASE_TOKEN is not set.', 'Make a release token with `robutler app token create` and give it to the job as a secret.');
  const answer = await ctx.api.request('POST', `/api/apps/${app.id}/release/ci`, { body: { ref, sha }, bearer: releaseToken, redact: [releaseToken] });
  return reportRelease(ctx, answer, { app: app.id, ref, sha });
}

// ---------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------

async function hooks(ctx: Ctx, words: string[]): Promise<unknown> {
  const [sub, which, ...extra] = words;
  if (sub === 'status' && which === undefined) {
    requireSignIn(ctx);
    const app = binding(ctx);
    const answer = await ctx.api.request('GET', `/api/folders/${app.id}/release-hooks`);
    const shown = pick({ hooks: record(answer.body.hooks) }, ['hooks']);
    if (Object.keys(shown).length === 0) ctx.say('No release hook has run.');
    for (const [k, v] of Object.entries(shown)) ctx.say(`${k.replace(/^hooks\./, '')}: ${v}`);
    return { hooks: shown };
  }
  if (sub !== 'run' || extra.length || !which || !['up', 'down', 'post'].includes(which)) throw usage('Usage: robutler app hooks run <up|down|post>, or robutler app hooks status');
  requireSignIn(ctx);
  const app = binding(ctx);
  try {
    const answer = await ctx.api.request('POST', `/api/folders/${app.id}/release-hooks`, { body: { action: which } });
    const state = text(answer.body.status, 60) || 'done';
    ctx.say(`The ${which} hook of ${app.name ?? app.id}: ${state}.`);
    return { hook: which, status: state };
  } catch (error) {
    if (error instanceof AppApiError && error.status === 400 && which !== 'post') {
      throw new AppCliError('not_available', `${ctx.api.portalHost} runs only the post hook on request; up and down run as part of a release.`);
    }
    throw error;
  }
}

// ---------------------------------------------------------------------------
// secrets, logs
// ---------------------------------------------------------------------------

const BINDING = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;
const FUNCTION = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const SECRET_MAX = 64 * 1024;

/** The path of one function of the app's agent. Both parts are checked, so neither can add a path segment. */
function functionPath(ctx: Ctx, app: AppBinding): string {
  const fn = str(ctx.flags, '--function');
  if (!fn || !FUNCTION.test(fn)) throw usage('Name the function: --function <name>.');
  const agent = (str(ctx.flags, '--agent') ?? app.agentId ?? '').toLowerCase();
  if (!isAppId(agent)) {
    throw new AppCliError('no_agent', "The app's agent is not known here.", 'Run `robutler app status` (it records the agent when the portal names it), or pass --agent <id>.');
  }
  return `/api/agents/${agent}/functions/${fn}`;
}

async function secrets(ctx: Ctx, words: string[]): Promise<unknown> {
  const [sub, name, ...extra] = words;
  if (sub === 'list') {
    if (name !== undefined) throw usage('Usage: robutler app secrets list --function <name>');
    requireSignIn(ctx);
    const answer = await ctx.api.request('GET', `${functionPath(ctx, binding(ctx))}/secret`, { accept: [404, 405] });
    if (answer.status !== 200) throw notOffered(ctx, 'listing secret names', "The function's page in the portal lists them.");
    // Names and whether each is set. Nothing else an answer carries is read, a value least of all.
    const names = list(answer.body.bindings)
      .map(record)
      .filter((b) => typeof b.name === 'string' && BINDING.test(b.name))
      .map((b) => ({ name: b.name as string, set: b.set === true }));
    if (names.length === 0) ctx.say('This function declares no secrets.');
    for (const b of names) ctx.say(`${b.name}  ${b.set ? 'set' : 'not set'}`);
    return { secrets: names };
  }
  if (sub !== 'set' && sub !== 'unset') throw usage('Usage: robutler app secrets set|unset <NAME> --function <name>, or secrets list --function <name>');
  if (sub === 'set' && (extra.length > 0 || (name ?? '').includes('='))) {
    throw new AppCliError('value_on_argv', 'A secret value is never taken from the command line: it would stay in the shell history and the process list.', 'Pipe it in: printf %s "$VALUE" | robutler app secrets set NAME --function <fn>', 2);
  }
  if (!name || extra.length > 0 || !BINDING.test(name)) throw usage('A secret name is a letter, then letters, digits or "_": API_KEY.');
  requireSignIn(ctx);
  const route = `${functionPath(ctx, binding(ctx))}/secret`;
  if (sub === 'unset') {
    await ctx.api.request('DELETE', `${route}?binding=${encodeURIComponent(name)}`);
    ctx.say(`Removed ${name}.`);
    return { secret: name, set: false };
  }
  const value = (await ctx.deps.readSecret(`Value for ${name} (not shown): `)).replace(/\r?\n$/, '');
  if (!value) throw new AppCliError('empty_value', 'No value arrived on standard input. Nothing was set.');
  if (Buffer.byteLength(value) > SECRET_MAX) throw new AppCliError('value_too_large', `A secret is at most ${SECRET_MAX / 1024} KB.`);
  await ctx.api.request('POST', route, { body: { binding: name, value }, redact: [value] });
  ctx.say(`Set ${name}.`);
  return { secret: name, set: true };
}

async function logs(ctx: Ctx): Promise<unknown> {
  requireSignIn(ctx);
  const limit = Number(str(ctx.flags, '--limit') ?? '25');
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) throw usage('--limit is a number from 1 to 200.');
  const answer = await ctx.api.request('GET', `${functionPath(ctx, binding(ctx))}/invocations?limit=${limit}`);
  const rows = list(answer.body.invocations)
    .map(record)
    .map((r) => ({ at: text(r.startedAt, 40), status: text(r.status, 40), error: text(r.errorCode, 80), ms: text(r.durationMs, 12) }));
  if (rows.length === 0) ctx.say('No runs yet.');
  for (const r of rows) ctx.say(`${r.at}  ${r.status.padEnd(10)}${r.ms ? `  ${r.ms} ms` : ''}${r.error ? `  ${r.error}` : ''}`);
  return { runs: rows };
}

// ---------------------------------------------------------------------------
// token
// ---------------------------------------------------------------------------

/**
 * The portal keeps release tokens to a person's own browser session. When it
 * says so, the command ends by pointing there: exit code 4, nothing on
 * standard output, so a pipeline that expected a token gets none.
 */
function inBrowser(ctx: Ctx, what: string): AppCliError {
  const url = `${ctx.api.baseUrl}/settings?tab=developer`;
  ctx.deps.openBrowser(url);
  return new AppCliError('continue_in_browser', `${ctx.api.portalHost} lets you ${what} only in your browser, signed in as yourself.`, `Open Settings, Developer, Release tokens: ${url}`, 4);
}

const tokenRow = (value: unknown) => {
  const r = record(value);
  return { id: text(r.id, 64), name: text(r.name, 120), app: text(r.app ?? r.appName ?? r.scope, 120), expiresAt: text(r.expiresAt, 40), lastUsedAt: text(r.lastUsedAt, 40), revoked: r.revokedAt != null || r.revoked === true };
};

async function token(ctx: Ctx, words: string[]): Promise<unknown> {
  const [sub, id, ...extra] = words;
  requireSignIn(ctx);
  if (sub === 'list' && id === undefined) {
    const answer = await ctx.api.request('GET', '/api/apps/release-tokens', { accept: [401, 403, 404] });
    if (answer.status === 404) throw notOffered(ctx, 'release tokens');
    if (answer.status !== 200) throw inBrowser(ctx, 'see your release tokens');
    const rows = list(answer.body.tokens).map(tokenRow);
    if (rows.length === 0) ctx.say('No release tokens.');
    for (const r of rows) ctx.say(`${r.id}  ${r.name}  ${r.app}  expires ${r.expiresAt || 'unknown'}${r.revoked ? '  revoked' : ''}`);
    return { tokens: rows };
  }
  if (sub === 'revoke' && id !== undefined && extra.length === 0) {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(id)) throw usage('That is not a token id. `robutler app token list` shows them.');
    const answer = await ctx.api.request('DELETE', `/api/apps/release-tokens/${id}`, { accept: [401, 403, 404] });
    if (answer.status === 404) throw new AppCliError('not_found', `No release token ${id} of yours on ${ctx.api.portalHost}, or release tokens are not offered there.`);
    if (answer.status !== 200) throw inBrowser(ctx, 'revoke a release token');
    ctx.say(`Revoked ${id}. It stops working now.`);
    return { revoked: id };
  }
  if (sub !== 'create' || id !== undefined) throw usage('Usage: robutler app token create [--name <text>] [--all-apps] [--days <n>], token list, or token revoke <id>');

  const days = Number(str(ctx.flags, '--days') ?? '90');
  if (!Number.isInteger(days) || days < 1 || days > 90) throw usage('--days is a number from 1 to 90.');
  const allApps = ctx.flags.get('--all-apps') === true;
  const app = allApps ? null : binding(ctx);
  await assertPerson(ctx.api, 'make a release token');
  const name = (str(ctx.flags, '--name') ?? `CI release${app?.name ? ` of ${app.name}` : ''}`).slice(0, 80);
  const answer = await ctx.api.request('POST', '/api/apps/release-tokens', { body: { name, app: app ? app.id : 'owner-apps', expiresInDays: days }, accept: [401, 403, 404] });
  if (answer.status === 404) throw notOffered(ctx, 'release tokens');
  if (answer.status === 401 || answer.status === 403) throw inBrowser(ctx, 'make a release token');
  const made = answer.body.token;
  // Printable and one line, or it is not printed at all.
  if (typeof made !== 'string' || !/^[A-Za-z0-9._~+/=-]{16,4096}$/.test(made)) throw new AppCliError('bad_answer', `${ctx.api.portalHost} answered without a usable token.`);
  const listing = tokenRow(answer.body.listing);
  // The explanation goes to stderr and the token alone to stdout, so it can be piped straight into a secret store.
  ctx.deps.err(`Release token ${listing.name || name}${listing.expiresAt ? `, expires ${listing.expiresAt}` : ''}.`);
  ctx.deps.err('It is shown once, below, and this command keeps no copy. Store it as the repository secret ROBUTLER_RELEASE_TOKEN.');
  if (ctx.json) return { token: made, id: listing.id, name: listing.name || name, expiresAt: listing.expiresAt };
  ctx.deps.out(made);
  return null;
}
