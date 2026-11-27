/**
 * `robutler app`, every command, against a local HTTP server that stands in
 * for the portal. No network, no keystore, no browser: `runApp` takes each of
 * those as a dependency.
 *
 * The security rules each have their own test: a release and a release token
 * are a person's act; a widening is never accepted without the hash that was
 * shown; a secret's value comes from standard input and is never printed; a
 * release token is written to standard output once and to no file; text from
 * the server is printed as data.
 */

import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import { clean, credentialKind, readBinding, writeBinding } from '../../../src/cli/app-api';
import { type AppDeps, type DraftHandle, parseAppArgs, runApp, securityNote } from '../../../src/cli/app-command';

const APP = '11111111-1111-4111-8111-111111111111';
const AGENT = '22222222-2222-4222-8222-222222222222';
const SHA = 'a'.repeat(40);
const HASH = 'b'.repeat(64);
const SECRET_VALUE = 'sk-live-do-not-print-7f3a';
const RELEASE_TOKEN = 'rrt_once-only-token-4c1d9e7a';

/** An unsigned token with this payload: enough for the CLI, which reads and never verifies. */
function jwt(payload: Record<string, unknown>): string {
  const part = (v: unknown) => Buffer.from(JSON.stringify(v)).toString('base64url');
  return `${part({ alg: 'RS256' })}.${part(payload)}.c2ln`;
}
const PERSON = jwt({ sub: 'u1', scopes: ['agents:own'] });
const AGENT_KEY = jwt({ sub: 'u1', agent_id: AGENT, scopes: ['read', 'write'] });
const API_KEY = 'rok_0123456789abcdef';

interface Seen {
  method: string;
  url: string;
  authorization: string | undefined;
  body: unknown;
}

/** The stand-in portal. `routes` maps "METHOD /path" to an answer; everything is recorded. */
let server: Server;
let baseUrl = '';
const seen: Seen[] = [];
let routes: Record<string, (req: Seen) => { status: number; body?: unknown }> = {};

beforeAll(async () => {
  server = createServer((req: IncomingMessage, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      const record: Seen = { method: req.method ?? '', url: req.url ?? '', authorization: req.headers.authorization, body: raw ? JSON.parse(raw) : undefined };
      seen.push(record);
      const handler = routes[`${record.method} ${record.url.split('?')[0]}`];
      const answer = handler ? handler(record) : { status: 404, body: { error: 'Not found' } };
      res.writeHead(answer.status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(answer.body ?? {}));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterAll(async () => {
  await new Promise((resolve) => server.close(resolve));
});

const dirs: string[] = [];
afterEach(() => {
  seen.length = 0;
  routes = {};
  for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
});

function folder(bound = true): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'robutler-app-'));
  dirs.push(dir);
  if (bound) writeBinding(dir, baseUrl, { id: APP, name: 'Platformer', agentId: AGENT });
  return dir;
}

interface Run {
  code: number;
  out: string[];
  err: string[];
  opened: string[];
  all: string;
}

async function run(argv: string[], over: Partial<AppDeps> = {}): Promise<Run> {
  const out: string[] = [];
  const err: string[] = [];
  const opened: string[] = [];
  const code = await runApp(argv, {
    cwd: over.cwd ?? folder(),
    env: {},
    baseUrl,
    token: PERSON,
    out: (line) => out.push(line),
    err: (line) => err.push(line),
    openBrowser: (url) => opened.push(url),
    readSecret: async () => {
      throw new Error('standard input was read');
    },
    openDraft: async () => {
      throw new Error('the draft was opened');
    },
    ...over,
  });
  return { code, out, err, opened, all: [...out, ...err].join('\n') };
}

const me = (accountType = 'human') => () => ({ status: 200, body: { user: { username: 'ada', accountType } } });

/** An in-memory draft: what `push` needs of a project. */
function fakeDraft(files: Record<string, string> = {}): DraftHandle & { files: Map<string, Buffer>; commits: number } {
  const store = new Map(Object.entries(files).map(([k, v]) => [k, Buffer.from(v)]));
  const draft = {
    files: store,
    commits: 0,
    head: 'c'.repeat(64) as string | null,
    async list(at = '') {
      const prefix = at ? `${at}/` : '';
      const names = new Map<string, { name: string; type: string; size?: number; sha256?: string }>();
      for (const [file, bytes] of store) {
        if (!file.startsWith(prefix)) continue;
        const rest = file.slice(prefix.length);
        const name = rest.split('/')[0];
        if (rest.includes('/')) names.set(name, { name, type: 'dir' });
        else names.set(name, { name, type: 'file', size: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
      }
      if (!at) names.set('lib', { name: 'lib', type: 'mount' });
      return [...names.values()];
    },
    async write(file: string, data: Uint8Array) {
      store.set(file, Buffer.from(data));
    },
    async delete(file: string) {
      store.delete(file);
    },
    async commit() {
      draft.commits += 1;
      draft.head = 'd'.repeat(64);
      return draft.head;
    },
  };
  return draft;
}

describe('text from the server is data', () => {
  it('removes escape sequences, control characters and direction overrides', () => {
    expect(clean('\u001b[31mred\u001b[0m \u001b]0;title\u0007x\u0008‮evil\u0000\r\nnext')).toBe('red xevil next');
    expect(clean('a'.repeat(50), 10)).toHaveLength(10);
    expect(clean({ a: '\u001b[2Jwipe' })).toBe('{"a":"\\u001b[2Jwipe"}');
  });

  it('prints an error message from the server without its escapes', async () => {
    routes[`GET /api/apps/${APP}`] = () => ({ status: 500, body: { error: { code: 'boom\u001b[31m', message: 'failed \u001b[2J\u001b[1;1Hnow\u0007' } } });
    const r = await run(['status']);
    expect(r.code).toBe(1);
    for (const line of r.err) expect(line).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/);
    expect(r.err[0]).toBe('failed now');
  });

  it('prints names from the server without their escapes', async () => {
    routes[`GET /api/apps/${APP}`] = () => ({ status: 200, body: { app: { id: APP, name: 'Nice\u001b[2K‮app', public: true, draft: 'e'.repeat(64), link: { repo: 'o/r\u001b[31m', branch: 'main' } } } });
    const r = await run(['status']);
    expect(r.code).toBe(0);
    for (const line of r.out) expect(line).not.toMatch(/[\u0000-\u001f\u007f-\u009f\u202e]/);
    expect(r.out[0]).toContain('Niceapp');
  });
});

describe('the credential', () => {
  it('tells a sign-in from a key by the token alone', () => {
    expect(credentialKind(PERSON)).toBe('sign-in');
    expect(credentialKind(AGENT_KEY)).toBe('agent-key');
    expect(credentialKind(API_KEY)).toBe('api-key');
    expect(credentialKind(jwt({ sub: 'u1', scopes: ['apps:release'] }))).toBe('release-token');
    expect(credentialKind(jwt({ sub: 'u1', daemon: { id: 'd' } }))).toBe('agent-key');
    expect(credentialKind(null)).toBe('none');
  });

  it('sends the sign-in as a bearer and says so when there is none', async () => {
    routes[`GET /api/apps/${APP}`] = () => ({ status: 200, body: { app: { id: APP, name: 'Platformer' } } });
    await run(['status']);
    expect(seen[0].authorization).toBe(`Bearer ${PERSON}`);
    const r = await run(['status'], { token: null });
    expect(r.code).toBe(1);
    expect(r.err.join(' ')).toContain('robutler login');
  });
});

describe('app create, use, link', () => {
  it('creates the app and binds the folder to it', async () => {
    const cwd = folder(false);
    routes['POST /api/apps'] = () => ({ status: 201, body: { ok: true, app: { id: APP, name: 'platformer', widgetId: AGENT, agentId: AGENT, openUrl: `${baseUrl}/w/x` } } });
    const r = await run(['create', 'platformer'], { cwd });
    expect(r.code).toBe(0);
    expect(seen[0].body).toEqual({ name: 'platformer' });
    expect(readBinding(cwd, baseUrl)).toEqual({ id: APP, name: 'platformer', widgetId: AGENT, agentId: AGENT });
    // The file holds ids and a name: nothing of the credential.
    expect(fs.readFileSync(path.join(cwd, '.robutler', 'app.json'), 'utf8')).not.toContain(PERSON);
  });

  it('refuses a second app for a bound folder, and says what to do when the portal has no create route', async () => {
    expect((await run(['create', 'again'])).err[0]).toContain('already the app Platformer');
    expect(seen).toHaveLength(0);
    const r = await run(['create', 'new-app'], { cwd: folder(false) });
    expect(r.code).toBe(1);
    expect(r.err.join(' ')).toContain('robutler app use <app id>');
  });

  it('binds a folder to an existing app only when the portal knows it', async () => {
    const cwd = folder(false);
    expect((await run(['use', APP], { cwd })).code).toBe(1);
    expect(readBinding(cwd, baseUrl)).toBeNull();
    routes[`GET /api/casfs/projects/${APP}`] = () => ({ status: 200, body: { ok: true, head: 'f'.repeat(64) } });
    expect((await run(['use', APP], { cwd })).code).toBe(0);
    expect(readBinding(cwd, baseUrl)).toEqual({ id: APP });
  });

  it('link opens the portal page in the browser and sends nothing itself', async () => {
    const cwd = folder();
    fs.writeFileSync(path.join(cwd, 'robutler.json'), JSON.stringify({ sourcePath: 'app' }));
    const r = await run(['link', 'robutlerai/app-littlejs-platformer', '--branch', 'main'], { cwd });
    expect(r.code).toBe(0);
    expect(seen).toHaveLength(0);
    expect(r.opened).toEqual([`${baseUrl}/api/casfs/projects/${APP}/git/github?direction=source&repo=robutlerai%2Fapp-littlejs-platformer&branch=main&folder=app&follow=1`]);
    expect(r.opened[0]).not.toContain(PERSON);
  });

  it('link refuses a URL, a path that climbs and a strange branch', async () => {
    expect((await run(['link', 'https://github.com/o/r'])).code).toBe(2);
    expect((await run(['link', 'o/r', '--path', '../x'])).code).toBe(2);
    expect((await run(['link', 'o/r', '--branch', 'a..b'])).code).toBe(2);
  });
});

describe('app push', () => {
  function appFolder(): string {
    const cwd = folder();
    fs.writeFileSync(path.join(cwd, 'robutler.json'), JSON.stringify({ sourcePath: 'app' }));
    fs.mkdirSync(path.join(cwd, 'app', 'img'), { recursive: true });
    fs.writeFileSync(path.join(cwd, 'app', 'widget.json'), '{"name":"p"}');
    fs.writeFileSync(path.join(cwd, 'app', 'index.html'), '<p>hi</p>');
    fs.writeFileSync(path.join(cwd, 'app', 'img', 'a.png'), 'png');
    fs.writeFileSync(path.join(cwd, 'README.md'), 'not served');
    return cwd;
  }

  it('writes what changed, removes what is gone, leaves mounts alone and commits once', async () => {
    const cwd = appFolder();
    const draft = fakeDraft({ 'index.html': '<p>hi</p>', 'old.js': 'x', 'img/a.png': 'old' });
    const r = await run(['push'], { cwd, openDraft: async () => draft });
    expect(r.code).toBe(0);
    expect([...draft.files.keys()].sort()).toEqual(['img/a.png', 'index.html', 'widget.json']);
    expect(draft.files.get('img/a.png')!.toString()).toBe('png');
    expect(draft.commits).toBe(1);
    expect(r.out[0]).toBe('Pushed to the draft of Platformer: 2 written, 1 removed, 1 unchanged.');
  });

  it('says so when the push renamed the draft from widget.json, and stays quiet when it did not', async () => {
    const cwd = appFolder();
    routes[`POST /api/apps/${APP}/draft-meta`] = () => ({ status: 200, body: { ok: true, app: APP, name: { from: 'Littlejs Platformer', to: 'Platformer\u001b[2J' }, description: { changed: true } } });
    const r = await run(['push'], { cwd, openDraft: async () => fakeDraft() });
    expect(r.code).toBe(0);
    expect(r.out).toContain('name: Littlejs Platformer -> Platformer');
    routes[`POST /api/apps/${APP}/draft-meta`] = () => ({ status: 200, body: { ok: true, app: APP, name: null, description: { changed: false } } });
    const quiet = await run(['push'], { cwd, openDraft: async () => fakeDraft() });
    expect(quiet.out.some((l) => l.startsWith('name:'))).toBe(false);
  });

  it('sends nothing on a dry run or when nothing changed', async () => {
    const cwd = appFolder();
    const draft = fakeDraft({ 'old.js': 'x' });
    expect((await run(['push', '--dry-run'], { cwd, openDraft: async () => draft })).code).toBe(0);
    expect([...draft.files.keys()]).toEqual(['old.js']);
    expect(draft.commits).toBe(0);
    await run(['push'], { cwd, openDraft: async () => draft });
    const again = await run(['push'], { cwd, openDraft: async () => draft });
    expect(again.out[0]).toContain('already holds these 3 files');
    expect(draft.commits).toBe(1);
  });

  it('refuses a folder with no widget.json and a symbolic link', async () => {
    const cwd = appFolder();
    expect((await run(['push', '.'], { cwd })).err[0]).toContain('no widget.json');
    fs.symlinkSync('/etc/hostname', path.join(cwd, 'app', 'leak'));
    const r = await run(['push'], { cwd, openDraft: async () => fakeDraft() });
    expect(r.code).toBe(1);
    expect(r.err[0]).toContain('symbolic link');
  });
});

describe('app status', () => {
  it('falls back to the project, the link and the releases, and records the agent it learns', async () => {
    routes[`GET /api/casfs/projects/${APP}`] = () => ({ status: 200, body: { ok: true, head: 'f'.repeat(64) } });
    routes[`GET /api/casfs/projects/${APP}/git`] = () => ({ status: 200, body: { ok: true, link: { provider: 'github', repo: 'o/r', branch: 'main', folder: 'app', ciRelease: { enabled: false, bound: true } } } });
    routes[`GET /api/folders/${APP}/releases`] = () => ({ status: 401, body: { ok: false } });
    const r = await run(['status', '--json']);
    expect(r.code).toBe(0);
    const doc = JSON.parse(r.out.join('\n'));
    expect(doc.data.draft).toBe('f'.repeat(64));
    expect(doc.data.link).toMatchObject({ repo: 'o/r', 'ciRelease.enabled': 'false' });
    expect(doc.data.notes).toHaveLength(1);
  });

  it('answers a missing app by name and exit 1, as one JSON document under --json', async () => {
    const r = await run(['status', '--json']);
    expect(r.code).toBe(1);
    expect(JSON.parse(r.out.join('\n')).error.code).toBe('app_not_found');
  });
});

describe('app release is a person\'s act', () => {
  const preview = (over: Record<string, unknown> = {}) => ({
    status: 200,
    body: { ok: true, outcome: 'preview', ref: 'v1', sha: SHA, repo: 'o/r', firstRelease: false, security: { hash: HASH, widenedCount: 0, widened: [] }, ...over },
  });

  it.each([
    ['an API key', API_KEY],
    ["an agent's key", AGENT_KEY],
    ['a release token', jwt({ sub: 'u1', scopes: ['apps:release'] })],
  ])('refuses %s before any request', async (_what, token) => {
    const r = await run(['release', '--ref', 'v1'], { token });
    expect(r.code).toBe(1);
    expect(r.err[0]).toContain('Only a person');
    expect(seen).toHaveLength(0);
  });

  it('refuses an account that is not a person, before the release route is asked', async () => {
    routes['GET /api/users/me'] = me('agent');
    const r = await run(['release', '--ref', 'v1', '--json']);
    expect(JSON.parse(r.out.join('\n')).error.code).toBe('not_a_person');
    expect(seen.map((s) => s.url)).toEqual(['/api/users/me']);
    routes['GET /api/users/me'] = () => ({ status: 200, body: { user: { username: 'x' } } });
    expect((await run(['release', '--ref', 'v1'])).code).toBe(1);
  });

  it('releases a tag that widens nothing, sending the hash that was shown', async () => {
    routes['GET /api/users/me'] = me();
    routes[`POST /api/apps/${APP}/release`] = (req) => ((req.body as { dryRun?: boolean }).dryRun ? preview() : { status: 200, body: { ok: true, outcome: 'released', version: '2' } });
    const r = await run(['release', '--ref', 'v1', '--expect-commit', SHA]);
    expect(r.code).toBe(0);
    expect(r.out.join('\n')).toContain(`security hash: ${HASH}`);
    expect(seen.map((s) => s.body)).toEqual([undefined, { ref: 'v1', sha: SHA, dryRun: true }, { ref: 'v1', sha: SHA, acceptSecurity: HASH }]);
    expect(r.out.at(-1)).toBe('Released as version 2.');
  });

  it('never accepts a widening by itself: it prints the changes and the hash and stops with exit 3', async () => {
    routes['GET /api/users/me'] = me();
    routes[`POST /api/apps/${APP}/release`] = () => preview({ security: { hash: HASH, widenedCount: 2, widened: [{ field: 'permissions', item: 'camera', message: 'May use the camera\u001b[2J' }, { field: 'connect', item: 'api.example.com', message: 'May call api.example.com' }] } });
    const r = await run(['release', '--ref', 'v1']);
    expect(r.code).toBe(3);
    expect(r.out.join('\n')).toContain('- May use the camera');
    expect(r.out.join('\n')).toContain(HASH);
    expect(r.err.join('\n')).toContain(`--accept-security ${HASH}`);
    expect(seen.filter((s) => s.url.endsWith('/release')).map((s) => (s.body as { dryRun?: boolean }).dryRun)).toEqual([true]);
  });

  it('refuses a hash that is not the one shown, and releases with the one that is', async () => {
    routes['GET /api/users/me'] = me();
    const widened = { hash: HASH, widenedCount: 1, widened: [{ field: 'connect', item: 'x', message: 'May call x' }] };
    routes[`POST /api/apps/${APP}/release`] = (req) => ((req.body as { dryRun?: boolean }).dryRun ? preview({ security: widened }) : { status: 200, body: { ok: true, outcome: 'released' } });
    const wrong = await run(['release', '--ref', 'v1', '--accept-security', 'c'.repeat(64)]);
    expect(wrong.code).toBe(3);
    expect(wrong.err[0]).toContain('not the hash of this release');
    expect(seen.filter((s) => (s.body as { acceptSecurity?: string } | undefined)?.acceptSecurity)).toHaveLength(0);
    const right = await run(['release', '--ref', 'v1', '--accept-security', HASH]);
    expect(right.code).toBe(0);
    expect(seen.at(-1)!.body).toEqual({ ref: 'v1', sha: SHA, acceptSecurity: HASH });
  });

  it('treats a first release as needing acceptance, and a count of zero beside listed changes as a widening', async () => {
    routes['GET /api/users/me'] = me();
    routes[`POST /api/apps/${APP}/release`] = () => preview({ firstRelease: true });
    expect((await run(['release', '--ref', 'v1'])).code).toBe(3);
    expect(securityNote({ hash: HASH, widenedCount: 0, widened: [{ message: 'x' }] })!.widenedCount).toBe(1);
    expect(securityNote({ hash: 'short', widened: [] })).toBeNull();
  });

  it('stops when the tag is not at the expected commit, or the answer carries no security note', async () => {
    routes['GET /api/users/me'] = me();
    routes[`POST /api/apps/${APP}/release`] = () => preview({ sha: 'd'.repeat(40) });
    const moved = await run(['release', '--ref', 'v1', '--expect-commit', SHA, '--accept-security', HASH]);
    expect(moved.err[0]).toContain('Nothing was released');
    routes[`POST /api/apps/${APP}/release`] = () => preview({ security: undefined });
    expect((await run(['release', '--ref', 'v1'])).err[0]).toContain('did not say what this release changes');
    expect(seen.filter((s) => (s.body as { acceptSecurity?: string } | undefined)?.acceptSecurity)).toHaveLength(0);
  });

  it('says the owner was asked when the portal answers 202', async () => {
    routes['GET /api/users/me'] = me();
    routes[`POST /api/apps/${APP}/release`] = (req) => ((req.body as { dryRun?: boolean }).dryRun ? preview() : { status: 202, body: { ok: true, outcome: 'approval_requested', approvalUrl: `${baseUrl}/approve/1` } });
    const r = await run(['release', '--ref', 'v1']);
    expect(r.code).toBe(0);
    expect(r.out.join('\n')).toContain('Nothing is released yet');
    expect(r.out.join('\n')).toContain('/approve/1');
  });

  it('checks the shape of a tag, a commit and a hash before asking anything', async () => {
    expect((await run(['release', '--ref', 'main'])).code).toBe(2);
    expect((await run(['release', '--expect-commit', 'abc123'])).code).toBe(2);
    expect((await run(['release', '--accept-security', 'yes'])).code).toBe(2);
    expect(seen).toHaveLength(0);
  });
});

describe('app release --ci', () => {
  it('sends the release token from the environment, never the sign-in, and never prints it', async () => {
    routes[`POST /api/apps/${APP}/release/ci`] = (req) => ({ status: 403, body: { ok: false, error: { code: 'refused', message: `bad ${req.authorization}` } } });
    const r = await run(['release', '--ci', '--ref', 'v1', '--expect-commit', SHA], { env: { ROBUTLER_RELEASE_TOKEN: RELEASE_TOKEN } });
    expect(seen[0].authorization).toBe(`Bearer ${RELEASE_TOKEN}`);
    expect(seen[0].body).toEqual({ ref: 'v1', sha: SHA });
    expect(r.code).toBe(1);
    expect(r.all).not.toContain(RELEASE_TOKEN);
    expect(r.all).not.toContain(PERSON);
  });

  it('reports released and approval_requested, and needs the tag, the commit and the token', async () => {
    routes[`POST /api/apps/${APP}/release/ci`] = () => ({ status: 200, body: { ok: true, outcome: 'released' } });
    expect((await run(['release', '--ci', '--ref', 'v1', '--expect-commit', SHA], { env: { ROBUTLER_RELEASE_TOKEN: RELEASE_TOKEN } })).out[0]).toBe('Released.');
    expect((await run(['release', '--ci', '--ref', 'v1'], { env: { ROBUTLER_RELEASE_TOKEN: RELEASE_TOKEN } })).code).toBe(2);
    expect((await run(['release', '--ci', '--ref', 'v1', '--expect-commit', SHA])).err[0]).toContain('ROBUTLER_RELEASE_TOKEN is not set');
    expect((await run(['release', '--ci', '--ref', 'v1', '--expect-commit', SHA, '--accept-security', HASH], { env: { ROBUTLER_RELEASE_TOKEN: RELEASE_TOKEN } })).code).toBe(2);
  });
});

describe('app secrets', () => {
  const route = `/api/agents/${AGENT}/functions/checkout/secret`;

  it('reads the value from standard input and never prints it', async () => {
    routes[`POST ${route}`] = () => ({ status: 200, body: { ok: true } });
    const r = await run(['secrets', 'set', 'STRIPE_KEY', '--function', 'checkout'], { readSecret: async () => `${SECRET_VALUE}\n` });
    expect(r.code).toBe(0);
    expect(seen[0].body).toEqual({ binding: 'STRIPE_KEY', value: SECRET_VALUE });
    expect(r.all).not.toContain(SECRET_VALUE);
    expect(r.out).toEqual(['Set STRIPE_KEY.']);
  });

  it.each([
    [['secrets', 'set', 'STRIPE_KEY', SECRET_VALUE, '--function', 'checkout']],
    [['secrets', 'set', `STRIPE_KEY=${SECRET_VALUE}`, '--function', 'checkout']],
    [['secrets', 'set', 'STRIPE_KEY', '--value', SECRET_VALUE, '--function', 'checkout']],
    [['secrets', 'set', 'STRIPE_KEY', `--value=${SECRET_VALUE}`, '--function', 'checkout']],
  ])('refuses a value on the command line: %j', async (argv) => {
    const r = await run(argv);
    expect(r.code).toBe(2);
    expect(r.err[0]).toContain('never taken from the command line');
    expect(r.all).not.toContain(SECRET_VALUE);
    expect(seen).toHaveLength(0);
  });

  it('keeps the value out of the error when the server sends it back', async () => {
    routes[`POST ${route}`] = (req) => ({ status: 400, body: { error: `rejected value ${(req.body as { value: string }).value}` } });
    const r = await run(['secrets', 'set', 'STRIPE_KEY', '--function', 'checkout', '--json'], { readSecret: async () => SECRET_VALUE });
    expect(r.code).toBe(1);
    expect(r.all).not.toContain(SECRET_VALUE);
    expect(r.all).toContain('[redacted]');
  });

  it('sets nothing from empty input', async () => {
    const r = await run(['secrets', 'set', 'STRIPE_KEY', '--function', 'checkout'], { readSecret: async () => '\n' });
    expect(r.code).toBe(1);
    expect(seen).toHaveLength(0);
  });

  it('lists names only, whatever else the answer carries', async () => {
    routes[`GET ${route}`] = () => ({ status: 200, body: { bindings: [{ name: 'STRIPE_KEY', set: true, value: SECRET_VALUE }, { name: 'OTHER', set: false }, { name: `bad name ${SECRET_VALUE}`, set: true }] } });
    const r = await run(['secrets', 'list', '--function', 'checkout', '--json']);
    expect(JSON.parse(r.out.join('\n')).data).toEqual({ secrets: [{ name: 'STRIPE_KEY', set: true }, { name: 'OTHER', set: false }] });
    expect(r.all).not.toContain(SECRET_VALUE);
  });

  it('unsets by name, and needs a function and the app\'s agent', async () => {
    routes[`DELETE ${route}`] = () => ({ status: 200, body: { ok: true } });
    expect((await run(['secrets', 'unset', 'STRIPE_KEY', '--function', 'checkout'])).code).toBe(0);
    expect(seen[0].url).toBe(`${route}?binding=STRIPE_KEY`);
    expect((await run(['secrets', 'unset', 'STRIPE_KEY'])).code).toBe(2);
    expect((await run(['secrets', 'unset', 'STRIPE_KEY', '--function', '../x'])).code).toBe(2);
    const cwd = folder(false);
    writeBinding(cwd, baseUrl, { id: APP });
    expect((await run(['secrets', 'unset', 'STRIPE_KEY', '--function', 'checkout'], { cwd })).err[0]).toContain("app's agent is not known");
  });
});

describe('app hooks and logs', () => {
  it('runs a hook and reports its state', async () => {
    routes[`POST /api/folders/${APP}/release-hooks`] = () => ({ status: 200, body: { ok: true, status: 'ran' } });
    const r = await run(['hooks', 'run', 'post']);
    expect(seen[0].body).toEqual({ action: 'post' });
    expect(r.out[0]).toBe('The post hook of Platformer: ran.');
    expect((await run(['hooks', 'run', 'sideways'])).code).toBe(2);
  });

  it('says so when the portal runs only the post hook on request', async () => {
    routes[`POST /api/folders/${APP}/release-hooks`] = () => ({ status: 400, body: { ok: false, error: { code: 'bad_request', message: 'Expected { action: "post" }' } } });
    expect((await run(['hooks', 'run', 'up'])).err[0]).toContain('runs only the post hook on request');
  });

  it('shows recent runs of a function', async () => {
    routes[`GET /api/agents/${AGENT}/functions/checkout/invocations`] = () => ({ status: 200, body: { invocations: [{ startedAt: '2030-01-01T00:00:00.000Z', status: 'error', errorCode: 'timeout\u001b[31m', durationMs: 12 }] } });
    const r = await run(['logs', '--function', 'checkout', '--limit', '5']);
    expect(seen[0].url).toContain('limit=5');
    expect(r.out[0]).toBe('2030-01-01T00:00:00.000Z  error       12 ms  timeout');
  });
});

describe('app token', () => {
  it('writes the token to standard output once, the explanation to standard error, and to no file', async () => {
    const cwd = folder();
    const home = folder(false);
    routes['GET /api/users/me'] = me();
    routes['POST /api/apps/release-tokens'] = () => ({ status: 201, body: { token: RELEASE_TOKEN, listing: { id: 't1', name: 'CI', expiresAt: '2030-03-01' } } });
    const before = process.env.HOME;
    process.env.HOME = home;
    let r: Run;
    try {
      r = await run(['token', 'create', '--days', '30'], { cwd });
    } finally {
      process.env.HOME = before;
    }
    expect(r.code).toBe(0);
    expect(r.out).toEqual([RELEASE_TOKEN]);
    expect(r.err.join('\n')).not.toContain(RELEASE_TOKEN);
    expect(seen.at(-1)!.body).toEqual({ name: 'CI release of Platformer', app: APP, expiresInDays: 30 });
    const written = (dir: string): string[] => fs.readdirSync(dir, { recursive: true, withFileTypes: true }).filter((e) => e.isFile()).map((e) => path.join(e.parentPath, e.name));
    for (const file of [...written(cwd), ...written(home)]) expect(fs.readFileSync(file, 'utf8')).not.toContain(RELEASE_TOKEN);
  });

  it.each([
    ['an API key', API_KEY],
    ["an agent's key", AGENT_KEY],
  ])('refuses %s before any request', async (_what, token) => {
    const r = await run(['token', 'create'], { token });
    expect(r.code).toBe(1);
    expect(r.err[0]).toContain('Only a person');
    expect(seen).toHaveLength(0);
  });

  it('refuses an agent account, and never asks for a token then', async () => {
    routes['GET /api/users/me'] = me('agent');
    routes['POST /api/apps/release-tokens'] = () => ({ status: 201, body: { token: RELEASE_TOKEN } });
    const r = await run(['token', 'create']);
    expect(r.code).toBe(1);
    expect(seen.map((s) => s.method)).toEqual(['GET']);
    expect(r.all).not.toContain(RELEASE_TOKEN);
  });

  it('points to the browser, with nothing on standard output, when the portal keeps tokens to a browser session', async () => {
    routes['GET /api/users/me'] = me();
    routes['POST /api/apps/release-tokens'] = () => ({ status: 403, body: { error: { code: 'not_a_person', message: 'Release tokens are managed by a person, signed in to the site in a browser.' } } });
    const r = await run(['token', 'create']);
    expect(r.code).toBe(4);
    expect(r.out).toEqual([]);
    expect(r.opened).toEqual([`${baseUrl}/settings?tab=developer`]);
    routes['GET /api/apps/release-tokens'] = () => ({ status: 403, body: { error: { code: 'not_a_person', message: 'x' } } });
    routes['DELETE /api/apps/release-tokens/t1'] = () => ({ status: 401, body: { error: 'Unauthorized' } });
    expect((await run(['token', 'list'])).code).toBe(4);
    expect((await run(['token', 'revoke', 't1'])).code).toBe(4);
  });

  it('says release tokens are not offered when the portal has them switched off', async () => {
    routes['GET /api/users/me'] = me();
    for (const argv of [['token', 'create'], ['token', 'list'], ['token', 'revoke', 't1']]) {
      const r = await run(argv);
      expect(r.code).toBe(1);
      expect(r.err[0]).toMatch(/does not offer release tokens|release tokens are not offered/);
      expect(r.opened).toEqual([]);
    }
  });

  it('prints nothing that is not one printable line as a token', async () => {
    routes['GET /api/users/me'] = me();
    routes['POST /api/apps/release-tokens'] = () => ({ status: 201, body: { token: 'tok\u001b[2J\nsecond-line-0123456789' } });
    const r = await run(['token', 'create']);
    expect(r.code).toBe(1);
    expect(r.out).toEqual([]);
  });

  it('lists and revokes', async () => {
    routes['GET /api/apps/release-tokens'] = () => ({ status: 200, body: { tokens: [{ id: 't1', name: 'CI', expiresAt: '2030-03-01', token: RELEASE_TOKEN }] } });
    routes['DELETE /api/apps/release-tokens/t1'] = () => ({ status: 200, body: { ok: true } });
    const listed = await run(['token', 'list']);
    expect(listed.out[0]).toContain('t1  CI');
    expect(listed.all).not.toContain(RELEASE_TOKEN);
    expect((await run(['token', 'revoke', 't1'])).out[0]).toContain('Revoked t1');
    expect((await run(['token', 'revoke', '../x'])).code).toBe(2);
  });
});

describe('arguments', () => {
  it('refuses an option it does not know, and prints the help', async () => {
    expect((await run(['status', '--force'])).code).toBe(2);
    expect(() => parseAppArgs(['push', '--ref'])).toThrow('Missing value for --ref');
    const help = await run(['--help']);
    expect(help.code).toBe(0);
    expect(help.out[0]).toContain('robutler app <command>');
    expect((await run([])).code).toBe(2);
    expect((await run(['frobnicate'])).err[0]).toBe('Unknown command: app frobnicate');
  });

  it('acts on --app or ROBUTLER_APP_ID instead of the bound app', async () => {
    const other = '33333333-3333-4333-8333-333333333333';
    routes[`GET /api/apps/${other}`] = () => ({ status: 200, body: { app: { id: other, name: 'Other' } } });
    expect((await run(['status', '--app', other])).out[0]).toContain('Other');
    expect((await run(['status'], { env: { ROBUTLER_APP_ID: other } })).out[0]).toContain('Other');
    expect((await run(['status', '--app', 'nope'])).code).toBe(2);
  });
});
