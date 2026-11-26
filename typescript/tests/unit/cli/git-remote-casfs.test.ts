/**
 * git-remote-casfs: the remote helper protocol, and a real `git clone` / `git push` through it.
 *
 * The protocol half drives `RemoteHelper` line by line against an in-memory API and a recorded git runner. The end to
 * end half (skipped when git is not installed) starts a small HTTP server that answers the three project git routes
 * over a bare repository, puts the helper on PATH, and runs real git: clone, edit, push (which must arrive as a pack
 * with no thin deltas and land on the caller's own branch), fetch.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { RemoteHelper, httpApi, parseCasfsUrl, type CasfsGitApi, type GitRunner } from '../../../src/cli/git-remote-casfs.js';

const PROJECT = '11111111-1111-4111-8111-111111111111';
const A = 'a'.repeat(40);
const B = 'b'.repeat(40);

describe('parseCasfsUrl', () => {
  it('takes casfs://<id> and casfs://<host>/<id>', () => {
    expect(parseCasfsUrl(`casfs://${PROJECT}`)).toEqual({ projectId: PROJECT, origin: null });
    expect(parseCasfsUrl(`casfs://robutler.example/${PROJECT.toUpperCase()}`)).toEqual({ projectId: PROJECT, origin: 'https://robutler.example' });
    expect(parseCasfsUrl(`casfs://127.0.0.1:4100/${PROJECT}`)).toEqual({ projectId: PROJECT, origin: 'http://127.0.0.1:4100' });
    for (const bad of ['casfs://not-a-uuid', `https://x/${PROJECT}`, `casfs://a/b/${PROJECT}`, `casfs://user@host/${PROJECT}`]) expect(() => parseCasfsUrl(bad)).toThrow();
  });
});

function recorder(responses: Record<string, string | Error> = {}): GitRunner & { calls: Array<{ args: string[]; input?: string }> } {
  const calls: Array<{ args: string[]; input?: string }> = [];
  return {
    calls,
    gitDir: () => '/nonexistent/.git',
    run: async (args, input) => {
      calls.push({ args, input: input === undefined ? undefined : Buffer.from(input).toString() });
      const r = responses[args.join(' ')] ?? responses[args[0]];
      if (r instanceof Error) throw r;
      return Buffer.from(r ?? '');
    },
  };
}

describe('the protocol', () => {
  const api = (over: Partial<CasfsGitApi> = {}): CasfsGitApi => ({
    refs: async () => ({ head: 'refs/heads/main', refs: [{ name: 'refs/heads/main', oid: A, casfsRef: 'main', kind: 'branch' }, { name: 'refs/tags/release/1', oid: B, casfsRef: 'release/1', kind: 'tag' }] }),
    fetch: async () => ({ pack: new Uint8Array([1, 2, 3]), commit: A, shallow: [], objects: 3 }),
    push: async () => ({ ref: 'people/me', commit: 'c'.repeat(64), outcome: 'created' }),
    ...over,
  });
  const session = (a: CasfsGitApi, git: GitRunner) => {
    const out: string[] = [];
    const notes: string[] = [];
    const h = new RemoteHelper('origin', a, git, { out: (t) => out.push(t), note: (t) => notes.push(t) });
    return { h, out, notes };
  };

  it('capabilities, options, list with HEAD', async () => {
    const { h, out } = session(api(), recorder());
    await h.line('capabilities');
    expect(out.join('')).toBe('fetch\npush\noption\n\n');
    out.length = 0;
    await h.line('option verbosity 1');
    await h.line('option depth 1');
    expect(out).toEqual(['ok\n', 'unsupported\n']);
    out.length = 0;
    await h.line('list for-push');
    expect(out.join('')).toBe(`${A} refs/heads/main\n${B} refs/tags/release/1\n@refs/heads/main HEAD\n\n`);
    await h.line('');
    expect(h.done).toBe(true);
  });

  it('fetch: one request per wanted commit, with the haves, indexed by index-pack; answered with a blank line', async () => {
    const asked: Array<{ ref: string; want: string; haves: string[] }> = [];
    const git = recorder({ 'for-each-ref': `${B} commit\n${'d'.repeat(40)} tag\n`, 'cat-file': new Error('missing') });
    const { h, out } = session(api({ fetch: async (ref, want, haves) => (asked.push({ ref, want, haves }), { pack: new Uint8Array([9]), commit: want, shallow: [], objects: 1 }) }), git);
    await h.line(`fetch ${A} refs/heads/main`);
    await h.line(`fetch ${A} HEAD`);
    expect(out).toEqual([]);
    await h.line('');
    expect(asked).toEqual([{ ref: 'refs/heads/main', want: A, haves: [B] }]);
    expect(git.calls.find((c) => c.args[0] === 'index-pack')?.args).toEqual(['index-pack', '--stdin']);
    expect(out).toEqual(['\n']);
  });

  it('fetch: a ref that moved between list and fetch is an error, not a silent other commit', async () => {
    const { h } = session(api({ fetch: async () => ({ pack: new Uint8Array(), commit: B, shallow: [], objects: 0 }) }), recorder({ 'for-each-ref': '', 'cat-file': new Error('missing') }));
    await h.line(`fetch ${A} refs/heads/main`);
    await expect(h.line('')).rejects.toThrow(/moved/);
  });

  it('push: a pack with no thin deltas, excluding what the remote holds; reports where it landed; refusals become error lines', async () => {
    const git = recorder({ 'rev-parse': `${'e'.repeat(40)}\n`, 'pack-objects': 'PACK', 'for-each-ref': `${A}\n`, 'cat-file': '' });
    const sent: Array<{ ref: string; oid: string }> = [];
    const { h, out, notes } = session(
      api({
        push: async (ref, oid) => {
          sent.push({ ref, oid });
          if (ref === 'refs/heads/main') throw new Error('git push never moves main; push to a branch of your own and merge in the project');
          return { ref: 'people/me', commit: 'c'.repeat(64), outcome: 'merged' };
        },
      }),
      git,
    );
    await h.line('list for-push');
    out.length = 0;
    await h.line('push refs/heads/work:refs/heads/feature');
    await h.line('push +refs/heads/work:refs/heads/main');
    await h.line('push :refs/heads/old');
    await h.line('');
    const pk = git.calls.find((c) => c.args[0] === 'pack-objects')!;
    expect(pk.args).toEqual(['pack-objects', '--revs', '--stdout', '--no-thin', '-q']);
    expect(pk.input).toContain(`${'e'.repeat(40)}\n`);
    expect(pk.input).toContain(`^${A}`);
    expect(sent.map((s) => s.ref)).toEqual(['refs/heads/feature', 'refs/heads/main']);
    const reply = out.join('');
    expect(reply).toContain('ok refs/heads/feature\n');
    expect(reply).toMatch(/error refs\/heads\/main git push never moves main/);
    expect(reply).toMatch(/error refs\/heads\/old deleting a branch/);
    expect(reply.endsWith('\n\n')).toBe(true);
    expect(notes[0]).toMatch(/landed on people\/me \(merged\)/);
  });

  it('the HTTP client sends the token and the grant, and names a 404 without saying which of missing or private', async () => {
    const seen: Array<{ url: string; headers: Record<string, string> }> = [];
    const fetchImpl = (async (url: string, init: RequestInit) => {
      seen.push({ url, headers: init.headers as Record<string, string> });
      return new Response(JSON.stringify({ ok: false, error: { code: 'Cas:NotFound', message: 'not found' } }), { status: 404 });
    }) as unknown as typeof fetch;
    const c = httpApi({ baseUrl: 'https://portal.example/', projectId: PROJECT, token: 'tok', grant: 'g', fetchImpl });
    await expect(c.refs()).rejects.toThrow(/not found, or you do not have access/);
    expect(seen[0].url).toBe(`https://portal.example/api/casfs/projects/${PROJECT}/git/refs`);
    expect(seen[0].headers.authorization).toBe('Bearer tok');
    expect(seen[0].headers['x-casfs-grant']).toBe('g');
  });
});

// ── End to end with real git ────────────────────────────────────────────────

const HAVE_GIT = spawnSync('git', ['--version']).status === 0;
const ROOT = resolve(__dirname, '../../..');
const TSX = join(ROOT, 'node_modules/.bin/tsx');

describe.skipIf(!HAVE_GIT)('git clone and push through the helper', () => {
  let dir = '';
  let server: Server;
  let port = 0;
  let pushes: Array<{ ref: string; thinRefused: boolean }> = [];
  const env = () => ({ ...process.env, PATH: `${join(dir, 'bin')}:${process.env.PATH}`, WEBAGENTS_TOKEN: 'test-token', GIT_AUTHOR_NAME: 'T', GIT_AUTHOR_EMAIL: 't@x', GIT_COMMITTER_NAME: 'T', GIT_COMMITTER_EMAIL: 't@x', GIT_TERMINAL_PROMPT: '0' });
  const git = (cwd: string, ...args: string[]) => execFileSync('git', args, { cwd, env: env(), stdio: ['ignore', 'pipe', 'pipe'] }).toString();
  const bare = () => join(dir, 'server.git');
  // Git that talks to the helper must run ASYNCHRONOUSLY: the routes are served by this process, and a synchronous
  // child would block the event loop the server answers on.
  const gitAsync = (cwd: string, ...args: string[]) =>
    new Promise<{ status: number | null; stderr: string }>((res) => {
      const p = spawn('git', args, { cwd, env: env(), stdio: ['ignore', 'ignore', 'pipe'] });
      let stderr = '';
      p.stderr.on('data', (b: Buffer) => (stderr += b.toString()));
      p.on('close', (status) => res({ status, stderr }));
    });

  beforeAll(async () => {
    // Under the caller's report folder when it names one, else the OS temp dir; only this directory is removed afterwards.
    const parent = process.env.CASFS_E2E_DIR ?? tmpdir();
    mkdirSync(parent, { recursive: true });
    dir = mkdtempSync(join(parent, 'git-remote-casfs-'));
    mkdirSync(join(dir, 'bin'));
    writeFileSync(join(dir, 'bin', 'git-remote-casfs'), `#!/bin/sh\nexec "${TSX}" "${join(ROOT, 'src/cli/git-remote-casfs.ts')}" "$@"\n`);
    chmodSync(join(dir, 'bin', 'git-remote-casfs'), 0o755);
    execFileSync('git', ['init', '-q', '--bare', bare()]);
    const seed = join(dir, 'seed');
    execFileSync('git', ['init', '-q', '-b', 'main', seed]);
    writeFileSync(join(seed, 'readme.md'), '# hello\n');
    mkdirSync(join(seed, 'src'));
    writeFileSync(join(seed, 'src', 'app.js'), 'console.log(1)\n');
    git(seed, 'add', '.');
    git(seed, 'commit', '-q', '-m', 'first');
    git(seed, 'push', '-q', bare(), 'main:refs/heads/main');

    // The three routes, over the bare repository.
    server = createServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(c as Buffer);
      const body = Buffer.concat(chunks);
      const path = (req.url ?? '').replace(`/api/casfs/projects/${PROJECT}/git`, '');
      if (req.headers.authorization !== 'Bearer test-token') return void res.writeHead(401).end('{}');
      const sg = (args: string[], input?: Buffer) => spawnSync('git', args, { cwd: bare(), input });
      if (path === '/refs') {
        const out = sg(['for-each-ref', '--format=%(objectname) %(refname)']).stdout.toString().trim();
        const refs = out ? out.split('\n').map((l) => { const [oid, name] = l.split(' '); return { name, oid, casfsRef: name.replace(/^refs\/(heads|tags)\//, ''), kind: 'branch' }; }) : [];
        return void res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true, head: 'refs/heads/main', refs }));
      }
      if (path === '/fetch') {
        const j = JSON.parse(body.toString()) as { want: string; haves: string[] };
        const have = j.haves.filter((h) => sg(['cat-file', '-e', h]).status === 0).map((h) => `^${h}`);
        const pack = sg(['pack-objects', '--revs', '--stdout', '-q'], Buffer.from([j.want, ...have].join('\n') + '\n')).stdout;
        return void res.writeHead(200, { 'x-casfs-git-commit': j.want, 'x-casfs-git-shallow': '', 'x-casfs-git-objects': String(pack.readUInt32BE(8)) }).end(pack);
      }
      if (path === '/push') {
        const ref = String(req.headers['x-casfs-git-ref']);
        const oid = String(req.headers['x-casfs-git-new']);
        if (ref === 'refs/heads/main') return void res.writeHead(403, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: false, error: { code: 'Cas:Forbidden', message: 'git push never moves main' } }));
        // No --fix-thin: a pack with a delta against an object it does not carry fails here, as on the server.
        const ip = sg(['index-pack', '--stdin'], body);
        pushes.push({ ref, thinRefused: ip.status !== 0 });
        if (ip.status !== 0) return void res.writeHead(400, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: false, error: { code: 'Cas:ArchiveInvalid', message: 'thin pack' } }));
        sg(['update-ref', 'refs/heads/people/me', oid]);
        return void res.writeHead(201, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true, landed: { ref: 'people/me', commit: 'c'.repeat(64), outcome: 'created' } }));
      }
      res.writeHead(404).end('{}');
    });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
    port = (server.address() as { port: number }).port;
  }, 60_000);

  afterAll(async () => {
    await new Promise<void>((r) => (server ? server.close(() => r()) : r()));
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  it('clones, pushes a change to a branch of its own (no thin pack), is refused on main, and fetches', async () => {
    const url = `casfs://127.0.0.1:${port}/${PROJECT}`;
    const client = join(dir, 'client');
    const cl = await gitAsync(dir, 'clone', '-q', url, client);
    expect(cl.status, cl.stderr).toBe(0);
    expect(readFileSync(join(client, 'readme.md'), 'utf8')).toBe('# hello\n');
    // A one-line change to a file the server holds: a delta candidate against the server's copy.
    writeFileSync(join(client, 'src', 'app.js'), 'console.log(1)\nconsole.log(2)\n');
    git(client, 'commit', '-qam', 'second');
    const r = await gitAsync(client, 'push', 'origin', 'HEAD:refs/heads/feature');
    expect(r.status, r.stderr).toBe(0);
    expect(r.stderr).toMatch(/landed on people\/me/);
    expect(pushes).toEqual([{ ref: 'refs/heads/feature', thinRefused: false }]);
    const head = git(client, 'rev-parse', 'HEAD').trim();
    expect(spawnSync('git', ['rev-parse', 'refs/heads/people/me'], { cwd: bare() }).stdout.toString().trim()).toBe(head);
    const main = await gitAsync(client, 'push', 'origin', 'HEAD:main');
    expect(main.status).not.toBe(0);
    expect(main.stderr).toMatch(/never moves main/);
    const f = await gitAsync(client, 'fetch', 'origin');
    expect(f.status, f.stderr).toBe(0);
    expect(git(client, 'rev-parse', 'refs/remotes/origin/people/me').trim()).toBe(head);
  }, 180_000);
});
