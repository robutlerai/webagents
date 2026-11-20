#!/usr/bin/env node
/**
 * git-remote-casfs: a git remote helper for Robutler CASFS projects.
 *
 *   git clone casfs://<project-id>
 *   git clone casfs://<host>/<project-id>
 *   git fetch / git pull / git push
 *
 * Git runs this program when a remote URL starts with `casfs://` and talks to it over stdin and stdout with the
 * remote helper protocol (see `gitremote-helpers(7)`). The helper speaks to the portal's project API with the same
 * login and portal address as every other `webagents` command (`webagents login`, `WEBAGENTS_TOKEN`,
 * `ROBUTLER_API_URL`, `platform.url`). An agent that holds a CASFS grant for the project passes it in `CASFS_GRANT`.
 *
 * WHAT IT DOES
 *   list          the refs the server lets this caller read: branches as `refs/heads/<name>`, tags as
 *                 `refs/tags/<name>`, and HEAD as `main`.
 *   fetch         asks the server for a pack of ONE ref (naming the commit git wants, and the commits this repository
 *                 already holds so the server leaves out what is here), indexes it with `git index-pack --stdin`, and
 *                 records the commits whose history the server did not send in `$GIT_DIR/shallow`.
 *   push          builds a pack with `git pack-objects --revs --no-thin` (every delta base is inside the pack; the
 *                 server accepts no thin packs) of the commits the remote does not hold, and sends it. The server
 *                 never moves `main`, `release` or a tag: a push lands on your own branch of the project, and the
 *                 helper prints where (`people/<you>` or `agent/<you>`). Merge it into `main` in the project.
 *
 * Exit status is 0 when git's own protocol finished; a refused push is reported to git as `error <ref> <why>`, which
 * git shows and turns into its own non-zero status.
 */

import { spawn } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const OID = /^[0-9a-f]{40}$/;

// ── URLs ────────────────────────────────────────────────────────────────────

export interface CasfsRemote {
  projectId: string;
  /** An explicit portal origin from the URL (`casfs://<host>/<id>`), or null for the configured one. */
  origin: string | null;
}

/** `casfs://<project-id>` or `casfs://<host>[:port]/<project-id>`. */
export function parseCasfsUrl(url: string): CasfsRemote {
  const m = /^casfs:\/\/(?:([A-Za-z0-9.-]+(?::\d{1,5})?)\/)?([0-9a-fA-F-]{36})\/?$/.exec(url.trim());
  if (!m || !UUID.test(m[2])) throw new Error(`not a CASFS remote: ${url} (expected casfs://<project-id> or casfs://<host>/<project-id>)`);
  const host = m[1] ?? null;
  const loopback = host !== null && /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(host);
  return { projectId: m[2].toLowerCase(), origin: host ? `${loopback ? 'http' : 'https'}://${host}` : null };
}

// ── The server ──────────────────────────────────────────────────────────────

export interface RefAd {
  name: string;
  oid: string;
  casfsRef: string;
  kind: 'branch' | 'tag';
}
export interface FetchedPack {
  pack: Uint8Array;
  commit: string;
  shallow: string[];
  objects: number;
}
export interface Landed {
  ref: string;
  commit: string;
  outcome: 'created' | 'fast-forward' | 'merged' | 'unchanged';
}

export interface CasfsGitApi {
  refs(): Promise<{ head: string | null; refs: RefAd[] }>;
  fetch(ref: string, want: string, haves: string[]): Promise<FetchedPack>;
  push(ref: string, newOid: string, pack: Uint8Array): Promise<Landed>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null,
  ) {
    super(message);
  }
}

/** The portal's git routes for one project. */
export function httpApi(opts: { baseUrl: string; projectId: string; token: string | null; grant?: string | null; fetchImpl?: typeof fetch }): CasfsGitApi {
  const f = opts.fetchImpl ?? fetch;
  const base = `${opts.baseUrl.replace(/\/+$/, '')}/api/casfs/projects/${opts.projectId}/git`;
  const headers = (extra: Record<string, string> = {}) => ({
    ...(opts.token ? { authorization: `Bearer ${opts.token}` } : {}),
    ...(opts.grant ? { 'x-casfs-grant': opts.grant } : {}),
    'user-agent': 'git-remote-casfs',
    ...extra,
  });
  const fail = async (res: Response): Promise<never> => {
    let code: string | null = null;
    let message = `HTTP ${res.status}`;
    try {
      const j = (await res.json()) as { error?: { code?: string; message?: string }; code?: string; message?: string };
      code = j.error?.code ?? j.code ?? null;
      message = j.error?.message ?? j.message ?? message;
    } catch {
      /* not JSON */
    }
    if (res.status === 401) message = 'not signed in: run `webagents login` (or set WEBAGENTS_TOKEN)';
    if (res.status === 404 && (code === null || code === 'Cas:NotFound')) message = 'project or ref not found, or you do not have access to it';
    throw new ApiError(message, res.status, code);
  };
  return {
    async refs() {
      const res = await f(`${base}/refs`, { headers: headers(), redirect: 'error' });
      if (!res.ok) return fail(res);
      const j = (await res.json()) as { head: string | null; refs: RefAd[] };
      return { head: j.head ?? null, refs: Array.isArray(j.refs) ? j.refs.filter((r) => OID.test(r.oid) && typeof r.name === 'string') : [] };
    },
    async fetch(ref, want, haves) {
      const res = await f(`${base}/fetch`, { method: 'POST', headers: headers({ 'content-type': 'application/json' }), body: JSON.stringify({ ref, want, haves }), redirect: 'error' });
      if (!res.ok) return fail(res);
      const commit = res.headers.get('x-casfs-git-commit') ?? '';
      const shallow = (res.headers.get('x-casfs-git-shallow') ?? '').split(',').filter((s) => OID.test(s));
      return { pack: new Uint8Array(await res.arrayBuffer()), commit, shallow, objects: Number(res.headers.get('x-casfs-git-objects') ?? '-1') };
    },
    async push(ref, newOid, pack) {
      const res = await f(`${base}/push`, {
        method: 'POST',
        headers: headers({ 'content-type': 'application/x-git-packed-objects', 'x-casfs-git-ref': ref, 'x-casfs-git-new': newOid }),
        body: pack as unknown as BodyInit,
        redirect: 'error',
      });
      if (!res.ok) return fail(res);
      return ((await res.json()) as { landed: Landed }).landed;
    },
  };
}

// ── Local git ───────────────────────────────────────────────────────────────

export interface GitRunner {
  /** Run git with `args`, feeding `input`; resolve stdout, reject on a non-zero exit. */
  run(args: string[], input?: Uint8Array | string): Promise<Buffer>;
  /** The repository's git directory. */
  gitDir(): string;
}

export function processGit(env: NodeJS.ProcessEnv = process.env): GitRunner {
  return {
    run: (args, input) =>
      new Promise((resolve, reject) => {
        const p = spawn('git', args, { env, stdio: ['pipe', 'pipe', 'pipe'] });
        const out: Buffer[] = [];
        const err: Buffer[] = [];
        p.stdout.on('data', (b: Buffer) => out.push(b));
        p.stderr.on('data', (b: Buffer) => err.push(b));
        p.on('error', reject);
        p.on('close', (code) => (code === 0 ? resolve(Buffer.concat(out)) : reject(new Error(`git ${args[0]} failed: ${Buffer.concat(err).toString().trim()}`))));
        p.stdin.on('error', () => undefined);
        p.stdin.end(input ?? '');
      }),
    gitDir: () => env.GIT_DIR ?? '.git',
  };
}

// ── The protocol ────────────────────────────────────────────────────────────

export interface HelperIo {
  /** Write to git (stdout). */
  out(text: string): void;
  /** Tell the person (stderr). */
  note(text: string): void;
}

/**
 * One helper session. Feed it git's lines with `line()`; it answers through `io.out`. Batches (`fetch`, `push`) are
 * collected until git's blank line and run together, as the protocol requires.
 */
export class RemoteHelper {
  private batch: { kind: 'fetch' | 'push'; lines: string[] } | null = null;
  private advertised = new Map<string, string>();
  private verbosity = 1;
  done = false;

  constructor(
    private readonly remoteName: string,
    private readonly api: CasfsGitApi,
    private readonly git: GitRunner,
    private readonly io: HelperIo,
  ) {}

  async line(raw: string): Promise<void> {
    const line = raw.replace(/\r?\n$/, '');
    if (this.batch) {
      if (line === '') {
        const b = this.batch;
        this.batch = null;
        if (b.kind === 'fetch') await this.runFetch(b.lines);
        else await this.runPush(b.lines);
        return;
      }
      if (!line.startsWith(`${this.batch.kind} `)) throw new Error(`unexpected command inside a ${this.batch.kind} batch: ${line}`);
      this.batch.lines.push(line);
      return;
    }
    if (line === '') {
      this.done = true;
      return;
    }
    const [cmd, ...rest] = line.split(' ');
    switch (cmd) {
      case 'capabilities':
        this.io.out('fetch\npush\noption\n\n');
        return;
      case 'option':
        this.io.out(this.option(rest[0] ?? '', rest.slice(1).join(' ')) + '\n');
        return;
      case 'list':
        await this.list();
        return;
      case 'fetch':
      case 'push':
        this.batch = { kind: cmd, lines: [line] };
        return;
      default:
        throw new Error(`unsupported command from git: ${cmd}`);
    }
  }

  private option(name: string, value: string): string {
    if (name === 'verbosity') {
      this.verbosity = Number(value) || 0;
      return 'ok';
    }
    if (name === 'progress' || name === 'followtags') return 'ok';
    return 'unsupported';
  }

  private async list(): Promise<void> {
    const { head, refs } = await this.api.refs();
    this.advertised = new Map(refs.map((r) => [r.name, r.oid]));
    let text = '';
    for (const r of refs) text += `${r.oid} ${r.name}\n`;
    if (head && this.advertised.has(head)) text += `@${head} HEAD\n`;
    this.io.out(text + '\n');
  }

  /** Commits this repository holds that the server may recognise: its view of this remote, and every ref it has. */
  private async haves(): Promise<string[]> {
    const out = (await this.git.run(['for-each-ref', '--format=%(objectname) %(objecttype)'])).toString();
    const oids = new Set<string>();
    for (const l of out.split('\n')) {
      const [oid, type] = l.split(' ');
      if (type === 'commit' && OID.test(oid)) oids.add(oid);
    }
    return [...oids].slice(0, 1000);
  }

  private async hasObject(oid: string): Promise<boolean> {
    try {
      await this.git.run(['cat-file', '-e', oid]);
      return true;
    } catch {
      return false;
    }
  }

  private async runFetch(lines: string[]): Promise<void> {
    const haves = await this.haves();
    const seen = new Set<string>();
    for (const l of lines) {
      const [, want, name] = l.split(' ');
      if (!OID.test(want) || seen.has(want)) continue;
      seen.add(want);
      if (await this.hasObject(want)) continue;
      const got = await this.api.fetch(name, want, haves);
      if (got.commit !== want) throw new Error(`${name} moved on the server while fetching; fetch again`);
      if (got.objects !== 0) await this.git.run(['index-pack', '--stdin'], got.pack);
      await this.recordShallow(got.shallow);
      if (this.verbosity > 1) this.io.note(`casfs: fetched ${name} (${got.objects} objects)`);
    }
    this.io.out('\n');
  }

  /** A commit whose parents the server did not send, and this repository does not hold, is a shallow boundary. */
  private async recordShallow(oids: string[]): Promise<void> {
    if (!oids.length) return;
    const file = join(this.git.gitDir(), 'shallow');
    const have = new Set(existsSync(file) ? readFileSync(file, 'utf8').split('\n').filter(Boolean) : []);
    const add: string[] = [];
    for (const oid of oids) {
      if (have.has(oid)) continue;
      const body = (await this.git.run(['cat-file', 'commit', oid])).toString();
      const parents = [...body.matchAll(/^parent ([0-9a-f]{40})$/gm)].map((m) => m[1]);
      let missing = false;
      for (const p of parents) if (!(await this.hasObject(p))) missing = true;
      if (missing) add.push(oid);
    }
    if (add.length) appendFileSync(file, add.map((o) => o + '\n').join(''));
  }

  private async runPush(lines: string[]): Promise<void> {
    let reply = '';
    for (const l of lines) {
      const spec = l.slice('push '.length);
      const force = spec.startsWith('+');
      const [src, dst] = (force ? spec.slice(1) : spec).split(':');
      if (!dst) {
        reply += `error ${src} a push names a destination branch\n`;
        continue;
      }
      if (!src) {
        reply += `error ${dst} deleting a branch through git is not supported; remove it in the project\n`;
        continue;
      }
      try {
        const newOid = (await this.git.run(['rev-parse', '--verify', `${src}^{commit}`])).toString().trim();
        // What the remote holds and this repository knows: everything reachable from it stays out of the pack.
        const exclude: string[] = [];
        for (const oid of new Set([...this.advertised.values(), ...(await this.trackingOids())])) if (await this.hasObject(oid)) exclude.push(`^${oid}`);
        const pack = await this.git.run(['pack-objects', '--revs', '--stdout', '--no-thin', '-q'], [newOid, ...exclude].join('\n') + '\n');
        const landed = await this.api.push(dst, newOid, pack);
        this.io.note(`casfs: ${dst} landed on ${landed.ref} (${landed.outcome}); fetch to see it, and merge it into main in the project`);
        reply += `ok ${dst}\n`;
      } catch (e) {
        const msg = (e as Error).message.replace(/\n/g, ' ');
        reply += `error ${dst} ${msg}\n`;
      }
    }
    this.io.out(reply + '\n');
  }

  private async trackingOids(): Promise<string[]> {
    const out = (await this.git.run(['for-each-ref', '--format=%(objectname)', `refs/remotes/${this.remoteName}/`]).catch(() => Buffer.alloc(0))).toString();
    return out.split('\n').filter((o) => OID.test(o));
  }
}

// ── Entry ───────────────────────────────────────────────────────────────────

/** Read git's commands line by line from stdin and answer on stdout until git closes the conversation. */
export async function serve(helper: RemoteHelper, input: NodeJS.ReadableStream = process.stdin): Promise<void> {
  let buf = '';
  for await (const chunk of input) {
    buf += chunk.toString();
    let i: number;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      await helper.line(line);
      if (helper.done) return;
    }
  }
}

async function main(argv: string[]): Promise<number> {
  const [remoteName, url] = [argv[2], argv[3] ?? argv[2]];
  if (!url) {
    process.stderr.write('usage: git-remote-casfs <remote> <casfs://project-id>\n(git runs this for you: git clone casfs://<project-id>)\n');
    return 2;
  }
  const remote = parseCasfsUrl(url);
  const [{ resolvePlatformUrl }, { getToken }] = await Promise.all([import('./config-store.js'), import('./credentials.js')]);
  const baseUrl = remote.origin ?? resolvePlatformUrl()[0];
  const token = await getToken().catch(() => null);
  const api = httpApi({ baseUrl, projectId: remote.projectId, token, grant: process.env.CASFS_GRANT ?? null });
  const helper = new RemoteHelper(remoteName && !remoteName.includes('://') ? remoteName : 'origin', api, processGit(), {
    out: (t) => process.stdout.write(t),
    note: (t) => process.stderr.write(t + '\n'),
  });
  await serve(helper);
  return 0;
}

/** True when this file is the program git ran (an installed bin is a symlink to it), not a module a test imported. */
function isEntry(): boolean {
  try {
    return !!process.argv[1] && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;
  } catch {
    return false;
  }
}

if (isEntry()) {
  main(process.argv).then(
    (code) => process.exit(code),
    (e: Error) => {
      process.stderr.write(`fatal: ${e.message}\n`);
      process.exit(128);
    },
  );
}
