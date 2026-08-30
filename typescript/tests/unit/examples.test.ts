/**
 * The examples cannot rot: the three minimal examples are EXECUTED here
 * against a stub portal (loopback only, no model call, no key), and the doc
 * snippets are asserted to be generated from these exact files.
 *
 * These now run the examples' OWN top-level code — `await serve(agent, ...)`
 * really binds a socket on port 0 — rather than reaching for a test-only
 * builder. There is no wrapper left to reach for: what the example does is
 * what the test drives.
 *
 * The examples import from 'webagents' (the documented specifier); vi.mock
 * aliases it onto src/ so the example modules load identically under the
 * SDK's own vitest config and the monorepo's.
 */

import { describe, it, expect, vi, afterAll } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { WebSocketServer, type WebSocket as WsSocket } from 'ws';

vi.mock('webagents', async () => await import('../../src/index'));

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.resolve(HERE, '../../examples');
const DOCS = path.resolve(HERE, '../../../docs');
const REPO = path.resolve(HERE, '../../..');

// Examples bind a real socket; port 0 keeps that to an ephemeral port. The
// key must not land in the developer's home directory either.
process.env.PORT = '0';
process.env.WEBAGENTS_KEYS_DIR = mkdtempSync(path.join(tmpdir(), 'webagents-example-keys-'));
process.env.OPENAI_API_KEY ??= 'test-key-not-used';
// The agent card's `url` is the address the agent publishes for itself, and an
// agent behind a proxy/tunnel is NOT reachable at the host a request happened
// to arrive on. Set before the import: the example reads the environment at
// module scope.
const PUBLIC_URL = 'https://agent.example.com';
process.env.WEBAGENTS_PUBLIC_URL = PUBLIC_URL;

function fakeAgentToken(withBinding = true): string {
  const seg = (obj: Record<string, unknown>) =>
    Buffer.from(JSON.stringify(obj)).toString('base64url');
  const payload: Record<string, unknown> = { sub: 'owner-1' };
  if (withBinding) payload.agent_id = 'agent-1';
  return `${seg({ alg: 'none' })}.${seg(payload)}.x`;
}

interface StubPortal {
  port: number;
  frames: Array<Record<string, unknown>>;
  waitFor(type: string, timeoutMs?: number): Promise<Record<string, unknown>>;
  /** Push a turn on demand, so a test can stub the agent's run first. */
  pushTurn(): void;
  close(): Promise<void>;
}

/**
 * A loopback `/ws` speaking the platform's REAL frames. Note what it does
 * with ids: the session it ACKs is `sess_1`, but the turn it later pushes
 * carries a PER-REQUEST `req_...` id the SDK has never seen, plus the frame's
 * own `agent` field — the fallback the bridge must not lose.
 */
async function startStubPortal(): Promise<StubPortal> {
  const wss = new WebSocketServer({ port: 0 });
  await new Promise<void>((resolve) => wss.on('listening', () => resolve()));

  const frames: Array<Record<string, unknown>> = [];
  const waiters: Array<{ type: string; resolve: (f: Record<string, unknown>) => void }> = [];
  let live: WsSocket | null = null;
  let agentName = 'mini';

  wss.on('connection', (socket: WsSocket) => {
    live = socket;
    socket.on('message', (raw) => {
      const frame = JSON.parse(String(raw)) as Record<string, unknown>;
      frames.push(frame);
      for (let i = waiters.length - 1; i >= 0; i -= 1) {
        if (waiters[i].type === frame.type) waiters.splice(i, 1)[0].resolve(frame);
      }
      if (frame.type === 'session.create') {
        agentName = (frame.session as { agent: string }).agent;
        socket.send(
          JSON.stringify({
            type: 'session.created',
            session_id: 'sess_1',
            session: { agent: agentName },
          }),
        );
      }
    });
  });

  return {
    port: (wss.address() as { port: number }).port,
    frames,
    pushTurn() {
      live?.send(
        JSON.stringify({
          type: 'input.text',
          session_id: 'req_stub_1',
          agent: agentName,
          text: 'Hi',
          messages: [
            { role: 'user', content: 'earlier turn' },
            { role: 'assistant', content: 'earlier answer' },
            { role: 'user', content: 'Hi' },
          ],
          payment_token: 'pt_test',
        }),
      );
    },
    waitFor(type, timeoutMs = 8000) {
      const existing = frames.find((f) => f.type === type);
      if (existing) return Promise.resolve(existing);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`timeout waiting for ${type}`)), timeoutMs);
        waiters.push({
          type,
          resolve: (f) => {
            clearTimeout(timer);
            resolve(f);
          },
        });
      });
    },
    close: () =>
      new Promise<void>((resolve) => {
        for (const client of wss.clients) client.terminate();
        wss.close(() => resolve());
      }),
  };
}

const closers: Array<() => Promise<void>> = [];
afterAll(async () => {
  for (const close of closers) await close().catch(() => {});
});

describe('own-url-minimal example', () => {
  it('really listens, and serves completions plus a registration-complete card', async () => {
    const example = await import('../../examples/own-url-minimal.ts');
    closers.push(() => example.server.close());

    (example.agent as unknown as { runStreaming: unknown }).runStreaming = async function* () {
      yield { type: 'delta', delta: 'ok' };
    };

    const base = `http://127.0.0.1:${example.server.port}`;
    expect(example.server.port).toBeGreaterThan(0);

    // Card at the ORIGIN (the platform resolves it origin-relative and
    // DISCARDS the agent path) and under the agent prefix, with
    // metadata.publicKey (SPKI PEM). `serve()` alone must satisfy this:
    // there is no wrapper left to add it.
    for (const p of ['/.well-known/agent.json', '/agents/mini/.well-known/agent.json']) {
      const res = await fetch(`${base}${p}`);
      expect(res.status, p).toBe(200);
      const card = (await res.json()) as {
        name: string;
        url?: string;
        metadata?: { publicKey?: string };
      };
      expect(card.name).toBe('mini');
      expect(card.metadata?.publicKey ?? '').toMatch(/^-----BEGIN PUBLIC KEY-----/);
      // The card must publish the CONFIGURED public URL, not the host the
      // request came in on. This assertion used to be absent, and the card
      // really did say `http://127.0.0.1:<ephemeral>/agents/mini` while
      // WEBAGENTS_PUBLIC_URL was set — registration would have pinned an
      // address nothing outside this process could dial.
      expect(card.url, p).toBe(PUBLIC_URL);
      expect(card.url, p).not.toContain('127.0.0.1');
    }

    const jwks = await fetch(`${base}/.well-known/jwks.json`);
    expect(jwks.status).toBe(200);
    expect(((await jwks.json()) as { keys: unknown[] }).keys.length).toBeGreaterThan(0);

    // The endpoint the platform dials. It runs the model on the owner's
    // credit, so an unauthenticated call is refused — this assertion used to
    // pin the opposite (200 with no Authorization at all), which made every
    // served port an open billable endpoint.
    const anonymous = await fetch(`${base}/agents/mini/chat/completions`, {
      method: 'POST',
      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hi' }], stream: true }),
      headers: { 'content-type': 'application/json' },
    });
    expect(anonymous.status).toBe(401);

    const completions = await fetch(`${base}/agents/mini/chat/completions`, {
      method: 'POST',
      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hi' }], stream: true }),
      headers: { 'content-type': 'application/json', authorization: 'Bearer test-token' },
    });
    expect(completions.status).toBe(200);
    expect(await completions.text()).toContain('ok');
  }, 20000);
});

describe('the agent card publishes the configured public URL', () => {
  /**
   * Separate from the example above on purpose: that one proves the ENV
   * fallback, this one proves `serve({ publicUrl })` actually reaches the
   * card builder. `serve()` used `publicUrl` only as the identity issuer and
   * never threaded it into `createFetchHandler`, so the card was built from
   * the REQUEST HOST — the one address an agent behind a proxy or in a
   * container is guaranteed NOT to be reachable at.
   */
  it('prefers serve({ publicUrl }) over the environment and over the request host', async () => {
    const { BaseAgent, serve } = await import('../../src/index');
    const previous = process.env.WEBAGENTS_PUBLIC_URL;
    process.env.WEBAGENTS_PUBLIC_URL = 'https://from-the-environment.example';
    try {
      const agent = new BaseAgent({ name: 'card', instructions: 'You are helpful.' });
      const server = await serve(agent, {
        port: 0,
        basePath: '/agents/card',
        publicUrl: 'https://configured.example.com/',
      });
      closers.push(() => server.close());

      const res = await fetch(`http://127.0.0.1:${server.port}/.well-known/agent.json`);
      const card = (await res.json()) as { url?: string };
      // Explicit config wins, with the trailing slash normalised away.
      expect(card.url).toBe('https://configured.example.com');
    } finally {
      process.env.WEBAGENTS_PUBLIC_URL = previous;
    }
  }, 20000);

  /**
   * The last tier, and the one the two SDKs used to disagree about: TS
   * answered the REQUEST ORIGIN (`http://10.0.0.9:7777/agents/card`) where
   * Python answered a relative `/{agent_name}`. They now agree on relative.
   *
   * The request origin comes from the Host header, so behind a proxy, a
   * tunnel or a container it is a wrong address published as fact — exactly
   * the failure `publicUrl` exists to prevent, reintroduced by the fallback.
   * A relative reference resolves against whatever origin the consumer
   * actually fetched the card from, which is by construction reachable.
   *
   * No consumer reads `card.url`. Precisely: the `AgentMetadata` interface
   * (portal `lib/auth/agent-auth.ts:71`) carries an index signature
   * `[key: string]: unknown` at line 82, so `url` IS carried through it —
   * nothing DEREFERENCES it. The only `metadata.` reads in that file are
   * `capabilities` (272, 344) and `publicKey` (485, 509), and the callable
   * address a registration is keyed on is
   * `composeAgentRegistrationUrl(iss, agent_path, sub)` (186), from the
   * agent's own signed token.
   */
  it('falls back to a RELATIVE basePath when nothing is configured, like Python', async () => {
    const { BaseAgent } = await import('../../src/index');
    const { createFetchHandler } = await import('../../src/server/handler');
    const previous = process.env.WEBAGENTS_PUBLIC_URL;
    delete process.env.WEBAGENTS_PUBLIC_URL;
    try {
      const agent = new BaseAgent({ name: 'card', instructions: 'You are helpful.' });
      const handler = createFetchHandler(agent, { basePath: '/agents/card' });
      const res = await handler(new Request('http://10.0.0.9:7777/.well-known/agent.json'));
      const card = (await res.json()) as { url?: string };
      expect(card.url).toBe('/agents/card');
      // The Host header must not leak into the published address at all.
      expect(card.url).not.toContain('10.0.0.9');

      // Resolving it the way any consumer would gets the right absolute URL,
      // from the origin it really reached the agent at.
      expect(new URL(card.url as string, 'https://proxy.example.com/.well-known/agent.json').toString())
        .toBe('https://proxy.example.com/agents/card');
    } finally {
      if (previous === undefined) delete process.env.WEBAGENTS_PUBLIC_URL;
      else process.env.WEBAGENTS_PUBLIC_URL = previous;
    }
  });

  /**
   * The whole precedence table, asserted as one thing so the two SDKs can be
   * compared line by line. The Python twin is
   * python/tests/server/test_card_url_resolution.py, which drives
   * `resolve_public_base_url` with the same cases and expects the same
   * answers.
   *
   * The whitespace rows are the ones that used to differ: TS trimmed a
   * whitespace-only configured value, Python published it verbatim. Note that
   * a whitespace-only `publicUrl` SUPPRESSES the environment variable rather
   * than falling through to it — an explicit (if useless) argument still beats
   * the environment, and both SDKs now agree on that too.
   */
  it('resolves the card url on the same precedence table as Python', async () => {
    const { BaseAgent } = await import('../../src/index');
    const { createFetchHandler } = await import('../../src/server/handler');
    const previous = process.env.WEBAGENTS_PUBLIC_URL;

    const cardUrl = async (publicUrl: string | undefined, env: string | undefined) => {
      if (env === undefined) delete process.env.WEBAGENTS_PUBLIC_URL;
      else process.env.WEBAGENTS_PUBLIC_URL = env;
      const agent = new BaseAgent({ name: 'card', instructions: 'You are helpful.' });
      const handler = createFetchHandler(agent, { basePath: '/card', ...(publicUrl === undefined ? {} : { publicUrl }) });
      const res = await handler(new Request('http://10.0.0.9:7777/.well-known/agent.json'));
      return ((await res.json()) as { url?: string }).url;
    };

    try {
      // configured, env, expected
      const table: Array<[string | undefined, string | undefined, string]> = [
        ['https://configured.example', 'https://env.example', 'https://configured.example'],
        ['https://configured.example/', undefined, 'https://configured.example'],
        ['https://configured.example///', undefined, 'https://configured.example'],
        [undefined, 'https://env.example/', 'https://env.example'],
        [undefined, undefined, '/card'],
        ['', 'https://env.example', 'https://env.example'],
        // Whitespace-only: trimmed to nothing, and it does NOT fall through
        // to the environment.
        ['   ', 'https://env.example', '/card'],
        ['   ', undefined, '/card'],
        [undefined, '   ', '/card'],
      ];
      for (const [configured, env, expected] of table) {
        expect(await cardUrl(configured, env), JSON.stringify([configured, env])).toBe(expected);
      }
    } finally {
      if (previous === undefined) delete process.env.WEBAGENTS_PUBLIC_URL;
      else process.env.WEBAGENTS_PUBLIC_URL = previous;
    }
  });
});

describe('portal-connect-minimal example', () => {
  it('serves a full turn over the bridge that serve() opened', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());

    // Set BEFORE the import: the example's skill reads the environment at
    // construction, which is the whole point of it having no wrapper.
    process.env.WEBAGENTS_PORTAL_URL = `ws://127.0.0.1:${portal.port}/ws`;
    process.env.WEBAGENTS_AGENT_TOKEN = fakeAgentToken();

    const seen: unknown[][] = [];
    const example = await import('../../examples/portal-connect-minimal.ts');
    closers.push(() => example.server.close());
    (example.agent as unknown as { runStreaming: unknown }).runStreaming = async function* (
      messages: unknown[],
    ) {
      seen.push(messages);
      yield { choices: [{ delta: { content: 'hello from mini' } }] };
    };

    const created = await portal.waitFor('session.create');
    expect((created.session as { agent: string }).agent).toBe('mini');
    expect((created.session as { token: string }).token).toBe(process.env.WEBAGENTS_AGENT_TOKEN);

    portal.pushTurn();
    const delta = await portal.waitFor('response.delta');
    expect((delta.delta as { text: string }).text).toBe('hello from mini');
    const done = await portal.waitFor('response.done');
    // The PER-REQUEST id, resolved through the frame's `agent` field.
    expect(done.session_id).toBe('req_stub_1');

    // Full history reaches the run — not a synthesized single turn.
    expect(seen[0]).toHaveLength(3);
  }, 20000);
});

describe('portal-connect-socket-only example', () => {
  it('opens the bridge with no HTTP server at all', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());

    process.env.WEBAGENTS_PORTAL_URL = `ws://127.0.0.1:${portal.port}/ws`;
    process.env.WEBAGENTS_AGENT_TOKEN = fakeAgentToken();

    const seen: unknown[][] = [];
    const example = await import('../../examples/portal-connect-socket-only.ts');
    closers.push(() => example.portal.stop());
    (example.agent as unknown as { runStreaming: unknown }).runStreaming = async function* (
      messages: unknown[],
    ) {
      seen.push(messages);
      yield { choices: [{ delta: { content: 'hello from mini' } }] };
    };

    // `main()` never resolves by design; start it and let it run.
    void example.main();

    await portal.waitFor('session.create');
    portal.pushTurn();
    const delta = await portal.waitFor('response.delta');
    expect((delta.delta as { text: string }).text).toBe('hello from mini');
    await portal.waitFor('response.done');
    expect(seen[0]).toHaveLength(3);
  }, 20000);
});

describe('the credential guard lives on the skill, not on a wrapper', () => {
  it('refuses an owner key with no agent binding, at start', async () => {
    const { checkAgentToken, PortalCredentialError } = await import('../../src/portal/connect');
    expect(() => checkAgentToken(fakeAgentToken(false))).toThrow(PortalCredentialError);
    expect(() => checkAgentToken(fakeAgentToken(false))).toThrow(/api-key/);
    expect(() => checkAgentToken(undefined)).toThrow(PortalCredentialError);
    expect(() => checkAgentToken(fakeAgentToken(true))).not.toThrow();
  });

  it('PortalConnectSkill.start() refuses before opening a socket', async () => {
    const { PortalConnectSkill } = await import(
      '../../src/skills/transport/portal-connect/skill'
    );
    const { PortalCredentialError } = await import('../../src/portal/connect');
    const skill = new PortalConnectSkill({
      portalUrl: 'ws://127.0.0.1:1/ws',
      token: fakeAgentToken(false),
    });
    skill.setAgent({ name: 'mini' } as never);
    await expect(skill.start()).rejects.toBeInstanceOf(PortalCredentialError);
    expect(skill.isStarted).toBe(false);
  });
});

describe('doc snippets are generated from these files', () => {
  function codeAfterHeader(file: string): string {
    const text = readFileSync(path.join(EXAMPLES, file), 'utf8');
    const m = text.match(/^\s*\/\*\*[\s\S]*?\*\/\s*\n/);
    return (m ? text.slice(m[0].length) : text).trim();
  }

  it('portal-connect.md carries the example files verbatim', () => {
    const doc = readFileSync(path.join(DOCS, 'skills/platform/portal-connect.md'), 'utf8');
    expect(doc).toContain(codeAfterHeader('portal-connect-minimal.ts'));
  });

  it('quickstart.md carries the own-url example verbatim', () => {
    const doc = readFileSync(path.join(DOCS, 'quickstart.md'), 'utf8');
    expect(doc).toContain(codeAfterHeader('own-url-minimal.ts'));
  });

  /**
   * The predecessor of this check read two files and looked for two exact
   * import strings. That is not a guard: it missed `.mdx`, every call shape,
   * every `webagents.portal` reference, and — because it never looked at
   * shipped source — a runtime `console.warn` in src/portal/connect.ts and a
   * stale comment in src/index.ts that both named the deleted API.
   *
   * The scan now covers docs (`.md` AND `.mdx`), `python/webagents/**`,
   * `typescript/src/**` and both examples trees. Patterns and the
   * exact-line allowlist for deliberate narrative live in
   * `scripts/removed-api-guard.json`, shared with the Python half
   * (python/tests/docs/test_doc_examples.py).
   */
  it('nothing shipped still names the deleted connect()/host() wrappers', () => {
    const config = JSON.parse(readFileSync(path.join(REPO, 'scripts/removed-api-guard.json'), 'utf8')) as {
      patterns: string[];
      roots: Array<{ dir: string; ext: string[] }>;
      files?: string[];
      allowlist: Record<string, string[]>;
    };
    const patterns = config.patterns.map((p) => new RegExp(p));

    const walk = (dir: string, exts: string[], out: string[]): string[] => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full, exts, out);
        else if (exts.includes(path.extname(entry.name))) out.push(full);
      }
      return out;
    };

    const scanned: string[] = [];
    for (const root of config.roots) {
      scanned.push(...walk(path.join(REPO, root.dir), root.ext, []).sort());
    }
    // Individually named files: the package READMEs and the other top-level
    // markdown cannot be expressed as roots (walking `python/` would descend
    // into .venv, walking the repo root into node_modules).
    for (const name of config.files ?? []) {
      scanned.push(path.join(REPO, name));
    }

    const offenders: string[] = [];
    for (const file of scanned) {
      const rel = path.relative(REPO, file).split(path.sep).join('/');
      const exempt = config.allowlist[rel] ?? [];
      readFileSync(file, 'utf8')
        .split('\n')
        .forEach((line, i) => {
          const stripped = line.trim();
          if (exempt.includes(stripped)) return;
          if (patterns.some((p) => p.test(line))) offenders.push(`${rel}:${i + 1}: ${stripped}`);
        });
    }
    expect(offenders, 'stale references to the deleted connect()/host() API').toEqual([]);
  });

  it('the guard actually fires — a planted call shape is caught', () => {
    const config = JSON.parse(readFileSync(path.join(REPO, 'scripts/removed-api-guard.json'), 'utf8')) as {
      patterns: string[];
    };
    const patterns = config.patterns.map((p) => new RegExp(p));
    for (const planted of [
      'const s = await connect(agent);',
      'await host(agent, { port: 3000 });',
      'import webagents.portal',
      "import { BaseAgent, connect } from 'webagents';",
      '`webagents.host()` serves the card',
    ]) {
      expect(patterns.some((p) => p.test(planted)), planted).toBe(true);
    }
    // ...and does not fire on the shapes that merely look like it.
    for (const benign of [
      '  async connect(): Promise<void> {',
      '  def connect(',
      '    await ws.connect(url);',
      '    const h = req.headers.host();',
    ]) {
      expect(patterns.some((p) => p.test(benign)), benign).toBe(false);
    }
  });

  /**
   * The guard's blind spots are the interesting part of it. Its roots
   * originally covered docs and shipped source only, leaving out the three
   * package READMEs — the most-read documents here, and the ones that ship
   * inside the npm and PyPI packages — CONTRIBUTING, RELEASE, CHANGELOG, and
   * BOTH test trees. A stale reference to the deleted wrapper, in a test
   * docstring, survived precisely because of that gap. Pin the coverage
   * rather than trusting the scan to have been pointed at the right places.
   */
  it('the guard looks where readers look', () => {
    const config = JSON.parse(readFileSync(path.join(REPO, 'scripts/removed-api-guard.json'), 'utf8')) as {
      roots: Array<{ dir: string; ext: string[] }>;
      files?: string[];
    };
    const covered = new Set([...config.roots.map((r) => r.dir), ...(config.files ?? [])]);
    for (const required of [
      'README.md',
      'python/README.md',
      'typescript/README.md',
      'CONTRIBUTING.md',
      'RELEASE.md',
      'CHANGELOG.md',
      'python/tests',
      'typescript/tests',
      'docs',
      'python/webagents',
      'typescript/src',
    ]) {
      expect(covered.has(required), `removed-api guard no longer covers ${required}`).toBe(true);
    }
    // A path that does not exist is a spelling that scans nothing.
    for (const root of config.roots) {
      expect(statSync(path.join(REPO, root.dir)).isDirectory(), root.dir).toBe(true);
    }
    for (const name of config.files ?? []) {
      expect(statSync(path.join(REPO, name)).isFile(), name).toBe(true);
    }
  });
});
