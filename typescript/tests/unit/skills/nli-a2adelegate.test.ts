/**
 * `delegate` reaches a peer over A2A v1.0 (2026-09-27, the a2a-delegate lane):
 * the route decided by `skills/nli/a2a-target.ts` from the shared fixture
 * (`python/tests/fixtures/a2a/delegate_routing.json`), a configured peer
 * called on loopback with its bearer and nothing of the caller's, an https
 * URL probed for a card, a card verified when its signature names a `jku`,
 * a redirect refused, and the platform path left as it was. The Python twin
 * is `python/tests/test_nli_a2adelegate.py`.
 */

import { afterAll, afterEach, describe, expect, it, vi } from 'vitest';
import { createServer, type Server } from 'node:http';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff } from '../../../src/core/decorators';
import type { Context, StructuredToolResult } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';
import { AgentIdentity } from '../../../src/crypto/identity';
import { A2ATransportSkill } from '../../../src/skills/transport/a2a/skill';
import { identityCardSigner, signAgentCard } from '../../../src/skills/transport/a2a/card';
import { NLISkill } from '../../../src/skills/nli/skill';
import {
  A2A_ATTACHMENTS_REFUSED,
  A2A_EMPTY_REPLY,
  A2A_FAILED,
  A2A_PROBE_TIMEOUT_MS,
  A2A_UNPAID_NOTE,
  classifyDelegateTarget,
  isLoopbackUrl,
} from '../../../src/skills/nli/a2a-target';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/a2a/delegate_routing.json'), 'utf8')) as {
  platform_base: string;
  peers: Record<string, { token?: string }>;
  cases: Array<{ name: string; target: string; expect: Record<string, unknown> }>;
  probe: { card_paths: string[]; timeout_seconds: number };
  verification: { refused: string };
  never_sent_headers: string[];
  unpaid_note: string;
  a2a_failed: string;
  attachments_refused: string;
  result: { empty_reply: string; data_keys: string[] };
};

// ---------------------------------------------------------------------------
// A peer agent, served on loopback through the SDK's own fetch handler
// ---------------------------------------------------------------------------

class EchoLLM extends Skill {
  reply = (texts: string[]) => `echo: ${texts.join(' ')}`;

  @handoff({ name: 'echo-llm' })
  async *processUAMP(events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text' && (e as { text: string }).text) texts.push((e as { text: string }).text);
    }
    const reply = this.reply(texts);
    yield createResponseDeltaEvent('r1', { type: 'text', text: reply });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: reply }]);
  }
}

interface Received {
  method: string;
  url: string;
  headers: Record<string, string>;
}

type Handler = (request: Request) => Promise<Response>;

const servers: Server[] = [];
afterAll(() => {
  for (const server of servers) server.close();
});

/** A loopback HTTP server that records every request's headers and answers through `handler`, set after the port is known. */
async function loopbackServer(): Promise<{ origin: string; received: Received[]; setHandler: (handler: Handler) => void }> {
  const received: Received[] = [];
  let handler: Handler = async () => new Response('not ready', { status: 503 });
  const server = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const headers = new Headers();
    for (const [name, value] of Object.entries(req.headers)) {
      if (value !== undefined) headers.set(name, Array.isArray(value) ? value.join(', ') : value);
    }
    received.push({ method: req.method ?? '', url: req.url ?? '', headers: Object.fromEntries(headers.entries()) });
    const method = req.method ?? 'GET';
    const request = new Request(`http://${req.headers.host}${req.url}`, {
      method,
      headers,
      ...(method === 'GET' || method === 'HEAD' ? {} : { body: new Uint8Array(Buffer.concat(chunks)) }),
    });
    const response = await handler(request);
    res.statusCode = response.status;
    response.headers.forEach((value, name) => res.setHeader(name, value));
    res.end(Buffer.from(await response.arrayBuffer()));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  servers.push(server);
  const { port } = server.address() as { port: number };
  return { origin: `http://127.0.0.1:${port}`, received, setHandler: (next) => { handler = next; } };
}

const PEER_PATH = '/agents/peer';

/** The peer agent behind `origin`, signed with an identity at its own URL when `signed`. */
async function peerAgent(origin: string, options: { signed?: boolean; peers?: Record<string, { token?: string }> } = {}) {
  const llm = new EchoLLM();
  const agent = new BaseAgent({ name: 'peer', description: 'A2A peer for the delegate test', skills: [llm, new A2ATransportSkill({ peers: options.peers ?? {} })] });
  let identity: AgentIdentity | undefined;
  if (options.signed !== false) {
    identity = new AgentIdentity({ agentId: 'peer', issuer: `${origin}${PEER_PATH}` });
    await identity.initialize();
    agent.identity = identity;
  }
  const handler = createFetchHandler(agent, { basePath: PEER_PATH, publicUrl: origin, identity });
  return { agent, llm, identity, handler };
}

/** The calling agent: an `a2a` skill with `peers` beside the NLI skill, so `delegate` reads them. */
function caller(peers: Record<string, { token?: string }>, nliConfig: Record<string, unknown> = {}) {
  const nli = new NLISkill({ baseUrl: FIXTURE.platform_base, transport: 'http', apiKey: 'platform-key-never-forwarded', ...nliConfig });
  const agent = new BaseAgent({ name: 'caller', skills: [new A2ATransportSkill({ peers }), nli] });
  return { agent, nli };
}

/** A run context carrying everything a peer must never see. */
function makeContext(overrides: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(overrides));
  return {
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: (key: string, value: unknown) => {
      store.set(key, value);
    },
    delete: (key: string) => {
      store.delete(key);
    },
    signal: new AbortController().signal,
    auth: { authenticated: true, user_id: 'user-1' },
    payment: { token: 'parent-payment-token', agentToken: 'agent-scoped-token', payerId: 'payer-1' },
    metadata: { authToken: 'caller-auth-token', apiKey: 'metadata-api-key', chatId: 'chat-1' },
  } as unknown as Context;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// The words and the routing, shared with Python
// ---------------------------------------------------------------------------

describe('the words shared with Python', () => {
  it('the sentences are the fixture’s', () => {
    expect(A2A_UNPAID_NOTE).toBe(FIXTURE.unpaid_note);
    expect(A2A_FAILED).toBe(FIXTURE.a2a_failed);
    expect(A2A_ATTACHMENTS_REFUSED).toBe(FIXTURE.attachments_refused);
    expect(A2A_EMPTY_REPLY).toBe(FIXTURE.result.empty_reply);
    expect(A2A_PROBE_TIMEOUT_MS).toBe(FIXTURE.probe.timeout_seconds * 1000);
  });

  it.each(FIXTURE.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    expect(classifyDelegateTarget(c.target, { peers: FIXTURE.peers, platformBase: FIXTURE.platform_base })).toEqual(c.expect);
  });

  it('knows a loopback URL', () => {
    for (const url of ['http://127.0.0.1:8765/agents/x', 'http://localhost/x', 'https://[::1]:9/x', 'http://a.localhost/x']) expect(isLoopbackUrl(url)).toBe(true);
    for (const url of ['https://peer.example/x', 'http://10.0.0.1/x', 'not a url']) expect(isLoopbackUrl(url)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// A configured peer, on loopback
// ---------------------------------------------------------------------------

describe('delegate to a configured a2a peer', () => {
  it('calls the peer over A2A with its bearer and nothing of the caller’s, verifies the signed card, and says nobody pays', async () => {
    const hop = await loopbackServer();
    const peer = await peerAgent(hop.origin);
    hop.setHandler(peer.handler);
    const peerUrl = `${hop.origin}${PEER_PATH}`;
    const { nli } = caller({ [peerUrl]: { token: 'peer-token' } });

    const result = (await nli.delegate({ agent: `${peerUrl}/`, message: 'hello peer', budget: 0.5, chat_id: 'ctx-1' }, makeContext())) as StructuredToolResult;
    expect(typeof result).toBe('object');
    expect(result.text).toBe(`echo: hello peer\n${FIXTURE.unpaid_note.replace('{url}', peerUrl)}`);
    const a2a = (result.data as { a2a: Record<string, unknown> }).a2a;
    expect(Object.keys(a2a)).toEqual(FIXTURE.result.data_keys);
    expect(a2a).toMatchObject({ url: peerUrl, rpcUrl: `${peerUrl}/a2a`, state: 'TASK_STATE_COMPLETED', verified: true });
    expect(typeof a2a.taskId).toBe('string');

    // The card first, then the send, then the key set for the card's jku; every request carries only the peer's bearer.
    const paths = hop.received.map((r) => `${r.method} ${r.url}`);
    expect(paths[0]).toBe(`GET ${PEER_PATH}${FIXTURE.probe.card_paths[0]}`);
    expect(paths).toContain(`POST ${PEER_PATH}/a2a`);
    expect(paths).toContain(`GET ${PEER_PATH}/.well-known/jwks.json`);
    for (const request of hop.received) {
      for (const name of FIXTURE.never_sent_headers) expect(request.headers[name], `${request.url} carried ${name}`).toBeUndefined();
      const values = Object.values(request.headers).join('\n');
      for (const secret of ['parent-payment-token', 'agent-scoped-token', 'caller-auth-token', 'metadata-api-key', 'platform-key-never-forwarded']) {
        expect(values, `${request.url} carried ${secret}`).not.toContain(secret);
      }
    }
    const send = hop.received.find((r) => r.method === 'POST')!;
    expect(send.headers.authorization).toBe('Bearer peer-token');
    expect(send.headers['a2a-version']).toBe('1.0');
    await peer.agent.cleanup();
  });

  it('an unsigned card is called unverified, and an empty reply reads as (no response)', async () => {
    const hop = await loopbackServer();
    const peer = await peerAgent(hop.origin, { signed: false });
    peer.llm.reply = () => '';
    hop.setHandler(peer.handler);
    const peerUrl = `${hop.origin}${PEER_PATH}`;
    const { nli } = caller({ [peerUrl]: { token: 'peer-token' } });
    const result = (await nli.delegate({ agent: peerUrl, message: 'anything' }, makeContext())) as StructuredToolResult;
    expect(result.text).toBe(`${FIXTURE.result.empty_reply}\n${FIXTURE.unpaid_note.replace('{url}', peerUrl)}`);
    expect((result.data as { a2a: { verified: unknown } }).a2a.verified).toBeNull();
    await peer.agent.cleanup();
  });

  it('refuses attachments for a peer, a card whose jku does not verify, and a redirect', async () => {
    const hop = await loopbackServer();
    const peer = await peerAgent(hop.origin);
    hop.setHandler(peer.handler);
    const peerUrl = `${hop.origin}${PEER_PATH}`;
    const { nli } = caller({ [peerUrl]: { token: 'peer-token' } });

    expect(await nli.delegate({ agent: peerUrl, message: 'with a file', attachments: ['11111111-1111-1111-1111-111111111111'] }, makeContext())).toBe(
      `Error: ${FIXTURE.attachments_refused.replace('{url}', peerUrl)}`,
    );
    expect(hop.received).toHaveLength(0);

    // The card is signed by a key the peer's key set does not publish.
    const stranger = new AgentIdentity({ agentId: 'stranger', issuer: `${hop.origin}${PEER_PATH}` });
    await stranger.initialize();
    hop.setHandler(async (request) => {
      const url = new URL(request.url);
      if (url.pathname === `${PEER_PATH}${FIXTURE.probe.card_paths[0]}`) {
        const card = await (await peer.handler(request)).json();
        delete (card as Record<string, unknown>).signatures;
        return Response.json(await signAgentCard(card as Record<string, unknown>, identityCardSigner(stranger)));
      }
      return peer.handler(request);
    });
    const refused = await nli.delegate({ agent: peerUrl, message: 'hello' }, makeContext());
    expect(refused).toContain(`Error: ${FIXTURE.a2a_failed.replace('{url}', peerUrl).replace('{reason}', '')}`);
    expect(refused).toContain(FIXTURE.verification.refused.replace('{card_url}', `${peerUrl}${FIXTURE.probe.card_paths[0]}`).replace('{reason}', ''));
    expect(refused).toContain('does not verify');
    expect(hop.received.some((r) => r.method === 'POST')).toBe(false);

    hop.setHandler(async () => new Response(null, { status: 302, headers: { location: 'https://elsewhere.example/' } }));
    const moved = await nli.delegate({ agent: peerUrl, message: 'hello' }, makeContext());
    expect(moved).toContain(`Error: ${FIXTURE.a2a_failed.replace('{url}', peerUrl).replace('{reason}', '')}`);
    expect(moved).toContain('redirect');
    await peer.agent.cleanup();
  });
});

// ---------------------------------------------------------------------------
// An https URL: probed for a card, and left on the platform path without one
// ---------------------------------------------------------------------------

describe('delegate to an https URL', () => {
  it('goes over A2A when the URL serves a v1.0 card, with no bearer', async () => {
    const origin = 'https://remote.example';
    const remote = new BaseAgent({ name: 'remote', skills: [new EchoLLM(), new A2ATransportSkill({})] });
    const identity = new AgentIdentity({ agentId: 'remote', issuer: `${origin}/agents/remote` });
    await identity.initialize();
    remote.identity = identity;
    const handler = createFetchHandler(remote, { basePath: '/agents/remote', publicUrl: origin, identity });
    const sent: Array<{ url: string; headers: Record<string, string> }> = [];
    vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      sent.push({ url: request.url, headers: Object.fromEntries(request.headers.entries()) });
      // The peer requires a credential on its billable route (the floor); a probed URL gets none from us, so the peer here is open.
      return handler(request);
    });
    const { nli } = caller({});
    const result = (await nli.delegate({ agent: `${origin}/agents/remote`, message: 'hi remote' }, makeContext())) as StructuredToolResult | string;
    // The floor refuses an anonymous POST to /a2a, which is the honest answer for a peer that demands a credential.
    expect(typeof result).toBe('string');
    expect(result).toContain(FIXTURE.a2a_failed.replace('{url}', `${origin}/agents/remote`).replace('{reason}', ''));
    expect(sent[0].url).toBe(`${origin}/agents/remote${FIXTURE.probe.card_paths[0]}`);
    expect(sent.every((s) => s.headers.authorization === undefined)).toBe(true);
    for (const s of sent) for (const name of FIXTURE.never_sent_headers) expect(s.headers[name]).toBeUndefined();
    await remote.cleanup();
  });

  it('stays on the platform path when the URL serves no card', async () => {
    const asked: string[] = [];
    vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      asked.push(request.url);
      return new Response('nope', { status: 404 });
    });
    const { nli } = caller({});
    const streamMock = vi.fn().mockImplementation(async function* () {
      yield 'from the platform path';
    });
    (nli as unknown as { streamMessage: unknown }).streamMessage = streamMock;
    // No parent token: the platform path mints nothing, so the only fetches are the probe's.
    const ctx = makeContext({ _agentic_messages: [] });
    (ctx as unknown as { payment: unknown }).payment = undefined;
    const result = await nli.delegate({ agent: 'https://nocard.example/agents/x', message: 'hi' }, ctx);
    expect(asked).toEqual(FIXTURE.probe.card_paths.map((p) => `https://nocard.example/agents/x${p}`));
    expect(streamMock).toHaveBeenCalledTimes(1);
    expect(streamMock.mock.calls[0][0]).toBe('https://nocard.example/agents/x');
    expect(typeof result === 'string' ? result : (result as StructuredToolResult).text).toContain('from the platform path');
  });

  it('never probes a platform agent or an @name', async () => {
    // The platform path's own calls (the child token from the delegate route) still go out; no card is ever asked for.
    const asked: string[] = [];
    vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
      asked.push(new Request(input, init).url);
      return Response.json({ token: 'child.jwt.token', tokenId: 'child-1', amountCredits: 0.1 });
    });
    // No parent token in these contexts: nothing is minted, so nothing at all should be fetched.
    const bare = () => {
      const ctx = makeContext({ _agentic_messages: [] });
      (ctx as unknown as { payment: unknown }).payment = undefined;
      return ctx;
    };
    const { nli } = caller({});
    const streamMock = vi.fn().mockImplementation(async function* () {
      yield 'ok';
    });
    (nli as unknown as { streamMessage: unknown }).streamMessage = streamMock;
    await nli.delegate({ agent: '@bob', message: 'hi' }, bare());
    await nli.delegate({ agent: `${FIXTURE.platform_base}/agents/bob`, message: 'hi' }, bare());
    expect(asked.filter((url) => url.includes('/.well-known/'))).toEqual([]);
    expect(streamMock).toHaveBeenCalledTimes(2);
  });
});
