/**
 * PortalDiscoverySkill signs with the agent's identity and needs no platform
 * key (2026-09-23).
 *
 * The platform's `authenticateAgentRequest` takes an RFC 9421 signed request
 * before it looks for a bearer, on every route this skill calls. Until this
 * change the skill only ever sent `Authorization: Bearer <WEBAGENTS_API_KEY>`,
 * so an agent that already held a signing identity was told to obtain a key
 * for a call the platform would have accepted signed. Pinned here, in order:
 *
 *   1. an identity signs, and no bearer rides beside the signature;
 *   2. the identity `serve()` / `WebAgentsServer.addAgent()` hands the agent
 *      is the one the skill reads;
 *   3. a configured key is still presented, and still works, when there is
 *      no identity, or when the identity cannot sign (loopback);
 *   4. neither is refused up front with a sentence that names both fixes.
 *
 * The signatures are verified here the way the platform verifies them:
 * rebuild the RFC 9421 base from the request AS SENT and check it under the
 * public key the identity publishes, so a regression in what is signed (not
 * merely whether headers are present) cannot pass.
 */

import { describe, it, expect, vi, beforeAll, beforeEach, afterEach, afterAll } from 'vitest';
import { createPublicKey, verify as cryptoVerify } from 'node:crypto';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { PortalDiscoverySkill, NO_DISCOVERY_CREDENTIAL, NO_DISCOVERY_SIGN_IN } from '../../../../src/skills/discovery/skill';
import { AgentIdentity } from '../../../../src/crypto/identity';
import type { SigningIdentity } from '../../../../src/crypto/http-signature';
import { BaseAgent } from '../../../../src/core/agent';

const PORTAL = 'https://platform.example.com';
const ISSUER = 'https://agent.example.com/agents/finder';
const KEY_SET = `${ISSUER}/.well-known/jwks.json`;

let identity: AgentIdentity;
beforeAll(async () => {
  identity = new AgentIdentity({ agentId: 'finder', issuer: ISSUER });
  await identity.initialize();
});

/** One request as the stub platform saw it. */
interface Sent {
  url: string;
  method: string;
  headers: Headers;
  body: string;
}

let sent: Sent[] = [];
const savedKey = process.env.WEBAGENTS_API_KEY;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

beforeEach(() => {
  sent = [];
  delete process.env.WEBAGENTS_API_KEY;
  // `signedFetch` calls fetch with a `Request`; the bearer path calls it with
  // a URL and an init. Normalise both to the request the platform would read.
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(input, init);
    sent.push({
      url: request.url,
      method: request.method,
      headers: new Headers(request.headers),
      body: request.body ? await request.text() : '',
    });
    const route = new URL(request.url).pathname;
    if (route === '/api/intents/search') {
      return json({ results: [{ id: 'i-1', intent: 'translate documents', agentId: 'a-1', similarity: 0.91 }] });
    }
    if (route === '/api/intents/create') return json({ results: [{ id: 'i-1' }], count: 1 }, 201);
    return json({ error: 'not found' }, 404);
  });
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  if (savedKey === undefined) delete process.env.WEBAGENTS_API_KEY;
  else process.env.WEBAGENTS_API_KEY = savedKey;
});

/** The covered component names of `sig1`, in order, from a `Signature-Input` value. */
function coveredComponents(signatureInput: string): string[] {
  const list = /^sig1=\(([^)]*)\)/.exec(signatureInput);
  if (!list) throw new Error(`no sig1 inner list in ${signatureInput}`);
  return [...list[1].matchAll(/"([^"]+)"(?:;key="[^"]*")?/g)].map((m) => m[1]);
}

/** Rebuild the RFC 9421 base from what was SENT, as the platform would. */
function rebuildBase(request: Sent, keySetUrl: string): string {
  const input = request.headers.get('signature-input')!;
  const params = input.slice(input.indexOf('=') + 1);
  const url = new URL(request.url);
  const lines: string[] = [];
  for (const name of coveredComponents(input)) {
    if (name === '@method') lines.push(`"@method": ${request.method}`);
    else if (name === '@authority') lines.push(`"@authority": ${url.host}`);
    else if (name === '@path') lines.push(`"@path": ${url.pathname}`);
    else if (name === '@query') lines.push(`"@query": ${url.search || '?'}`);
    else if (name === 'content-digest') lines.push(`"content-digest": ${request.headers.get('content-digest')}`);
    else if (name === 'signature-agent') lines.push(`"signature-agent";key="sig1": "${keySetUrl}";type=jwks_uri`);
    else lines.push(`"${name}": ${request.headers.get(name)!.trim()}`);
  }
  lines.push(`"@signature-params": ${params}`);
  return lines.join('\n');
}

/** True when `sig1` verifies under the key `signer` publishes for the base rebuilt from the request. */
function verifies(request: Sent, signer: AgentIdentity, keySetUrl = KEY_SET): boolean {
  const m = /^sig1=:([A-Za-z0-9+/=]+):$/.exec(request.headers.get('signature')!);
  if (!m) return false;
  const key = createPublicKey({ key: signer.getJwks().keys[0] as never, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(rebuildBase(request, keySetUrl), 'ascii'), key, Buffer.from(m[1], 'base64'));
}

function expectSigned(request: Sent, signer: AgentIdentity, keySetUrl = KEY_SET): void {
  expect(request.headers.get('authorization')).toBeNull();
  expect(request.headers.get('signature-agent')).toBe(`sig1="${keySetUrl}";type=jwks_uri`);
  expect(request.headers.get('signature-input')).toMatch(
    /^sig1=\("@method" "@authority" "@path" "@query"( "content-digest")? "signature-agent";key="sig1"\);created=\d+;expires=\d+;keyid="[A-Za-z0-9_-]{43}";alg="ed25519";nonce="[A-Za-z0-9+/]{86}==";tag="web-bot-auth"$/,
  );
  expect(/keyid="([^"]+)"/.exec(request.headers.get('signature-input')!)![1]).toBe(signer.kid);
  expect(verifies(request, signer, keySetUrl)).toBe(true);
}

function expectBearer(request: Sent, key: string): void {
  expect(request.headers.get('authorization')).toBe(`Bearer ${key}`);
  expect(request.headers.get('signature-input')).toBeNull();
  expect(request.headers.get('signature')).toBeNull();
  expect(request.headers.get('signature-agent')).toBeNull();
}

const intentsSearch = () => sent.find((r) => new URL(r.url).pathname === '/api/intents/search');

describe('PortalDiscoverySkill credential: an identity signs', () => {
  it('signs the search with the identity it is given, sends no bearer, and the platform can verify it', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, identity });
    expect(skill.credential()).toEqual({ kind: 'signature', identity });

    const result = await skill.search({ query: 'translate a contract into German', types: ['intents'] });

    const request = intentsSearch()!;
    expect(request).toBeDefined();
    expect(request.method).toBe('POST');
    expect(request.headers.get('content-digest')).toMatch(/^sha-256=:[A-Za-z0-9+/]+=*:$/);
    expect(JSON.parse(request.body)).toEqual({ query: 'translate a contract into German', limit: 10 });
    expectSigned(request, identity);
    expect(result.intents).toEqual([{ id: 'i-1', intent: 'translate documents', agentId: 'a-1', similarity: 0.91 }]);
  });

  it('reads the identity the host left on the agent, so a served agent needs no configuration', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL });
    const agent = new BaseAgent({ name: 'finder', skills: [skill] });
    expect(skill.credential().kind).toBe('none');

    // What `serve()` does before `initialize()` (typescript/src/server/node.ts).
    agent.identity = identity;
    expect(skill.credential()).toEqual({ kind: 'signature', identity });

    await skill.search({ query: 'translate a contract', types: ['intents'] });
    expectSigned(intentsSearch()!, identity);
  });

  it('signs rather than sending the key when it has both', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, identity, apiKey: 'rok_configured' });
    expect(skill.credential().kind).toBe('signature');

    await skill.search({ query: 'translate', types: ['intents'] });
    const request = intentsSearch()!;
    expectSigned(request, identity);
    expect(request.headers.get('authorization')).toBeNull();
  });

  it('publishIntents() signs the create call and reports the platform status', async () => {
    const skill = new PortalDiscoverySkill({
      portalUrl: PORTAL,
      identity,
      intents: ['translate documents between English and German'],
      description: 'Translates English and German text.',
    });

    const result = await skill.publishIntents();
    expect(result).toEqual({ ok: true, status: 201 });

    const request = sent.find((r) => new URL(r.url).pathname === '/api/intents/create')!;
    expect(request.method).toBe('POST');
    expect(JSON.parse(request.body)).toEqual({
      intents: ['translate documents between English and German'],
      description: 'Translates English and German text.',
      capabilities: [],
      commands: [],
    });
    expectSigned(request, identity);
  });

  it('autoPublish on initialize signs with the identity the agent was handed', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, autoPublish: true, intents: ['translate documents'] });
    const agent = new BaseAgent({ name: 'finder', skills: [skill] });
    agent.identity = identity;

    await agent.initialize();

    const request = sent.find((r) => new URL(r.url).pathname === '/api/intents/create')!;
    expect(request).toBeDefined();
    expectSigned(request, identity);
  });
});

describe('PortalDiscoverySkill credential: a key is a bearer', () => {
  it('presents a configured key when it has no identity', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, apiKey: 'rok_configured' });
    expect(skill.credential()).toEqual({ kind: 'bearer', key: 'rok_configured' });

    const result = await skill.search({ query: 'translate', types: ['intents'] });
    expectBearer(intentsSearch()!, 'rok_configured');
    expect(result.intents).toHaveLength(1);
  });

  it('takes the key from WEBAGENTS_API_KEY', async () => {
    process.env.WEBAGENTS_API_KEY = 'rok_from_env';
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL });
    expect(skill.credential()).toEqual({ kind: 'bearer', key: 'rok_from_env' });

    await skill.publishIntents().then(() => undefined, () => undefined);
    // No intents configured: nothing was sent, and nothing was refused for want of a credential.
    expect(sent).toHaveLength(0);
    await skill.search({ query: 'translate', types: ['intents'] });
    expectBearer(intentsSearch()!, 'rok_from_env');
  });

  it('falls back to the key when the identity cannot sign, and says why when it cannot fall back', async () => {
    // What `serve()` leaves on an agent with no WEBAGENTS_PUBLIC_URL: a
    // loopback issuer the platform refuses by name before resolving anything.
    const loopback: SigningIdentity = { issuer: 'http://localhost:8000/agents/finder', getHeldKeys: () => [] };

    const withKey = new PortalDiscoverySkill({ portalUrl: PORTAL, identity: loopback, apiKey: 'rok_configured' });
    expect(withKey.credential()).toEqual({ kind: 'bearer', key: 'rok_configured' });
    await withKey.search({ query: 'translate', types: ['intents'] });
    expectBearer(intentsSearch()!, 'rok_configured');

    const withoutKey = new PortalDiscoverySkill({ portalUrl: PORTAL, identity: loopback });
    const credential = withoutKey.credential();
    expect(credential.kind).toBe('none');
    expect((credential as { reason: string }).reason).toContain('WEBAGENTS_PUBLIC_URL');
  });
});

describe('PortalDiscoverySkill credential: neither is refused, naming the fix', () => {
  it('search() answers the refusal and dials nothing', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL });
    expect(skill.credential()).toEqual({ kind: 'none', reason: NO_DISCOVERY_CREDENTIAL });

    const result = await skill.search({ query: 'translate', types: ['intents', 'agents', 'posts'] });

    expect(result).toEqual({ error: NO_DISCOVERY_CREDENTIAL });
    // The ways a person at the CLI has, the same sentence in both SDKs.
    expect(result.error).toContain('`webagents publish`');
    expect(result.error).toContain('WEBAGENTS_PUBLIC_URL');
    expect(result.error).toContain('WEBAGENTS_AGENT_TOKEN');
    expect(sent).toHaveLength(0);
  });

  it('publishIntents() answers the refusal and dials nothing', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, intents: ['translate documents'] });

    const result = await skill.publishIntents();

    expect(result).toEqual({ ok: false, status: 0, error: NO_DISCOVERY_CREDENTIAL });
    expect(sent).toHaveLength(0);
  });

  it('a whitespace-only key is no key', () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, apiKey: '   ' });
    expect(skill.credential().kind).toBe('none');
  });
});

describe('PortalDiscoverySkill credential: in the chat, the signed-in person (2026-09-25)', () => {
  it('search() presents the person\'s token when the agent has no credential of its own', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, personToken: async () => 'person-token' });

    await skill.search({ query: 'translate', types: ['intents'] });

    expect(sent).toHaveLength(1);
    expect(sent[0].headers.get('authorization')).toBe('Bearer person-token');
  });

  it('the agent\'s own key comes first: it searches as itself', async () => {
    const person = vi.fn(async () => 'person-token');
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, apiKey: 'rok_agent', personToken: person });

    await skill.search({ query: 'translate', types: ['intents'] });

    expect(sent[0].headers.get('authorization')).toBe('Bearer rok_agent');
    expect(person).not.toHaveBeenCalled();
  });

  it('it stands in for an identity that cannot sign from a loopback URL', async () => {
    const loopback: SigningIdentity = { issuer: 'http://localhost:8000/agents/finder', getHeldKeys: () => [] };
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, identity: loopback, personToken: async () => 'person-token' });

    await skill.search({ query: 'translate', types: ['intents'] });

    expect(sent[0].headers.get('authorization')).toBe('Bearer person-token');
    expect(sent[0].headers.get('signature-input')).toBeNull();
  });

  it('nobody signed in: the chat\'s sentence, and nothing dialled', async () => {
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, personToken: async () => null });

    const result = await skill.search({ query: 'translate', types: ['intents'] });

    expect(result).toEqual({ error: NO_DISCOVERY_SIGN_IN });
    expect(sent).toHaveLength(0);
  });

  it('publishIntents() never speaks as the person', async () => {
    const person = vi.fn(async () => 'person-token');
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL, intents: ['translate documents'], personToken: person });

    const result = await skill.publishIntents();

    expect(result).toEqual({ ok: false, status: 0, error: NO_DISCOVERY_CREDENTIAL });
    expect(person).not.toHaveBeenCalled();
    expect(sent).toHaveLength(0);
  });
});

describe('the hosts hand the agent its identity', () => {
  let keysDir: string;
  const closers: Array<() => Promise<void>> = [];

  beforeAll(async () => {
    keysDir = await mkdtemp(path.join(tmpdir(), 'webagents-discovery-keys-'));
  });

  afterAll(async () => {
    for (const close of closers) await close().catch(() => {});
    await rm(keysDir, { recursive: true, force: true });
  });

  it('serve() sets agent.identity to the identity it publishes, before initialize() runs', async () => {
    const { serve } = await import('../../../../src/server/node');
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL });
    let credentialAtInitialize: string | undefined;
    const originalInitialize = skill.initialize.bind(skill);
    skill.initialize = async () => {
      credentialAtInitialize = skill.credential().kind;
      await originalInitialize();
    };
    const agent = new BaseAgent({ name: 'finder', skills: [skill] });

    const server = await serve(agent, {
      port: 0,
      basePath: '/agents/finder',
      publicUrl: 'https://agent.example.com',
      keysDir,
      heartbeat: false,
    });
    closers.push(() => server.close());

    expect(agent.identity).toBe(server.identity);
    expect(server.identity.issuer).toBe(ISSUER);
    expect(credentialAtInitialize).toBe('signature');
    expect(skill.credential()).toEqual({ kind: 'signature', identity: server.identity });

    // And the signature names the key set this very server publishes.
    await skill.search({ query: 'translate', types: ['intents'] });
    expectSigned(intentsSearch()!, server.identity);
  });

  it('WebAgentsServer.addAgent() does the same for every agent it holds an identity for', async () => {
    const { WebAgentsServer } = await import('../../../../src/server/multi');
    const skill = new PortalDiscoverySkill({ portalUrl: PORTAL });
    const agent = new BaseAgent({ name: 'multi-finder', skills: [skill] });
    const server = new WebAgentsServer({
      port: 0,
      identity: { publicUrl: 'https://agents.example.com', keysDir },
    });

    await server.addAgent('multi-finder', agent);

    expect(agent.identity).toBe(server.getIdentity('multi-finder'));
    expect(skill.credential().kind).toBe('signature');
    await skill.search({ query: 'translate', types: ['intents'] });
    expectSigned(intentsSearch()!, server.getIdentity('multi-finder')!, 'https://agents.example.com/agents/multi-finder/.well-known/jwks.json');
  });
});
