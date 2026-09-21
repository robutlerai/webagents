/**
 * The agent card contract (ADR 0038 step 5, W2 design sections 3.3 and 9.1,
 * 2026-09-17), for BOTH servers: `createFetchHandler` and `WebAgentsServer`.
 *
 * The platform reads the card once, at registration, and checks three
 * fields by string comparison: `client_id` equals the URL the card was
 * fetched from, `url` equals the principal (the agent URL every signature
 * names), `jwks_uri` equals the key set the signature was resolved through.
 * A card failing any of them is `card_not_self_naming` or
 * `card_key_set_mismatch`. So these tests pin that all three are DERIVED from
 * the identity's issuer, that the key set really is served at `jwks_uri`,
 * and that the bearer-era PEM (`publicKey`, `metadata.publicKey`) is gone.
 *
 * The origin-level copy of the card is gone with the PEM: it existed for the
 * origin-level fallback the platform deleted (S-015) and cannot name itself
 * for a prefixed agent. A server with an EMPTY basePath still serves the
 * card at the origin, because that is its prefix.
 *
 * `WebAgentsServer` served no card at all before this day, so a multi-agent
 * TypeScript host could not register whatever else was right (design
 * section 12, item 2); its card is the same document from `card.ts`.
 */

import { describe, it, expect } from 'vitest';
import { AgentIdentity } from '../../../src/crypto/identity';
import { signatureAgentValue } from '../../../src/crypto/http-signature';
import { createFetchHandler } from '../../../src/server/handler';
import { WebAgentsServer } from '../../../src/server/multi';
import { buildAgentCard } from '../../../src/server/card';
import type { IAgent } from '../../../src/core/types';

const AGENT_URL = 'https://agent.example.com/agents/mini';

const mini = {
  name: 'mini',
  description: 'A small agent',
  getToolDefinitions: () => [
    { type: 'function', function: { name: 'add', description: 'Add two numbers', parameters: {} } },
  ],
} as unknown as IAgent;

async function identityAt(issuer: string): Promise<AgentIdentity> {
  const identity = new AgentIdentity({ agentId: 'mini', issuer });
  await identity.initialize();
  return identity;
}

interface Card {
  name: string;
  description?: string;
  client_id: string;
  url: string;
  jwks_uri?: string;
  capabilities: { streaming: boolean; pushNotifications: boolean };
  authentication: { schemes: string[] };
  skills: Array<{ id: string; name: string }>;
  publicKey?: unknown;
  metadata?: unknown;
}

describe('buildAgentCard', () => {
  it('derives client_id, url and jwks_uri from one principal', () => {
    const card = buildAgentCard(mini, { principal: `${AGENT_URL}/`, signs: true });
    expect(card).toEqual({
      name: 'mini',
      description: 'A small agent',
      client_id: `${AGENT_URL}/.well-known/agent.json`,
      url: AGENT_URL,
      jwks_uri: `${AGENT_URL}/.well-known/jwks.json`,
      capabilities: { streaming: true, pushNotifications: false },
      authentication: { schemes: ['HTTPSig'] },
      skills: [{ id: 'add', name: 'add', description: 'Add two numbers' }],
    });
  });

  it('names no key set and claims Bearer, not HTTPSig, for an agent that cannot sign', () => {
    const card = buildAgentCard(mini, { principal: '/agents/mini', signs: false });
    expect(card.client_id).toBe('/agents/mini/.well-known/agent.json');
    expect(card.url).toBe('/agents/mini');
    expect('jwks_uri' in card).toBe(false);
    expect(card.authentication.schemes).toEqual(['Bearer']);
    expect(buildAgentCard(mini, { principal: '/', signs: false }).url).toBe('/');
    expect(buildAgentCard(mini, { principal: '', signs: false }).client_id).toBe('/.well-known/agent.json');
  });
});

describe('createFetchHandler card contract', () => {
  it('serves a self-naming card under basePath whose jwks_uri is the key set the signer names', async () => {
    const identity = await identityAt(AGENT_URL);
    const handler = createFetchHandler(mini, { basePath: '/agents/mini', identity });

    const res = await handler(new Request(`${AGENT_URL}/.well-known/agent.json`));
    expect(res.status).toBe(200);
    const card = (await res.json()) as Card;
    expect(card.client_id).toBe(`${AGENT_URL}/.well-known/agent.json`);
    expect(card.url).toBe(AGENT_URL);
    expect(card.jwks_uri).toBe(`${AGENT_URL}/.well-known/jwks.json`);
    expect(card.jwks_uri).toBe(signatureAgentValue(identity.issuer, 'dictionary-typed'));
    expect(card.authentication.schemes).toEqual(['HTTPSig']);
    expect(card.skills.map((s) => s.id)).toEqual(['add']);
    // The PEM shapes are deleted, not deprecated.
    expect('publicKey' in card).toBe(false);
    expect('metadata' in card).toBe(false);

    // The key set at `jwks_uri` lists the signing key by thumbprint, no `alg`.
    const jwks = await handler(new Request(card.jwks_uri!));
    expect(jwks.status).toBe(200);
    expect(jwks.headers.get('cache-control')).toBe('public, max-age=3600');
    const { keys } = (await jwks.json()) as { keys: Array<Record<string, string>> };
    expect(keys).toHaveLength(1);
    expect(keys[0].kid).toBe(identity.kid);
    expect(keys[0].alg).toBeUndefined();
  });

  it('follows the identity, which signs, when publicUrl would say something else', async () => {
    const identity = await identityAt(AGENT_URL);
    const handler = createFetchHandler(mini, {
      basePath: '/agents/mini',
      identity,
      publicUrl: 'https://other.example.com',
    });
    const card = (await (await handler(new Request(`${AGENT_URL}/.well-known/agent.json`))).json()) as Card;
    expect(card.url).toBe(AGENT_URL);
    expect(card.client_id).toBe(`${AGENT_URL}/.well-known/agent.json`);
  });

  it('no longer serves the origin-level copy for a prefixed agent', async () => {
    const identity = await identityAt(AGENT_URL);
    const handler = createFetchHandler(mini, { basePath: '/agents/mini', identity });
    const res = await handler(new Request('https://agent.example.com/.well-known/agent.json'));
    expect(res.status).toBe(404);
  });

  it('serves the card at the origin when basePath is empty, because that IS its prefix', async () => {
    const identity = await identityAt('https://agent.example.com');
    const handler = createFetchHandler(mini, { identity });
    const res = await handler(new Request('https://agent.example.com/.well-known/agent.json'));
    expect(res.status).toBe(200);
    const card = (await res.json()) as Card;
    expect(card.client_id).toBe('https://agent.example.com/.well-known/agent.json');
    expect(card.url).toBe('https://agent.example.com');
    expect(card.jwks_uri).toBe('https://agent.example.com/.well-known/jwks.json');
  });

  it('composes publicUrl + basePath without an identity, and names no key set', async () => {
    const handler = createFetchHandler(mini, { basePath: '/agents/mini', publicUrl: 'https://agent.example.com/' });
    const card = (await (await handler(new Request('https://x/agents/mini/.well-known/agent.json'))).json()) as Card;
    expect(card.url).toBe(AGENT_URL);
    expect(card.client_id).toBe(`${AGENT_URL}/.well-known/agent.json`);
    expect('jwks_uri' in card).toBe(false);
    expect(card.authentication.schemes).toEqual(['Bearer']);
  });

  it('falls back to basePath as a relative reference, never to the request host', async () => {
    const previous = process.env.WEBAGENTS_PUBLIC_URL;
    delete process.env.WEBAGENTS_PUBLIC_URL;
    try {
      const handler = createFetchHandler(mini, { basePath: '/agents/mini' });
      const card = (await (await handler(new Request('http://127.0.0.1:8816/agents/mini/.well-known/agent.json'))).json()) as Card;
      expect(card.url).toBe('/agents/mini');
      expect(card.client_id).toBe('/agents/mini/.well-known/agent.json');
    } finally {
      if (previous !== undefined) process.env.WEBAGENTS_PUBLIC_URL = previous;
    }
  });
});

describe('WebAgentsServer card contract', () => {
  const echo = {
    name: 'echo',
    description: 'An echo agent',
    getCapabilities: () => ({}),
    getToolDefinitions: () => [],
  } as unknown as IAgent;

  it('serves a self-naming card per agent, at the route the router mounts it on', async () => {
    const server = new WebAgentsServer({
      port: 0,
      logging: false,
      identity: { publicUrl: 'https://agents.example.com/', keysDir: null },
    });
    await server.addAgent('echo', echo);
    const identity = server.getIdentity('echo')!;
    expect(identity.issuer).toBe('https://agents.example.com/agents/echo');

    const res = await server.getApp().request('/agents/echo/.well-known/agent.json');
    expect(res.status).toBe(200);
    const card = (await res.json()) as Card;
    expect(card.name).toBe('echo');
    expect(card.client_id).toBe('https://agents.example.com/agents/echo/.well-known/agent.json');
    expect(card.url).toBe('https://agents.example.com/agents/echo');
    expect(card.jwks_uri).toBe('https://agents.example.com/agents/echo/.well-known/jwks.json');
    expect(card.jwks_uri).toBe(signatureAgentValue(identity.issuer, 'dictionary-typed'));
    expect(card.authentication.schemes).toEqual(['HTTPSig']);
    expect('publicKey' in card).toBe(false);

    const jwks = await server.getApp().request('/agents/echo/.well-known/jwks.json');
    const { keys } = (await jwks.json()) as { keys: Array<Record<string, string>> };
    expect(keys[0].kid).toBe(identity.kid);
    expect(keys[0].alg).toBeUndefined();
  });

  it('includes the server basePath in the principal, because the router does', async () => {
    const server = new WebAgentsServer({
      port: 0,
      logging: false,
      basePath: '/api',
      identity: { publicUrl: 'https://agents.example.com', keysDir: null },
    });
    await server.addAgent('echo', echo);
    const res = await server.getApp().request('/api/agents/echo/.well-known/agent.json');
    expect(res.status).toBe(200);
    const card = (await res.json()) as Card;
    expect(card.url).toBe('https://agents.example.com/api/agents/echo');
    expect(card.client_id).toBe('https://agents.example.com/api/agents/echo/.well-known/agent.json');
    expect(server.getIdentity('echo')!.issuer).toBe(card.url);
  });

  it('serves a relative, key-set-less card for a server with no identity', async () => {
    const server = new WebAgentsServer({ port: 0, logging: false });
    await server.addAgent('plain', { ...echo, name: 'plain' } as unknown as IAgent);
    const res = await server.getApp().request('/agents/plain/.well-known/agent.json');
    expect(res.status).toBe(200);
    const card = (await res.json()) as Card;
    expect(card.url).toBe('/agents/plain');
    expect(card.client_id).toBe('/agents/plain/.well-known/agent.json');
    expect('jwks_uri' in card).toBe(false);
    expect(card.authentication.schemes).toEqual(['Bearer']);
    const jwks = await server.getApp().request('/agents/plain/.well-known/jwks.json');
    expect(jwks.status).toBe(404);
  });

  it('the built-in card wins over a transport skill card, as it does in createFetchHandler', async () => {
    const withSkillCard = {
      ...echo,
      getHttpHandler: (path: string, method: string) =>
        path === '/.well-known/agent.json' && method === 'GET'
          ? { handler: async () => new Response(JSON.stringify({ name: 'skill card' })) }
          : undefined,
    } as unknown as IAgent;
    const server = new WebAgentsServer({
      port: 0,
      logging: false,
      identity: { publicUrl: 'https://agents.example.com', keysDir: null },
    });
    await server.addAgent('echo', withSkillCard);
    const card = (await (await server.getApp().request('/agents/echo/.well-known/agent.json')).json()) as Card;
    expect(card.name).toBe('echo');
    expect(card.client_id).toBe('https://agents.example.com/agents/echo/.well-known/agent.json');
  });

  it('spells the card the way the platform derives the principal, whatever spelling publicUrl uses (2026-09-18)', async () => {
    // The platform reads the principal off Signature-Agent through a WHATWG
    // parse (host lowercased, :443 dropped) and compares by string equality.
    // The signer already spelled its value from `url.origin`; the card echoed
    // publicUrl verbatim, so this shape verified and was then refused
    // `card_not_self_naming`.
    const server = new WebAgentsServer({
      port: 0,
      logging: false,
      identity: { publicUrl: 'https://Agents.Example.com:443/', keysDir: null },
    });
    await server.addAgent('echo', echo);
    const identity = server.getIdentity('echo')!;
    expect(identity.issuer).toBe('https://agents.example.com/agents/echo');
    const card = (await (await server.getApp().request('/agents/echo/.well-known/agent.json')).json()) as Card;
    expect(card.url).toBe('https://agents.example.com/agents/echo');
    expect(card.client_id).toBe('https://agents.example.com/agents/echo/.well-known/agent.json');
    expect(card.jwks_uri).toBe(signatureAgentValue(identity.issuer, 'dictionary-typed'));
    expect(card.jwks_uri).toBe(signatureAgentValue('https://Agents.Example.com:443/agents/echo', 'dictionary-typed'));
  });
});

describe('buildAgentCard spelling (2026-09-18)', () => {
  const agent = { name: 'x', getCapabilities: () => ({}), getToolDefinitions: () => [] } as unknown as IAgent;

  it('canonicalises an absolute principal and leaves a relative one alone', () => {
    const absolute = buildAgentCard(agent, { principal: 'https://Agent.Example.COM:443/agents/x/', signs: true });
    expect(absolute.url).toBe('https://agent.example.com/agents/x');
    expect(absolute.client_id).toBe('https://agent.example.com/agents/x/.well-known/agent.json');
    expect(absolute.jwks_uri).toBe('https://agent.example.com/agents/x/.well-known/jwks.json');
    const relative = buildAgentCard(agent, { principal: '/agents/x/', signs: false });
    expect(relative.url).toBe('/agents/x');
    expect(relative.client_id).toBe('/agents/x/.well-known/agent.json');
    const port = buildAgentCard(agent, { principal: 'https://agent.example.com:8443/agents/x', signs: true });
    expect(port.url).toBe('https://agent.example.com:8443/agents/x');
  });
});
