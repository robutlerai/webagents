/**
 * The agent card publishes the SPKI PEM at BOTH `publicKey` and
 * `metadata.publicKey` (build plan 1M-00, ADR-0038 step 1).
 *
 * The platform's verifier reads the card's TOP-LEVEL `publicKey`
 * (portal `lib/auth/agent-auth.ts`: `metadata?.publicKey`, where `metadata`
 * is the whole fetched card). This SDK wrote only the nested
 * `metadata.publicKey`, so a card that satisfied every test in this
 * repository carried no key for the one reader that matters, and no
 * SDK-served agent could auto-register. `serve-identity.test.ts` pins that
 * the key survives a restart; this file pins WHERE it is published.
 */

import { describe, it, expect } from 'vitest';
import { loadOrCreateAgentIdentity } from '../../../src/crypto/identity-store';
import { createFetchHandler } from '../../../src/server/handler';
import type { IAgent } from '../../../src/core/types';

const agent = { name: 'mini' } as unknown as IAgent;

async function card(): Promise<{ publicKey?: string; metadata?: { publicKey?: string } }> {
  const identity = await loadOrCreateAgentIdentity(agent.name, {
    issuer: 'https://agent.example.com',
    keysDir: null,
  });
  const handler = createFetchHandler(agent, { basePath: '/agents/mini', identity });
  const res = await handler(new Request('https://agent.example.com/.well-known/agent.json'));
  return (await res.json()) as { publicKey?: string; metadata?: { publicKey?: string } };
}

describe('agent card public key placement', () => {
  it('publishes the key at the top level, where the platform reads it', async () => {
    const c = await card();
    expect(c.publicKey).toMatch(/^-----BEGIN PUBLIC KEY-----/);
  });

  it('keeps the nested placement for readers of that shape, and the two never differ', async () => {
    const c = await card();
    expect(c.metadata?.publicKey).toMatch(/^-----BEGIN PUBLIC KEY-----/);
    expect(c.metadata?.publicKey).toBe(c.publicKey);
  });

  it('publishes neither when the server has no identity, never an empty string', async () => {
    const handler = createFetchHandler(agent, { basePath: '/agents/mini' });
    const res = await handler(new Request('https://agent.example.com/.well-known/agent.json'));
    const c = (await res.json()) as Record<string, unknown>;
    expect('publicKey' in c).toBe(false);
    expect('metadata' in c).toBe(false);
  });
});
