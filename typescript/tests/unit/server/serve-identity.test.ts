/**
 * `serve()` must keep the SAME agent key across restarts.
 *
 * Registration stores the card's `metadata.publicKey` in
 * `agent_registrations.publicKey` and verifies every later AOAuth token
 * against that stored copy. A key generated per boot therefore works until
 * the first restart and then fails every verification — with a card that
 * still looks perfectly well-formed, which is what makes it expensive to
 * find.
 *
 * This used to be a property of the `host()` wrapper, so the DOCUMENTED
 * server (`serve()`) published an ephemeral key. It is now a property of the
 * server itself, which is what these tests pin.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { loadOrCreateAgentIdentity } from '../../../src/crypto/identity-store';
import { createFetchHandler } from '../../../src/server/handler';
import type { IAgent } from '../../../src/core/types';

const agent = { name: 'mini' } as unknown as IAgent;

/** One "boot": resolve the identity the way `serve()` does, then read the
 *  card the way the platform does. */
async function cardPublicKey(keysDir: string | null): Promise<string> {
  const identity = await loadOrCreateAgentIdentity(agent.name, {
    issuer: 'https://agent.example.com',
    keysDir,
  });
  const handler = createFetchHandler(agent, { basePath: '/agents/mini', identity });
  const res = await handler(new Request('https://agent.example.com/.well-known/agent.json'));
  const card = (await res.json()) as { metadata?: { publicKey?: string } };
  return card.metadata?.publicKey ?? '';
}

describe('served agent identity', () => {
  let keysDir: string;

  beforeEach(async () => {
    keysDir = await mkdtemp(path.join(tmpdir(), 'webagents-keys-'));
  });

  afterEach(async () => {
    await rm(keysDir, { recursive: true, force: true });
  });

  it('persists the key and serves the same public key on the next start', async () => {
    const first = await cardPublicKey(keysDir);
    expect(first).toMatch(/^-----BEGIN PUBLIC KEY-----/);

    const files = await readdir(keysDir);
    expect(files).toHaveLength(1);

    // A second "boot" against the same directory.
    const second = await cardPublicKey(keysDir);
    expect(second).toBe(first);
  });

  it('reads WEBAGENTS_KEYS_DIR when no directory is passed', async () => {
    const previous = process.env.WEBAGENTS_KEYS_DIR;
    process.env.WEBAGENTS_KEYS_DIR = keysDir;
    try {
      const first = await cardPublicKey(undefined as unknown as null);
      const second = await cardPublicKey(undefined as unknown as null);
      expect(first).toMatch(/^-----BEGIN PUBLIC KEY-----/);
      expect(second).toBe(first);
    } finally {
      if (previous === undefined) delete process.env.WEBAGENTS_KEYS_DIR;
      else process.env.WEBAGENTS_KEYS_DIR = previous;
    }
  });

  it('still works with an explicitly ephemeral key, and that key differs per start', async () => {
    const first = await cardPublicKey(null);
    const second = await cardPublicKey(null);
    expect(first).toMatch(/^-----BEGIN PUBLIC KEY-----/);
    expect(second).not.toBe(first);
  });
});
