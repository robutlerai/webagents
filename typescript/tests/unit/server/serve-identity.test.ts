/**
 * `serve()` must keep the SAME agent key across restarts.
 *
 * Registration pins the key set the platform fetched from
 * `{agentUrl}/.well-known/jwks.json`, keyed by RFC 7638 thumbprint, and
 * verifies every later signed request against that stored set (ADR 0038
 * step 5, 2026-09-17; before that day it pinned the card's PEM, with the
 * same consequence). A key generated per boot therefore works until the
 * first restart and then fails every verification, with a card and a key
 * set that still look perfectly well-formed, which is what makes it
 * expensive to find.
 *
 * This used to be a property of the `host()` wrapper, so the DOCUMENTED
 * server (`serve()`) published an ephemeral key. It is now a property of the
 * server itself, which is what these tests pin: the thumbprint the served
 * key set lists is the same one after a second boot against the same
 * directory.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { loadOrCreateAgentIdentity } from '../../../src/crypto/identity-store';
import { createFetchHandler } from '../../../src/server/handler';
import type { IAgent } from '../../../src/core/types';

const agent = { name: 'mini' } as unknown as IAgent;
const AGENT_URL = 'https://agent.example.com/agents/mini';
const THUMBPRINT = /^[A-Za-z0-9_-]{43}$/;

/** One "boot": resolve the identity the way `serve()` does, then read the
 *  card and the key set it names, the way the platform does. */
async function servedKid(keysDir: string | null): Promise<string> {
  const identity = await loadOrCreateAgentIdentity(agent.name, { issuer: AGENT_URL, keysDir });
  const handler = createFetchHandler(agent, { basePath: '/agents/mini', identity });
  const card = (await (await handler(new Request(`${AGENT_URL}/.well-known/agent.json`))).json()) as {
    jwks_uri?: string;
  };
  expect(card.jwks_uri).toBe(`${AGENT_URL}/.well-known/jwks.json`);
  const res = await handler(new Request(card.jwks_uri!));
  const { keys } = (await res.json()) as { keys: Array<{ kid: string }> };
  return keys[0]?.kid ?? '';
}

describe('served agent identity', () => {
  let keysDir: string;

  beforeEach(async () => {
    keysDir = await mkdtemp(path.join(tmpdir(), 'webagents-keys-'));
  });

  afterEach(async () => {
    await rm(keysDir, { recursive: true, force: true });
  });

  it('persists the key and serves the same thumbprint on the next start', async () => {
    const first = await servedKid(keysDir);
    expect(first).toMatch(THUMBPRINT);

    const files = await readdir(keysDir);
    expect(files).toHaveLength(1);

    // A second "boot" against the same directory.
    const second = await servedKid(keysDir);
    expect(second).toBe(first);
  });

  it('reads WEBAGENTS_KEYS_DIR when no directory is passed', async () => {
    const previous = process.env.WEBAGENTS_KEYS_DIR;
    process.env.WEBAGENTS_KEYS_DIR = keysDir;
    try {
      const first = await servedKid(undefined as unknown as null);
      const second = await servedKid(undefined as unknown as null);
      expect(first).toMatch(THUMBPRINT);
      expect(second).toBe(first);
    } finally {
      if (previous === undefined) delete process.env.WEBAGENTS_KEYS_DIR;
      else process.env.WEBAGENTS_KEYS_DIR = previous;
    }
  });

  it('still works with an explicitly ephemeral key, and that key differs per start', async () => {
    const first = await servedKid(null);
    const second = await servedKid(null);
    expect(first).toMatch(THUMBPRINT);
    expect(second).not.toBe(first);
  });
});
