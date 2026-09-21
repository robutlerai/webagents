/**
 * The well-known signatures directory (2026-09-19).
 *
 * `SignRequestOptions.form: 'legacy-string'` sends a bare ORIGIN as
 * `Signature-Agent`. The platform resolves a bare origin to
 * `{origin}/.well-known/http-message-signatures-directory` and refuses any
 * answer not served as `application/http-message-signatures-directory+json`
 * (lib/auth/web-bot-auth/discovery.ts in the portal). Neither SDK server
 * served that path, so the form failed discovery against a webagents-hosted
 * agent every time. These tests resolve the signer's own header the way the
 * platform does and ask the server for what it names, on all three server
 * shapes, and apply the platform's directory rules to the answer.
 */

import { describe, it, expect, vi } from 'vitest';
import { createHash } from 'node:crypto';
import { loadOrCreateAgentIdentity } from '../../../src/crypto/identity-store';
import { signMessage } from '../../../src/crypto/http-signature';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { WebAgentsServer } from '../../../src/server/multi';
import {
  DIRECTORY_MAX_KEYS,
  DIRECTORY_MEDIA_TYPE,
  DIRECTORY_WELL_KNOWN_PATH,
  keyDirectoryResponse,
} from '../../../src/server/key-directory';
import type { IAgent } from '../../../src/core/types';

const agent = {
  name: 'mini',
  description: 'A test agent',
  getCapabilities: () => ({}),
  getToolDefinitions: () => [],
  getHttpHandler: () => undefined,
  getWebSocketHandler: () => undefined,
} as unknown as IAgent;

interface DirectoryKey {
  kty: string;
  crv: string;
  x: string;
  kid?: string;
  use?: string;
  alg?: string;
  d?: string;
}

/** RFC 7638 over `{crv, kty, x}`, as the platform computes it. */
function thumbprint(key: DirectoryKey): string {
  return createHash('sha256').update(JSON.stringify({ crv: key.crv, kty: key.kty, x: key.x })).digest('base64url');
}

/** The platform's reading of a directory answer (discovery.ts): the media type, then the entry rules. */
async function platformReads(res: Response): Promise<string[]> {
  expect(res.status).toBe(200);
  expect((res.headers.get('content-type') ?? '').split(';')[0].trim().toLowerCase()).toBe('application/http-message-signatures-directory+json');
  expect(res.headers.get('cache-control')).toMatch(/max-age=\d+/);
  const { keys } = (await res.json()) as { keys: DirectoryKey[] };
  expect(keys.length).toBeGreaterThan(0);
  expect(keys.length).toBeLessThanOrEqual(16);
  for (const key of keys) {
    expect(key).toMatchObject({ kty: 'OKP', crv: 'Ed25519', use: 'sig' });
    // At the well-known directory a `kid` MUST equal the thumbprint, or the whole set is refused.
    expect(key.kid).toBe(thumbprint(key));
    expect(key.d).toBeUndefined();
  }
  return keys.map((k) => k.kid!);
}

/** The URL the platform dials for a legacy-string `Signature-Agent` value: the bare origin plus the well-known path. */
function directoryUrlOf(signatureAgent: string): string {
  const origin = JSON.parse(signatureAgent) as string;
  expect(new URL(origin).origin).toBe(origin);
  return `${origin}/.well-known/http-message-signatures-directory`;
}

describe('the signatures directory a legacy-string signer names', () => {
  it('names the exact path and media type the platform asks for', () => {
    expect(DIRECTORY_WELL_KNOWN_PATH).toBe('/.well-known/http-message-signatures-directory');
    expect(DIRECTORY_MEDIA_TYPE).toBe('application/http-message-signatures-directory+json');
    expect(DIRECTORY_MAX_KEYS).toBe(16);
  });

  it('createFetchHandler: what the signer names is served, at the origin, whatever the base path, and lists the signing key', async () => {
    const identity = await loadOrCreateAgentIdentity('mini', { issuer: 'https://agent.example.com/agents/mini', keysDir: null });
    const signed = await signMessage(identity, { method: 'POST', url: 'https://robutler.ai/api/auth/cli/token', body: '{}' }, { form: 'legacy-string' });
    const keyid = /keyid="([^"]+)"/.exec(signed.headers['signature-input'])![1];

    const handler = createFetchHandler(agent, { basePath: '/agents/mini', identity });
    const kids = await platformReads(await handler(new Request(directoryUrlOf(signed.headers['signature-agent']))));
    expect(kids).toContain(keyid);

    // Not under the prefix: a bare origin cannot name a path.
    expect((await handler(new Request('https://agent.example.com/agents/mini/.well-known/http-message-signatures-directory'))).status).toBe(404);
    // GET only, and the key set at its own path keeps its own media type.
    expect((await handler(new Request('https://agent.example.com/.well-known/http-message-signatures-directory', { method: 'POST' }))).status).not.toBe(200);
    expect((await handler(new Request('https://agent.example.com/agents/mini/.well-known/jwks.json'))).headers.get('content-type')).toBe('application/json');
  });

  it('createFetchHandler: an agent with no signing identity answers 404, never an empty directory', async () => {
    const handler = createFetchHandler(agent, { basePath: '/agents/mini' });
    const res = await handler(new Request('https://agent.example.com/.well-known/http-message-signatures-directory'));
    expect(res.status).toBe(404);
  });

  it('serve() (createAgentApp): the directory is reachable through the app, with no credential', async () => {
    const identity = await loadOrCreateAgentIdentity('mini', { issuer: 'https://agent.example.com/agents/mini', keysDir: null });
    const { app } = createAgentApp(agent, { basePath: '/agents/mini', identity });
    const kids = await platformReads(await app.request('/.well-known/http-message-signatures-directory'));
    expect(kids).toEqual([identity.kid]);
  });

  it('WebAgentsServer: one origin directory lists every hosted agent, and follows agents added and removed after boot', async () => {
    const server = new WebAgentsServer({ port: 0, logging: false, basePath: '/api', identity: { publicUrl: 'https://agents.example.com/', keysDir: null } });
    expect((await server.getApp().request('/.well-known/http-message-signatures-directory')).status).toBe(404);

    await server.addAgent('echo', { ...agent, name: 'echo' } as unknown as IAgent);
    await server.addAgent('mini', agent);
    const kids = await platformReads(await server.getApp().request('/.well-known/http-message-signatures-directory'));
    expect(kids).toEqual([server.getIdentity('echo')!.kid, server.getIdentity('mini')!.kid]);

    // What a legacy-string signature from either agent names is this one document.
    const signed = await signMessage(server.getIdentity('mini')!, { method: 'GET', url: 'https://robutler.ai/api/user' }, { form: 'legacy-string' });
    expect(directoryUrlOf(signed.headers['signature-agent'])).toBe('https://agents.example.com/.well-known/http-message-signatures-directory');

    server.removeAgent('echo');
    expect(await platformReads(await server.getApp().request('/.well-known/http-message-signatures-directory'))).toEqual([server.getIdentity('mini')!.kid]);
  });

  it('lists one entry per thumbprint and only Ed25519 signing keys, and warns above the platform cap instead of dropping a key', () => {
    const key = (n: number) => ({ kty: 'OKP' as const, crv: 'Ed25519' as const, x: `x${n}`.padEnd(43, 'A'), kid: `kid-${n}`, use: 'sig' as const });
    const source = (...keys: Array<ReturnType<typeof key>>) => ({ getJwks: () => ({ keys }) });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const many = Array.from({ length: 17 }, (_, i) => source(key(i), key(i)));
      const res = keyDirectoryResponse([undefined, ...many]);
      expect(res.status).toBe(200);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0][0])).toContain('17 keys');
    } finally {
      warn.mockRestore();
    }
  });
});
