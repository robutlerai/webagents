/**
 * The operator key that decides an agent's OWNER is inside the signature
 * (S-184, fixed 2026-09-19).
 *
 * `registerWithPlatform` sends the operator's platform API key in
 * `X-Robutler-Owner-Key`, and the platform resolves the agent's owner from
 * it. Until that day the header rode OUTSIDE the covered components, so
 * whoever could rewrite headers between the agent and the platform's TLS
 * edge (an egress gateway, a sidecar) could swap in a key of their own and
 * register the agent under their account while the agent's signature still
 * verified. These tests pin the SDK half: whenever the header is sent it is a
 * covered component, its value is in the signature base, and a swapped value
 * no longer verifies. The platform half (honouring the header on a signed
 * request only when it is covered) lives in the portal.
 */

import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import { createPublicKey, verify as cryptoVerify } from 'node:crypto';
import { registerWithPlatform, OWNER_KEY_HEADER } from '../../../src/server/registration';
import { AgentIdentity } from '../../../src/crypto/identity';

const PLATFORM = 'https://platform.example.com';
const ISSUER = 'https://agent.example.com/agents/demo';
const KEY_SET = `${ISSUER}/.well-known/jwks.json`;

let identity: AgentIdentity;
beforeAll(async () => {
  identity = new AgentIdentity({ agentId: 'demo', issuer: ISSUER });
  await identity.initialize();
});

function okResponse(owned: boolean) {
  return new Response(JSON.stringify({ access_token: 'tok', user_id: 'u-1', username: 'example.com.agents.demo', owned }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** The covered component names of `sig1`, in order, from a `Signature-Input` value. */
function coveredComponents(signatureInput: string): string[] {
  const list = /^sig1=\(([^)]*)\)/.exec(signatureInput);
  if (!list) throw new Error(`no sig1 inner list in ${signatureInput}`);
  return [...list[1].matchAll(/"([^"]+)"(?:;key="[^"]*")?/g)].map((m) => m[1]);
}

/** Rebuild the RFC 9421 base for the registering request from what was SENT, as the platform would. */
function rebuildBase(request: Request, ownerKeyValue: string | null): string {
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
    else if (name === 'signature-agent') lines.push(`"signature-agent";key="sig1": "${KEY_SET}";type=jwks_uri`);
    else lines.push(`"${name}": ${ownerKeyValue ?? request.headers.get(name)!.trim()}`);
  }
  lines.push(`"@signature-params": ${params}`);
  return lines.join('\n');
}

function verifies(request: Request, base: string): boolean {
  const m = /^sig1=:([A-Za-z0-9+/=]+):$/.exec(request.headers.get('signature')!);
  if (!m) return false;
  const key = createPublicKey({ key: identity.getJwks().keys[0] as never, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(base, 'ascii'), key, new Uint8Array(Buffer.from(m[1], 'base64')));
}

describe('registerWithPlatform: the owner key is covered by the signature (S-184)', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let warn: ReturnType<typeof vi.spyOn>;
  const savedEnv = process.env.ROBUTLER_API_KEY;

  beforeEach(() => {
    delete process.env.ROBUTLER_API_KEY;
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => okResponse(true));
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    warn.mockRestore();
    if (savedEnv === undefined) delete process.env.ROBUTLER_API_KEY;
    else process.env.ROBUTLER_API_KEY = savedEnv;
  });

  const sentRequest = () => fetchSpy.mock.calls[0][0] as Request;

  it('names the header it covers', () => {
    expect(OWNER_KEY_HEADER).toBe('X-Robutler-Owner-Key');
  });

  it('covers x-robutler-owner-key, after content-digest and before the signature-agent member', async () => {
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, ownerApiKey: '  rok_operator  ' });
    expect(result).toMatchObject({ ok: true, owned: true });

    const request = sentRequest();
    expect(request.headers.get('x-robutler-owner-key')).toBe('rok_operator');
    expect(coveredComponents(request.headers.get('signature-input')!)).toEqual([
      '@method',
      '@authority',
      '@path',
      '@query',
      'content-digest',
      'x-robutler-owner-key',
      'signature-agent',
    ]);
    // The signature verifies over a base that carries the key's value...
    const base = rebuildBase(request, null);
    expect(base).toContain('\n"x-robutler-owner-key": rok_operator\n');
    expect(verifies(request, base)).toBe(true);
    // ...and not over the same request with another operator's key swapped in.
    expect(verifies(request, rebuildBase(request, 'rok_attacker'))).toBe(false);
  });

  it('covers it when the key comes from ROBUTLER_API_KEY', async () => {
    process.env.ROBUTLER_API_KEY = 'rok_from_env';
    await registerWithPlatform(identity, { platformUrl: PLATFORM });
    const request = sentRequest();
    expect(request.headers.get('x-robutler-owner-key')).toBe('rok_from_env');
    expect(coveredComponents(request.headers.get('signature-input')!)).toContain('x-robutler-owner-key');
    expect(verifies(request, rebuildBase(request, null))).toBe(true);
  });

  it('sends no header and covers nothing extra when there is no owner key', async () => {
    fetchSpy.mockImplementation(async () => okResponse(false));
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, ownerApiKey: '   ' });
    expect(result).toMatchObject({ ok: true, owned: false });

    const request = sentRequest();
    expect(request.headers.get('x-robutler-owner-key')).toBeNull();
    expect(coveredComponents(request.headers.get('signature-input')!)).toEqual([
      '@method',
      '@authority',
      '@path',
      '@query',
      'content-digest',
      'signature-agent',
    ]);
    expect(verifies(request, rebuildBase(request, null))).toBe(true);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('UNCLAIMED'))).toBe(true);
  });

  // S-190 (fixed 2026-09-19): coverage stops tampering, not disclosure. Fetch
  // strips only `Authorization` and `Cookie` across origins, so under the
  // default `redirect: 'follow'` the operator's key went wherever a Location
  // pointed. The mode is pinned on the request that is actually sent, with
  // and without a key, so the rule cannot quietly become conditional.
  it('never follows a redirect on the registering request, which carries the operator key', async () => {
    await registerWithPlatform(identity, { platformUrl: PLATFORM, ownerApiKey: 'rk_owner' });
    expect(sentRequest().redirect).toBe('error');

    fetchSpy.mockClear();
    await registerWithPlatform(identity, { platformUrl: PLATFORM });
    expect(sentRequest().redirect).toBe('error');
  });

  it('reports a redirect as a failed registration rather than throwing', async () => {
    // What undici raises for a 3xx under `redirect: 'error'`.
    fetchSpy.mockImplementation(async () => {
      throw new TypeError('fetch failed: unexpected redirect');
    });
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, ownerApiKey: 'rk_owner' });
    expect(result).toMatchObject({ ok: false, status: 0 });
    expect(String((result as { error?: string }).error)).toContain('redirect');
  });
});
