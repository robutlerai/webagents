/**
 * `registerWithPlatform` persisting its bearer, and reading it back.
 *
 * WHY THIS MATTERS MORE THAN A CACHE NORMALLY WOULD: the token the platform
 * answers with is good for seven days, carries `agents:own`, and is minted
 * with no `jti`, so nothing revokes it (portal security log S-037). An agent
 * that re-registers on every restart mints another one of those each time and
 * leaves the previous ones live until they lapse. Reading one back mints
 * none, so the reuse path is a security property rather than a speed-up.
 *
 * The other half these pin is that the store stays OPTIONAL. Registration
 * must survive a missing store, a store that throws on read, and a store that
 * throws on write, because an agent on a box with no keystore still has to
 * come up.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { registerWithPlatform, PLATFORM_TOKEN_SECRET } from '../../../src/server/registration';

const PLATFORM = 'https://platform.example.com';
const ISSUER = 'https://agent.example.com';

/** A structurally valid JWT with a chosen `exp`. Signature is a placeholder. */
function tokenExpiringIn(seconds: number): string {
  const b64 = (value: object) =>
    Buffer.from(JSON.stringify(value)).toString('base64url').replace(/=+$/, '');
  return `${b64({ alg: 'RS256' })}.${b64({
    exp: Math.floor(Date.now() / 1000) + seconds,
    scopes: ['agents:own'],
  })}.placeholder-signature-not-verified-locally`;
}

const STORED = tokenExpiringIn(6 * 24 * 3600);
const MINTED = tokenExpiringIn(7 * 24 * 3600);

const identity = {
  issuer: ISSUER,
  mintToken: async () => 'dummy-aoauth-assertion-not-a-real-token',
};

/** An in-memory store of the shape registration accepts. */
function memoryStore(seed: Record<string, string> = {}) {
  const items = new Map(Object.entries(seed));
  return {
    items,
    get: vi.fn(async (name: string) => items.get(name) ?? null),
    set: vi.fn(async (name: string, value: string) => {
      items.set(name, value);
      return 'keystore' as const;
    }),
    delete: vi.fn(async (name: string) => items.delete(name)),
  };
}

function okResponse() {
  return new Response(
    JSON.stringify({ access_token: MINTED, user_id: 'u-1', username: 'example.com.agents.demo' }),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  );
}

describe('registerWithPlatform with a secret store', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(okResponse());
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    warn.mockRestore();
  });

  it('reuses a stored bearer and calls nothing', async () => {
    const secrets = memoryStore({ [PLATFORM_TOKEN_SECRET]: STORED });
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });

    expect(result).toMatchObject({ ok: true, status: 0, reused: true, accessToken: STORED });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(secrets.set).not.toHaveBeenCalled();
  });

  it('persists a freshly minted bearer under the default name', async () => {
    const secrets = memoryStore();
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });

    expect(result).toMatchObject({ ok: true, status: 200, reused: false, stored: 'saved' });
    expect(result.accessToken).toBe(MINTED);
    expect(secrets.items.get(PLATFORM_TOKEN_SECRET)).toBe(MINTED);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('honours a custom secret name', async () => {
    const secrets = memoryStore();
    await registerWithPlatform(identity, {
      platformUrl: PLATFORM,
      secrets,
      tokenName: 'staging_platform_token',
    });
    expect(secrets.items.has('staging_platform_token')).toBe(true);
    expect(secrets.items.has(PLATFORM_TOKEN_SECRET)).toBe(false);
  });

  it('re-registers when the stored bearer has lapsed, and drops the dead one', async () => {
    const expired = tokenExpiringIn(-60);
    const secrets = memoryStore({ [PLATFORM_TOKEN_SECRET]: expired });

    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });

    expect(secrets.delete).toHaveBeenCalledWith(PLATFORM_TOKEN_SECRET);
    expect(result.reused).toBe(false);
    expect(result.accessToken).toBe(MINTED);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('treats a bearer inside the expiry skew as spent', async () => {
    // 60 seconds left, default skew 300: usable once, then dead mid-run.
    const secrets = memoryStore({ [PLATFORM_TOKEN_SECRET]: tokenExpiringIn(60) });
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });
    expect(result.reused).toBe(false);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('re-registers a bearer with no readable exp rather than assuming it is eternal', async () => {
    const secrets = memoryStore({ [PLATFORM_TOKEN_SECRET]: 'not-a-jwt' });
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });
    expect(result.reused).toBe(false);
    expect(secrets.delete).toHaveBeenCalledWith(PLATFORM_TOKEN_SECRET);
  });

  it('registers again when refresh is asked for, even with a live stored bearer', async () => {
    const secrets = memoryStore({ [PLATFORM_TOKEN_SECRET]: STORED });
    const result = await registerWithPlatform(identity, {
      platformUrl: PLATFORM,
      secrets,
      refresh: true,
    });
    expect(result.reused).toBe(false);
    expect(result.accessToken).toBe(MINTED);
    expect(secrets.items.get(PLATFORM_TOKEN_SECRET)).toBe(MINTED);
  });

  it('registers normally with no store at all', async () => {
    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM });
    expect(result).toMatchObject({ ok: true, status: 200, stored: 'not-requested' });
    expect(result.accessToken).toBe(MINTED);
  });

  it('registers anyway when the store cannot be read', async () => {
    const secrets = memoryStore();
    secrets.get.mockRejectedValue(new Error('keystore locked'));

    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });

    expect(result.ok).toBe(true);
    expect(result.accessToken).toBe(MINTED);
    expect(warn.mock.calls.map((c) => String(c[0])).join('\n')).toContain('keystore locked');
  });

  it('hands back a working bearer when the store cannot be written', async () => {
    const secrets = memoryStore();
    secrets.set.mockRejectedValue(new Error('disk full'));

    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });

    // Registration SUCCEEDED. Throwing away a live credential over a storage
    // failure would be the worse outcome, so it is reported instead.
    expect(result.ok).toBe(true);
    expect(result.accessToken).toBe(MINTED);
    expect(result.stored).toContain('disk full');
  });

  it('stores nothing when registration fails', async () => {
    fetchSpy.mockResolvedValue(new Response('unauthorized', { status: 401 }));
    const secrets = memoryStore();

    const result = await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });

    expect(result.ok).toBe(false);
    expect(result.status).toBe(401);
    expect(secrets.set).not.toHaveBeenCalled();
    expect(secrets.items.size).toBe(0);
  });

  it('never logs the bearer it stored', async () => {
    const secrets = memoryStore();
    secrets.set.mockRejectedValue(new Error('disk full'));
    await registerWithPlatform(identity, { platformUrl: PLATFORM, secrets });
    expect(warn.mock.calls.map((c) => String(c[0])).join('\n')).not.toContain(MINTED);
  });
});
