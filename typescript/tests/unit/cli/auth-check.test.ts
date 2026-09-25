/**
 * `webagents login` validates the key before claiming success.
 *
 * It used to make NO network call at all: it wrote whatever was typed to disk
 * and printed "Authenticated successfully", so a mistyped or expired key was
 * reported as a successful login and then failed later, somewhere unrelated,
 * with an error that pointed at the wrong thing (2026-09-23).
 *
 * `fetch` is injected rather than mocked globally, so these never touch the
 * network and never depend on a portal being up.
 */

import { describe, it, expect } from 'vitest';
import { validateToken } from '../../../src/cli/auth-check';

function respondWith(status: number, body: unknown): typeof fetch {
  return (async () =>
    new Response(typeof body === 'string' ? body : JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    })) as unknown as typeof fetch;
}

describe('validateToken', () => {
  it('accepts a good key and reports the user it belongs to', async () => {
    const result = await validateToken(
      'https://robutler.ai',
      'rok_good',
      respondWith(200, { access_token: 'jwt.for.cli', username: 'tester' }),
    );
    expect(result.ok).toBe(true);
    expect(result.username).toBe('tester');
    // The exchanged token is what gets stored: scoped `agents:own` and expiring
    // in 7 days, rather than the long-lived key the user pasted.
    expect(result.accessToken).toBe('jwt.for.cli');
  });

  it('rejects a key the portal refuses', async () => {
    for (const status of [401, 403]) {
      const result = await validateToken('https://robutler.ai', 'rok_bad', respondWith(status, {}));
      expect(result.ok).toBe(false);
      expect(result.reason).toMatch(/rejected/);
    }
  });

  it('reports a server error as a server error, not as a bad key', async () => {
    // "your key is wrong" and "the portal is down" need different reactions.
    const result = await validateToken('https://robutler.ai', 'rok_x', respondWith(500, {}));
    expect(result.ok).toBe(false);
    expect(result.reason).toMatch(/answered 500/);
  });

  it('does not claim validity when it could not ask', async () => {
    const offline = (async () => {
      throw new Error('getaddrinfo ENOTFOUND');
    }) as unknown as typeof fetch;
    const result = await validateToken('https://robutler.ai', 'rok_x', offline);
    expect(result.ok).toBe(false);
    expect(result.reason).toMatch(/could not reach/);
  });

  it('treats an unparseable 2xx as success', async () => {
    // The key was accepted; the caller only needs to know it works.
    const result = await validateToken('https://robutler.ai', 'rok_x', respondWith(200, 'not json'));
    expect(result.ok).toBe(true);
  });

  it('does not double up slashes in the portal URL', async () => {
    let seen = '';
    const capture = (async (url: string) => {
      seen = url;
      return new Response('{}', { status: 200 });
    }) as unknown as typeof fetch;
    await validateToken('https://robutler.ai/', 'rok_x', capture);
    expect(seen).toBe('https://robutler.ai/api/auth/cli/token');
  });
});
