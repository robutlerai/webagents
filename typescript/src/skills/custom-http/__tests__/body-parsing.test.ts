/**
 * A fn without `permissions.rawBody` must still receive a non-JSON POST body.
 *
 * The original code called `request.json()` first and, on failure, tried
 * `request.clone().text()` — but `.json()` had already consumed the stream,
 * so the clone yielded nothing and the body arrived as `null`. Every browser
 * form post (application/x-www-form-urlencoded) was therefore invisible to
 * the function: a landing page saw an empty submission and answered "that
 * email does not look right" no matter what was typed.
 */

import { describe, it, expect } from 'vitest';

/** Mirrors the skill: clone FIRST, then try JSON, then fall back to text. */
async function readBody(request: Request): Promise<unknown> {
  const forText = request.clone();
  let body: unknown = await request.json().catch(() => null);
  if (body === null) {
    const text = await forText.text().catch(() => '');
    if (text) body = text;
  }
  return body;
}

describe('non-JSON request bodies survive', () => {
  it('keeps a urlencoded form body', async () => {
    const req = new Request('https://x.test/', {
      method: 'POST',
      headers: { 'content-type': 'application/x-www-form-urlencoded' },
      body: 'name=Dana&email=dana%40example.com',
    });
    expect(await readBody(req)).toBe('name=Dana&email=dana%40example.com');
  });

  it('still parses JSON into an object', async () => {
    const req = new Request('https://x.test/', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ a: 1 }),
    });
    expect(await readBody(req)).toEqual({ a: 1 });
  });

  it('keeps plain text', async () => {
    const req = new Request('https://x.test/', { method: 'POST', body: 'hello' });
    expect(await readBody(req)).toBe('hello');
  });

  it('an empty body stays null', async () => {
    const req = new Request('https://x.test/', { method: 'POST' });
    expect(await readBody(req)).toBeNull();
  });
});
