/**
 * Signing in through the browser (2026-09-24, `src/cli/browser-login.ts`).
 *
 * The chat's "sign in to use Robutler's models" and `webagents login` at a
 * terminal both wait here for the portal's redirect. Pinned against a real
 * loopback server: the token is taken only with the `state` this terminal
 * made, Deny ends the wait with a reason rather than a five-minute timeout,
 * Ctrl+C (the abort signal) ends it too, and the page it answers takes the
 * token out of the address bar and escapes what it prints.
 */

import { describe, expect, it } from 'vitest';
import { browserLogin, callbackPage } from '../../../src/cli/browser-login';

/** A port for this test, away from the real 8789 and from other runs. */
function freshPort(): number {
  return 20000 + Math.floor(Math.random() * 20000);
}

/** Starts a sign-in and hands back what the browser would open. */
function start(options: { signal?: AbortSignal } = {}) {
  const port = freshPort();
  const lines: string[] = [];
  let opened!: (url: string) => void;
  const openedOnce = new Promise<string>((resolve) => {
    opened = resolve;
  });
  const pending = browserLogin('https://portal.example/', {
    port,
    timeoutMs: 10_000,
    open: (url) => opened(url),
    print: (line) => lines.push(line),
    ...options,
  });
  return { pending, openedOnce, port, lines };
}

async function redirect(port: number, query: Record<string, string>) {
  const url = new URL(`http://127.0.0.1:${port}/callback`);
  for (const [key, value] of Object.entries(query)) url.searchParams.set(key, value);
  const res = await fetch(url);
  return { status: res.status, body: await res.text() };
}

describe('browserLogin', () => {
  it('opens the portal with its port and state, and takes the token that comes back with that state', async () => {
    const run = start();
    const url = new URL(await run.openedOnce);
    expect(`${url.origin}${url.pathname}`).toBe('https://portal.example/cli/auth');
    expect(url.searchParams.get('port')).toBe(String(run.port));
    const state = url.searchParams.get('state')!;
    expect(state.length).toBeGreaterThanOrEqual(16);
    // The URL is printed too, for when no browser opens.
    expect(run.lines.join('\n')).toContain(url.toString());

    const page = await redirect(run.port, { token: 'jwt.fresh', username: 'dev', state });
    expect(page.status).toBe(200);
    expect(page.body).toContain('Signed in as @dev');
    await expect(run.pending).resolves.toEqual({ token: 'jwt.fresh', username: 'dev' });
  });

  it('refuses a token that arrives without its state, and keeps waiting for the real one', async () => {
    const run = start();
    const state = new URL(await run.openedOnce).searchParams.get('state')!;
    const forged = await redirect(run.port, { token: 'jwt.someone-else', state: 'not-the-state' });
    expect(forged.status).toBe(400);
    const real = await redirect(run.port, { token: 'jwt.mine', state });
    expect(real.status).toBe(200);
    await expect(run.pending).resolves.toMatchObject({ token: 'jwt.mine' });
  });

  it('ends the wait when the person denies in the browser', async () => {
    const run = start();
    const state = new URL(await run.openedOnce).searchParams.get('state')!;
    // Attached before the redirect, which is what rejects it.
    const outcome = expect(run.pending).rejects.toThrow('Sign-in was cancelled in the browser.');
    const page = await redirect(run.port, { error: 'access_denied', state });
    expect(page.body).toContain('Sign-in cancelled');
    await outcome;
  });

  it('ends the wait on Ctrl+C, and frees the port', async () => {
    const controller = new AbortController();
    const run = start({ signal: controller.signal });
    await run.openedOnce;
    controller.abort();
    await expect(run.pending).rejects.toThrow('Sign-in cancelled.');
    await expect(fetch(`http://127.0.0.1:${run.port}/callback`)).rejects.toThrow();
  });
});

describe('the page the browser lands on', () => {
  it('takes the token out of the address bar and sends no referrer', () => {
    const html = callbackPage('Signed in', 'Return to your terminal.', 'ok');
    expect(html).toContain('history.replaceState(null,"","/callback")');
    expect(html).toContain('<meta name="referrer" content="no-referrer">');
  });

  it('escapes what it prints', () => {
    const html = callbackPage('Signed in as @<img src=x onerror=alert(1)>', '"quoted" & \'single\'', 'ok');
    expect(html).not.toContain('<img');
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
    expect(html).toContain('&quot;quoted&quot; &amp; &#39;single&#39;');
  });
});
