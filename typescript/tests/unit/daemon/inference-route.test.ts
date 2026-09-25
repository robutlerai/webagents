/**
 * The daemon serves inference, and refuses it without a credential.
 *
 * Two defects in one place, both fixed 2026-09-23:
 *
 *   1. There was NO inference route at all. The daemon could list a locally
 *      registered agent and nothing could ever talk to it: no chat/completions,
 *      no streaming, no UAMP. A registry you cannot address is a directory.
 *   2. There was NO credential floor, unlike `server/node.ts:126-132`, and CORS
 *      ran with `cors()` defaults, i.e. `Access-Control-Allow-Origin: *`. That
 *      was survivable only while nothing billable was served. Adding the route
 *      without the floor would have turned a structural gap into a live one, so
 *      both land together. Logged as S-214.
 *
 * Driven through `app.fetch` rather than a real socket: the floor and the route
 * are both request-shaped, so nothing here needs a port.
 */

import { describe, it, expect } from 'vitest';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { BaseAgent } from '../../../src/core/agent';

function daemonWithAgent(name = 'probe') {
  const daemon = new WebAgentsDaemon({ port: 0, watch: false, cron: false });
  daemon.registry.registerLocal(
    new BaseAgent({ name, instructions: 'You are a probe.' }) as never,
  );
  return daemon;
}

const AUTHED = { 'content-type': 'application/json', authorization: 'Bearer test-token' };
const ANON = { 'content-type': 'application/json' };

function call(daemon: WebAgentsDaemon, path: string, init?: RequestInit) {
  return daemon.app.fetch(new Request(`http://localhost${path}`, init));
}

describe('daemon credential floor', () => {
  it('refuses anonymous inference with 401', async () => {
    const daemon = daemonWithAgent();
    const res = await call(daemon, '/agents/probe/chat/completions', {
      method: 'POST',
      headers: ANON,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });
    expect(res.status).toBe(401);
    expect((await res.json()).error.code).toBe('unauthorized');
  });

  it('refuses before reading the body, so a malformed anonymous request is still 401', async () => {
    // The floor decides from method and path alone. An anonymous caller must
    // not be able to make the daemon parse arbitrary bytes, and anonymous plus
    // malformed must be 401 rather than 400.
    const daemon = daemonWithAgent();
    const res = await call(daemon, '/agents/probe/chat/completions', {
      method: 'POST',
      headers: ANON,
      body: '{not json',
    });
    expect(res.status).toBe(401);
  });

  it('leaves non-billable routes open', async () => {
    const daemon = daemonWithAgent();
    expect((await call(daemon, '/health')).status).toBe(200);
    expect((await call(daemon, '/agents')).status).toBe(200);
  });
});

describe('daemon CORS', () => {
  it('does not echo an arbitrary origin', async () => {
    // `cors()` with defaults answered `*`, so any page open in the developer's
    // browser could enumerate and deregister their local agents.
    const daemon = daemonWithAgent();
    const res = await call(daemon, '/agents', { headers: { Origin: 'https://evil.example' } });
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });

  it('allows an origin the operator opted into', async () => {
    const daemon = new WebAgentsDaemon({
      port: 0,
      watch: false,
      cron: false,
      allowedOrigins: ['https://studio.example'],
    });
    const res = await call(daemon, '/agents', { headers: { Origin: 'https://studio.example' } });
    expect(res.headers.get('access-control-allow-origin')).toBe('https://studio.example');
  });
});

describe('daemon inference route', () => {
  it('404s an unknown agent', async () => {
    const daemon = daemonWithAgent();
    const res = await call(daemon, '/agents/nope/chat/completions', {
      method: 'POST',
      headers: AUTHED,
      body: JSON.stringify({ messages: [] }),
    });
    expect(res.status).toBe(404);
    expect((await res.json()).error).toBe('Agent not found');
  });

  it('distinguishes a remotely registered agent from a missing one', async () => {
    // A remote entry has a url and no instance. "Not found" would send the
    // caller looking for a registration that is right there.
    const daemon = daemonWithAgent();
    daemon.registry.registerRemote('far', 'https://far.example', {} as never);
    const res = await call(daemon, '/agents/far/chat/completions', {
      method: 'POST',
      headers: AUTHED,
      body: JSON.stringify({ messages: [] }),
    });
    expect(res.status).toBe(404);
    expect((await res.json()).error).toMatch(/registered remotely/);
  });

  it('rejects a malformed body and a non-array messages field separately', async () => {
    const daemon = daemonWithAgent();

    const badJson = await call(daemon, '/agents/probe/chat/completions', {
      method: 'POST',
      headers: AUTHED,
      body: '{not json',
    });
    expect(badJson.status).toBe(400);
    expect((await badJson.json()).error).toBe('Body must be JSON');

    const badMessages = await call(daemon, '/agents/probe/chat/completions', {
      method: 'POST',
      headers: AUTHED,
      body: JSON.stringify({ messages: 'nope' }),
    });
    expect(badMessages.status).toBe(400);
    expect((await badMessages.json()).error).toMatch(/must be an array/);
  });

  it('reaches the agent once authenticated', async () => {
    // The probe has no LLM skill, so running it fails: a 500 here is the agent
    // answering, which is the point. It proves the floor is not swallowing
    // authenticated traffic.
    const daemon = daemonWithAgent();
    const res = await call(daemon, '/agents/probe/chat/completions', {
      method: 'POST',
      headers: AUTHED,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });
    expect(res.status).toBe(500);
  });
});
