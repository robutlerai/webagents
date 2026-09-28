/**
 * S-284 (2026-09-26): the TypeScript daemon bound `0.0.0.0` by default, and
 * `POST /agents/register` and `DELETE /agents/:name` took no credential, so
 * anyone on the network could deregister the owner's agents or add a remote
 * agent to the registry. The credential floor gated only the billable route.
 *
 * Fixed to match the Python daemon: bind `127.0.0.1` by default, and require
 * on register and remove the credential the floor already demands for
 * `chat/completions` (`hasCredential`). Python's active daemon has no register
 * route, so this is a TypeScript-only parity fix; there is no shared fixture.
 *
 * Driven through `app.fetch`, like `inference-route.test.ts`: the gate is
 * request-shaped, so nothing here needs a port.
 */

import { describe, expect, it } from 'vitest';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { BaseAgent } from '../../../src/core/agent';

const AUTHED = { 'content-type': 'application/json', authorization: 'Bearer test-token' };
const ANON = { 'content-type': 'application/json' };

function daemon(): WebAgentsDaemon {
  return new WebAgentsDaemon({ port: 0, watch: false, cron: false, healthChecks: false });
}

function call(d: WebAgentsDaemon, path: string, init?: RequestInit) {
  return d.app.fetch(new Request(`http://localhost${path}`, init));
}

const REMOTE = JSON.stringify({ name: 'far', url: 'https://far.example', capabilities: {} });

describe('S-284: the daemon binds loopback by default', () => {
  it('defaults hostname to 127.0.0.1, not 0.0.0.0', () => {
    const hostname = (daemon() as unknown as { config: { hostname: string } }).config.hostname;
    expect(hostname).toBe('127.0.0.1');
  });

  it('still honours an explicit hostname the operator opts into', () => {
    const d = new WebAgentsDaemon({ port: 0, hostname: '0.0.0.0', watch: false, cron: false, healthChecks: false });
    expect((d as unknown as { config: { hostname: string } }).config.hostname).toBe('0.0.0.0');
  });
});

describe('S-284: register and remove require the floor credential', () => {
  it('refuses anonymous register with 401', async () => {
    const res = await call(daemon(), '/agents/register', { method: 'POST', headers: ANON, body: REMOTE });
    expect(res.status).toBe(401);
    expect((await res.json()).error.code).toBe('unauthorized');
  });

  it('refuses anonymous register before reading the body', async () => {
    const res = await call(daemon(), '/agents/register', { method: 'POST', headers: ANON, body: '{not json' });
    expect(res.status).toBe(401);
  });

  it('refuses anonymous remove with 401, and does not remove', async () => {
    const d = daemon();
    d.registry.registerLocal(new BaseAgent({ name: 'keep', instructions: 'x' }) as never);
    const res = await call(d, '/agents/keep', { method: 'DELETE', headers: ANON });
    expect(res.status).toBe(401);
    expect(d.registry.get('keep')).toBeTruthy();
  });

  it('registers a remote agent with a credential', async () => {
    const d = daemon();
    const res = await call(d, '/agents/register', { method: 'POST', headers: AUTHED, body: REMOTE });
    expect(res.status).toBe(200);
    expect((await res.json()).success).toBe(true);
    expect(d.registry.get('far')).toBeTruthy();
  });

  it('removes an agent with a credential', async () => {
    const d = daemon();
    d.registry.registerLocal(new BaseAgent({ name: 'goner', instructions: 'x' }) as never);
    const res = await call(d, '/agents/goner', { method: 'DELETE', headers: AUTHED });
    expect(res.status).toBe(200);
    expect((await res.json()).success).toBe(true);
    expect(d.registry.get('goner')).toBeUndefined();
  });

  it('a bare Bearer with nothing after it is not a credential', async () => {
    const res = await call(daemon(), '/agents/register', {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer' },
      body: REMOTE,
    });
    expect(res.status).toBe(401);
  });

  it('404s a missing agent once authenticated, not 401', async () => {
    const res = await call(daemon(), '/agents/nope', { method: 'DELETE', headers: AUTHED });
    expect(res.status).toBe(404);
  });
});

describe('S-284: the control-plane reads stay open', () => {
  it('leaves health, the agent list and the schedule list anonymous', async () => {
    const d = daemon();
    expect((await call(d, '/health')).status).toBe(200);
    expect((await call(d, '/agents')).status).toBe(200);
    expect((await call(d, '/agents/cron')).status).toBe(200);
  });
});
