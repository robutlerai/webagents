/**
 * Phase 5b: the TypeScript CLI is a client of the daemon, not a second
 * implementation of it (2026-09-23).
 *
 * The contract these pin is small but load-bearing:
 *
 *   * the paths match `webagentsd`, which builds its router with
 *     `url_prefix="/agents"`. Getting this wrong is invisible until someone
 *     runs the Python daemon, which is the one that ships.
 *   * "not running" and "said no" are different errors with different next
 *     steps, and a CLI that conflates them sends people to the wrong place.
 *
 * The paths below were MEASURED against a running `webagents daemon start`
 * (2026-09-23), not inferred from the source:
 *
 *     GET /health        200
 *     GET /agents/       200  {"agents": [...], "count": n}
 *     GET /agents        307  -> /agents/
 *     GET /agents/cron   200  {"jobs": []}
 *     GET /cron          404
 *
 * The 307 is why the client sends the trailing slash, and the 404 is why the
 * TypeScript daemon moved its cron routes under /agents.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  DaemonClient,
  DaemonRequestError,
  DaemonUnreachableError,
} from '../../../src/cli/daemon-client.js';

const calls: Array<{ url: string; method: string }> = [];
let respond: (url: string, method: string) => Response;

beforeEach(() => {
  calls.length = 0;
  respond = () => new Response('{}', { status: 200 });
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    calls.push({ url: String(input), method });
    return respond(String(input), method);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('the routes match webagentsd', () => {
  it('lists agents at /agents/ with the trailing slash', async () => {
    // FastAPI registers the router's `/` under the prefix, so the path is
    // `/agents/`. Without the slash it answers 307, and a redirect is a
    // needless way to lose a method.
    respond = () => new Response('[]', { status: 200 });
    await new DaemonClient({ port: 8765 }).listAgents();

    expect(calls[0].url).toBe('http://127.0.0.1:8765/agents/');
  });

  it('reads cron from /agents/cron, not /cron', async () => {
    respond = () => new Response('{"jobs": []}', { status: 200 });
    await new DaemonClient().listCronJobs();

    expect(calls[0].url).toContain('/agents/cron');
  });

  it('encodes an agent name rather than pasting it into the path', async () => {
    await new DaemonClient().getAgent('team/bot');

    expect(calls[0].url).toBe('http://127.0.0.1:8765/agents/team%2Fbot');
  });

  it('honours daemon.host and daemon.port', () => {
    expect(new DaemonClient({ host: 'localhost', port: 9999 }).baseUrl).toBe(
      'http://localhost:9999',
    );
  });
});

describe('the two shapes of failure stay apart', () => {
  it('reports an unreachable daemon with the command that starts one', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('fetch failed');
    });

    await expect(new DaemonClient().health()).rejects.toThrow(DaemonUnreachableError);
    await expect(new DaemonClient().health()).rejects.toThrow(/webagents daemon/);
  });

  it('reports a refusal from a daemon that IS running as a request error', async () => {
    respond = () => new Response('no such agent', { status: 404 });

    const error = await new DaemonClient().getAgent('nope').catch((e) => e);
    expect(error).toBeInstanceOf(DaemonRequestError);
    expect(error.status).toBe(404);
  });

  it('isRunning answers false instead of throwing', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('fetch failed');
    });

    await expect(new DaemonClient().isRunning()).resolves.toBe(false);
  });
});

describe('response shapes', () => {
  it('accepts a bare array of agents', async () => {
    respond = () => new Response('[{"name": "a"}]', { status: 200 });
    await expect(new DaemonClient().listAgents()).resolves.toEqual([{ name: 'a' }]);
  });

  it('accepts an {agents: [...]} envelope', async () => {
    // The two daemons do not agree, and the client should not care.
    respond = () => new Response('{"agents": [{"name": "b"}]}', { status: 200 });
    await expect(new DaemonClient().listAgents()).resolves.toEqual([{ name: 'b' }]);
  });

  it('treats an unparseable 200 as a request error rather than returning junk', async () => {
    respond = () => new Response('<html>proxy</html>', { status: 200 });
    await expect(new DaemonClient().listAgents()).rejects.toThrow(DaemonRequestError);
  });
});
