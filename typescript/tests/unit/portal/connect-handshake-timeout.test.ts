/**
 * A portal that accepts the TCP connection and never finishes the WebSocket
 * handshake must not hold the bridge's reconnect loop for ever. Before
 * `handshakeTimeoutMs` the attempt had no bound: no `open`, no `error`, no
 * `close`, so `runOneConnection` never resolved, and an agent whose process
 * looked healthy stayed silently offline after "reconnecting (attempt 1)"
 * (observed for 37 hours, 2026-10-19 to 2026-10-20). With the bound, each
 * stalled attempt ends as a socket error and the loop goes on counting, so a
 * finite budget is spent and reported, and an infinite one keeps trying.
 */

import { describe, it, expect, vi } from 'vitest';
import { createServer } from 'http';
import { runPortalBridge } from '../../../src/portal/connect';
import type { IAgent } from '../../../src/core/types';

function fakeAgentToken(): string {
  const seg = (obj: Record<string, unknown>) =>
    Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1', agent_id: 'agent-1' })}.x`;
}

const agent = {
  name: 'stalled',
  getCapabilities: () => ({}),
  run: async () => ({ content: 'ok' }),
  runStreaming: async function* () {
    yield { type: 'delta' as const, delta: 'ok' };
  },
} as unknown as IAgent;

describe('runPortalBridge: a handshake that never completes', () => {
  it('times out each attempt, keeps counting, and gives up after the budget', async () => {
    // Accept the upgrade request and never answer it: the socket stays half open.
    const held: import('net').Socket[] = [];
    const server = createServer((_req, res) => { res.statusCode = 404; res.end(); });
    server.on('upgrade', (_req, socket) => { held.push(socket); });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
    const port = (server.address() as { port: number }).port;

    const errors: string[] = [];
    const warns: string[] = [];
    const errSpy = vi.spyOn(console, 'error').mockImplementation((...a: unknown[]) => { errors.push(a.map(String).join(' ')); });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation((...a: unknown[]) => { warns.push(a.map(String).join(' ')); });
    try {
      const t0 = Date.now();
      await Promise.race([
        runPortalBridge(agent, {
          portalUrl: `http://127.0.0.1:${port}`,
          token: fakeAgentToken(),
          reconnectDelayS: 0.05,
          maxReconnectAttempts: 2,
          handshakeTimeoutMs: 200,
        }),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error('the bridge never gave up: the stalled handshake held the loop')), 5000)),
      ]);
      expect(Date.now() - t0).toBeLessThan(5000);
      // Three attempts (the first plus two reconnects), each ended by the timeout.
      expect(errors.filter((e) => /handshake has timed out/i.test(e)).length).toBe(3);
      expect(warns.filter((w) => /reconnecting \(attempt (1|2)\)/.test(w)).length).toBe(2);
      expect(errors.some((e) => /giving up after 2 reconnect attempts/.test(e))).toBe(true);
    } finally {
      errSpy.mockRestore();
      warnSpy.mockRestore();
      for (const s of held) s.destroy();
      await new Promise<void>((r) => server.close(() => r()));
    }
  });
});
