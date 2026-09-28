/**
 * Realtime in `serve()` (gap-closure plan item 1.5, 2026-09-26). Before this,
 * the TypeScript Realtime skill had only an `on_connection` hook that nothing
 * in `serve()` called with a socket (`/realtime` 404ed at the upgrade), and a
 * chat turn ran every `on_connection` hook with `{ metadata: {} }`. Now:
 *
 *  - a served agent answers a WebSocket at `/realtime` (behind the credential
 *    floor and the origin check) with `session.created`, as the Python server
 *    does through `@websocket("/realtime")`;
 *  - the session gets the caller's metadata and auth from the handshake, in
 *    the shape the portal's voice relay drives the skill with
 *    (`lib/voice/relay.ts`: `{ ws, metadata: { transport: 'realtime' } }`),
 *    which this test doubles;
 *  - `processUAMP` passes the run's own metadata to `on_connection` hooks.
 *
 * The path and the first event are pinned by
 * `python/tests/fixtures/acp/acp_protocol.json` (`realtime`).
 */

import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

import { BaseAgent } from '../../../src/core/agent';
import { handoff, hook } from '../../../src/core/decorators';
import { Skill } from '../../../src/core/skill';
import type { Context, HookData } from '../../../src/core/types';
import { serve } from '../../../src/server/node';
import { RealtimeTransportSkill } from '../../../src/skills/transport/realtime/skill';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REALTIME = (
  JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/acp/acp_protocol.json'), 'utf8')) as {
    realtime: { ws_path: string; first_event: string; hook_metadata: Record<string, string> };
  }
).realtime;

/** The widget side, as the relay's socket and the bridge tests present it. */
class WidgetFakeWS {
  sent: string[] = [];
  private listeners: Record<string, Array<(ev: unknown) => void>> = {};
  addEventListener(type: string, cb: (ev: unknown) => void) {
    (this.listeners[type] ||= []).push(cb);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    (this.listeners.close || []).forEach((cb) => cb({}));
  }
  json(): Array<Record<string, unknown>> {
    return this.sent.map((s) => JSON.parse(s) as Record<string, unknown>);
  }
}

describe('a served agent answers a Realtime session at /realtime', () => {
  it('opens the socket behind the floor and sends session.created', async () => {
    const skill = new RealtimeTransportSkill();
    const agent = new BaseAgent({ name: 'voice', instructions: 'x', skills: [skill] });
    expect(agent.listWebSocketEndpoints().map((e) => e.path)).toContain(REALTIME.ws_path);
    const handle = await serve(agent, { port: 0, hostname: '127.0.0.1', heartbeat: false, logging: false });
    try {
      const anonymous = new WebSocket(`ws://127.0.0.1:${handle.port}${REALTIME.ws_path}`);
      const refused = await new Promise<number>((resolve) => {
        anonymous.on('unexpected-response', (_req, res) => resolve(res.statusCode ?? 0));
        anonymous.on('error', () => undefined);
      });
      expect(refused).toBe(401);

      const ws = new WebSocket(`ws://127.0.0.1:${handle.port}${REALTIME.ws_path}?token=any-credential`);
      const first = await new Promise<Record<string, unknown>>((resolve, reject) => {
        ws.on('message', (data) => resolve(JSON.parse(data.toString()) as Record<string, unknown>));
        ws.on('error', reject);
      });
      expect(first.type).toBe(REALTIME.first_event);
      expect(first.session).toMatchObject({ id: expect.any(String), status: 'active' });
      ws.close();
    } finally {
      await handle.close();
    }
  }, 20_000);
});

describe('the session gets the caller, in the relay’s shape', () => {
  it('serveRealtime hands the upgrade context and its metadata to the hook', async () => {
    const skill = new RealtimeTransportSkill();
    const spy = vi.spyOn(skill, 'handleRealtimeConnection');
    const ws = new WidgetFakeWS();
    const context = {
      auth: { authenticated: true, user_id: 'u1' },
      metadata: { userAgent: 'zed', ip: '10.0.0.1', path: REALTIME.ws_path, method: 'GET' },
    } as unknown as Context;
    skill.serveRealtime(ws as unknown as WebSocket, context);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(spy).toHaveBeenCalledTimes(1);
    const [data, ctx] = spy.mock.calls[0];
    expect(ctx).toBe(context);
    expect(data.ws).toBe(ws);
    expect(data.metadata).toEqual({ userAgent: 'zed', ip: '10.0.0.1', method: 'GET', ...REALTIME.hook_metadata });
    expect(ws.json()[0]?.type).toBe(REALTIME.first_event);
    ws.close();
  });

  it('the relay’s own call shape still works: { ws, metadata: { transport } } and any context', async () => {
    const skill = new RealtimeTransportSkill();
    const ws = new WidgetFakeWS();
    await skill.handleRealtimeConnection({ ws, metadata: { transport: 'realtime' } } as unknown as HookData, {} as Context);
    expect(ws.json()[0]?.type).toBe(REALTIME.first_event);
    ws.close();
  });
});

describe('processUAMP passes the run’s metadata to on_connection hooks', () => {
  it('a hook sees what the transport put on the run, not {}', async () => {
    const seen: Array<Record<string, unknown> | undefined> = [];

    class Recorder extends Skill {
      @hook({ lifecycle: 'on_connection', priority: 50 })
      async record(data: HookData, _context: Context): Promise<void> {
        seen.push(data.metadata);
      }
    }

    class StubLLM extends Skill {
      @handoff({ name: 'stub-llm' })
      async *processUAMP(_events: ClientEvent[], _context: Context): AsyncGenerator<ServerEvent> {
        const id = generateEventId();
        yield { type: 'response.created', event_id: generateEventId(), response_id: id } as ServerEvent;
        yield createResponseDeltaEvent(id, { type: 'text', text: 'ok' });
        yield createResponseDoneEvent(id, [{ type: 'text', text: 'ok' }]);
      }
    }

    const agent = new BaseAgent({ name: 'meta', instructions: 'x', skills: [new StubLLM(), new Recorder()] });
    await agent.run([{ role: 'user', content: 'hi' }], { metadata: { transport: 'realtime', path: '/realtime' } });
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ transport: 'realtime', path: '/realtime' });
  });
});
