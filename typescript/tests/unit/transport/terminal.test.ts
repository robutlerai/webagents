/**
 * TerminalRouter (`workspace.terminal` namespace) unit tests.
 *
 * Covers:
 *   - `isFor` envelope predicate
 *   - Host=unsupported on plain Node returns `not_supported`
 *   - `bad_payload` for malformed input
 *   - `duplicate_session` on second open with the same session_id
 *   - `unsupported_version` when extension_version > VERSION
 *   - `concurrency_limit` at the 9th open
 *   - Paused rolling buffer (overflow drops oldest, resume drains)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  MAX_SESSIONS_PER_ROUTER,
  NAMESPACE,
  PAUSED_BUFFER_BYTES,
  TerminalRouter,
  VERSION,
  type EnvelopeSender,
  type OutgoingPayload,
} from '../../../src/transport/terminal/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface TauriMock {
  invoke: ReturnType<typeof vi.fn>;
  events: {
    listen: ReturnType<typeof vi.fn>;
  };
  /** Re-emit a Tauri `pty:data` event for tests. */
  emitData(ptySid: string, data: string): void;
  /** Re-emit a Tauri `pty:exit` event for tests. */
  emitExit(ptySid: string, code: number | null, signal?: string | null): void;
}

function createTauriMock(opts: { openImpl?: () => Promise<string> } = {}): TauriMock {
  const dataListeners: Array<(ev: { payload: { ptySid: string; data: string } }) => void> = [];
  const exitListeners: Array<(ev: { payload: { ptySid: string; code: number | null; signal?: string | null } }) => void> = [];
  let nextPtySid = 0;

  const invoke = vi.fn(async (cmd: string, _args?: Record<string, unknown>) => {
    if (cmd === 'pty_open') {
      if (opts.openImpl) return opts.openImpl();
      return `pty_${nextPtySid++}`;
    }
    if (cmd === 'pty_write' || cmd === 'pty_resize' || cmd === 'pty_close') return undefined;
    return undefined;
  });

  const listen = vi.fn(async (event: string, handler: (ev: unknown) => void) => {
    if (event === 'pty:data') dataListeners.push(handler as never);
    else if (event === 'pty:exit') exitListeners.push(handler as never);
    return () => {
      const arr = event === 'pty:data' ? dataListeners : exitListeners;
      const i = arr.indexOf(handler as never);
      if (i >= 0) arr.splice(i, 1);
    };
  });

  return {
    invoke,
    events: { listen },
    emitData(ptySid, data) {
      for (const l of dataListeners) l({ payload: { ptySid, data } });
    },
    emitExit(ptySid, code, signal = null) {
      for (const l of exitListeners) l({ payload: { ptySid, code, signal } });
    },
  };
}

function captureSender(): { send: EnvelopeSender; out: OutgoingPayload[] } {
  const out: OutgoingPayload[] = [];
  return { send: (p) => out.push(p), out };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('TerminalRouter.isFor', () => {
  it('matches the workspace.terminal envelope', () => {
    expect(
      TerminalRouter.isFor({ type: 'extension.message', namespace: NAMESPACE }),
    ).toBe(true);
  });

  it('rejects other extension namespaces', () => {
    expect(
      TerminalRouter.isFor({ type: 'extension.message', namespace: 'workspace.vnc' }),
    ).toBe(false);
  });

  it('rejects non-extension UAMP frames', () => {
    expect(TerminalRouter.isFor({ type: 'session.create' })).toBe(false);
    expect(TerminalRouter.isFor({ type: 'response.delta' })).toBe(false);
  });

  it('rejects junk', () => {
    expect(TerminalRouter.isFor({})).toBe(false);
    expect(TerminalRouter.isFor({ type: 'extension.message' })).toBe(false);
  });
});

describe('TerminalRouter (host=unsupported)', () => {
  let router: TerminalRouter;
  beforeEach(() => {
    router = new TerminalRouter({ host: 'unsupported' });
  });

  it('returns not_supported on first open', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u1', cols: 80, rows: 24 },
      send,
    );
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ type: 'err', code: 'not_supported', session_id: 't1' });
  });

  it('does not import Tauri when host=unsupported', async () => {
    const importTauri = vi.fn();
    const r = new TerminalRouter({ host: 'unsupported', importTauri });
    const { send } = captureSender();
    await r.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    expect(importTauri).not.toHaveBeenCalled();
  });
});

describe('TerminalRouter payload validation', () => {
  let router: TerminalRouter;
  let tauri: TauriMock;
  beforeEach(() => {
    tauri = createTauriMock();
    router = new TerminalRouter({
      host: 'tauri',
      importTauri: async () => tauri,
    });
  });

  it('emits bad_payload for missing type', async () => {
    const { send, out } = captureSender();
    await router.handlePayload({ session_id: 't1' }, send);
    expect(out[0]).toMatchObject({ type: 'err', code: 'bad_payload' });
  });

  it('emits bad_payload for non-numeric cols on open', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 'eighty', rows: 24 },
      send,
    );
    expect(out[0]).toMatchObject({ type: 'err', code: 'bad_payload' });
  });

  it('emits bad_payload for missing session_id', async () => {
    const { send, out } = captureSender();
    await router.handlePayload({ type: 'in', data: 'aGk=' }, send);
    expect(out[0]).toMatchObject({ type: 'err', code: 'bad_payload' });
    // session_id falls back to '' when not extractable
    expect(out[0]).toMatchObject({ session_id: '' });
  });

  it('emits bad_payload for unknown type', async () => {
    const { send, out } = captureSender();
    await router.handlePayload({ type: 'totally-made-up', session_id: 't1' }, send);
    expect(out[0]).toMatchObject({ type: 'err', code: 'bad_payload' });
  });
});

describe('TerminalRouter version policy', () => {
  it('rejects extension_version > VERSION', async () => {
    const tauri = createTauriMock();
    const router = new TerminalRouter({
      host: 'tauri',
      importTauri: async () => tauri,
    });
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
      { extension_version: VERSION + 1 },
    );
    expect(out[0]).toMatchObject({ type: 'err', code: 'unsupported_version', session_id: 't1' });
  });

  it('accepts extension_version === VERSION', async () => {
    const tauri = createTauriMock();
    const router = new TerminalRouter({
      host: 'tauri',
      importTauri: async () => tauri,
    });
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
      { extension_version: VERSION },
    );
    expect(out.find((o) => o.type === 'ready')).toBeDefined();
  });
});

describe('TerminalRouter session lifecycle', () => {
  let tauri: TauriMock;
  let router: TerminalRouter;
  beforeEach(() => {
    tauri = createTauriMock();
    router = new TerminalRouter({
      host: 'tauri',
      importTauri: async () => tauri,
    });
  });

  it('sends ready after pty_open succeeds', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    expect(out).toEqual([{ type: 'ready', session_id: 't1' }]);
    expect(tauri.invoke).toHaveBeenCalledWith('pty_open', expect.any(Object));
  });

  it('rejects duplicate session_id', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    expect(out.filter((o) => o.type === 'ready')).toHaveLength(1);
    expect(out.find((o) => o.type === 'err')).toMatchObject({
      type: 'err',
      code: 'duplicate_session',
    });
  });

  it('emits concurrency_limit on the 9th open', async () => {
    const { send, out } = captureSender();
    for (let i = 0; i < MAX_SESSIONS_PER_ROUTER; i++) {
      await router.handlePayload(
        { type: 'open', session_id: `t${i}`, peer_id: 'u', cols: 80, rows: 24 },
        send,
      );
    }
    out.length = 0;
    await router.handlePayload(
      { type: 'open', session_id: 'overflow', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    expect(out[0]).toMatchObject({
      type: 'err',
      code: 'concurrency_limit',
      session_id: 'overflow',
    });
  });

  it('emits pty_open_failed when invoke throws', async () => {
    const failing = createTauriMock({
      openImpl: () => Promise.reject(new Error('osspawn EAGAIN')),
    });
    const r = new TerminalRouter({ host: 'tauri', importTauri: async () => failing });
    const { send, out } = captureSender();
    await r.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    expect(out[0]).toMatchObject({
      type: 'err',
      code: 'pty_open_failed',
      session_id: 't1',
    });
  });

  it('forwards out frames from pty:data', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    out.length = 0;
    tauri.emitData('pty_0', 'aGVsbG8=');
    expect(out[0]).toEqual({ type: 'out', session_id: 't1', data: 'aGVsbG8=' });
  });

  it('emits exit on pty:exit and removes the session', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    out.length = 0;
    tauri.emitExit('pty_0', 0);
    expect(out[0]).toMatchObject({ type: 'exit', session_id: 't1', code: 0 });
    // Session is gone — closing again is a no-op (and importantly does not
    // throw). We assert by closing and observing no `err` frame.
    out.length = 0;
    await router.handlePayload({ type: 'close', session_id: 't1' }, send);
    expect(out).toEqual([]);
  });
});

describe('TerminalRouter pause/resume buffer', () => {
  let tauri: TauriMock;
  let router: TerminalRouter;
  beforeEach(() => {
    tauri = createTauriMock();
    router = new TerminalRouter({ host: 'tauri', importTauri: async () => tauri });
  });

  it('drops nothing while under cap, drains on resume in order', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    await router.handlePayload({ type: 'pause', session_id: 't1' }, send);
    out.length = 0;
    tauri.emitData('pty_0', 'AAA=');
    tauri.emitData('pty_0', 'BBB=');
    expect(out).toEqual([]);
    await router.handlePayload({ type: 'resume', session_id: 't1' }, send);
    expect(out).toEqual([
      { type: 'out', session_id: 't1', data: 'AAA=' },
      { type: 'out', session_id: 't1', data: 'BBB=' },
    ]);
  });

  it('rolls oldest off when paused buffer exceeds cap', async () => {
    const { send } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    await router.handlePayload({ type: 'pause', session_id: 't1' }, send);
    // Push enough chunks to exceed PAUSED_BUFFER_BYTES (each chunk ~64 KiB)
    const chunkSize = 64 * 1024;
    const chunk = 'A'.repeat(chunkSize);
    const total = Math.ceil(PAUSED_BUFFER_BYTES / chunkSize) + 4;
    for (let i = 0; i < total; i++) {
      tauri.emitData('pty_0', chunk);
    }
    expect(router._peekPausedBufferBytes('t1')).toBeLessThanOrEqual(
      PAUSED_BUFFER_BYTES + chunkSize,
    );
    expect(router._peekPausedBufferBytes('t1')).toBeGreaterThan(0);
  });

  it('ignores data for sessions other than the matching ptySid', async () => {
    const { send, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    out.length = 0;
    tauri.emitData('some_other_pty', 'XXXX');
    expect(out).toEqual([]);
  });
});

describe('TerminalRouter shutdown', () => {
  it('closes every open session and is idempotent', async () => {
    const tauri = createTauriMock();
    const router = new TerminalRouter({
      host: 'tauri',
      importTauri: async () => tauri,
    });
    const { send } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't1', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    await router.handlePayload(
      { type: 'open', session_id: 't2', peer_id: 'u', cols: 80, rows: 24 },
      send,
    );
    expect(router._peekSessionCount()).toBe(2);
    await router.shutdown('test');
    expect(router._peekSessionCount()).toBe(0);
    expect(tauri.invoke).toHaveBeenCalledWith('pty_close', expect.any(Object));
    // Idempotent
    await router.shutdown('again');
    // Post-shutdown payloads are dropped silently
    const { send: send2, out } = captureSender();
    await router.handlePayload(
      { type: 'open', session_id: 't3', peer_id: 'u', cols: 80, rows: 24 },
      send2,
    );
    expect(out).toEqual([]);
  });
});
