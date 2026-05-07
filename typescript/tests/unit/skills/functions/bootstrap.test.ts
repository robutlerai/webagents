/**
 * Pure-JS unit tests for `js-v1-bootstrap.ts`.
 *
 * The bootstrap is normally executed inside an `isolated-vm` context;
 * here we run it inside a Node `vm.runInNewContext` sandbox with stub
 * `__hostXxx` references. That gives us a fast, dependency-free
 * smoke check that the pure-JS classes (Headers, Request, Response,
 * AbortController, AbortSignal, EventTarget, eval/Function blockers)
 * load and behave to spec.
 *
 * URLPattern + crypto.subtle exercise host-bridge `Reference` objects
 * which we can't model under plain `vm`, so they're skipped here and
 * covered by the integration test with real `isolated-vm`.
 */

import { describe, it, expect } from 'vitest';
import vm from 'node:vm';
import { ISOLATE_BOOTSTRAP } from '../../../../src/executor/runtimes/js-v1-bootstrap';

interface Sandbox {
  // In `vm.createContext`, the sandbox object IS `globalThis`, so
  // bootstrap-installed bindings are reachable as direct properties.
  eval: (src?: string) => unknown;
  Function: new (...a: unknown[]) => unknown;
  Event: new (type: string, init?: { bubbles?: boolean; cancelable?: boolean }) => {
    type: string;
    defaultPrevented: boolean;
  };
  EventTarget: new () => {
    addEventListener(type: string, l: (e: unknown) => void, opts?: unknown): void;
    removeEventListener(type: string, l: (e: unknown) => void, opts?: unknown): void;
    dispatchEvent(e: { type: string }): boolean;
  };
  AbortSignal: {
    new (): { aborted: boolean; reason: unknown; addEventListener(t: string, l: (e: unknown) => void): void };
    abort(reason?: unknown): { aborted: boolean; reason: unknown };
  };
  AbortController: new () => { signal: { aborted: boolean; reason: unknown }; abort(reason?: unknown): void };
  Headers: new (init?: unknown) => {
    append(k: string, v: string): void;
    get(k: string): string | null;
    has(k: string): boolean;
    set(k: string, v: string): void;
    delete(k: string): void;
    forEach(cb: (v: string, k: string) => void): void;
    getSetCookie(): string[];
  };
  Request: new (input: unknown, init?: unknown) => {
    url: string;
    method: string;
    headers: { get(k: string): string | null };
    text(): Promise<string>;
    json(): Promise<unknown>;
    clone(): unknown;
    bodyUsed: boolean;
  };
  Response: {
    new (body?: unknown, init?: unknown): {
      status: number;
      ok: boolean;
      headers: { get(k: string): string | null };
      text(): Promise<string>;
      json(): Promise<unknown>;
    };
    json(data: unknown, init?: unknown): { status: number; headers: { get(k: string): string | null }; text(): Promise<string> };
    error(): { type: string; status: number };
    redirect(url: string, status?: number): { status: number; headers: { get(k: string): string | null } };
  };
  console: { log(...a: unknown[]): void };
  setTimeout: typeof setTimeout;
}

function makeSandbox(): Sandbox {
  // Stub `__hostXxx` references so the bootstrap script doesn't crash
  // when it tries to wire console / crypto / structuredClone. The
  // tests below don't exercise those paths.
  const noopRef = {
    apply: () => undefined,
    applySync: () => undefined,
    applyIgnored: () => undefined,
  };
  const sandbox = {
    __hostConsoleDebug: noopRef,
    __hostConsoleInfo: noopRef,
    __hostConsoleWarn: noopRef,
    __hostConsoleError: noopRef,
    __hostRandomUUID: noopRef,
    __hostGetRandomValues: noopRef,
    __hostSubtle: noopRef,
    __hostStructuredClone: noopRef,
    __hostUrlPatternCreate: noopRef,
    __hostUrlPatternTest: noopRef,
    __hostUrlPatternExec: noopRef,
    TextEncoder,
    TextDecoder,
    URL,
    URLSearchParams,
    setTimeout,
    clearTimeout,
    Promise,
  } as Record<string, unknown>;
  vm.createContext(sandbox);
  vm.runInContext(ISOLATE_BOOTSTRAP, sandbox);
  return sandbox as unknown as Sandbox;
}

describe('ISOLATE_BOOTSTRAP — pure-JS Web Platform classes', () => {
  it('blocks eval and Function constructor', () => {
    const sb = makeSandbox();
    expect(() => sb.eval('1+1')).toThrow(/EVAL_DENIED/);
    expect(() => new sb.Function('return 1')).toThrow(/FUNCTION_DENIED/);
  });

  describe('Headers', () => {
    it('case-insensitively gets/sets/has/deletes', () => {
      const sb = makeSandbox();
      const h = new sb.Headers();
      h.set('Content-Type', 'application/json');
      expect(h.get('content-type')).toBe('application/json');
      expect(h.has('CONTENT-TYPE')).toBe(true);
      h.delete('content-type');
      expect(h.get('content-type')).toBe(null);
    });

    it('appends and combines values; preserves Set-Cookie list', () => {
      const sb = makeSandbox();
      const h = new sb.Headers();
      h.append('x-test', 'a');
      h.append('x-test', 'b');
      expect(h.get('x-test')).toBe('a, b');
      h.append('Set-Cookie', 'foo=1');
      h.append('Set-Cookie', 'bar=2');
      expect(h.getSetCookie()).toEqual(['foo=1', 'bar=2']);
    });

    it('accepts plain object init', () => {
      const sb = makeSandbox();
      const h = new sb.Headers({ Authorization: 'Bearer x' });
      expect(h.get('authorization')).toBe('Bearer x');
    });

    it('iterates entries sorted by lowercased key', () => {
      const sb = makeSandbox();
      const h = new sb.Headers();
      h.set('B', '2');
      h.set('a', '1');
      const entries: string[] = [];
      h.forEach((v, k) => entries.push(`${k}=${v}`));
      expect(entries).toEqual(['a=1', 'b=2']);
    });
  });

  describe('AbortController / AbortSignal / EventTarget', () => {
    it('signal starts un-aborted; abort() flips and dispatches event', () => {
      const sb = makeSandbox();
      const ac = new sb.AbortController();
      expect(ac.signal.aborted).toBe(false);
      let fired = 0;
      ac.signal.addEventListener('abort', () => { fired++; });
      ac.abort('boom');
      expect(ac.signal.aborted).toBe(true);
      expect(ac.signal.reason).toBe('boom');
      expect(fired).toBe(1);
      ac.abort('again');
      expect(fired).toBe(1);
    });

    it('AbortSignal.abort returns a pre-aborted signal', () => {
      const sb = makeSandbox();
      const s = sb.AbortSignal.abort('preset');
      expect(s.aborted).toBe(true);
      expect(s.reason).toBe('preset');
    });

    it('EventTarget honours { once: true } and removeEventListener', () => {
      const sb = makeSandbox();
      const t = new sb.EventTarget();
      let count = 0;
      const onTick = () => { count++; };
      t.addEventListener('tick', onTick);
      t.dispatchEvent(new sb.Event('tick'));
      t.dispatchEvent(new sb.Event('tick'));
      expect(count).toBe(2);
      t.removeEventListener('tick', onTick);
      t.dispatchEvent(new sb.Event('tick'));
      expect(count).toBe(2);

      let onceCount = 0;
      t.addEventListener('one', () => { onceCount++; }, { once: true });
      t.dispatchEvent(new sb.Event('one'));
      t.dispatchEvent(new sb.Event('one'));
      expect(onceCount).toBe(1);
    });
  });

  describe('Request', () => {
    it('captures url, method, headers; body round-trips via text()', async () => {
      const sb = makeSandbox();
      const r = new sb.Request('https://example.com/api', {
        method: 'POST',
        headers: { 'X-Trace': 'abc' },
        body: 'hello',
      });
      expect(r.url).toBe('https://example.com/api');
      expect(r.method).toBe('POST');
      expect(r.headers.get('x-trace')).toBe('abc');
      expect(r.headers.get('content-type')).toMatch(/text\/plain/i);
      expect(await r.text()).toBe('hello');
      expect(r.bodyUsed).toBe(true);
    });

    it('serialises object body as JSON with content-type set', async () => {
      const sb = makeSandbox();
      const r = new sb.Request('https://example.com', { method: 'POST', body: { x: 1 } });
      expect(r.headers.get('content-type')).toMatch(/application\/json/i);
      expect(await r.json()).toEqual({ x: 1 });
    });

    it('clone() preserves url + method + headers', () => {
      // Body round-trip in clone() depends on `instanceof ArrayBuffer`
      // matching across realms, which doesn't hold under node:vm
      // (host TextEncoder produces host ArrayBuffers). Real isolated-vm
      // runs everything in one realm — covered by the integration test.
      const sb = makeSandbox();
      const r = new sb.Request('https://example.com', {
        method: 'PUT',
        headers: { 'X-Token': 'abc' },
      });
      const c = r.clone() as unknown as {
        url: string;
        method: string;
        headers: { get(k: string): string | null };
      };
      expect(c.url).toBe('https://example.com');
      expect(c.method).toBe('PUT');
      expect(c.headers.get('x-token')).toBe('abc');
    });
  });

  describe('Response', () => {
    it('default status is 200, ok is true', async () => {
      const sb = makeSandbox();
      const r = new sb.Response('hi');
      expect(r.status).toBe(200);
      expect(r.ok).toBe(true);
      expect(await r.text()).toBe('hi');
    });

    it('Response.json sets the content-type and serialises', async () => {
      const sb = makeSandbox();
      const r = sb.Response.json({ a: 1 });
      expect(r.headers.get('content-type')).toMatch(/application\/json/i);
      expect(await r.text()).toBe('{"a":1}');
    });

    it('Response.error and Response.redirect produce expected shapes', () => {
      const sb = makeSandbox();
      const e = sb.Response.error();
      expect(e.type).toBe('error');
      expect(e.status).toBe(0);
      const r = sb.Response.redirect('https://example.com/foo', 308);
      expect(r.status).toBe(308);
      expect(r.headers.get('location')).toBe('https://example.com/foo');
    });
  });
});
