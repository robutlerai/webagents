/**
 * One line when a request arrives and one when it is answered (2026-09-25).
 *
 * The same lines in both SDKs (`python/webagents/server/core/request_log.py`):
 *
 *   <-- POST /chat/completions
 *   --> POST /chat/completions 200 15ms
 *
 * Hono's `logger()` printed these with the status in colour into pipes and
 * files as well, and the Python server printed nothing, so `webagents serve`
 * looked different in each. The status is coloured only on a terminal, and
 * never with NO_COLOR set.
 */

import type { MiddlewareHandler } from 'hono';

const STATUS_COLOURS: Record<number, number> = { 2: 32, 3: 36, 4: 33, 5: 31 };

function coloured(): boolean {
  return typeof process !== 'undefined' && Boolean(process.stdout?.isTTY) && !process.env.NO_COLOR;
}

/** The status, coloured by class on a terminal. */
export function statusText(status: number): string {
  const code = STATUS_COLOURS[Math.floor(status / 100)];
  return code && coloured() ? `\x1b[${code}m${status}\x1b[0m` : String(status);
}

/** Milliseconds under a second, whole seconds from one. */
export function elapsedText(ms: number): string {
  return ms < 1000 ? `${ms}ms` : `${Math.round(ms / 1000)}s`;
}

export function requestLog(print: (line: string) => void = (line) => console.log(line)): MiddlewareHandler {
  return async (c, next) => {
    const url = c.req.url;
    const path = url.slice(url.indexOf('/', url.indexOf('://') + 3));
    print(`<-- ${c.req.method} ${path}`);
    const started = Date.now();
    await next();
    print(`--> ${c.req.method} ${path} ${statusText(c.res.status)} ${elapsedText(Date.now() - started)}`);
  };
}
