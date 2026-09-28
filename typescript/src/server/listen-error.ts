/**
 * A port that is already in use, said in one sentence (2026-09-26, the
 * new-developer e2e run): `webagents serve --port <busy>` died with Node's
 * unhandled `'error'` event, a stack trace and `Node.js v24.7.0`, because
 * nothing listened for the server's `'error'`. The three listeners (`serve()`,
 * the daemon, `serveMcpHttp`) now turn `EADDRINUSE` into a `PortInUseError`
 * whose message is the one sentence the Python CLI prints too
 * (`cli/listen.py`), pinned by `python/tests/fixtures/cli/listen.json`; the
 * CLI prints it and exits 1 (`cli/index.ts`, `orListenError`).
 */

/** The sentence, with `{port}` and `{host}` (the fixture's `port_in_use`). */
export const PORT_IN_USE = 'Port {port} on {host} is already in use. Stop the program using it, or pass --port with a free one.';

export class PortInUseError extends Error {
  readonly code = 'EADDRINUSE';
  constructor(
    readonly host: string,
    readonly port: number,
  ) {
    super(PORT_IN_USE.replace('{port}', String(port)).replace('{host}', host));
    this.name = 'PortInUseError';
  }
}

/** `PortInUseError` for a listen error that is `EADDRINUSE`; any other error as it is. */
export function listenError(error: unknown, host: string, port: number): unknown {
  if (error instanceof PortInUseError) return error;
  if ((error as { code?: unknown } | null)?.code === 'EADDRINUSE') return new PortInUseError(host, port);
  return error;
}
