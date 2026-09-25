/**
 * The machine-readable output contract, TypeScript side.
 *
 * Mirrors `python/webagents/cli/output.py` EXACTLY, because the whole point of
 * one command surface across two SDKs is that a script does not have to know
 * which one it is talking to. If these two ever drift, the contract is worth
 * less than no contract at all.
 *
 * When `--json` is passed:
 *
 *   * stdout carries EXACTLY ONE JSON document and nothing else: no banners,
 *     no progress, no tables, no colour.
 *   * every diagnostic goes to stderr, where a pipeline can ignore or capture
 *     it separately.
 *   * a failure still prints one JSON document, with `ok: false` and an
 *     `error` object, and exits non-zero. A script must never have to parse a
 *     stack trace to find out what happened.
 *
 * The envelope:
 *
 *     {"ok": true,  "data": {...}}
 *     {"ok": false, "error": {"code": "...", "message": "...", "fix": "..."}}
 *
 * `fix` carries the same next-step text a human would have been shown. An
 * agent reading this output is exactly the audience that benefits from being
 * told what to do rather than only what broke.
 */

/** Whether this invocation asked for machine-readable output. */
export function jsonEnabled(program: { opts(): Record<string, unknown> }): boolean {
  return Boolean(program.opts().json);
}

/** Print the success envelope. The ONLY thing written to stdout. */
export function emit(data: unknown): void {
  process.stdout.write(`${JSON.stringify({ ok: true, data }, null, 2)}\n`);
}

/** Print the error envelope and exit non-zero. Always ONE document. */
export function fail(code: string, message: string, fix = '', exitCode = 1): never {
  const error: Record<string, string> = { code, message };
  if (fix) error.fix = fix;
  process.stdout.write(`${JSON.stringify({ ok: false, error }, null, 2)}\n`);
  process.exit(exitCode);
}

/** A human-facing aside that must never pollute stdout. */
export function note(message: string): void {
  process.stderr.write(`${message}\n`);
}
