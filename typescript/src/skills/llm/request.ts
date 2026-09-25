/**
 * The request to a model provider, and what its failure says.
 *
 * Node's `fetch` rejects with the bare message "fetch failed" and keeps the
 * reason (a refused connection, an unknown host, a certificate) in `cause`,
 * so a mistyped `OPENAI_BASE_URL` reached the person as "Error: fetch failed"
 * and nothing else (found walking a first run, 2026-09-24). The failure now
 * names the server and the reason. Only the origin is named: a path or query
 * can carry a key on some gateways.
 *
 * FOR THE TERMINAL ONLY. The CLI switches the detail on for a chat or a `-p`
 * run (setRequestErrorDetail); nothing else does. A served agent answers a
 * failed run with a fixed sentence and a reference now, keeping the message
 * for its own log (S-228, `server/error-reply.ts`), but the upstream host and
 * address still have no business in a reply, and the log line says what
 * failed without them.
 */

let detailed = false;

/** On for a CLI chat or `-p` run, where the person at the terminal reads the error. */
export function setRequestErrorDetail(on: boolean): void {
  detailed = on;
}

/** The reason a request failed, from `error.cause` when fetch put it there. */
export function describeRequestError(error: unknown, url?: string): string {
  const err = error as (Error & { cause?: unknown }) | undefined;
  const message = err?.message || String(error);
  const cause = err?.cause as (Error & { code?: string; errors?: Array<{ message?: string }> }) | undefined;
  // A refused dual-stack connection is an AggregateError with an empty message.
  const reason = (cause && (cause.message || cause.errors?.[0]?.message || cause.code)) || message;
  let origin = '';
  try {
    origin = url ? new URL(url).origin : '';
  } catch {
    origin = '';
  }
  return origin ? `Could not reach ${origin}: ${reason}` : reason;
}

/** `fetch` for a model request; at the terminal, its failure says what failed (see above). */
export async function fetchModel(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (error) {
    // A cancelled request stays what it is; the caller knows it cancelled.
    if (!detailed || (error as Error)?.name === 'AbortError') throw error;
    throw new Error(describeRequestError(error, url), { cause: error });
  }
}
