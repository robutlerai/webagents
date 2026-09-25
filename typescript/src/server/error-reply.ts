/**
 * What a caller is told when a run fails, and what the log keeps
 * (2026-09-24, S-228).
 *
 * THE LEAK. The servers answered a failed run with the error's own message:
 * `{ error: { code: 'completions_error', message: (error as Error).message } }`
 * from `handler.ts`, the same for `uamp_error` and `handler_error` there and in
 * `node.ts`. That message is whatever the failing code put in it. The LLM
 * skills throw `<Provider> API returned <status>: <first 200 chars of the
 * body>` (`skills/llm/openai/skill.ts`), and OpenAI's 401 body quotes a masked
 * form of the rejected key; a tool's error carries paths on the agent's host.
 * Anyone who could make a served agent's run fail could read it.
 *
 * THE REPLY NOW is the portal's S-028 fix, and the Python SDK's
 * (`python/webagents/server/core/error_reply.py`): the full error goes to the
 * server's log under a short reference, and the caller gets a fixed sentence
 * carrying the same reference. Status codes and body shapes are unchanged; only
 * the `message` is.
 *
 * WHAT KEEPS ITS TEXT: errors thrown to be shown, which are this SDK's auth
 * refusals and `PaymentRequiredError` (matched by name, as `handler.ts`
 * already matches auth errors, so this module adds no import edge into the
 * skills). And, when a caller passes `{ detail: true }`, everything: only the
 * daemon does that, and only while it listens on loopback, where its caller is
 * the developer's own CLI.
 */

/** What a caller reads when a run fails, ahead of the reference. */
export const INTERNAL_ERROR_MESSAGE = 'The agent could not complete this request.';

/** This SDK's refusals, whose messages are written for the caller. */
const SHOWN = new Set(['AuthenticationError', 'AuthorizationError', 'PaymentRequiredError']);

/** Eight hex characters: short enough to read out, unique enough to grep for. */
export function newReference(): string {
  const bytes = new Uint8Array(4);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Whether `error` was thrown to be shown to the caller (see the file comment). */
export function isMeantToBeShown(error: unknown): boolean {
  const name = (error as { name?: unknown } | null)?.name;
  return typeof name === 'string' && SHOWN.has(name);
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * What the caller reads for a failed run: the error's own message when it is
 * meant to be shown or `detail` is on, otherwise the fixed sentence and a fresh
 * reference. Every failure that is not meant to be shown is logged, stack
 * included, under that reference, whichever of the two the caller gets.
 */
export function replyText(error: unknown, where: string, options: { detail?: boolean } = {}): string {
  if (isMeantToBeShown(error)) return messageOf(error);
  const reference = newReference();
  console.error(`[webagents] ${where} failed [ref ${reference}]:`, error);
  if (options.detail) return messageOf(error);
  return `${INTERNAL_ERROR_MESSAGE} Reference: ${reference}`;
}

/**
 * Whether a listening address is this machine only. Literal addresses and
 * `localhost` only: a name that resolves to loopback today may not tomorrow.
 */
export function isLoopbackAddress(hostname: string | undefined): boolean {
  if (!hostname) return false;
  const host = hostname.trim().replace(/^\[|\]$/g, '').toLowerCase();
  return host === 'localhost' || host === '::1' || /^127(\.\d{1,3}){3}$/.test(host);
}
