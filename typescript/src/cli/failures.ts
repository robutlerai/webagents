/**
 * What a failed turn says, in the chat and in `-p` (2026-09-25).
 *
 * A headline in plain words and one line of what to do. Until now each CLI
 * printed the same refusal its own way: asked for a model by a CLI sign-in, a
 * Robutler that does not fund one answered, and this chat headed the error
 * with the platform's protocol text ("session.create requires X-Payment-Token
 * in session.extensions [WebSocket closed unexpectedly (code=4001)]") while
 * the Python chat printed its own sentence, the original cause in brackets,
 * and then the same sentence again as the hint.
 *
 * One function in each SDK now decides both lines, and both run the same
 * cases (`python/tests/fixtures/cli/failure_presentation.json`); the Python
 * twin is `webagents/cli/repl/failures.py`. The proxy skill still raises its
 * own words (the socket's close code stays in the message, which the portal
 * reads), so the rules match either SDK's wording.
 */

import { cliCommand } from './config-store';

/** Functions, not constants: the command names the active profile (`cliCommand`). */
export const signInHint = (): string => `Use a provider key: ${cliCommand('secrets set OPENAI_API_KEY')}.`;
export const creditsHint = (): string =>
  `Add credits in Robutler, or use a provider key of your own (${cliCommand('secrets set OPENAI_API_KEY')}).`;
export const EXPIRED_HINT = 'Your Robutler sign-in may have expired. Type /login to sign in again.';
export const UNREACHABLE_HINT = 'Check the network, or platform.url.';

/** The socket's close code the proxy skill appends to the platform's reason. */
const CLOSE_NOTE = /\s*\[WebSocket closed unexpectedly \(code=\d+\)\]\s*$/;
const UNREACHABLE =
  /could not reach|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|fetch failed|websocket error|connection timeout|connection refused|connect call failed|timed out|nodename nor servname|name or service not known/i;

export interface FailureText {
  headline: string;
  hint?: string;
  /** A stable code for Robutler's own refusals, which `-p` reports. */
  code?: string;
}

/**
 * The headline and hint for a failed turn. `proxyUrl` is the Robutler socket
 * when the turn ran on Robutler's models; `genericHint` is the chat's advice
 * for a provider's own errors.
 */
export function presentFailure(
  message: string,
  options: { proxyUrl?: string; genericHint?: (message: string) => string | undefined } = {},
): FailureText {
  const text = (message ?? '').trim() || 'The model returned an error.';
  const { proxyUrl } = options;
  if (proxyUrl !== undefined) {
    // A Robutler from before CLI sign-ins asks for a payment token alone; a
    // later one also names the Bearer token `webagents login` stores.
    if ((text.includes('X-Payment-Token') && !text.includes('Bearer')) || text.includes('does not run models for a CLI sign-in')) {
      return { headline: `Robutler at ${proxyUrl} does not run models for a CLI sign-in.`, hint: signInHint(), code: 'sign_in_not_supported' };
    }
    if (/not enough credits|insufficient credits|payment_required|\b4002\b/i.test(text)) {
      return { headline: 'Not enough credits to start a model call.', hint: creditsHint(), code: 'insufficient_credits' };
    }
    if (/payment token|bearer token|unauthori[sz]ed|\b4001\b|\b401\b|expired/i.test(text)) {
      return { headline: 'Robutler did not accept your sign-in.', hint: EXPIRED_HINT, code: 'sign_in_refused' };
    }
    if (UNREACHABLE.test(text)) {
      return { headline: `Could not reach Robutler at ${proxyUrl}.`, hint: UNREACHABLE_HINT, code: 'unreachable' };
    }
  }
  const headline = (text.replace(CLOSE_NOTE, '').split('\n')[0] ?? '').trim() || 'The model returned an error.';
  const hint = options.genericHint?.(text);
  return hint ? { headline, hint } : { headline };
}
