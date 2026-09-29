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

import { findProvider } from '../skills/llm/providers';
import { TOOL_LOOP, TOOL_ROUND_LIMIT, toolLoopSentence, toolRoundLimitSentence } from '../core/tool-budget';
import { cliCommand } from './config-store';

/**
 * The key variable the hints name for `model` (2026-09-27): its provider's,
 * when this SDK has a client for it; OPENAI_API_KEY otherwise, the chat's
 * historical default. Every hint said OPENAI_API_KEY, on a Google model too.
 * `auto/balanced` names no provider, so it keeps the default.
 */
export function providerKeyFor(model?: string): string {
  const slash = model?.indexOf('/') ?? -1;
  const provider = model && slash > 0 ? findProvider(model.slice(0, slash)) : undefined;
  return provider?.credential === 'api-key' && provider.envVar ? provider.envVar : 'OPENAI_API_KEY';
}

/** Functions, not constants: the command names the active profile (`cliCommand`). */
export const signInHint = (model?: string): string => `Use a provider key: ${cliCommand(`secrets set ${providerKeyFor(model)}`)}.`;
export const creditsHint = (model?: string): string =>
  `Add credits in Robutler, or use a provider key of your own (${cliCommand(`secrets set ${providerKeyFor(model)}`)}).`;
export const EXPIRED_HINT = 'Your Robutler sign-in may have expired. Type /login to sign in again.';

/**
 * A hint as `-p` says it (2026-09-28, the e2e pass): `/login` is a chat
 * command, so `-p` names the command a shell runs instead. The Python
 * `for_prompt` says the same (fixture `cli/final_sdk_low_items.json`).
 */
export function forPrompt(hint: string | undefined): string | undefined {
  if (hint === EXPIRED_HINT) return `Your Robutler sign-in may have expired. Run \`${cliCommand('login')}\` to sign in again.`;
  return hint;
}
export const UNREACHABLE_HINT = 'Check the network, or platform.url.';

/** What to do about a reply with nothing in it: the same line in both chats. */
export const EMPTY_REPLY_HINT = '/model <provider/model> tries another model.';

/**
 * What the chat says about a turn that produced no answer (2026-09-27).
 *
 * The Python agent used to answer an empty completion with a canned apology
 * about "content filtering" and add it to the conversation, whatever had
 * happened (the real reason, MALFORMED_FUNCTION_CALL, was in the platform's
 * log alone), and a reply that was only thinking said nothing at all. Both
 * chats now print one truthful line: the provider's own finish reason, that
 * the prompt was blocked, that the request was sent twice, or that the model
 * only produced thinking. The cases both SDKs run are in
 * `python/tests/fixtures/cli/chat_fixes_empty_reply.json`; the Python twin is
 * `cli/repl/failures.py` (`present_empty_reply`).
 */
export function presentEmptyReply(
  finish: { reason?: string | null; blocked?: boolean; retried?: boolean; rounds?: number | null; tool?: string | null } = {},
  options: { thinking?: boolean } = {},
): FailureText {
  // `tool_round_limit` and `tool_loop` are the AGENT's reasons, not the
  // provider's (2026-09-28, `core/tool-budget.ts`): the turn spent its tool
  // rounds, or called `tool` three times in a row with the same arguments and
  // the same result, and its last, tool-less call brought no answer. The Python chat named the finish
  // reason of the model's last tool call instead ("the provider reported STOP").
  if (finish.reason === TOOL_ROUND_LIMIT) {
    return { headline: toolRoundLimitSentence(finish.rounds ?? undefined), hint: EMPTY_REPLY_HINT };
  }
  if (finish.reason === TOOL_LOOP) {
    return { headline: toolLoopSentence(finish.tool ?? undefined), hint: EMPTY_REPLY_HINT };
  }
  const reason = finish.reason ?? undefined;
  const twice = finish.retried ? ', twice' : '';
  let headline: string;
  if (finish.blocked) {
    headline = `The model returned no answer: the provider blocked the prompt${reason ? ` (${reason})` : ''}.`;
  } else if (options.thinking) {
    headline = `The model returned no answer: it only produced thinking${reason ? ` (the provider reported ${reason}${twice})` : ''}.`;
  } else if (reason) {
    headline = `The model returned no answer: the provider reported ${reason}${twice}.`;
  } else {
    headline = 'The model returned no answer, and the provider gave no reason.';
  }
  return { headline, hint: EMPTY_REPLY_HINT };
}

/** The socket's close code the proxy skill appends to the platform's reason. */
const CLOSE_NOTE = /\s*\[WebSocket closed unexpectedly \(code=\d+\)\]\s*$/;
/**
 * Robutler's own credit refusals that say more than "not enough credits"
 * (2026-09-27, the portal's `cliCreditsRefusal`): what the call needs
 * reserved, and what unfinished calls hold and when it returns. The chat used
 * to print "Not enough credits to start a model call." over every one of
 * them, beside a balance that was plainly there. Shown as Robutler wrote
 * them: the first sentence is the headline, the rest the hint. The Python
 * twin is `_SPECIFIC_CREDITS` in `cli/repl/failures.py`.
 */
const SPECIFIC_CREDITS = /^Not enough credits (?:for this model call|to start a model call right now)\b/;
const HELD_CREDITS = 'held by unfinished model calls';
/** The first sentence, and the rest: a full stop followed by space or the end (not the point in "0.358"). */
const FIRST_SENTENCE = /^([\s\S]*?\.)(?=\s|$)\s*([\s\S]*)$/;
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
  options: { proxyUrl?: string; genericHint?: (message: string) => string | undefined; model?: string } = {},
): FailureText {
  const text = (message ?? '').trim() || 'The model returned an error.';
  const { proxyUrl, model } = options;
  if (proxyUrl !== undefined) {
    // The platform's own provider key was refused (a 401, 402 or 429 from
    // the maker, `providerErrorForClient` in the portal, 2026-09-27): not the
    // person's problem, so the way out is another model.
    if (text.toLowerCase().includes('unavailable right now')) {
      return { headline: 'This model is unavailable right now.', hint: EMPTY_REPLY_HINT, code: 'provider_unavailable' };
    }
    // A Robutler from before CLI sign-ins asks for a payment token alone; a
    // later one also names the Bearer token `webagents login` stores.
    if ((text.includes('X-Payment-Token') && !text.includes('Bearer')) || text.includes('does not run models for a CLI sign-in')) {
      return { headline: `Robutler at ${proxyUrl} does not run models for a CLI sign-in.`, hint: signInHint(model), code: 'sign_in_not_supported' };
    }
    const said = text.replace(CLOSE_NOTE, '').trim();
    if (SPECIFIC_CREDITS.test(said)) {
      const sentence = FIRST_SENTENCE.exec(said);
      const headline = sentence ? sentence[1] : said;
      const rest = sentence ? sentence[2].trim() : '';
      if (said.includes(HELD_CREDITS)) {
        return rest ? { headline, hint: rest, code: 'credits_held' } : { headline, code: 'credits_held' };
      }
      return { headline, hint: creditsHint(model), code: 'insufficient_credits' };
    }
    if (/not enough credits|insufficient credits|payment_required|\b4002\b/i.test(text)) {
      return { headline: 'Not enough credits to start a model call.', hint: creditsHint(model), code: 'insufficient_credits' };
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
