/**
 * Per-hop budgets for `delegate` (webagents gap-closure plan 2.3, 2026-09-26),
 * the twin of `python/webagents/agents/skills/robutler/nli/budget.py`; the
 * fixture `python/tests/fixtures/payments/delegate_budget.json` pins the
 * parameter, the receipt and the tree for both SDKs.
 *
 * WHAT CHANGED. A self-hosted TypeScript agent forwarded ITS OWN payment
 * token to every agent it delegated to (`X-Payment-Token`), so a hop could
 * spend whatever that token held, and nothing recorded what it did spend.
 * The portal already derives a child per hop for hosted agents (S-009); the
 * open SDK now does the same through the platform's delegate route,
 * authenticated as the agent, for the `budget` the model names (default
 * 0.1 credits, at most 5). The parent's balance is never handed over whole:
 * with no parent, no platform key, or a refusal (depth, balance, policy),
 * the hop fails closed and says why.
 *
 * EACH HOP RETURNS A RECEIPT: the child token's balance is read back after
 * the hop, and `budget - remaining` is what the hop spent. The line is
 * appended to the tool result so the model and the owner see the cost. The
 * receipt names the child by a FINGERPRINT (S-303, 2026-09-26: the first 12
 * hex characters of the SHA-256 of its id), never by the token id: the id is
 * a settle bearer, which the platform's `resolveToLockId` auto-locks with no
 * audience check, so a receipt printing it let any signed-in reader of the
 * transcript settle the remainder to themselves.
 *
 * THE OWNER READS THE RUN'S BUDGET TREE at `GET /api/payments/tokens/{id}/tree`
 * (the root token and every child derived from it) and prints it with
 * `webagents budget <tokenId>` (`../../cli/budget-tree.ts`); the ids are in
 * the owner's own token list on the platform, not in a transcript.
 */

import { createHash } from 'node:crypto';
import { DEFAULT_PLATFORM_URL } from '../platform-url';

/** The `budget` parameter of `delegate`, as the fixture pins it. */
export const DELEGATE_BUDGET_PARAMETER = {
  name: 'budget',
  description:
    'The most this hop may spend, in credits (default 0.1, at most 5). A child token for exactly this budget is minted for the agent; the hop is refused when the budget cannot be derived.',
  default: 0.1,
  max: 5,
} as const;

/** A budget the model named, bounded: the default when absent, refused above the maximum. */
export function resolveDelegateBudget(value: unknown): { ok: true; budget: number } | { ok: false; error: string } {
  if (value === undefined || value === null) return { ok: true, budget: DELEGATE_BUDGET_PARAMETER.default };
  const budget = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(budget) || budget <= 0) return { ok: false, error: 'budget must be a positive number of credits' };
  if (budget > DELEGATE_BUDGET_PARAMETER.max) return { ok: false, error: `budget ${formatCredits(budget)} exceeds maximum allowed (${DELEGATE_BUDGET_PARAMETER.max} credits)` };
  return { ok: true, budget };
}

/** The budget a payment token was minted with: its unverified `payment.balance` claim, or undefined. */
export function tokenBudgetOf(jwt: string): number | undefined {
  const parts = jwt.split('.');
  if (parts.length < 2) return undefined;
  try {
    const payload = JSON.parse(new TextDecoder().decode(base64urlBytes(parts[1]))) as { payment?: { balance?: unknown } };
    const balance = payload.payment?.balance;
    return typeof balance === 'number' && Number.isFinite(balance) && balance > 0 ? balance : undefined;
  } catch {
    return undefined;
  }
}

/** A credit amount as the receipts and the tree print it: at most 9 decimals, no trailing zeros. */
export function formatCredits(value: number): string {
  return String(Number(value.toFixed(9)));
}

export interface DelegateReceipt {
  budget: number;
  spent: number;
  remaining: number;
  /** The child token's fingerprint (`tokenFingerprint`), never its id (S-303). */
  token: string;
}

/** The first 12 hex characters of the SHA-256 of a token id: names a token without being a bearer for it (S-303). */
export function tokenFingerprint(tokenId: string): string {
  return createHash('sha256').update(tokenId, 'utf8').digest('hex').slice(0, 12);
}

/** The receipt line appended to a hop's result. */
export function formatDelegateReceipt(receipt: DelegateReceipt): string {
  return `[delegate receipt: budget ${formatCredits(receipt.budget)} credits, spent ${formatCredits(receipt.spent)}, remaining ${formatCredits(receipt.remaining)}, token ${receipt.token}]`;
}

/** The `jti` of a payment token, unverified: the token row's id, for the receipt and the tree. */
export function tokenIdOf(jwt: string): string | undefined {
  const parts = jwt.split('.');
  if (parts.length < 2) return undefined;
  try {
    const payload = JSON.parse(new TextDecoder().decode(base64urlBytes(parts[1]))) as { jti?: unknown };
    return typeof payload.jti === 'string' ? payload.jti : undefined;
  } catch {
    return undefined;
  }
}

function base64urlBytes(text: string): Uint8Array {
  const padded = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
  const binary = atob(padded);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

export interface MintChildOptions {
  platformUrl?: string;
  /** The agent's platform key: the delegate route charges the caller in the parent's audience. */
  apiKey: string;
  parentToken: string;
  /** The callee, a username or an id. */
  delegateTo: string;
  budget: number;
  fetch?: typeof fetch;
  timeoutMs?: number;
}

export type MintChildResult = { ok: true; token: string; tokenId?: string; amountCredits: number } | { ok: false; error: string };

/**
 * A child token for one hop, from the platform's delegate route. The route
 * bounds the amount by the parent's depth and balance and the payer's policy
 * for the callee; a refusal is this hop's refusal, never a reason to forward
 * the parent.
 */
export async function mintChildToken(options: MintChildOptions): Promise<MintChildResult> {
  const base = (options.platformUrl ?? DEFAULT_PLATFORM_URL).replace(/\/+$/, '');
  const fetchImpl = options.fetch ?? ((input: string | URL | Request, init?: RequestInit) => fetch(input, init));
  let res: Response;
  try {
    res = await fetchImpl(`${base}/api/payments/delegate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${options.apiKey}` },
      body: JSON.stringify({ parentToken: options.parentToken, delegateTo: options.delegateTo.replace(/^@/, ''), amount: options.budget }),
      signal: AbortSignal.timeout(options.timeoutMs ?? 10_000),
    });
  } catch (error) {
    return { ok: false, error: `the platform could not be reached to derive the budget: ${(error as Error).message}` };
  }
  const body = (await res.json().catch(() => ({}))) as { token?: unknown; tokenId?: unknown; amountCredits?: unknown; amountDollars?: unknown; error?: unknown };
  if (!res.ok || typeof body.token !== 'string' || !body.token) {
    return { ok: false, error: typeof body.error === 'string' && body.error ? body.error : `the delegate route answered ${res.status}` };
  }
  const amount = typeof body.amountCredits === 'number' ? body.amountCredits : typeof body.amountDollars === 'number' ? body.amountDollars : options.budget;
  return { ok: true, token: body.token, tokenId: typeof body.tokenId === 'string' ? body.tokenId : tokenIdOf(body.token), amountCredits: amount };
}

/** The child's balance after the hop, from the platform's verify route, as a receipt. */
export async function readChildReceipt(options: {
  platformUrl?: string;
  apiKey?: string;
  childToken: string;
  budget: number;
  tokenId?: string;
  fetch?: typeof fetch;
}): Promise<DelegateReceipt | null> {
  const base = (options.platformUrl ?? DEFAULT_PLATFORM_URL).replace(/\/+$/, '');
  const fetchImpl = options.fetch ?? ((input: string | URL | Request, init?: RequestInit) => fetch(input, init));
  try {
    const res = await fetchImpl(`${base}/api/payments/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}) },
      body: JSON.stringify({ token: options.childToken }),
      signal: AbortSignal.timeout(8_000),
    });
    const body = (await res.json().catch(() => ({}))) as { balanceCredits?: unknown; balanceDollars?: unknown };
    const remaining = typeof body.balanceCredits === 'number' ? body.balanceCredits : typeof body.balanceDollars === 'number' ? body.balanceDollars : null;
    if (remaining === null) return null;
    const bounded = Math.min(Math.max(remaining, 0), options.budget);
    const tokenId = options.tokenId ?? tokenIdOf(options.childToken);
    return {
      budget: options.budget,
      spent: Number((options.budget - bounded).toFixed(9)),
      remaining: Number(bounded.toFixed(9)),
      token: tokenId ? tokenFingerprint(tokenId) : 'unknown',
    };
  } catch {
    return null;
  }
}
