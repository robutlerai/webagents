/**
 * `webagents budget <tokenId>`: the budget tree of a run (webagents
 * gap-closure plan 2.3, 2026-09-26), in the words the Python CLI uses
 * (`python/webagents/cli/budget_tree.py`), so the two print the same lines;
 * the fixture `python/tests/fixtures/payments/delegate_budget.json` pins them.
 *
 * The tree comes from `GET /api/payments/tokens/{id}/tree` on the platform:
 * the root token and every child a hop derived from it, each with its budget,
 * what was charged against it and what it has left. Credits, never a currency.
 */

import { cliCommand, resolvePlatformUrl } from './config-store';

export interface BudgetTreeNode {
  id: string;
  audience?: string[];
  status?: string;
  budgetCredits: number;
  spentCredits: number;
  remainingCredits: number;
  chain?: string[];
  children?: BudgetTreeNode[];
}

/** A credit amount as the tree prints it: at most 9 decimals, no trailing zeros. */
export function formatCredits(value: number): string {
  return String(Number(Number(value).toFixed(9)));
}

/** One line per token, two spaces per level; `audience=any` for an unrestricted token. */
export function renderBudgetTree(root: BudgetTreeNode): string[] {
  const lines: string[] = [];
  const walk = (node: BudgetTreeNode, depth: number): void => {
    const audience = node.audience && node.audience.length > 0 ? node.audience.join(',') : 'any';
    lines.push(
      `${'  '.repeat(depth)}${node.id} audience=${audience} budget=${formatCredits(node.budgetCredits)} spent=${formatCredits(node.spentCredits)} remaining=${formatCredits(node.remainingCredits)} ${node.status ?? 'unknown'}`,
    );
    for (const child of node.children ?? []) walk(child, depth + 1);
  };
  walk(root, 0);
  return lines;
}

/** The totals line under the tree: budgets and spend summed over every token. */
export function budgetTreeTotals(root: BudgetTreeNode): string {
  let budget = 0;
  let spent = 0;
  let count = 0;
  const walk = (node: BudgetTreeNode): void => {
    budget += Number(node.budgetCredits) || 0;
    spent += Number(node.spentCredits) || 0;
    count += 1;
    for (const child of node.children ?? []) walk(child);
  };
  walk(root);
  return `total budget=${formatCredits(budget)} spent=${formatCredits(spent)} across ${count} token${count === 1 ? '' : 's'}`;
}

export type BudgetTreeResult =
  | { ok: true; tree: BudgetTreeNode; lines: string[]; totals: string }
  | { ok: false; code: string; message: string; fix: string };

/** Fetch and render the tree for `tokenId` with the CLI's sign-in. */
export async function budgetTree(tokenId: string, options: { fetch?: typeof fetch } = {}): Promise<BudgetTreeResult> {
  const { getToken } = await import('./credentials.js');
  const [portalUrl] = resolvePlatformUrl();
  const host = portalUrl.replace(/^https?:\/\//, '');
  const token = await getToken();
  if (!token) return { ok: false, code: 'not_signed_in', message: `Not signed in to ${host}.`, fix: `Run \`${cliCommand('login')}\`.` };
  if (!/^[0-9a-f-]{36}$/i.test(tokenId)) return { ok: false, code: 'bad_token_id', message: `${tokenId} is not a payment token id.`, fix: 'Pass a payment token id from your token list on the platform; a delegate receipt names only a fingerprint.' };
  const fetchImpl = options.fetch ?? ((input: string | URL | Request, init?: RequestInit) => fetch(input, init));
  let res: Response;
  try {
    res = await fetchImpl(`${portalUrl}/api/payments/tokens/${tokenId}/tree`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(8000),
    });
  } catch (error) {
    return { ok: false, code: 'unreachable', message: `Could not reach ${host}: ${(error as Error).message}`, fix: 'Check the network, or `webagents config get platform.url`.' };
  }
  if (res.status === 401) return { ok: false, code: 'expired', message: `Your sign-in on ${host} has expired.`, fix: `Run \`${cliCommand('login')}\`.` };
  if (res.status === 404) return { ok: false, code: 'not_found', message: `No budget tree for ${tokenId} on ${host}: not a token of yours.`, fix: '' };
  if (!res.ok) return { ok: false, code: 'http_error', message: `${host} answered ${res.status}.`, fix: '' };
  const data = (await res.json().catch(() => ({}))) as { tree?: BudgetTreeNode };
  if (!data.tree || typeof data.tree.id !== 'string') return { ok: false, code: 'bad_answer', message: `${host} answered without a tree.`, fix: '' };
  return { ok: true, tree: data.tree, lines: renderBudgetTree(data.tree), totals: budgetTreeTotals(data.tree) };
}
