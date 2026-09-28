/**
 * `webagents budget <tokenId>` (webagents gap-closure plan 2.3, 2026-09-26):
 * the tree the platform answers is printed one line per token, as the
 * fixture `python/tests/fixtures/payments/delegate_budget.json` pins it for
 * both CLIs, with a totals line; sign-in and answer failures are named.
 */

import { describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/cli/credentials.js', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getToken: async () => 'cli-token',
}));

import { budgetTree, budgetTreeTotals, formatCredits, renderBudgetTree } from '../../../src/cli/budget-tree';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/payments/delegate_budget.json'), 'utf8')) as {
  tree: { route: string; sample: Parameters<typeof renderBudgetTree>[0]; rendered: string[]; totals_line: string };
};

describe('rendering', () => {
  it('prints the fixture tree line for line, and the totals', () => {
    expect(renderBudgetTree(FIXTURE.tree.sample)).toEqual(FIXTURE.tree.rendered);
    expect(budgetTreeTotals(FIXTURE.tree.sample)).toBe(FIXTURE.tree.totals_line);
  });

  it('formats credits without trailing zeros or float noise', () => {
    expect(formatCredits(0.1)).toBe('0.1');
    expect(formatCredits(1)).toBe('1');
    expect(formatCredits(0.1 + 0.2)).toBe('0.3');
    expect(formatCredits(0.0123)).toBe('0.0123');
  });
});

describe('the command', () => {
  it('fetches the tree with the sign-in and renders it', async () => {
    const seen: string[] = [];
    const result = await budgetTree(FIXTURE.tree.sample.id, {
      fetch: (async (input: string | URL | Request, init?: RequestInit) => {
        seen.push(String(input));
        expect((init?.headers as Record<string, string>).Authorization).toBe('Bearer cli-token');
        return Response.json({ tree: FIXTURE.tree.sample });
      }) as unknown as typeof fetch,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(seen[0]).toMatch(new RegExp(FIXTURE.tree.route.replace('{id}', FIXTURE.tree.sample.id).replace(/\//g, '\\/') + '$'));
    expect(result.lines).toEqual(FIXTURE.tree.rendered);
    expect(result.totals).toBe(FIXTURE.tree.totals_line);
  });

  it('names a token that is not yours, an expired sign-in, and a bad id', async () => {
    const answer = (status: number) => (async () => new Response('{}', { status })) as unknown as typeof fetch;
    expect(await budgetTree(FIXTURE.tree.sample.id, { fetch: answer(404) })).toMatchObject({ ok: false, code: 'not_found' });
    expect(await budgetTree(FIXTURE.tree.sample.id, { fetch: answer(401) })).toMatchObject({ ok: false, code: 'expired' });
    expect(await budgetTree('nope', { fetch: answer(200) })).toMatchObject({ ok: false, code: 'bad_token_id' });
  });
});
