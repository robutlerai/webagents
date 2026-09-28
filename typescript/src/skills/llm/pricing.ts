/**
 * What a model call costs, in credits, for the chat's footer (2026-09-26,
 * gap-closure plan item 2.4).
 *
 * Two sources, and the footer says which it used:
 *
 *  * Robutler's models REPORT the cost: the platform's `response.done` usage
 *    carries `total_cost`, the credits its settle deducted for the call (B4,
 *    2026-09-28; the platform side ships with the final-billing lane), or the
 *    older `cost` (`{input_cost, output_cost, total_cost, currency:
 *    'credits'}`, the `CostInfo` of `uamp/types.ts`), and that number is
 *    shown as it is, the charge rather than an estimate. `total_cost` wins
 *    when both are there.
 *  * A provider key has no bill to read, so the cost is ESTIMATED from the
 *    table below: the provider's list price per 1M tokens, in credits (1
 *    credit = 1 USD, the settled rate), input tokens times the input price
 *    plus output tokens times the output price. Cache reads and long-context
 *    tiers are not counted, which is why an estimate is written with a tilde.
 *    A model the table does not know shows tokens alone; so does a local
 *    model (`ollama/...`), which costs nothing.
 *
 * THE TABLE ROTS. The rows are the provider list prices as the platform's own
 * catalog records them on 2026-09-26; before a release, check each against
 * the provider's list. The Python SDK keeps the same table
 * (`llm/pricing.py`), and both must equal the shared fixture
 * `python/tests/fixtures/w2ops/cost.json`, so a price cannot change in one
 * SDK only.
 *
 * Credits copy (CLAUDE.md): credits are named as credits; the number is the
 * cost of the person's own use, never a payment to anyone.
 */

/** `provider/model` -> [credits per 1M input tokens, credits per 1M output tokens]. */
export const PROVIDER_LIST_PRICES: Readonly<Record<string, readonly [number, number]>> = {
  'openai/gpt-4o-mini': [0.15, 0.6],
  'openai/gpt-4.1': [2.0, 8.0],
  'openai/o3': [2.0, 8.0],
  'openai/o4-mini': [1.1, 4.4],
  'openai/gpt-5.4-nano': [0.2, 1.25],
  'openai/gpt-5.4-mini': [0.75, 4.5],
  'openai/gpt-5.4': [2.5, 15.0],
  'openai/gpt-5.5': [5.0, 30.0],
  'openai/gpt-5.6-luna': [0.2, 1.2],
  'openai/gpt-5.6-terra': [2.0, 12.0],
  'openai/gpt-5.6-sol': [4.0, 20.0],
  'openai/gpt-6-astra': [10.0, 50.0],
  'anthropic/claude-haiku-4-5': [1.0, 5.0],
  'anthropic/claude-sonnet-4-6': [3.0, 15.0],
  'anthropic/claude-opus-4-6': [5.0, 25.0],
  'anthropic/claude-opus-4-7': [5.0, 25.0],
  'anthropic/claude-opus-4-8': [5.0, 25.0],
  'anthropic/claude-sonnet-5': [2.0, 10.0],
  'anthropic/claude-opus-5': [5.0, 25.0],
  'google/gemini-2.5-flash': [0.3, 2.5],
  'google/gemini-2.5-pro': [1.25, 10.0],
  'google/gemini-3-flash': [0.5, 3.0],
  'google/gemini-3.1-flash-lite': [0.25, 1.5],
  'google/gemini-3.1-pro': [2.0, 12.0],
  'google/gemini-3.5-flash-lite': [0.3, 2.5],
  'google/gemini-3.5-flash': [1.5, 9.0],
  'google/gemini-3.6-flash': [1.5, 7.5],
  'google/gemini-3.7-flash': [1.5, 7.5],
  'google/gemini-3.8-flash': [1.5, 7.5],
  'xai/grok-3': [3.0, 15.0],
  'xai/grok-3-mini': [0.3, 0.5],
  'xai/grok-4-0709': [3.0, 15.0],
  'xai/grok-4-fast-reasoning': [0.2, 0.5],
  'xai/grok-4-fast-non-reasoning': [0.2, 0.5],
  'xai/grok-code-fast-1': [0.2, 1.5],
  'xai/grok-4.3': [1.25, 2.5],
  'xai/grok-4.20-reasoning': [1.25, 2.5],
  'xai/grok-4.20-non-reasoning': [1.25, 2.5],
  'fireworks/deepseek-v3p2': [0.56, 1.68],
  'fireworks/deepseek-v3p1': [0.56, 1.68],
  'fireworks/deepseek-r1': [0.56, 1.68],
  'fireworks/kimi-k3': [3.0, 15.0],
  'fireworks/kimi-k2p6': [0.95, 4.0],
  'fireworks/glm-5': [1.0, 3.2],
  'fireworks/qwen3-8b': [0.2, 0.2],
  'fireworks/gpt-oss-120b': [0.15, 0.6],
  'fireworks/gpt-oss-20b': [0.07, 0.3],
  'fireworks/llama-v3p3-70b-instruct': [0.9, 0.9],
  'fireworks/minimax-m2p5': [0.3, 1.2],
};

/**
 * The table row a model id finds: exact, else without a trailing `-YYYYMMDD`
 * date (`claude-haiku-4-5-20251001`), else without a `:tag`. A bare id with
 * no provider finds nothing: the provider is part of the price.
 */
export function priceRowFor(model: string | undefined): string | undefined {
  if (!model || !model.includes('/')) return undefined;
  if (model in PROVIDER_LIST_PRICES) return model;
  const undated = model.replace(/-\d{8}$/, '');
  if (undated in PROVIDER_LIST_PRICES) return undated;
  const untagged = undated.replace(/:[^/]*$/, '');
  if (untagged in PROVIDER_LIST_PRICES) return untagged;
  return undefined;
}

/** The estimate for a call, in credits, or `undefined` for a model the table does not know. */
export function estimateCostCredits(model: string | undefined, inputTokens: number, outputTokens: number): number | undefined {
  const row = priceRowFor(model);
  if (!row) return undefined;
  const [input, output] = PROVIDER_LIST_PRICES[row];
  return (inputTokens * input + outputTokens * output) / 1_000_000;
}

/**
 * A number of credits as the footer writes it: up to four decimals, trailing
 * zeros dropped, thousands grouped, and `<0.0001` for a positive amount too
 * small to show.
 */
export function formatCredits(credits: number): string {
  if (!(credits > 0)) return '0';
  if (credits < 0.0001) return '<0.0001';
  // Half up on the decimal value (0.00425 -> 0.0043), as the Python footer
  // rounds; the binary value of 0.00425 * 10000 sits a hair under 42.5.
  const rounded = Math.round((credits + Number.EPSILON) * 10_000) / 10_000;
  const [whole, fraction = ''] = rounded.toFixed(4).split('.');
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const trimmed = fraction.replace(/0+$/, '');
  return trimmed ? `${grouped}.${trimmed}` : grouped;
}

/** The footer's words: `0.0042 credits` as reported, `~0.0042 credits` as estimated. */
export function costWords(credits: number, estimated: boolean): string {
  return `${estimated ? '~' : ''}${formatCredits(credits)} credits`;
}

/** A usage number that can be credits: finite, not negative. */
function creditsNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
}

/**
 * The credits a platform-reported usage carries, when it carries any:
 * `total_cost` (the charge, B4), else `cost.total_cost` in credits. Pinned
 * with the Python `reported_cost_credits` by
 * `python/tests/fixtures/w2ops/final_sdk_total_cost.json`.
 */
export function reportedCostCredits(usage: { total_cost?: unknown; cost?: { total_cost?: number; currency?: string } } | undefined): number | undefined {
  const charged = creditsNumber(usage?.total_cost);
  if (charged !== undefined) return charged;
  const cost = usage?.cost;
  const total = creditsNumber(cost?.total_cost);
  if (!cost || total === undefined) return undefined;
  if (cost.currency && cost.currency.toLowerCase() !== 'credits') return undefined;
  return total;
}

/**
 * A conversation's running cost: what was reported, what was estimated,
 * and whether any part is an estimate (the tilde). `undefined` credits
 * means nothing to show: no model in the table, or a local model.
 */
export interface RunningCost {
  credits: number;
  estimated: boolean;
  /** Whether any call added to it; false shows tokens alone. */
  known: boolean;
}

export const NO_COST: RunningCost = { credits: 0, estimated: false, known: false };

/**
 * One turn's usage added to the running cost: the platform's number when
 * the usage carries one, else the estimate for `model`, else nothing.
 */
export function addTurnCost(
  running: RunningCost,
  model: string | undefined,
  usage: { input_tokens?: number; output_tokens?: number; total_cost?: unknown; cost?: { total_cost?: number; currency?: string } } | undefined,
): RunningCost {
  if (!usage) return running;
  const reported = reportedCostCredits(usage);
  if (reported !== undefined) return { credits: running.credits + reported, estimated: running.estimated, known: true };
  const estimate = estimateCostCredits(model, usage.input_tokens ?? 0, usage.output_tokens ?? 0);
  if (estimate === undefined) return running;
  return { credits: running.credits + estimate, estimated: true, known: true };
}
