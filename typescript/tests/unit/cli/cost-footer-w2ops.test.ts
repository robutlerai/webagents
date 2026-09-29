/**
 * The run's cost in credits, in the chat footer next to the tokens
 * (2026-09-26, gap-closure plan item 2.4, lane w2-ops), against the shared
 * fixture `python/tests/fixtures/w2ops/cost.json`, which the Python suite
 * reads too (`tests/cli/test_cost_footer_w2ops.py`): the price table, how a
 * model finds its row, the estimate, the number's format, and what the
 * footer, /status and the goodbye line show.
 *
 * The chat is driven with its counters set directly (a turn's usage is what
 * `streamToTerminal` adds; the footer reads the totals), under a throwaway
 * HOME with the FILE secrets backend, no key and no sign-in.
 *
 * Only a turn that ran through Robutler costs credits (2026-09-29): the
 * footer said `~<0.0001 credits` for an `openai/` turn on the person's own
 * key, where Robutler spends nothing. The fixture's `access` says which; the
 * chat's `turnCostModel` decides, and a key's turn shows tokens alone.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import {
  NO_COST,
  PROVIDER_LIST_PRICES,
  addTurnCost,
  costWords,
  estimateCostCredits,
  formatCredits,
  priceRowFor,
  reportedCostCredits,
  type RunningCost,
} from '../../../src/skills/llm/pricing';
import { TurnPrinter } from '../../../src/cli/render';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/w2ops/cost.json'), 'utf8')) as {
  list_prices: Record<string, [number, number]>;
  lookup: Array<{ model: string; row: string | null }>;
  estimates: Array<{ model: string; input_tokens: number; output_tokens: number; credits: number | null }>;
  format: { credits: Array<{ value: number; text: string }>; words: { reported: string; estimated: string } };
  footer: Array<{
    case: string;
    model: string;
    access: 'proxy' | 'direct';
    reported: number | null;
    input_tokens: number;
    output_tokens: number;
    footer: string;
    status: string;
    goodbye: string;
  }>;
  resume: { case: string; estimated: boolean; credits: number };
};

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let printed: string[] = [];

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-cost-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '200';
  process.chdir(tempDir('wa-cost-project-'));
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

describe('the price table (fixture list_prices)', () => {
  it('equals the fixture, row for row', () => {
    expect(Object.fromEntries(Object.entries(PROVIDER_LIST_PRICES).map(([k, v]) => [k, [...v]]))).toEqual(FIXTURE.list_prices);
  });

  it.each(FIXTURE.lookup)('finds the row for $model', ({ model, row }) => {
    expect(priceRowFor(model) ?? null).toBe(row);
  });

  it.each(FIXTURE.estimates)('estimates $model at $input_tokens in and $output_tokens out', ({ model, input_tokens, output_tokens, credits }) => {
    const estimate = estimateCostCredits(model, input_tokens, output_tokens);
    if (credits === null) expect(estimate).toBeUndefined();
    else expect(estimate).toBeCloseTo(credits, 10);
  });
});

describe('the number (fixture format)', () => {
  it.each(FIXTURE.format.credits)('writes $value as $text', ({ value, text }) => {
    expect(formatCredits(value)).toBe(text);
  });

  it('names credits as credits, with a tilde for an estimate', () => {
    expect(costWords(0.0042, false)).toBe(FIXTURE.format.words.reported.replace('{credits}', '0.0042'));
    expect(costWords(0.0042, true)).toBe(FIXTURE.format.words.estimated.replace('{credits}', '0.0042'));
  });

  it('reads a platform-reported cost in credits, and nothing in another currency', () => {
    expect(reportedCostCredits({ cost: { input_cost: 0.001, output_cost: 0.0032, total_cost: 0.0042, currency: 'credits' } })).toBe(0.0042);
    expect(reportedCostCredits({ cost: { input_cost: 1, output_cost: 1, total_cost: 2, currency: 'USD' } })).toBeUndefined();
    expect(reportedCostCredits({})).toBeUndefined();
    expect(reportedCostCredits(undefined)).toBeUndefined();
  });

  it('the printer carries a reported cost, and none when the platform sent none', () => {
    const printer = new TurnPrinter({ live: false, color: false, out: { write: () => true } as unknown as NodeJS.WriteStream });
    printer.feed({ type: 'done', response: { content: '', content_items: [], usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2, cost: { input_cost: 0, output_cost: 0, total_cost: 0.01, currency: 'credits' } } } });
    printer.feed({ type: 'done', response: { content: '', content_items: [], usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } });
    expect(printer.usageCostCredits).toBe(0.01);
    const quiet = new TurnPrinter({ live: false, color: false, out: { write: () => true } as unknown as NodeJS.WriteStream });
    quiet.feed({ type: 'done', response: { content: '', content_items: [], usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } });
    expect(quiet.usageCostCredits).toBeNull();
  });
});

/**
 * The chat's counters after one turn of the case, as `streamToTerminal` sets
 * them: the estimate's model is the chat's own decision (`turnCostModel`,
 * 2026-09-29), undefined on the person's own key.
 */
function chatAfter(c: (typeof FIXTURE.footer)[number], repl: InteractiveREPL): void {
  const inner = repl as unknown as {
    inputTokens: number;
    outputTokens: number;
    sessionTokens: number;
    turns: number;
    cost: RunningCost;
    sessionCost: RunningCost;
    messages: unknown[];
    modelAccess: { kind: string; model?: string } | undefined;
    turnCostModel(): string | undefined;
  };
  inner.modelAccess = { kind: c.access, model: c.model };
  inner.inputTokens = c.input_tokens;
  inner.outputTokens = c.output_tokens;
  inner.sessionTokens = c.input_tokens + c.output_tokens;
  inner.turns = 1;
  inner.messages = [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'user', content: 'c' }];
  const usage = {
    input_tokens: c.input_tokens,
    output_tokens: c.output_tokens,
    ...(c.reported !== null ? { cost: { total_cost: c.reported, currency: 'credits' } } : {}),
  };
  const model = inner.turnCostModel();
  expect(model !== undefined, c.case).toBe(c.access === 'proxy');
  inner.cost = addTurnCost(NO_COST, model, usage);
  inner.sessionCost = addTurnCost(NO_COST, model, usage);
}

const resumeCase = () => FIXTURE.footer.find((c) => c.case === FIXTURE.resume.case)!;

describe('the footer, /status and the goodbye line (fixture footer)', () => {
  it.each(FIXTURE.footer)('$case', async (c) => {
    const repl = new InteractiveREPL({});
    await repl.initialize();
    chatAfter(c, repl);
    const inner = repl as unknown as { footerParts(): string[]; goodbye(): void; handleInput(line: string): Promise<void> };
    // The footer: agent, model, the tokens and cost, the folder.
    const parts = inner.footerParts();
    expect(parts.slice(2, -1).join(', ')).toBe(c.footer);
    // /status: the Conversation row.
    printed = [];
    await inner.handleInput('/status');
    const status = printed.join('\n').split('\n').find((line) => line.includes('Conversation'));
    expect(status?.replace(/^\s*Conversation\s+/, '').trim()).toBe(c.status);
    // The goodbye line: replies, tokens, cost, duration.
    printed = [];
    inner.goodbye();
    const goodbye = printed.join('\n').replace(/\x1b\[[0-9;]*m/g, '');
    const middle = goodbye.split(' · ').slice(1, -1).join(' · ');
    expect(middle).toBe(c.goodbye);
  });

  it('a new conversation starts its cost over, and a resumed one brings its cost back', async () => {
    const repl = new InteractiveREPL({});
    await repl.initialize();
    chatAfter(resumeCase(), repl);
    const inner = repl as unknown as {
      cost: RunningCost;
      saveConversation(): void;
      sessionDir(): string;
      sessionId: string;
      startNewConversation(say?: boolean): void;
      commandResume(args: string): Promise<void>;
    };
    inner.saveConversation();
    const id = inner.sessionId;
    inner.startNewConversation(false);
    expect(inner.cost).toEqual(NO_COST);
    await inner.commandResume(id.slice(0, 8));
    expect(inner.cost.known).toBe(true);
    expect(inner.cost.estimated).toBe(FIXTURE.resume.estimated);
    expect(inner.cost.credits).toBeCloseTo(FIXTURE.resume.credits, 10);
  });

  it("a turn on the person's own key costs no credits; a failover member says its own route", async () => {
    // `~<0.0001 credits` stood in the footer for an `openai/` turn on the
    // person's own OPENAI_API_KEY (2026-09-29): Robutler spent nothing.
    const repl = new InteractiveREPL({});
    await repl.initialize();
    const inner = repl as unknown as {
      modelAccess: { kind: string; model?: string } | undefined;
      agent: { skills?: Array<{ name?: string; answeredModel?: string; answeredViaRobutler?: boolean; notes?: string[] }> } | undefined;
      turnRanThroughRobutler(): boolean;
      turnCostModel(): string | undefined;
    };
    inner.modelAccess = { kind: 'direct', model: 'openai/gpt-4o-mini' };
    expect(inner.turnRanThroughRobutler()).toBe(false);
    expect(inner.turnCostModel()).toBeUndefined();
    inner.modelAccess = { kind: 'proxy', model: 'openai/gpt-4o-mini' };
    expect(inner.turnRanThroughRobutler()).toBe(true);
    expect(inner.turnCostModel()).toBe('openai/gpt-4o-mini');
    // A fallback that ran on a key under a Robutler primary, and the reverse.
    const saved = inner.agent;
    inner.agent = { skills: [{ name: 'failover', answeredModel: 'anthropic/claude-haiku-4-5', answeredViaRobutler: false, notes: [] }] };
    expect(inner.turnCostModel()).toBeUndefined();
    inner.modelAccess = { kind: 'direct', model: 'openai/gpt-4o-mini' };
    inner.agent = { skills: [{ name: 'failover', answeredModel: 'auto/balanced', answeredViaRobutler: true, notes: [] }] };
    expect(inner.turnCostModel()).toBe('auto/balanced');
    inner.agent = saved;
  });
});
