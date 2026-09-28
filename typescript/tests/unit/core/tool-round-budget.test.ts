/**
 * The tool-round budget of one turn (2026-09-28), against the shared fixture
 * `python/tests/fixtures/agent_loop/tool_round_budget.json`, which the Python
 * suite reads too (`tests/test_tool_round_budget.py`).
 *
 * The owner asked the Python built-in agent "nice": five rounds of tools (its
 * hard-coded cap), no answer, and a chat line blaming the provider. The Python
 * agent now takes this agent's budget and warning; pinned here, for both:
 *
 *  - the default, the warning round and the warning's words are the fixture's;
 *  - at the cap the agent still ends with the `max_iterations` error (the code
 *    the portal reads), and its `details.finish` carries the reason the chat
 *    says, `tool_round_limit`, with the rounds that ran;
 *  - an answer in the LAST round is an answer, with no such error after it;
 *  - the chat and `-p` say "The agent stopped after N tool rounds without an
 *    answer." instead of an error line;
 *  - both ROBUTLER.md copies carry the small-talk rule.
 */

import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, handoff } from '../../../src/core/decorators.js';
import type { AgenticMessage, Context } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createSessionCreateEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import type { ContentItem } from '../../../src/uamp/types.js';
import {
  DEFAULT_MAX_TOOL_ITERATIONS,
  REPEAT_LIMIT,
  TOOL_ROUND_LIMIT,
  budgetWarning,
  budgetWarningRound,
  canonicalArguments,
  continueQuestion,
  finalAnswerMessage,
  loopAnswerMessage,
  parseMaxToolRounds,
  toolLoopSentence,
  toolRoundLimitOf,
} from '../../../src/core/tool-budget';
import { completionBody } from '../../../src/server/handler';
import { MAX_TOOL_ROUNDS_ENV, ROUNDS_WORDS, effectiveMaxToolRounds } from '../../../src/core/tool-budget';
import { AGENT_FILE_KEYS, parseAgentMarkdown } from '../../../src/agents/index';
import { InteractiveREPL } from '../../../src/cli/app';
import { EMPTY_REPLY_HINT } from '../../../src/cli/failures';
import { TurnPrinter } from '../../../src/cli/render';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../../..');
const FIXTURE = JSON.parse(
  readFileSync(path.join(ROOT, 'python/tests/fixtures/agent_loop/tool_round_budget.json'), 'utf8'),
) as {
  default_limit: number;
  finish_reason: string;
  warning: string;
  warn_at: Array<{ limit: number; round: number }>;
  scenario: { limit: number; model_calls: number; warned_at_call: number; final_call: number; rounds: number; answer: string; headline: string; answered: string };
  final_answer: string;
  repeat_limit: number;
  loop_answer: string;
  loop_finish_reason: string;
  same_arguments: Array<{ a: string; b: string; same: boolean }>;
  continue_question: string;
  continue_message: string;
  refusals: Array<{ name: string; value: unknown; says: string }>;
  accepted: Array<{ value: unknown; rounds: number }>;
  loop: { limit: number; tool: string; model_calls: number; rounds: number; says: string };
};
const SCENARIO = FIXTURE.scenario;

/**
 * A model that asks for `add` whenever it has tools (new arguments each call,
 * or the same ones with `sameArgs`), answers at call `answerAt`, and answers
 * the fixture's answer when it has no tools (or nothing, with `silentLast`).
 */
function keepsAskingForTools(options: { answerAt?: number; sameArgs?: boolean; silentLast?: boolean } = {}) {
  let calls = 0;
  const seen: AgenticMessage[][] = [];
  const toolsOffered: boolean[] = [];
  class KeepsAsking extends Skill {
    @handoff({ name: 'keeps-asking' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      calls++;
      seen.push([...(context.get<AgenticMessage[]>('_agentic_messages') ?? [])]);
      const hasTools = (context.get<unknown[]>('_agentic_tools') ?? []).length > 0;
      toolsOffered.push(hasTools);
      const responseId = generateEventId();
      yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
      const output: ContentItem[] = [];
      const text = !hasTools ? (options.silentLast ? '' : SCENARIO.answer) : options.answerAt !== undefined && calls >= options.answerAt ? 'done' : undefined;
      if (text !== undefined) {
        if (text) {
          yield createResponseDeltaEvent(responseId, { type: 'text', text });
          output.push({ type: 'text', text });
        }
      } else {
        const args = options.sameArgs ? '{"a":1,"b":1}' : `{"a":${calls},"b":1}`;
        output.push({ type: 'tool_call', tool_call: { id: `call_${calls}`, name: 'add', arguments: args } });
      }
      yield createResponseDoneEvent(responseId, output);
    }
  }
  return { skill: new KeepsAsking(), calls: () => calls, seen, toolsOffered };
}

class MathTools extends Skill {
  @tool({ description: 'Add two numbers' })
  async add(params: { a: number; b: number }, _c: Context): Promise<number> {
    return params.a + params.b;
  }
}

function inputEvents(text: string): ClientEvent[] {
  return [createSessionCreateEvent({ modalities: ['text'] }), createInputTextEvent(text), createResponseCreateEvent()];
}

async function collect(gen: AsyncGenerator<ServerEvent>): Promise<ServerEvent[]> {
  const events: ServerEvent[] = [];
  for await (const event of gen) events.push(event);
  return events;
}

const warnings = (conversation: AgenticMessage[]): string[] =>
  conversation.filter((m) => m.role === 'system' && String(m.content).includes('tool-call budget')).map((m) => String(m.content));

describe('the budget both SDKs share', () => {
  it('the default, the reason and the words are the fixture\'s', () => {
    expect(DEFAULT_MAX_TOOL_ITERATIONS).toBe(FIXTURE.default_limit);
    expect(TOOL_ROUND_LIMIT).toBe(FIXTURE.finish_reason);
    expect(budgetWarning(7, 9)).toBe(FIXTURE.warning.replace('{used}', '7').replace('{limit}', '9'));
    expect(FIXTURE.warning).not.toContain('—');
    expect((new BaseAgent() as unknown as { maxToolIterations: number }).maxToolIterations).toBe(FIXTURE.default_limit);
  });

  it('the limits, the refusals and the question are the fixture\'s', () => {
    for (const c of FIXTURE.refusals) expect(() => parseMaxToolRounds(c.value, c.name)).toThrow(c.says);
    for (const c of FIXTURE.accepted) expect(parseMaxToolRounds(c.value)).toBe(c.rounds);
    expect(continueQuestion(7)).toBe(FIXTURE.continue_question.replace('{rounds}', '7'));
    for (const c of (FIXTURE as unknown as { continue_questions: Array<{ rounds: number; asks: string }> }).continue_questions) {
      expect(continueQuestion(c.rounds)).toBe(c.asks);
    }
  });

  it.each(FIXTURE.warn_at.map((row) => [row.limit, row.round] as const))('a budget of %i warns at round %i', (limit, round) => {
    expect(budgetWarningRound(limit)).toBe(round);
  });
});

describe('a model that keeps asking for tools', () => {
  it('answers at the cap with tools off, warned at the fixture\'s round', async () => {
    const model = keepsAskingForTools();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: SCENARIO.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    vi.restoreAllMocks();

    expect(model.calls()).toBe(SCENARIO.model_calls);
    expect(model.toolsOffered).toEqual([...Array(SCENARIO.rounds).fill(true), false]);
    model.seen.forEach((conversation, index) => {
      const call = index + 1;
      expect(warnings(conversation)).toEqual(
        call >= SCENARIO.warned_at_call ? [budgetWarning(SCENARIO.warned_at_call, SCENARIO.limit)] : [],
      );
    });
    const finalMessage = FIXTURE.final_answer.replace('{limit}', String(SCENARIO.limit));
    expect(finalAnswerMessage(SCENARIO.limit)).toBe(finalMessage);
    expect(model.seen[SCENARIO.final_call - 1].at(-1)).toEqual({ role: 'system', content: finalMessage });

    // The last call's answer carries the turn's finish; no error follows it.
    expect(events.find((e) => e.type === 'response.error')).toBeUndefined();
    const done = events.find((e) => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBe(TOOL_ROUND_LIMIT);
    expect(done.response.finish_rounds).toBe(SCENARIO.rounds);
  });

  it('a last call with no answer ends with max_iterations and the finish', async () => {
    const model = keepsAskingForTools({ silentLast: true });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: SCENARIO.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    vi.restoreAllMocks();
    const error = events.find((e) => e.type === 'response.error') as unknown as { error: { code: string; details?: unknown } };
    expect(error.error.code).toBe('max_iterations');
    expect(error.error.details).toEqual({ finish: { reason: TOOL_ROUND_LIMIT, blocked: false, retried: false, rounds: SCENARIO.rounds } });
  });

  it('the same call three times stops early with the loop reason', async () => {
    const model = keepsAskingForTools({ sameArgs: true });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: FIXTURE.loop.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    vi.restoreAllMocks();
    expect(model.calls()).toBe(FIXTURE.loop.model_calls);
    expect(model.toolsOffered.at(-1)).toBe(false);
    expect(model.seen.at(-1)!.at(-1)).toEqual({ role: 'system', content: loopAnswerMessage('add') });
    expect(loopAnswerMessage(FIXTURE.loop.tool)).toBe(FIXTURE.loop_answer.replace('{tool}', FIXTURE.loop.tool));
    const done = events.find((e) => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBe(FIXTURE.loop_finish_reason);
    expect(done.response.finish_tool).toBe('add');
    expect(done.response.finish_rounds).toBe(FIXTURE.loop.rounds);
    expect(toolLoopSentence(FIXTURE.loop.tool)).toBe(FIXTURE.loop.says);
  });

  it('same arguments are the fixture\'s', () => {
    expect(REPEAT_LIMIT).toBe(FIXTURE.repeat_limit);
    for (const c of FIXTURE.same_arguments) expect(canonicalArguments(c.a) === canonicalArguments(c.b)).toBe(c.same);
  });

  it('an answer in the last round is an answer, with no error after it', async () => {
    const model = keepsAskingForTools({ answerAt: SCENARIO.limit });
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: SCENARIO.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    expect(model.calls()).toBe(SCENARIO.limit);
    expect(events.find((e) => e.type === 'response.error')).toBeUndefined();
    const done = events.find((e) => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBeUndefined();
  });

  it('run() and runStreaming() report the finish, and the server says it', async () => {
    const model = keepsAskingForTools();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: SCENARIO.limit });
    const response = await agent.run([{ role: 'user', content: 'nice' }]);
    const chunks = [];
    for await (const chunk of agent.runStreaming([{ role: 'user', content: 'nice' }])) chunks.push(chunk);
    vi.restoreAllMocks();
    expect(response.content).toBe(SCENARIO.answer);
    expect(response.finish).toEqual({ reason: TOOL_ROUND_LIMIT, rounds: SCENARIO.rounds });
    const done = chunks.find((c) => c.type === 'done');
    expect(done?.response?.finish).toEqual({ reason: TOOL_ROUND_LIMIT, rounds: SCENARIO.rounds });
    expect(chunks.find((c) => c.type === 'error')).toBeUndefined();
    expect(completionBody(response, 'stub').webagents_finish).toEqual({ reason: TOOL_ROUND_LIMIT, blocked: false, retried: false, rounds: SCENARIO.rounds });
    expect(toolRoundLimitOf(new Error('boom'))).toBeUndefined();
  });
});

describe('what the chat and -p say at the cap', () => {
  function capError(rounds: number): Error {
    const err = new Error(`Agent reached maximum tool iterations (${rounds}). Stopping to prevent infinite loop.`);
    Object.assign(err, { code: 'max_iterations', details: { finish: { reason: TOOL_ROUND_LIMIT, blocked: false, retried: false, rounds } } });
    return err;
  }

  function draw(chunks: Array<Record<string, unknown>>): string {
    const writes: string[] = [];
    const out = { write: (s: string) => { writes.push(s); return true; }, isTTY: false };
    const printer = new TurnPrinter({ out: out as unknown as NodeJS.WriteStream, color: false, live: false });
    printer.start();
    for (const chunk of chunks) printer.feed(chunk as never);
    printer.finish();
    return writes.join('');
  }

  it('the chat says the agent stopped, not that the model failed', () => {
    const output = draw([{ type: 'error', error: capError(SCENARIO.rounds) }]);
    expect(output).toContain(SCENARIO.headline);
    expect(output).toContain(EMPTY_REPLY_HINT);
    expect(output).not.toContain('maximum tool iterations');
    expect(output).not.toContain('provider reported');
  });

  it('after an answer at the cap the printer adds nothing (the chat asks); a loop is said', () => {
    const answered = draw([
      { type: 'delta', delta: 'Two folders so far.' },
      { type: 'done', response: { content: 'Two folders so far.', finish: { reason: TOOL_ROUND_LIMIT, rounds: SCENARIO.rounds } } },
    ]);
    expect(answered).toContain('Two folders so far.');
    expect(answered).not.toContain('without an answer');
    const looped = draw([
      { type: 'delta', delta: 'The folder is empty.' },
      { type: 'done', response: { content: 'The folder is empty.', finish: { reason: 'tool_loop', rounds: 3, tool: FIXTURE.loop.tool } } },
    ]);
    expect(looped).toContain(FIXTURE.loop.says);
  });

  it('-p gives the same sentence with its own code; other errors are unchanged', () => {
    const fake = { explainFailure: (message: string) => ({ headline: `explained: ${message}` }) };
    const explain = (error: unknown) => InteractiveREPL.prototype.explainTurnError.call(fake as never, error);
    expect(explain(capError(SCENARIO.rounds))).toEqual({ headline: SCENARIO.headline, hint: EMPTY_REPLY_HINT, code: TOOL_ROUND_LIMIT });
    expect(explain(new Error('boom'))).toEqual({ headline: 'explained: boom' });
  });
});

describe('the budget is settable', () => {
  const BUDGET = FIXTURE as unknown as {
    env: string;
    rounds_words: Record<string, string>;
    precedence: Array<{ env: string | null; file: number | null; rounds: number; source: string }>;
    agent_file: { key: string; good: { text: string; rounds: number }; bad: { text: string; says: string } };
  };

  it('the flag wins over the file, and the default is fifty', () => {
    expect(MAX_TOOL_ROUNDS_ENV).toBe(BUDGET.env);
    for (const c of BUDGET.precedence) {
      const env = c.env ? { [BUDGET.env]: c.env } : {};
      expect(effectiveMaxToolRounds(c.file ?? undefined, env)).toEqual({ rounds: c.rounds, source: c.source });
    }
  });

  it('the agent file key is read, and a bad value refused with the sentence', () => {
    expect(AGENT_FILE_KEYS).toContain(BUDGET.agent_file.key);
    expect(parseAgentMarkdown(BUDGET.agent_file.good.text, 'AGENT.md').maxToolRounds).toBe(BUDGET.agent_file.good.rounds);
    expect(() => parseAgentMarkdown(BUDGET.agent_file.bad.text, 'AGENT.md')).toThrow(BUDGET.agent_file.bad.says);
  });

  it('the chat\'s words are the fixture\'s', () => {
    expect({ ...ROUNDS_WORDS }).toEqual(BUDGET.rounds_words);
  });
});

describe('the built-in agent answers small talk without tools', () => {
  const RULE =
    '- Answer greetings, thanks and small talk directly, without tools. Use a tool only when the request needs one, ' +
    'and do not explore the folder unless the person asks you to.';

  it('both ROBUTLER.md copies carry the rule and match', () => {
    const typescriptCopy = readFileSync(path.join(ROOT, 'typescript/src/agents/ROBUTLER.md'), 'utf8');
    const pythonCopy = readFileSync(path.join(ROOT, 'python/webagents/agents/builtin/ROBUTLER.md'), 'utf8');
    expect(typescriptCopy).toBe(pythonCopy);
    expect(typescriptCopy.split('## How to answer')[1]).toContain(RULE);
    expect(typescriptCopy).not.toContain('—');
  });
});
