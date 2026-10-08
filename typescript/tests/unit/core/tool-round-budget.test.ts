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
  RepeatedCalls,
  TOOL_LOOP,
  TOOL_ROUND_LIMIT,
  TurnBudget,
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
  one_detector: { results_seen: number; old_nudge: string };
  repeats: Array<{ name: string; calls: Array<[string, Record<string, unknown>, string]>; looped_at: number | null }>;
  edit_and_rerun: { limit: number; run_tool: string; write_tool: string; results: string[]; model_calls: number; rounds: number; answer: string };
};
const SCENARIO = FIXTURE.scenario;
const RERUN = FIXTURE.edit_and_rerun;

/**
 * A model that asks for `add` whenever its tools are on (new arguments each
 * call, or the same ones with `sameArgs`), answers at call `answerAt`, and
 * answers the fixture's answer on the wrap-up call (or nothing, with
 * `silentLast`). The wrap-up call is told apart by the budget's system
 * message, the way a real model reads it: the tool definitions stay listed
 * on that call so the provider's cached prefix survives, and the agent
 * passes `_agentic_tool_choice: 'none'` instead. With `ignoresWrapUp` the
 * model keeps asking for tools whenever any are listed, as a model that
 * never read the wrap-up would.
 */
function keepsAskingForTools(options: { answerAt?: number; sameArgs?: boolean; silentLast?: boolean; ignoresWrapUp?: boolean } = {}) {
  let calls = 0;
  const seen: AgenticMessage[][] = [];
  const toolsOffered: boolean[] = [];
  const toolChoices: string[] = [];
  class KeepsAsking extends Skill {
    @handoff({ name: 'keeps-asking' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      calls++;
      const conversation = [...(context.get<AgenticMessage[]>('_agentic_messages') ?? [])];
      seen.push(conversation);
      const hasTools = (context.get<unknown[]>('_agentic_tools') ?? []).length > 0;
      toolsOffered.push(hasTools);
      toolChoices.push(String(context.get<string>('_agentic_tool_choice') ?? ''));
      const wrapUp = conversation.some((m) => m.role === 'system' && (String(m.content).startsWith('You have used all') || String(m.content) === loopAnswerMessage('add')));
      const toolsOff = options.ignoresWrapUp ? !hasTools : (!hasTools || wrapUp);
      const responseId = generateEventId();
      yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
      const output: ContentItem[] = [];
      const text = toolsOff ? (options.silentLast ? '' : SCENARIO.answer) : options.answerAt !== undefined && calls >= options.answerAt ? 'done' : undefined;
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
  return { skill: new KeepsAsking(), calls: () => calls, seen, toolsOffered, toolChoices };
}

class MathTools extends Skill {
  @tool({ description: 'Add two numbers' })
  async add(params: { a: number; b: number }, _c: Context): Promise<number> {
    return params.a + params.b;
  }
}

/**
 * A model that runs a script, rewrites it and runs it again, twice, then
 * answers: the same `run_script` call three times with an edit in between
 * each (the data-analysis turn of 2026-09-29, fixture `edit_and_rerun`).
 */
function editsAndReruns() {
  let calls = 0;
  const toolsOffered: boolean[] = [];
  const sequence = [RERUN.run_tool, RERUN.write_tool, RERUN.run_tool, RERUN.write_tool, RERUN.run_tool];
  class EditsAndReruns extends Skill {
    @handoff({ name: 'edits-and-reruns' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      calls++;
      const hasTools = (context.get<unknown[]>('_agentic_tools') ?? []).length > 0;
      toolsOffered.push(hasTools);
      const responseId = generateEventId();
      yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
      const output: ContentItem[] = [];
      const name = hasTools ? sequence[calls - 1] : undefined;
      if (name === undefined) {
        yield createResponseDeltaEvent(responseId, { type: 'text', text: RERUN.answer });
        output.push({ type: 'text', text: RERUN.answer });
      } else {
        const args = name === RERUN.run_tool ? '{"command":"python3 analyze.py"}' : `{"file_path":"analyze.py","content":"attempt ${calls}"}`;
        output.push({ type: 'tool_call', tool_call: { id: `call_${calls}`, name, arguments: args } });
      }
      yield createResponseDoneEvent(responseId, output);
    }
  }
  return { skill: new EditsAndReruns(), calls: () => calls, toolsOffered };
}

/** `run_script` answers the fixture's results in turn; `write_script` always says the same thing. */
class ScriptTools extends Skill {
  private runs = 0;

  @tool({ name: 'run_script', description: 'Runs the script' })
  async run_script(_params: { command?: string }, _c: Context): Promise<string> {
    const result = RERUN.results[Math.min(this.runs, RERUN.results.length - 1)]!;
    this.runs += 1;
    return result;
  }

  @tool({ name: 'write_script', description: 'Writes the script' })
  async write_script(_params: { file_path?: string; content?: string }, _c: Context): Promise<string> {
    return 'Successfully overwrote file: analyze.py';
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
    // The last call lists the same tools as the ones before it (the cached
    // prefix survives) and says `none` for the tool choice instead.
    expect(model.toolsOffered).toEqual(Array(SCENARIO.model_calls).fill(true));
    expect(model.toolChoices).toEqual([...Array(SCENARIO.rounds).fill('auto'), 'none']);
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

  it('a last call with no answer is made once more with the tools off, then ends with max_iterations and the finish', async () => {
    const model = keepsAskingForTools({ silentLast: true });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: SCENARIO.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    vi.restoreAllMocks();
    // One more call than the fixture's, with no tools at all: the shape
    // every model answered before the tools stayed listed on the last call.
    expect(model.calls()).toBe(SCENARIO.model_calls + 1);
    expect(model.toolsOffered).toEqual([...Array(SCENARIO.model_calls).fill(true), false]);
    expect(model.seen.at(-1)).toEqual(model.seen.at(-2));
    const error = events.find((e) => e.type === 'response.error') as unknown as { error: { code: string; details?: unknown } };
    expect(error.error.code).toBe('max_iterations');
    expect(error.error.details).toEqual({ finish: { reason: TOOL_ROUND_LIMIT, blocked: false, retried: false, rounds: SCENARIO.rounds } });
  });

  it('a model that ignores the wrap-up and asks for a tool is not run; it gets one tool-less call and answers there', async () => {
    const model = keepsAskingForTools({ ignoresWrapUp: true });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: SCENARIO.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    vi.restoreAllMocks();
    expect(model.calls()).toBe(SCENARIO.model_calls + 1);
    expect(model.toolsOffered.at(-1)).toBe(false);
    // The ignored call's tool call ran nothing: one result per budgeted round.
    const toolResults = events.filter((e) => e.type === 'response.delta' && (e as unknown as { delta: { type: string } }).delta.type === 'tool_result');
    expect(toolResults).toHaveLength(SCENARIO.rounds);
    expect(events.find((e) => e.type === 'response.error')).toBeUndefined();
    const done = events.find((e) => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBe(TOOL_ROUND_LIMIT);
  });

  it('the same call three times stops early with the loop reason', async () => {
    const model = keepsAskingForTools({ sameArgs: true });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const agent = new BaseAgent({ skills: [model.skill, new MathTools()], maxToolIterations: FIXTURE.loop.limit });
    const events = await collect(agent.processUAMP(inputEvents('nice')));
    vi.restoreAllMocks();
    expect(model.calls()).toBe(FIXTURE.loop.model_calls);
    expect(model.toolsOffered.at(-1)).toBe(true);
    expect(model.toolChoices.at(-1)).toBe('none');
    expect(model.seen.at(-1)!.at(-1)).toEqual({ role: 'system', content: loopAnswerMessage('add') });
    expect(loopAnswerMessage(FIXTURE.loop.tool)).toBe(FIXTURE.loop_answer.replace('{tool}', FIXTURE.loop.tool));
    const done = events.find((e) => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBe(FIXTURE.loop_finish_reason);
    expect(done.response.finish_tool).toBe('add');
    expect(done.response.finish_rounds).toBe(FIXTURE.loop.rounds);
    expect(toolLoopSentence(FIXTURE.loop.tool)).toBe(FIXTURE.loop.says);
    // ONE detector (2026-09-29, fixture `one_detector`): every result reaches
    // the model as the tool returned it (`add` 1+1, stringified: "2"), the
    // client's tool_result events carry the same, and only the wrap-up
    // message is added. The older detector rewrote the third result here.
    const one = FIXTURE.one_detector;
    const toolMessages = model.seen.at(-1)!.filter((m) => m.role === 'tool').map((m) => String(m.content));
    expect(toolMessages).toEqual(Array(one.results_seen).fill('2'));
    const results = events
      .filter((e) => e.type === 'response.delta' && (e as unknown as { delta: { type: string } }).delta.type === 'tool_result')
      .map((e) => (e as unknown as { delta: { tool_result: { result: string } } }).delta.tool_result.result);
    expect(results).toEqual(Array(one.results_seen).fill('2'));
    for (const m of model.seen.at(-1)!) expect(String(m.content)).not.toContain(one.old_nudge);
    const systemMessages = model.seen.at(-1)!.filter((m) => m.role === 'system').map((m) => String(m.content));
    expect(systemMessages.at(-1)).toBe(loopAnswerMessage('add'));
  });

  it('recordCall answers the streak, which starts over with a different call', () => {
    // So the loop can trace a repeat before the stop (the Python `record_call` too).
    const budget = new TurnBudget(10);
    expect(budget.recordCall('look', { path: '.' }, 'src')).toBe(1);
    expect(budget.recordCall('look', { path: '.' }, 'src')).toBe(2);
    expect(budget.recordCall('read', { path: 'a' }, 'text')).toBe(1);
    expect(budget.recordCall('look', { path: '.' }, 'src')).toBe(1);
    expect(budget.recordCall('look', { path: '.' }, 'src')).toBe(2);
    expect(budget.recordCall('look', { path: '.' }, 'src')).toBe(REPEAT_LIMIT);
    expect(budget.beginCall([])).toBe(true);
  });

  it('same arguments are the fixture\'s', () => {
    expect(REPEAT_LIMIT).toBe(FIXTURE.repeat_limit);
    for (const c of FIXTURE.same_arguments) expect(canonicalArguments(c.a) === canonicalArguments(c.b)).toBe(c.same);
  });

  // A repeat counts only when nothing changed in between (2026-09-29, the
  // data-analysis turn of the skills e2e): `python3 analyze.py`, the script
  // rewritten, run again, twice, was stopped as a `tool_loop` at the third
  // run because the count read only the name and the arguments.
  it('a repeat counts only when nothing changed in between', () => {
    for (const c of FIXTURE.repeats) {
      const repeats = new RepeatedCalls();
      const answered = c.calls.map(([name, args, result]) => repeats.record(name, args, result));
      const first = answered.findIndex((tool) => tool !== undefined);
      expect(first === -1 ? null : first + 1, c.name).toBe(c.looped_at);
    }
  });

  it('an edit-and-rerun cycle is not a loop: every call keeps its tools and the answer is the model\'s own', async () => {
    const model = editsAndReruns();
    const agent = new BaseAgent({ skills: [model.skill, new ScriptTools()], maxToolIterations: RERUN.limit });
    const events = await collect(agent.processUAMP(inputEvents('should we roll it out?')));
    expect(model.calls()).toBe(RERUN.model_calls);
    expect(model.toolsOffered).toEqual(Array(RERUN.model_calls).fill(true));
    expect(events.find((e) => e.type === 'response.error')).toBeUndefined();
    const done = events.find((e) => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBeUndefined();
    const texts = events
      .filter((e) => e.type === 'response.delta')
      .map((e) => (e as unknown as { delta: { type: string; text?: string } }).delta)
      .filter((d) => d.type === 'text')
      .map((d) => d.text ?? '')
      .join('');
    expect(texts).toBe(RERUN.answer);
    // `run()` reports no agent finish either, so `-p` exits 0 with the answer.
    const again = editsAndReruns();
    const fresh = new BaseAgent({ skills: [again.skill, new ScriptTools()], maxToolIterations: RERUN.limit });
    const response = await fresh.run([{ role: 'user', content: 'should we roll it out?' }]);
    expect(again.calls()).toBe(RERUN.model_calls);
    expect(response.finish).toBeUndefined();
    expect(response.content).toBe(RERUN.answer);
    expect(TOOL_LOOP).toBe(FIXTURE.loop_finish_reason);
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
