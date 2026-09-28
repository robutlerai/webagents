/**
 * The tool-round budget of one turn (2026-09-28): `maxToolIterations`, the
 * same in both SDKs (the Python twin is `webagents/agents/core/tool_budget.py`).
 *
 * The agentic loop calls the model, runs the tools it asks for, and calls it
 * again, until the model answers or the budget is spent. The Python agent
 * hard-coded five rounds and stopped silently after the fifth: asked "nice",
 * the built-in agent explored the folder for five rounds and the turn ended
 * with no answer, while the chat blamed the provider ("the provider reported
 * STOP", the finish reason of the model's last tool call).
 *
 * Both SDKs now share one budget (default 50), warn the model at the same
 * round in the same words (half way through a budget of 20 or less, so the
 * warning fires before a short cap; 80% of a larger one), and end a turn that
 * spent the budget with the same reason, `tool_round_limit`, which the chat
 * says as "The agent stopped after N tool rounds without an answer."
 * (`cli/failures.ts`, `presentEmptyReply`). Here the reason travels as the
 * `max_iterations` error's `details.finish`: that error code is what the
 * portal and other callers already read. The cases both SDKs run are in
 * `python/tests/fixtures/agent_loop/tool_round_budget.json`.
 *
 * THE CAP IS NEVER A SILENT STOP (the owner, 2026-09-28): a turn that reaches
 * its budget makes ONE more model call with tools off and a wrap-up message,
 * so the model answers from what it gathered; the turn's finish reason is
 * still `tool_round_limit`, on its `response.done` (`finish_reason`,
 * `finish_rounds`). A turn that makes the same tool call, arguments and all,
 * `REPEAT_LIMIT` times stops the same way early, with `tool_loop`. Only when
 * that last call brings no answer does the turn end with the `max_iterations`
 * error, its finish in `details.finish`. The budget is `max_tool_rounds` in
 * the agent file, `--max-tool-rounds` on the command line and `/rounds` in
 * the chat (`parseMaxToolRounds`).
 */

/** The budget when the agent names none (`AgentConfig.maxToolIterations`). */
export const DEFAULT_MAX_TOOL_ITERATIONS = 50;
/** The finish reason of a turn that spent its budget. */
export const TOOL_ROUND_LIMIT = 'tool_round_limit';
/** The finish reason of a turn stopped for repeating one tool call. */
export const TOOL_LOOP = 'tool_loop';
/** How many identical calls (same tool, same arguments) one turn may make. */
export const REPEAT_LIMIT = 3;
/** The bounds of `max_tool_rounds` (the agent file, `--max-tool-rounds`, `/rounds`). */
export const MIN_TOOL_ROUNDS = 1;
export const MAX_TOOL_ROUNDS = 1000;
/** What a Yes to the chat's question sends as the next message. */
export const CONTINUE_MESSAGE = 'Keep going.';

/** The round at which the model is told to stop calling tools. */
export function budgetWarningRound(limit: number): number {
  const fraction = limit <= 20 ? 0.5 : 0.8;
  return Math.max(1, Math.ceil(limit * fraction));
}

/** The system message the model gets at that round, word for word the Python agent's. */
export function budgetWarning(used: number, limit: number): string {
  return (
    `You have used ${used}/${limit} of your tool-call budget for this response. ` +
    'Stop delegating and calling tools. Summarize what you have done so far and ' +
    'deliver the final answer to the user now. Do not start new workflows.'
  );
}

/**
 * What a turn that spent its budget says: the chat's headline (fixture
 * `cli/chat_fixes_empty_reply.json`).
 */
export function toolRoundLimitSentence(rounds?: number, answered = false): string {
  const tail = answered ? '.' : ' without an answer.';
  if (typeof rounds === 'number' && Number.isInteger(rounds) && rounds >= 0) {
    return `The agent stopped after ${rounds} tool round${rounds === 1 ? '' : 's'}${tail}`;
  }
  return `The agent stopped at its tool-round limit${tail}`;
}

/** The finish a turn the agent itself ended reports: the proxy skill's shape, plus the rounds that ran. */
export interface AgentFinish {
  reason: typeof TOOL_ROUND_LIMIT | typeof TOOL_LOOP;
  blocked: false;
  retried: false;
  rounds: number;
  /** The call a `tool_loop` repeated. */
  tool?: string;
}

export function toolRoundLimitFinish(rounds: number): AgentFinish {
  return { reason: TOOL_ROUND_LIMIT, blocked: false, retried: false, rounds };
}

export function toolLoopFinish(tool: string, rounds: number): AgentFinish {
  return { reason: TOOL_LOOP, blocked: false, retried: false, rounds, tool };
}

/** Whether a finish reason is the agent's own (the budget), not the provider's. */
export function isAgentFinish(reason: unknown): reason is AgentFinish['reason'] {
  return reason === TOOL_ROUND_LIMIT || reason === TOOL_LOOP;
}

/** The system message of the last, tool-less call of a turn that spent its budget. */
export function finalAnswerMessage(limit: number): string {
  return (
    `You have used all ${limit} tool rounds for this response, and tools are now off. ` +
    'Answer the user now from what you have gathered so far: what you found, what is still open, ' +
    'and what to try next.'
  );
}

/** The system message of the last, tool-less call of a turn stopped for repeating itself. */
export function loopAnswerMessage(tool: string): string {
  return (
    `You called ${tool} ${REPEAT_LIMIT} times with the same arguments, and tools are now off. ` +
    'Answer the user now from what you have gathered so far, and say what is still open.'
  );
}

/** What a turn stopped for repeating a tool call says, answer or not. */
export function toolLoopSentence(tool?: string): string {
  return tool
    ? `The agent stopped early: it called ${tool} ${REPEAT_LIMIT} times with the same arguments.`
    : 'The agent stopped early: it repeated the same tool call.';
}

/** What the interactive chat asks after a turn that spent its budget. */
export function continueQuestion(rounds: number): string {
  return `Used ${rounds} tool round${rounds === 1 ? '' : 's'}. Keep going? [Y/n]`;
}

/**
 * The finish an error carries when the agent ended the turn and its last
 * call brought no answer (the `max_iterations` error, `details.finish`),
 * else undefined. `runStreaming` and `run()` put the event's details on the
 * Error they yield or throw.
 */
export function agentFinishOf(error: unknown): { reason: string; rounds?: number; tool?: string } | undefined {
  const finish = (error as { details?: { finish?: { reason?: unknown; rounds?: unknown; tool?: unknown } } } | null | undefined)?.details?.finish;
  if (!finish || !isAgentFinish(finish.reason)) return undefined;
  return {
    reason: finish.reason,
    ...(typeof finish.rounds === 'number' ? { rounds: finish.rounds } : {}),
    ...(typeof finish.tool === 'string' && finish.tool ? { tool: finish.tool } : {}),
  };
}

/** Kept for the callers of the first version: the cap's finish only. */
export function toolRoundLimitOf(error: unknown): { reason: string; rounds?: number } | undefined {
  const finish = agentFinishOf(error);
  return finish?.reason === TOOL_ROUND_LIMIT ? { reason: finish.reason, ...(finish.rounds !== undefined ? { rounds: finish.rounds } : {}) } : undefined;
}

/** A value with its object keys sorted, for comparing arguments whatever their order. */
function sortedKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortedKeys);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value as Record<string, unknown>).sort().map((k) => [k, sortedKeys((value as Record<string, unknown>)[k])]));
  }
  return value;
}

/**
 * A tool call's arguments as one string, key order aside: parsed JSON is
 * written back with sorted keys, anything else is kept as given.
 */
export function canonicalArguments(args: unknown): string {
  let value = args;
  if (typeof args === 'string') {
    if (!args.trim()) value = {};
    else {
      try {
        value = JSON.parse(args);
      } catch {
        return args;
      }
    }
  }
  try {
    return JSON.stringify(sortedKeys(value)) ?? String(args);
  } catch {
    return String(args);
  }
}

/**
 * Counts the calls of one turn by tool and arguments. `record` answers the
 * tool's name once one call has been made `REPEAT_LIMIT` times.
 */
export class RepeatedCalls {
  private readonly counts = new Map<string, number>();

  record(name: string, args: unknown): string | undefined {
    const key = `${name}\u0000${canonicalArguments(args)}`;
    const count = (this.counts.get(key) ?? 0) + 1;
    this.counts.set(key, count);
    return count >= REPEAT_LIMIT ? name : undefined;
  }
}

/** The sentence a bad `max_tool_rounds` is refused with. */
export function roundsRefusal(name: string, value: unknown): string {
  let shown: string;
  try {
    shown = JSON.stringify(value) ?? JSON.stringify(String(value));
  } catch {
    shown = JSON.stringify(String(value));
  }
  return `${name} must be a whole number from ${MIN_TOOL_ROUNDS} to ${MAX_TOOL_ROUNDS}, not ${shown}.`;
}

/**
 * A tool-round budget from the agent file, a flag or `/rounds`: a whole
 * number (or its digits) within the bounds, else an Error with the sentence.
 */
export function parseMaxToolRounds(value: unknown, name = 'max_tool_rounds'): number {
  let n: number | undefined;
  if (typeof value === 'number' && Number.isInteger(value)) n = value;
  else if (typeof value === 'string' && /^\s*\d+\s*$/.test(value)) n = Number(value.trim());
  if (n === undefined || n < MIN_TOOL_ROUNDS || n > MAX_TOOL_ROUNDS) throw new Error(roundsRefusal(name, value));
  return n;
}

/**
 * One turn's tool rounds, the way the agent loop counts them (the Python
 * twin is `TurnBudget` in `agents/core/tool_budget.py`).
 *
 * `beginCall` runs before every model call. It counts the round, sends the
 * budget warning at its round, and decides when the NEXT call is the turn's
 * last, with tools off: the budget is spent (`tool_round_limit`), or one
 * tool call was made `REPEAT_LIMIT` times (`tool_loop`). Either way the
 * wrap-up message goes in first, after every tool result of the round (a
 * system message between an assistant's tool calls and their results is
 * refused by OpenAI-shaped APIs). `final` is then the turn's finish.
 */
export class TurnBudget {
  readonly warnAt: number;
  warned = false;
  rounds = 0;
  final: AgentFinish | undefined;
  private readonly repeats = new RepeatedCalls();
  private looped: string | undefined;

  constructor(readonly limit: number) {
    this.warnAt = budgetWarningRound(limit);
  }

  /** True when this model call is the turn's last, with tools off. */
  beginCall(conversation: Array<{ role: 'system'; content: string } | object>): boolean {
    if (!this.final && this.looped !== undefined) {
      this.final = toolLoopFinish(this.looped, this.rounds);
      conversation.push({ role: 'system', content: loopAnswerMessage(this.looped) });
    } else if (!this.final && this.rounds >= this.limit) {
      this.final = toolRoundLimitFinish(this.rounds);
      conversation.push({ role: 'system', content: finalAnswerMessage(this.limit) });
    }
    if (this.final) return true;
    this.rounds += 1;
    if (!this.warned && this.rounds >= this.warnAt) {
      this.warned = true;
      conversation.push({ role: 'system', content: budgetWarning(this.rounds, this.limit) });
    }
    return false;
  }

  /** After the model asked for a tool: the first call to reach the repeat limit is remembered. */
  recordCall(name: string, args: unknown): void {
    const repeated = this.repeats.record(name, args);
    if (repeated !== undefined && this.looped === undefined) this.looped = repeated;
  }
}

/**
 * Where `--max-tool-rounds` puts its value for the agent builders (and a
 * child process): the flag is validated once, at the root.
 */
export const MAX_TOOL_ROUNDS_ENV = 'WEBAGENTS_MAX_TOOL_ROUNDS';

/** What the chat says about the budget (`/rounds`, `/status`): the fixture's `rounds_words`, the Python chat's `ROUNDS_WORDS`. */
export const ROUNDS_WORDS: Readonly<Record<string, string>> = {
  show: 'Tool rounds: {rounds} per turn ({source}).',
  showHint: 'Change with /rounds <n>; add --save to keep it in the agent file.',
  set: 'Tool rounds set to {rounds} per turn, for this chat.',
  setHint: '/rounds {rounds} --save keeps it in {file}.',
  kept: 'Tool rounds {rounds} kept in {file}.',
  already: '{file} already keeps max_tool_rounds: {rounds}.',
  status: '{rounds} per turn ({source})',
  session: 'set in this chat',
  flag: '--max-tool-rounds',
  file: '{file}',
  default: 'the default',
};

/** A `ROUNDS_WORDS` sentence with its holes filled. */
export function roundsWords(key: string, values: Record<string, string | number> = {}): string {
  return (ROUNDS_WORDS[key] ?? key).replace(/\{(\w+)\}/g, (hole, name: string) => (name in values ? String(values[name]) : hole));
}

/** Where the budget came from, in the chat's words. */
export function roundsSourceWords(source: string, file?: string): string {
  return roundsWords(source, { file: file ?? 'the agent file' });
}

/**
 * The budget an agent built here runs with, and where it came from, most
 * specific first: `--max-tool-rounds` (its environment value), the agent
 * file's `max_tool_rounds`, the default. The chat's `/rounds` sits above all
 * three. A caller of a served agent has no say: no request field reaches it.
 */
export function effectiveMaxToolRounds(
  fileValue: number | undefined,
  env: Record<string, string | undefined> = process.env,
): { rounds: number; source: 'flag' | 'file' | 'default' } {
  const flag = env[MAX_TOOL_ROUNDS_ENV];
  if (flag !== undefined && flag.trim() !== '') return { rounds: parseMaxToolRounds(flag, '--max-tool-rounds'), source: 'flag' };
  if (fileValue !== undefined) return { rounds: fileValue, source: 'file' };
  return { rounds: DEFAULT_MAX_TOOL_ITERATIONS, source: 'default' };
}
