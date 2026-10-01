/**
 * Compaction of a conversation that is filling the model's context
 * (2026-09-29, the owner: "what's the best strategy for auto-compaction? ...
 * logic/settings with good defaults and command and api surface too?").
 *
 * WHY HERE. Compaction lived in the `memory` skill (a `before_llm_call`
 * hook), so an agent without `memory` never compacted and a long chat ran
 * until the provider refused it; and the hook compacted the run's copy of the
 * history while the chat kept and re-sent the whole of it, so past the
 * threshold every turn paid for a new summary and saved another episode. It
 * is now the agent's: the chat compacts its own history between turns (and
 * keeps the result), a run compacts only as a safety stop inside one long
 * turn, and `BaseAgent.compact()` is there for any other host. The memory
 * skill keeps the summary as an episode (`onCompaction`), once per compaction.
 *
 * THE POLICY (`CompactionPolicy`, the agent file's `compaction:` block):
 * `auto` (true), `at` (0.8 of the window, or tokens above 1), `keep` (0.25:
 * the recent part kept as it is), `hard` (0.95: where a run compacts by itself
 * inside one long turn), `clear_tool_results` (true), `model` (the summary's
 * model, default the agent's own), `instructions` (what the summary should
 * also keep) and `window` (for a model the table below does not know).
 *
 * THE STRATEGY, cheapest first: nothing under `at`; past it, (1) the output of
 * earlier tool calls is cleared (a one-line note names the tool and the size),
 * which is all when it brings the conversation under three quarters of `at`;
 * (2) otherwise the earlier part becomes one summary written by the model, an
 * earlier summary rolled into it; (3) if no summary can be made, the oldest
 * whole turns are dropped until it fits, and a note says how many. `/compact`
 * forces step 2. The system prompt is never touched. Counting is
 * `ceil(characters / 4) + 4` per message, the same in both SDKs.
 *
 * The Python twin is `python/webagents/agents/core/context_compaction.py`;
 * both are pinned by `python/tests/fixtures/context/compaction.json`.
 */

export const WORDS = {
  compacted:
    'Compacted the conversation: {summarized} earlier messages became a summary, the last {kept} stay as they were. Context {percent}% full.',
  cleared: 'Made room by clearing the output of {cleared} earlier tool calls. Context {percent}% full.',
  dropped: 'Could not summarize the conversation ({reason}); dropped its {dropped} oldest messages instead. Context {percent}% full.',
  nothing: 'Nothing to compact yet: the conversation is {percent}% of the context.',
  stub: '[output cleared to make room: {tool}, {chars} characters]',
  summaryPrefix: 'Summary of the earlier part of this conversation, written when it grew long:\n',
  droppedNote: '[{dropped} earlier messages were dropped to make room]',
  instructions:
    'Summarize the conversation below so that it can be continued from the summary alone. Keep: what the person asked for; what was decided, and why; what was done (files, commands, results); where things stand and what is still open; and the names, numbers, paths and preferences worth keeping. Plain prose or short lists, under 400 words, and nothing else.',
  focus: 'Pay particular attention to: {focus}',
  emptySummary: 'the model answered with nothing',
} as const;

/**
 * The context windows compaction measures against, by model prefix, first
 * match wins. A claim to check when the providers' catalogs change; a model
 * not listed gets `DEFAULT_WINDOW`, and `compaction.window` overrides both.
 */
export const CONTEXT_WINDOWS: ReadonlyArray<readonly [string, number]> = [
  ['openai/gpt-4.1', 1047576],
  ['openai/gpt-5', 400000],
  ['openai/o1', 200000],
  ['openai/o3', 200000],
  ['openai/o4', 200000],
  ['openai/', 128000],
  ['anthropic/', 200000],
  ['google/', 1048576],
  ['xai/grok-4-fast', 2000000],
  ['xai/grok-4', 256000],
  ['xai/', 131072],
  ['fireworks/', 131072],
  ['ollama/', 32768],
];
export const DEFAULT_WINDOW = 128000;
/**
 * Set in a run's metadata while it writes a compaction summary (the agent
 * summarizes in a run of its own), so nothing in it compacts again and the
 * memory notes stay out of it.
 */
export const COMPACTION_RUN = 'context_compaction';

/** A tool output shorter than this is left alone by the clearing step. */
export const CLEAR_MIN_CHARS = 400;
/** Where compaction aims to land, as a part of `at`, so it does not run again on the next turn. */
export const SETTLE = 0.75;

export const POLICY_KEYS = ['auto', 'at', 'keep', 'hard', 'clear_tool_results', 'model', 'instructions', 'window'] as const;
export const POLICY_WORDS = {
  notMapping: 'compaction must be a mapping of auto, at, keep, hard, clear_tool_results, model, instructions and window.',
  unknownKey: 'compaction: unknown key "{key}". It takes auto, at, keep, hard, clear_tool_results, model, instructions and window.',
  notBool: 'compaction.{key} must be true or false.',
  notAmount: 'compaction.{key} must be a part of the context between 0 and 1, or a number of tokens.',
  notText: 'compaction.{key} must be text.',
  notWindow: 'compaction.window must be a number of tokens.',
  atBelowHard: 'compaction.at must be below compaction.hard.',
} as const;

export interface CompactionPolicy {
  auto: boolean;
  at: number;
  keep: number;
  hard: number;
  clearToolResults: boolean;
  model?: string;
  instructions?: string;
  window?: number;
}

export const DEFAULT_POLICY: Readonly<CompactionPolicy> = { auto: true, at: 0.8, keep: 0.25, hard: 0.95, clearToolResults: true };

/** A `compaction:` block that cannot be read; the message says why. */
export class CompactionPolicyError extends Error {}

/** The agent file's `compaction:` block as a policy; null or absent is every default. */
export function parsePolicy(raw: unknown): CompactionPolicy {
  if (raw === null || raw === undefined) return { ...DEFAULT_POLICY };
  if (typeof raw !== 'object' || Array.isArray(raw)) throw new CompactionPolicyError(POLICY_WORDS.notMapping);
  const policy: CompactionPolicy = { ...DEFAULT_POLICY };
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!(POLICY_KEYS as readonly string[]).includes(key)) throw new CompactionPolicyError(POLICY_WORDS.unknownKey.replace('{key}', key));
    if (key === 'auto' || key === 'clear_tool_results') {
      if (typeof value !== 'boolean') throw new CompactionPolicyError(POLICY_WORDS.notBool.replace('{key}', key));
      if (key === 'auto') policy.auto = value;
      else policy.clearToolResults = value;
    } else if (key === 'at' || key === 'keep' || key === 'hard') {
      if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || (value > 1 && !Number.isInteger(value))) {
        throw new CompactionPolicyError(POLICY_WORDS.notAmount.replace('{key}', key));
      }
      policy[key] = value;
    } else if (key === 'model' || key === 'instructions') {
      if (typeof value !== 'string' || !value.trim()) throw new CompactionPolicyError(POLICY_WORDS.notText.replace('{key}', key));
      policy[key] = value.trim();
    } else {
      if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) throw new CompactionPolicyError(POLICY_WORDS.notWindow);
      policy.window = value;
    }
  }
  if (policy.at <= 1 === policy.hard <= 1 && policy.at >= policy.hard) throw new CompactionPolicyError(POLICY_WORDS.atBelowHard);
  return policy;
}

/**
 * The policy an agent runs under: the agent file's `compaction:` block; or,
 * without one, the `memory` skill's older `compaction.threshold` (tokens) as
 * its `at`, so a file that set it keeps it; or every default.
 */
export function effectivePolicy(parsed: CompactionPolicy | undefined, legacyThreshold: number | undefined): CompactionPolicy {
  if (parsed) return parsed;
  if (legacyThreshold) return { ...DEFAULT_POLICY, at: legacyThreshold, hard: Math.max(legacyThreshold + 1, Math.trunc(legacyThreshold * 1.2)) };
  return { ...DEFAULT_POLICY };
}

/** The context window of `provider/model` (`CONTEXT_WINDOWS`), or `override`. */
export function contextWindow(model: string | undefined | null, override?: number): number {
  if (override) return override;
  const name = (model ?? '').toLowerCase();
  return CONTEXT_WINDOWS.find(([prefix]) => name.startsWith(prefix))?.[1] ?? DEFAULT_WINDOW;
}

/** A policy amount in tokens: a fraction of the window, or tokens already. */
export function tokensOf(amount: number, window: number): number {
  return amount > 1 ? Math.trunc(amount) : Math.trunc(amount * window);
}

// -- counting -------------------------------------------------------------------------------

export interface CompactMessage {
  role: string;
  content?: unknown;
  tool_calls?: unknown;
  tool_call_id?: string;
  name?: string;
  [key: string]: unknown;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string' ? (part as { text: string }).text : '')).join('');
  }
  return '';
}

function toolCallsOf(message: CompactMessage): Array<Record<string, unknown>> {
  return Array.isArray(message.tool_calls) ? (message.tool_calls as unknown[]).filter((c): c is Record<string, unknown> => !!c && typeof c === 'object') : [];
}

function callParts(call: Record<string, unknown>): [string, string] {
  const fn = call.function && typeof call.function === 'object' ? (call.function as Record<string, unknown>) : {};
  return [typeof fn.name === 'string' ? fn.name : '', typeof fn.arguments === 'string' ? fn.arguments : ''];
}

/** `ceil(characters / 4) + 4` for one message (file comment). */
export function estimateMessageTokens(message: CompactMessage): number {
  let chars = textOf(message.content).length;
  for (const call of toolCallsOf(message)) {
    const [name, args] = callParts(call);
    chars += name.length + args.length;
  }
  return Math.ceil(chars / 4) + 4;
}

export function estimateTokens(messages: readonly CompactMessage[]): number {
  return messages.reduce((sum, m) => sum + estimateMessageTokens(m), 0);
}

export function percentOf(tokens: number, window: number): number {
  return window ? Math.min(999, Math.round((100 * tokens) / window)) : 0;
}

/** The earlier turns as the summarizing model reads them: one line per message, one per tool call. */
export function transcriptOf(messages: readonly CompactMessage[]): string {
  const lines: string[] = [];
  for (const m of messages) {
    const text = textOf(m.content);
    if (text) lines.push(`${m.role}: ${text}`);
    for (const call of toolCallsOf(m)) {
      const [name, args] = callParts(call);
      lines.push(`${m.role} called ${name || 'tool'}(${args})`);
    }
  }
  return lines.join('\n');
}

// -- the steps ------------------------------------------------------------------------------

function isCompactionNote(message: CompactMessage): boolean {
  return message.role === 'system' && textOf(message.content).startsWith(WORDS.summaryPrefix);
}

/** The agent's system prompt (the first message, when it is one) and the rest. */
function splitHead(messages: readonly CompactMessage[]): [CompactMessage[], CompactMessage[]] {
  if (messages.length && messages[0].role === 'system' && !isCompactionNote(messages[0])) return [[messages[0]], messages.slice(1)];
  return [[], [...messages]];
}

/**
 * Where the recent part starts: walking back from the end while it fits in
 * `keepTokens` (the last message always kept), never after `protectFrom` (the
 * turn in progress), and never on a tool result, so a call and its results
 * stay together.
 */
export function recentCut(body: readonly CompactMessage[], keepTokens: number, protectFrom?: number): number {
  if (!body.length) return 0;
  let cut = body.length - 1;
  let used = estimateMessageTokens(body[cut]);
  while (cut > 0 && used + estimateMessageTokens(body[cut - 1]) <= keepTokens) {
    cut -= 1;
    used += estimateMessageTokens(body[cut]);
  }
  if (protectFrom !== undefined) cut = Math.min(cut, Math.max(0, protectFrom));
  while (cut > 0 && body[cut].role === 'tool') cut -= 1;
  return cut;
}

/** `messages` with each tool output of `CLEAR_MIN_CHARS` or more replaced by a one-line note, and how many. */
export function clearToolResults(messages: readonly CompactMessage[]): [CompactMessage[], number] {
  const names = new Map<string, string>();
  let cleared = 0;
  const out = messages.map((m) => {
    for (const call of toolCallsOf(m)) if (typeof call.id === 'string') names.set(call.id, callParts(call)[0] || 'tool');
    const text = textOf(m.content);
    if (m.role === 'tool' && text.length >= CLEAR_MIN_CHARS && !text.startsWith('[output cleared')) {
      cleared += 1;
      const tool = names.get(String(m.tool_call_id)) ?? (typeof m.name === 'string' ? m.name : 'tool');
      return { ...m, content: WORDS.stub.replace('{tool}', tool).replace('{chars}', String(text.length)) };
    }
    return m;
  });
  return [out, cleared];
}

/** `older` without its oldest whole turns, dropped until `fits` says the rest fits; and how many went. */
export function dropOldestTurns(older: readonly CompactMessage[], fits: (rest: CompactMessage[]) => boolean): [CompactMessage[], number] {
  let rest = [...older];
  let dropped = 0;
  while (rest.length && !fits(rest)) {
    let end = rest.findIndex((m, i) => i > 0 && m.role === 'user');
    if (end < 0) end = rest.length;
    dropped += end;
    rest = rest.slice(end);
  }
  return [rest, dropped];
}

export type Summarizer = (transcript: string, instructions: string) => Promise<string>;

export type CompactionStage = 'none' | 'cleared' | 'summarized' | 'dropped';

/** What compaction did. */
export interface Compaction {
  stage: CompactionStage;
  messages: CompactMessage[];
  before: number;
  after: number;
  window: number;
  summary?: string;
  summarized: number;
  kept: number;
  cleared: number;
  dropped: number;
  reason?: string;
}

function outcome(stage: CompactionStage, messages: CompactMessage[], before: number, window: number, extra: Partial<Compaction> = {}): Compaction {
  return { stage, messages, before, after: stage === 'none' ? before : estimateTokens(messages), window, summarized: 0, kept: 0, cleared: 0, dropped: 0, ...extra };
}

/** The chat's one line for a compaction (`WORDS`). */
export function compactionSentence(c: Compaction): string {
  const percent = String(percentOf(c.after, c.window));
  if (c.stage === 'summarized') return WORDS.compacted.replace('{summarized}', String(c.summarized)).replace('{kept}', String(c.kept)).replace('{percent}', percent);
  if (c.stage === 'cleared') return WORDS.cleared.replace('{cleared}', String(c.cleared)).replace('{percent}', percent);
  if (c.stage === 'dropped') {
    return WORDS.dropped.replace('{reason}', c.reason ?? WORDS.emptySummary).replace('{dropped}', String(c.dropped)).replace('{percent}', percent);
  }
  return WORDS.nothing.replace('{percent}', String(percentOf(c.before, c.window)));
}

export function summaryInstructions(policy: CompactionPolicy, focus?: string): string {
  const parts: string[] = [WORDS.instructions];
  if (policy.instructions) parts.push(policy.instructions);
  if (focus && focus.trim()) parts.push(WORDS.focus.replace('{focus}', focus.trim()));
  return parts.join('\n');
}

/**
 * The conversation after compaction (file comment). `force` is `/compact`:
 * it summarizes whatever the size. `protectFrom` is the index, in `messages`,
 * where the turn in progress starts: nothing from there on is touched.
 * `threshold` defaults to the policy's `at`.
 */
export async function compactMessages(
  messages: readonly CompactMessage[],
  policy: CompactionPolicy,
  window: number,
  summarize: Summarizer,
  options: { force?: boolean; focus?: string; protectFrom?: number; threshold?: number } = {},
): Promise<Compaction> {
  const before = estimateTokens(messages);
  const limit = options.threshold ?? tokensOf(policy.at, window);
  const none = outcome('none', [...messages], before, window);
  if (!options.force && before <= limit) return none;
  const settle = Math.trunc(tokensOf(policy.at, window) * SETTLE);
  const [head, body] = splitHead(messages);
  const protect = options.protectFrom === undefined ? undefined : options.protectFrom - head.length;
  const lastAsk = turnStart(body);
  // `/compact`: everything before the latest exchange becomes the summary,
  // whatever the `keep` budget would have kept.
  const cut =
    options.force && lastAsk ? (protect === undefined ? lastAsk : Math.min(lastAsk, Math.max(0, protect))) : recentCut(body, tokensOf(policy.keep, window), protect);
  const older = body.slice(0, cut);
  const recent = body.slice(cut);
  if (!older.length) return none;
  const [clearedOlder, cleared] = policy.clearToolResults ? clearToolResults(older) : [[...older], 0];
  if (cleared && !options.force) {
    const candidate = [...head, ...clearedOlder, ...recent];
    if (estimateTokens(candidate) <= settle) return outcome('cleared', candidate, before, window, { cleared, kept: recent.length });
  }
  let summary = '';
  let reason: string | undefined;
  try {
    summary = (await summarize(transcriptOf(clearedOlder), summaryInstructions(policy, options.focus))).trim();
  } catch (error) {
    reason = (error as Error)?.message || (error as Error)?.name || 'error';
  }
  if (summary) {
    const out = [...head, { role: 'system', content: `${WORDS.summaryPrefix}${summary}` }, ...recent];
    return outcome('summarized', out, before, window, { summary, summarized: older.length, kept: recent.length, cleared });
  }
  const [rest, dropped] = dropOldestTurns(clearedOlder, (r) => estimateTokens([...head, ...r, ...recent]) + 12 <= settle);
  if (!dropped) {
    if (!cleared) return { ...none, reason: reason ?? WORDS.emptySummary };
    return outcome('cleared', [...head, ...clearedOlder, ...recent], before, window, { cleared, kept: recent.length, reason: reason ?? WORDS.emptySummary });
  }
  const out = [...head, { role: 'system', content: WORDS.droppedNote.replace('{dropped}', String(dropped)) }, ...rest, ...recent];
  return outcome('dropped', out, before, window, { dropped, cleared, kept: recent.length, reason: reason ?? WORDS.emptySummary });
}

/** The index of the last message from the person: where the turn in progress starts. */
export function turnStart(messages: readonly CompactMessage[]): number | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) if (messages[i].role === 'user') return i;
  return undefined;
}
