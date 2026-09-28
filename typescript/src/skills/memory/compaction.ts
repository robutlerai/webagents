/**
 * Automatic context compaction (gap-closure plan items 2.1 and 2.4,
 * 2026-09-26): past a token threshold, the older turns of the working
 * conversation become one summary written by the agent's own model, and the
 * last `keep` turns stay verbatim. The summary also goes into the caller's
 * episodic memory (`skill.ts`), so a compacted conversation is still
 * searchable later.
 *
 * Pure functions here, so both agents can be pinned by the same cases
 * (`python/tests/fixtures/memory_tool/definition.json`, `compaction`); the
 * Python twin is `skills/local/memory/memory_compaction.py`.
 *
 * WHAT IS COUNTED. The estimate is `ceil(characters / 4)` over a message's
 * text and its tool calls' names and arguments, plus 4 per message. It is a
 * budget guard, not a tokenizer: the point is to compact well before the
 * model's window, and the same estimate in both SDKs.
 *
 * WHERE THE CUT GOES. The first system message (the agent's prompt) is never
 * summarized. Of the rest, the last `keep` messages are kept, and the cut
 * moves earlier while it would land on a tool result, so a tool call and its
 * results are never split. An earlier summary sits among the older turns and
 * is rolled into the new one.
 */

export interface CompactableMessage {
  role: string;
  content?: unknown;
  tool_calls?: unknown;
  [key: string]: unknown;
}

export const COMPACTION_PREFIX = 'Summary of the earlier part of this conversation, written by you when it grew long:\n';

export const COMPACTION_INSTRUCTIONS =
  'Summarize the conversation below for your own later use: what was asked, what was decided, what is still open, and any names, numbers or preferences worth keeping. Write plain prose, under 300 words, and nothing else.';

export const DEFAULT_COMPACTION_THRESHOLD = 60000;
export const DEFAULT_COMPACTION_KEEP = 12;

interface ToolCallLike {
  function?: { name?: unknown; arguments?: unknown };
}

function toolCallsOf(message: CompactableMessage): ToolCallLike[] {
  return Array.isArray(message.tool_calls) ? (message.tool_calls as ToolCallLike[]) : [];
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string' ? (part as { text: string }).text : ''))
      .join('');
  }
  return '';
}

/** The estimate of one message (file comment). */
export function estimateMessageTokens(message: CompactableMessage): number {
  let chars = textOf(message.content).length;
  for (const call of toolCallsOf(message)) {
    const name = typeof call.function?.name === 'string' ? call.function.name : '';
    const args = typeof call.function?.arguments === 'string' ? call.function.arguments : '';
    chars += name.length + args.length;
  }
  return Math.ceil(chars / 4) + 4;
}

export function estimateTokens(messages: readonly CompactableMessage[]): number {
  return messages.reduce((sum, m) => sum + estimateMessageTokens(m), 0);
}

/** The older turns as the summarizing model reads them: one line per message, one per tool call. */
export function transcriptOf(messages: readonly CompactableMessage[]): string {
  const lines: string[] = [];
  for (const m of messages) {
    const text = textOf(m.content);
    if (text) lines.push(`${m.role}: ${text}`);
    for (const call of toolCallsOf(m)) {
      const name = typeof call.function?.name === 'string' ? call.function.name : 'tool';
      const args = typeof call.function?.arguments === 'string' ? call.function.arguments : '';
      lines.push(`${m.role} called ${name}(${args})`);
    }
  }
  return lines.join('\n');
}

export interface CompactionPlan {
  /** The leading system message, kept as is; absent when the conversation has none. */
  head: CompactableMessage | null;
  older: CompactableMessage[];
  recent: CompactableMessage[];
}

/** Where the cut goes, or null when there is nothing to summarize (file comment). */
export function planCompaction(
  messages: readonly CompactableMessage[],
  options: { threshold: number; keep: number },
): CompactionPlan | null {
  if (messages.length === 0 || estimateTokens(messages) <= options.threshold) return null;
  const head = messages[0].role === 'system' ? messages[0] : null;
  const body = head ? messages.slice(1) : [...messages];
  const keep = Math.max(0, Math.floor(options.keep));
  if (body.length <= keep) return null;
  let cut = body.length - keep;
  while (cut > 0 && body[cut].role === 'tool') cut -= 1;
  if (cut <= 0) return null;
  return { head, older: body.slice(0, cut), recent: body.slice(cut) };
}

export interface CompactionResult {
  compacted: boolean;
  messages: CompactableMessage[];
  summary?: string;
  summarized: number;
}

/**
 * The conversation after compaction: the head, one system message carrying
 * the summary, then the recent turns. `summarize` is the agent's model (or a
 * stub under test); an empty summary leaves the conversation as it was.
 */
export async function compactConversation(
  messages: readonly CompactableMessage[],
  options: { threshold: number; keep: number; summarize: (transcript: string, instructions: string) => Promise<string> },
): Promise<CompactionResult> {
  const plan = planCompaction(messages, options);
  if (!plan) return { compacted: false, messages: [...messages], summarized: 0 };
  const summary = (await options.summarize(transcriptOf(plan.older), COMPACTION_INSTRUCTIONS)).trim();
  if (!summary) return { compacted: false, messages: [...messages], summarized: 0 };
  const out: CompactableMessage[] = [];
  if (plan.head) out.push(plan.head);
  out.push({ role: 'system', content: `${COMPACTION_PREFIX}${summary}` });
  out.push(...plan.recent);
  return { compacted: true, messages: out, summary, summarized: plan.older.length };
}
