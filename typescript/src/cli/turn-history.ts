/**
 * What the chat keeps of a turn's tool calls, and how much of it goes back to
 * the model (2026-09-28).
 *
 * WHY. The chat kept only the person's message and the final answer of each
 * turn; the tool calls and what they returned were dropped. On the next
 * message the model had no record of the folder it had listed or the files it
 * had read, and a tool-eager model (Gemini 3.5 Flash, behind `auto/balanced`)
 * listed and read them again, every message: three messages of small talk cost
 * 21k, 27k and 45k tokens. The chat now keeps each turn's tool rounds between
 * the message and the answer, the way the agent loop itself sends them.
 *
 * THE BUDGET. Kept results add up, so what goes back to the model keeps the
 * newest results whole up to `TOOL_HISTORY_BUDGET_CHARS` and replaces older
 * ones with a one-line note that says what was left out; the call itself stays,
 * so the model still knows what it did and can call again. The conversation on
 * disk keeps everything; only the copy sent is trimmed.
 *
 * ONE BEHAVIOUR IN BOTH SDKS: `python/webagents/cli/turn_history.py`, and the
 * cases in `python/tests/fixtures/chat/turn_history.json`.
 */

import type { StreamChunk } from '../core/types';
import type { Message } from '../uamp/types';

/** Characters of tool results the model gets back whole, newest first. */
export const TOOL_HISTORY_BUDGET_CHARS = 40_000;

/** What a result past the budget becomes (agent-facing). */
export function leftOutResult(chars: number): string {
  return `[An earlier tool result was left out to save space (${chars} characters). Call the tool again if you need it.]`;
}

interface Round {
  calls: { id: string; name: string; arguments: string }[];
  results: Map<string, string>;
}

/**
 * Collects one turn's tool calls and results from its stream, in rounds: a call
 * that arrives after a result of the current round starts the next round. A call
 * that never got a result (the turn was stopped) is not kept, so every kept call
 * has its result, as the providers require.
 */
export class TurnRecorder {
  private rounds: Round[] = [];

  observe(chunk: StreamChunk): void {
    if (chunk.type === 'tool_call' && chunk.tool_call?.id) {
      let round = this.rounds[this.rounds.length - 1];
      if (!round || round.results.size > 0) {
        round = { calls: [], results: new Map() };
        this.rounds.push(round);
      }
      const { id, name, arguments: args } = chunk.tool_call;
      round.calls.push({ id, name, arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}) });
    } else if (chunk.type === 'tool_result' && chunk.tool_result?.call_id) {
      const round = this.rounds.find((r) => r.calls.some((c) => c.id === chunk.tool_result!.call_id));
      if (round) round.results.set(chunk.tool_result.call_id, String(chunk.tool_result.result ?? ''));
    }
  }

  /** The rounds as the agent loop sends them: an assistant message with the calls, then one tool message per result. */
  messages(): Message[] {
    const out: Message[] = [];
    for (const round of this.rounds) {
      const answered = round.calls.filter((c) => round.results.has(c.id));
      if (!answered.length) continue;
      out.push({
        role: 'assistant',
        content: null,
        tool_calls: answered.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } })),
      } as unknown as Message);
      for (const c of answered) {
        out.push({ role: 'tool', tool_call_id: c.id, name: c.name, content: round.results.get(c.id)! } as Message);
      }
    }
    return out;
  }
}

/**
 * The copy of the conversation sent to the model: tool results whole, newest
 * first, until `budget` characters; older ones replaced by `leftOutResult`.
 */
export function historyForModel(messages: Message[], budget: number = TOOL_HISTORY_BUDGET_CHARS): Message[] {
  let left = budget;
  const out = messages.slice();
  for (let i = out.length - 1; i >= 0; i--) {
    const m = out[i];
    if (m.role !== 'tool' || typeof m.content !== 'string') continue;
    if (m.content.length <= left) {
      left -= m.content.length;
      continue;
    }
    left = 0;
    out[i] = { ...m, content: leftOutResult(m.content.length) };
  }
  return out;
}

/** The person's and the agent's words in a conversation: what "N messages" counts. */
export function spokenCount(messages: Message[]): number {
  return messages.filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim()).length;
}
