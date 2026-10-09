/**
 * Tool-call ids in a replayed history.
 *
 * Every provider wire pairs a function call with its output by id, and none
 * of them accepts an empty one: the Responses API answers
 * `Invalid 'input[N].call_id': expected a nonempty string`, Anthropic checks
 * `tool_use.id` against `^[a-zA-Z0-9_-]+$`, Converse checks `toolUseId`, and
 * Gemini pairs `functionCall` with `functionResponse` by position and count.
 * A history can still arrive with empty ids: a row persisted by a client that
 * never minted one, a store that trimmed the id column, a transcript edited
 * by hand. One such row used to fail the whole request, on every turn, for as
 * long as it stayed inside the history window.
 *
 * `normalizeToolHistory` runs first in every adapter's message conversion:
 *
 *   - A call with an empty id gets a STABLE one, derived from the call's name
 *     and arguments and from its index in its message (`call_<hash>_<i>`,
 *     suffixed when the same bytes already produced that id earlier in the
 *     history). The same history therefore yields the same bytes on every
 *     request. A random id per request would read as a different history to
 *     a provider's prompt cache, which matches the request's leading bytes
 *     exactly, and would miss the cache on every turn.
 *   - An output with an empty id is paired BY ORDER with the first
 *     still-unanswered call of the most recent assistant turn, preferring a
 *     call whose name matches the output's `name`. Outputs follow their
 *     calls in the order the calls were made, which is the one fact an
 *     id-less output still carries.
 *   - With `unpaired: 'drop'` (the default) a call no output answers, with a
 *     message after it, is left out, the assistant's text beside it kept:
 *     every wire above refuses that shape, and sending an item the provider
 *     will refuse is never better than sending the history without it. So
 *     is an id-less output that pairs with nothing, since there is no valid
 *     item to send for it. With `unpaired: 'keep'` both stay, for an adapter
 *     that applies its own policy to them.
 *   - Everything else is replayed as it came, as the other SDKs' adapters
 *     do: a call in the LAST message of the history, answered or not (the
 *     shape of a turn whose outputs are about to be appended), and an
 *     output with an id of its own, whether or not a call declared it (the
 *     provider judges it).
 *
 * Messages are never mutated: a message that needs a change is copied.
 */
import type { Message } from './types';

type ToolCall = NonNullable<Message['tool_calls']>[number];

export interface NormalizeToolHistoryOptions {
  /** What to do with a call no output answers, or an output no call declared. Default `drop`. */
  unpaired?: 'drop' | 'keep';
}

/** 32-bit FNV-1a of a string, as 8 hex characters. Cheap, stable, needs no crypto. */
function fnv1a32(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

/** The id derived for an id-less call: its name and arguments hashed, its index in the message, and a suffix when taken. */
function derivedCallId(call: ToolCall, index: number, taken: Set<string>): string {
  const name = call.function?.name ?? '';
  const args = call.function?.arguments ?? '';
  const base = `call_${fnv1a32(`${name}\u0000${args}`)}_${index}`;
  let id = base;
  for (let n = 2; taken.has(id); n++) id = `${base}_${n}`;
  return id;
}

function hasId(id: unknown): id is string {
  return typeof id === 'string' && id.length > 0;
}

/**
 * Give every replayed function call and its output a matching non-empty id,
 * and (by default) leave out what cannot be paired. See the file comment.
 */
export function normalizeToolHistory(messages: Message[], options: NormalizeToolHistoryOptions = {}): Message[] {
  const unpaired = options.unpaired ?? 'drop';
  if (!messages.some((m) => (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) || m.role === 'tool')) {
    return messages;
  }

  // Pass one: assign ids, pair outputs with calls.
  const taken = new Set<string>();
  for (const m of messages) {
    for (const tc of m.tool_calls ?? []) if (hasId(tc.id)) taken.add(tc.id);
  }
  /** Every call in the history, by its (possibly derived) id. */
  const calls = new Map<string, { answered: boolean }>();
  /** The calls of the most recent assistant turn, in order, for pairing an id-less output. */
  let recent: Array<{ id: string; name: string }> = [];
  /** Per message index: the ids its calls ended up with (assistant) or the id its output pairs to (tool). */
  const callIds = new Map<number, string[]>();
  const outputId = new Map<number, string | null>();

  messages.forEach((m, index) => {
    if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) {
      const ids: string[] = [];
      recent = [];
      m.tool_calls.forEach((tc, i) => {
        const id = hasId(tc.id) ? tc.id : derivedCallId(tc, i, taken);
        taken.add(id);
        ids.push(id);
        calls.set(id, { answered: false });
        recent.push({ id, name: tc.function?.name ?? '' });
      });
      callIds.set(index, ids);
      return;
    }
    if (m.role !== 'tool') {
      // Outputs follow their calls directly; a message in between ends the
      // run an id-less output could pair into.
      recent = [];
      return;
    }
    let id: string | null = null;
    if (hasId(m.tool_call_id)) {
      // Kept as it came; it answers the call of that id when there is one.
      id = m.tool_call_id;
      const call = calls.get(id);
      if (call) call.answered = true;
      outputId.set(index, id);
      return;
    } else {
      const open = recent.filter((c) => !calls.get(c.id)!.answered);
      const byName = m.name ? open.find((c) => c.name === m.name) : undefined;
      id = (byName ?? open[0])?.id ?? null;
    }
    if (id) calls.get(id)!.answered = true;
    outputId.set(index, id);
  });

  // Pass two: rewrite, dropping or keeping what is unpaired. The input
  // array is returned when no message needed a change.
  const out: Message[] = [];
  let changed = false;
  messages.forEach((m, index) => {
    if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) {
      const ids = callIds.get(index)!;
      const kept: ToolCall[] = [];
      const trailing = index === messages.length - 1;
      m.tool_calls.forEach((tc, i) => {
        const id = ids[i];
        if (unpaired === 'drop' && !trailing && !calls.get(id)!.answered) return;
        kept.push(tc.id === id ? tc : { ...tc, id });
      });
      const unchanged = kept.length === m.tool_calls.length && kept.every((tc, i) => tc === m.tool_calls![i]);
      if (unchanged) {
        out.push(m);
        return;
      }
      changed = true;
      if (kept.length > 0) {
        out.push({ ...m, tool_calls: kept });
      } else {
        // Every call dropped: keep the assistant's words, if there were any.
        const { tool_calls: _calls, ...rest } = m;
        const hasText = typeof m.content === 'string' ? m.content.trim().length > 0 : Array.isArray(m.content) && m.content.length > 0;
        const hasItems = Array.isArray(m.content_items) && m.content_items.length > 0;
        if (hasText || hasItems) out.push(rest);
      }
      return;
    }
    if (m.role === 'tool') {
      const id = outputId.get(index) ?? null;
      if (id === null) {
        if (unpaired === 'keep') out.push(m);
        else changed = true;
        return;
      }
      if (m.tool_call_id === id) out.push(m);
      else {
        changed = true;
        out.push({ ...m, tool_call_id: id });
      }
      return;
    }
    out.push(m);
  });
  return changed ? out : messages;
}
