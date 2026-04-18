/**
 * History Conversation Seeding Tests
 *
 * Verifies the `_history_conversation` context key plumbing in
 * `BaseAgent.processUAMP`:
 *
 * - When set, the history is **prepended** to the events-derived
 *   conversation so the new user turn (from `input.text`) is preserved.
 * - When `_initial_conversation` is set, it wins (existing run() semantics
 *   are unchanged).
 * - Duplicate leading system messages are coalesced.
 *
 * This is the regression test for the delegate-subchat history preload
 * — without `_history_conversation`, a sub-agent reusing an existing
 * delegate sub-chat would start every turn cold and redo work it
 * already did (re-create files, etc.).
 */
import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { handoff } from '../../../src/core/decorators.js';
import type { Context, AgenticMessage } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import type { ContentItem } from '../../../src/uamp/types.js';

/**
 * Mock LLM that captures whatever conversation BaseAgent passes to it
 * via `_agentic_messages`, so tests can assert the merged shape.
 */
function createCapturingLLM() {
  const captured: AgenticMessage[][] = [];

  class CapturingLLM extends Skill {
    @handoff({ name: 'capturing-llm' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      const msgs = context.get<AgenticMessage[]>('_agentic_messages');
      if (msgs) captured.push([...msgs]);

      const responseId = generateEventId();
      yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
      yield createResponseDeltaEvent(responseId, { type: 'text', text: 'ok' });
      yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'ok' } as ContentItem]);
    }
  }

  return { skill: new CapturingLLM(), captured };
}

function eventsForNewTurn(text: string): ClientEvent[] {
  return [
    createSessionCreateEvent({ modalities: ['text'] }),
    createInputTextEvent(text),
    createResponseCreateEvent(),
  ];
}

async function drain(gen: AsyncGenerator<ServerEvent>): Promise<void> {
  for await (const _ of gen) {
    void _;
  }
}

describe('_history_conversation seeding', () => {
  it('prepends history to the events-derived new turn', async () => {
    const { skill, captured } = createCapturingLLM();
    const agent = new BaseAgent({ skills: [skill] });

    const history: AgenticMessage[] = [
      { role: 'user', content: 'first turn from prior chat' },
      { role: 'assistant', content: 'prior assistant reply' },
    ];
    (agent as unknown as { context: Context }).context.set('_history_conversation', history);

    await drain(agent.processUAMP(eventsForNewTurn('second turn')));

    expect(captured.length).toBeGreaterThan(0);
    const conv = captured[0];
    expect(conv.map(m => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(conv[0].content).toBe('first turn from prior chat');
    expect(conv[1].content).toBe('prior assistant reply');
    // The events-derived new turn is preserved at the tail.
    const lastUser = conv[2];
    expect(lastUser.role).toBe('user');
    const items = lastUser.content_items ?? [];
    expect(items.some((i): i is ContentItem & { text: string } => i.type === 'text' && (i as { text?: string }).text === 'second turn')).toBe(true);
  });

  it('clears the key after one use (next turn does not double-prepend)', async () => {
    const { skill, captured } = createCapturingLLM();
    const agent = new BaseAgent({ skills: [skill] });

    const history: AgenticMessage[] = [
      { role: 'user', content: 'old' },
    ];
    (agent as unknown as { context: Context }).context.set('_history_conversation', history);

    await drain(agent.processUAMP(eventsForNewTurn('first new')));
    await drain(agent.processUAMP(eventsForNewTurn('second new')));

    expect(captured.length).toBe(2);
    expect(captured[0].length).toBe(2); // history(1) + new(1)
    expect(captured[1].length).toBe(1); // just the new turn — history wasn't re-injected
  });

  it('drops a duplicate leading system message from events when history already has one', async () => {
    const { skill, captured } = createCapturingLLM();
    const agent = new BaseAgent({
      instructions: 'you are an agent',
      skills: [skill],
    });

    const history: AgenticMessage[] = [
      { role: 'system', content: 'history-system' },
      { role: 'user', content: 'old user' },
    ];
    (agent as unknown as { context: Context }).context.set('_history_conversation', history);

    await drain(agent.processUAMP(eventsForNewTurn('new user')));

    const conv = captured[0];
    const systemMsgs = conv.filter(m => m.role === 'system');
    expect(systemMsgs.length).toBe(1);
    expect(systemMsgs[0].content).toBe('history-system');
  });

  it('_initial_conversation takes precedence over _history_conversation', async () => {
    const { skill, captured } = createCapturingLLM();
    const agent = new BaseAgent({ skills: [skill] });

    const ctx = (agent as unknown as { context: Context }).context;
    ctx.set('_history_conversation', [
      { role: 'user', content: 'history-only' },
    ]);
    ctx.set('_initial_conversation', [
      { role: 'user', content: 'initial-only' },
    ]);

    await drain(agent.processUAMP(eventsForNewTurn('events new turn')));

    const conv = captured[0];
    expect(conv.length).toBe(1);
    expect(conv[0].content).toBe('initial-only');
  });

  it('with no _history_conversation, behaves exactly like before (events-only)', async () => {
    const { skill, captured } = createCapturingLLM();
    const agent = new BaseAgent({ skills: [skill] });

    await drain(agent.processUAMP(eventsForNewTurn('only this')));

    const conv = captured[0];
    expect(conv.length).toBe(1);
    expect(conv[0].role).toBe('user');
    const items = conv[0].content_items ?? [];
    expect(items.some(i => i.type === 'text' && (i as { text?: string }).text === 'only this')).toBe(true);
  });
});
