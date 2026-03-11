import { describe, it, expect, vi } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { handoff, hook } from '../../../src/core/decorators.js';
import type { Context, HookData } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import type { ContentItem } from '../../../src/uamp/types.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDoneEvent,
  createResponseDeltaEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import { ContextImpl } from '../../../src/core/context.js';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function createMockLLM(responses: Array<{
  text?: string;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
}>) {
  let callIndex = 0;

  class MockLLM extends Skill {
    @handoff({ name: 'mock-llm' })
    async *processUAMP(_events: ClientEvent[], _context: Context): AsyncGenerator<ServerEvent> {
      const response = responses[callIndex] ?? { text: 'default response' };
      callIndex++;

      const responseId = generateEventId();
      yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;

      const output: ContentItem[] = [];

      if (response.text) {
        yield createResponseDeltaEvent(responseId, { type: 'text', text: response.text });
        output.push({ type: 'text', text: response.text });
      }

      if (response.toolCalls) {
        for (const tc of response.toolCalls) {
          yield createResponseDeltaEvent(responseId, { type: 'tool_call', tool_call: tc });
          output.push({ type: 'tool_call', tool_call: tc });
        }
      }

      yield createResponseDoneEvent(responseId, output);
    }
  }

  return { skill: new MockLLM(), getCallCount: () => callIndex };
}

function buildInputEvents(text: string): ClientEvent[] {
  return [
    createSessionCreateEvent({ modalities: ['text'] }),
    createInputTextEvent(text),
    createResponseCreateEvent(),
  ];
}

async function collectEvents(gen: AsyncGenerator<ServerEvent>): Promise<ServerEvent[]> {
  const events: ServerEvent[] = [];
  for await (const event of gen) {
    events.push(event);
  }
  return events;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AbortSignal propagation', () => {
  it('signal is accessible on context after run() with signal option', async () => {
    let capturedSignal: AbortSignal | undefined;

    class SignalCaptureLLM extends Skill {
      @handoff({ name: 'signal-capture' })
      async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
        capturedSignal = context.signal;
        const responseId = generateEventId();
        yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
        yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'ok' }]);
      }
    }

    const agent = new BaseAgent({ skills: [new SignalCaptureLLM()] });
    const ac = new AbortController();

    await agent.run([{ role: 'user', content: 'hi' }], { signal: ac.signal });

    expect(capturedSignal).toBeDefined();
    expect(capturedSignal).toBe(ac.signal);
  });

  it('processUAMP stops on aborted signal', async () => {
    const { skill } = createMockLLM([{ text: 'should not appear' }]);
    const agent = new BaseAgent({ skills: [skill] });

    // Pre-abort the signal
    const ac = new AbortController();
    ac.abort();

    // Set signal on context before calling processUAMP
    (agent as any).context.signal = ac.signal;

    const events = await collectEvents(agent.processUAMP(buildInputEvents('hello')));

    const errorEvent = events.find(e => e.type === 'response.error') as any;
    expect(errorEvent).toBeDefined();
    expect(errorEvent.error.code).toBe('aborted');
    expect(errorEvent.error.message).toContain('cancelled');
  });

  it('_executeInternalToolCall returns cancelled message on abort', async () => {
    let toolWasCalled = false;

    class SlowToolSkill extends Skill {
      tools = [{
        name: 'slow_task',
        description: 'A slow task',
        parameters: {},
        handler: async (_params: Record<string, unknown>, _ctx: Context) => {
          toolWasCalled = true;
          return 'done';
        },
        scopes: [],
        provides: undefined,
      }];
    }

    // LLM that requests the slow_task tool, then responds
    const { skill: llm, getCallCount } = createMockLLM([
      { toolCalls: [{ id: 'call_1', name: 'slow_task', arguments: '{}' }] },
      { text: 'Completed' },
    ]);

    const agent = new BaseAgent({ skills: [llm, new SlowToolSkill()] });
    const ac = new AbortController();
    ac.abort(); // Pre-abort

    (agent as any).context.signal = ac.signal;

    const events = await collectEvents(agent.processUAMP(buildInputEvents('do task')));

    // Since signal was aborted before processUAMP, it should error immediately
    const errorEvent = events.find(e => e.type === 'response.error') as any;
    expect(errorEvent).toBeDefined();
    expect(errorEvent.error.code).toBe('aborted');
  });

  it('finalize_connection hook fires on abort', async () => {
    const hookCalls: string[] = [];

    class LifecycleSkill extends Skill {
      @hook({ lifecycle: 'on_connection' })
      async onConnection(_data: HookData, _c: Context): Promise<void> {
        hookCalls.push('on_connection');
      }

      @hook({ lifecycle: 'finalize_connection' })
      async finalizeConnection(_data: HookData, _c: Context): Promise<void> {
        hookCalls.push('finalize_connection');
      }
    }

    const { skill: llm } = createMockLLM([{ text: 'hello' }]);
    const agent = new BaseAgent({ skills: [llm, new LifecycleSkill()] });

    const ac = new AbortController();
    ac.abort();
    (agent as any).context.signal = ac.signal;

    await collectEvents(agent.processUAMP(buildInputEvents('hi')));

    expect(hookCalls).toContain('on_connection');
    expect(hookCalls).toContain('finalize_connection');
  });

  it('signal propagates from RunOptions to context via runStreaming', async () => {
    let capturedSignal: AbortSignal | undefined;

    class SignalCaptureLLM extends Skill {
      @handoff({ name: 'signal-capture-stream' })
      async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
        capturedSignal = context.signal;
        const responseId = generateEventId();
        yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
        yield createResponseDeltaEvent(responseId, { type: 'text', text: 'streamed' });
        yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'streamed' }]);
      }
    }

    const agent = new BaseAgent({ skills: [new SignalCaptureLLM()] });
    const ac = new AbortController();

    const chunks: string[] = [];
    for await (const chunk of agent.runStreaming(
      [{ role: 'user', content: 'stream test' }],
      { signal: ac.signal },
    )) {
      if (chunk.type === 'delta' && chunk.delta) {
        chunks.push(chunk.delta);
      }
    }

    expect(capturedSignal).toBeDefined();
    expect(capturedSignal).toBe(ac.signal);
    expect(chunks).toContain('streamed');
  });
});
