/**
 * Repeated Tool Call Detection Tests
 *
 * Tests the repeated tool call nudge logic in the agentic loop:
 * - 2 identical calls: no nudge, tools execute normally
 * - 3+ identical calls: tool result replaced with nudge message
 * - Different tool calls interspersed: counter resets
 * - Same tool name but different args: no nudge
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, handoff } from '../../../src/core/decorators.js';
import type { Context, AgenticMessage } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDoneEvent,
  createResponseDeltaEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import type { ContentItem } from '../../../src/uamp/types.js';

function createSequenceLLM(responses: Array<{
  text?: string;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
}>) {
  let callIndex = 0;
  const capturedConversations: AgenticMessage[][] = [];

  class SequenceLLM extends Skill {
    @handoff({ name: 'seq-llm' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      const messages = context.get<AgenticMessage[]>('_agentic_messages');
      if (messages) capturedConversations.push([...messages]);

      const response = responses[callIndex] ?? { text: 'done' };
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

  return {
    skill: new SequenceLLM(),
    getCallCount: () => callIndex,
    getCapturedConversations: () => capturedConversations,
  };
}

class SearchSkill extends Skill {
  callLog: string[] = [];

  @tool({ description: 'Search for something' })
  async search(params: { query: string }, _c: Context): Promise<string> {
    this.callLog.push(params.query);
    return `Results for: ${params.query}`;
  }

  @tool({ description: 'Get info about a topic' })
  async getInfo(params: { topic: string }, _c: Context): Promise<string> {
    this.callLog.push(`info:${params.topic}`);
    return `Info about: ${params.topic}`;
  }
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
  for await (const event of gen) events.push(event);
  return events;
}

describe('Repeated Tool Call Detection', () => {
  it('2 identical calls execute normally without nudge', async () => {
    const searchSkill = new SearchSkill();
    const args = '{"query":"test"}';
    const { skill: llm, getCapturedConversations } = createSequenceLLM([
      { toolCalls: [{ id: 'c1', name: 'search', arguments: args }] },
      { toolCalls: [{ id: 'c2', name: 'search', arguments: args }] },
      { text: 'Done' },
    ]);

    const agent = new BaseAgent({ skills: [llm, searchSkill], maxToolIterations: 5 });
    await collectEvents(agent.processUAMP(buildInputEvents('search twice')));

    expect(searchSkill.callLog).toEqual(['test', 'test']);
    const convos = getCapturedConversations();
    const lastConvo = convos[convos.length - 1];
    const toolResults = lastConvo.filter(m => m.role === 'tool');
    for (const tr of toolResults) {
      expect(tr.content).not.toContain('same arguments');
    }
  });

  it('3 identical calls trigger nudge on the 3rd', async () => {
    const searchSkill = new SearchSkill();
    const args = '{"query":"newww"}';
    const { skill: llm, getCapturedConversations } = createSequenceLLM([
      { toolCalls: [{ id: 'c1', name: 'search', arguments: args }] },
      { toolCalls: [{ id: 'c2', name: 'search', arguments: args }] },
      { toolCalls: [{ id: 'c3', name: 'search', arguments: args }] },
      { text: 'I could not find it.' },
    ]);

    const agent = new BaseAgent({ skills: [llm, searchSkill], maxToolIterations: 6 });
    await collectEvents(agent.processUAMP(buildInputEvents('search loop')));

    const convos = getCapturedConversations();
    const lastConvo = convos[convos.length - 1];
    const toolResults = lastConvo.filter(m => m.role === 'tool');

    const nudge = toolResults.find(tr =>
      typeof tr.content === 'string' && tr.content.includes('same arguments')
    );
    expect(nudge).toBeDefined();
  });

  it('different tool calls interspersed reset the counter', async () => {
    const searchSkill = new SearchSkill();
    const searchArgs = '{"query":"x"}';
    const infoArgs = '{"topic":"y"}';
    const { skill: llm, getCapturedConversations } = createSequenceLLM([
      { toolCalls: [{ id: 'c1', name: 'search', arguments: searchArgs }] },
      { toolCalls: [{ id: 'c2', name: 'getInfo', arguments: infoArgs }] },
      { toolCalls: [{ id: 'c3', name: 'search', arguments: searchArgs }] },
      { toolCalls: [{ id: 'c4', name: 'getInfo', arguments: infoArgs }] },
      { text: 'Done' },
    ]);

    const agent = new BaseAgent({ skills: [llm, searchSkill], maxToolIterations: 6 });
    await collectEvents(agent.processUAMP(buildInputEvents('interspersed')));

    const convos = getCapturedConversations();
    const lastConvo = convos[convos.length - 1];
    const toolResults = lastConvo.filter(m => m.role === 'tool');
    for (const tr of toolResults) {
      expect(tr.content).not.toContain('same arguments');
    }
  });

  it('same tool name but different args does not trigger nudge', async () => {
    const searchSkill = new SearchSkill();
    const { skill: llm, getCapturedConversations } = createSequenceLLM([
      { toolCalls: [{ id: 'c1', name: 'search', arguments: '{"query":"a"}' }] },
      { toolCalls: [{ id: 'c2', name: 'search', arguments: '{"query":"b"}' }] },
      { toolCalls: [{ id: 'c3', name: 'search', arguments: '{"query":"c"}' }] },
      { text: 'Done' },
    ]);

    const agent = new BaseAgent({ skills: [llm, searchSkill], maxToolIterations: 5 });
    await collectEvents(agent.processUAMP(buildInputEvents('different args')));

    const convos = getCapturedConversations();
    const lastConvo = convos[convos.length - 1];
    const toolResults = lastConvo.filter(m => m.role === 'tool');
    for (const tr of toolResults) {
      expect(tr.content).not.toContain('same arguments');
    }
  });

  it('payment_exhausted flag breaks the loop gracefully', async () => {
    const searchSkill = new SearchSkill();
    
    let callIndex = 0;
    class PaymentExhaustLLM extends Skill {
      @handoff({ name: 'exhaust-llm' })
      async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
        callIndex++;
        if (callIndex === 2) {
          context.set('_payment_exhausted', true);
        }
        const responseId = generateEventId();
        yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
        const tc = { id: `c${callIndex}`, name: 'search', arguments: '{"query":"test"}' };
        yield createResponseDeltaEvent(responseId, { type: 'tool_call', tool_call: tc });
        yield createResponseDoneEvent(responseId, [{ type: 'tool_call', tool_call: tc }]);
      }
    }

    const agent = new BaseAgent({ skills: [new PaymentExhaustLLM(), searchSkill], maxToolIterations: 10 });
    const events = await collectEvents(agent.processUAMP(buildInputEvents('exhaust')));

    const errorEvent = events.find(e => e.type === 'response.error');
    expect(errorEvent).toBeDefined();
    expect((errorEvent as any).error.code).toBe('payment_exhausted');
    expect(callIndex).toBe(2);
  });
});
