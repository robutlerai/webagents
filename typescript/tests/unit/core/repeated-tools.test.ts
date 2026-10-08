/**
 * Repeated tool calls in the agentic loop: ONE detector (2026-09-29).
 *
 * This file pinned the older, round-level nudge (the third identical round's
 * tool result rewritten into "You have called this tool 3 times with the same
 * arguments..."). That detector is gone: it lived beside the budget's
 * `RepeatedCalls` (`src/core/tool-budget.ts`) with a different rule, and on a
 * true loop the model got a rewritten third result under a wrap-up message
 * that said all three results were the same. What holds now, and what these
 * tests pin:
 *   - 2 identical calls: the tools run, every result is the tool's own;
 *   - 3 identical calls IN A ROW with the same result: the third result is
 *     still the tool's own, and the NEXT model call is the turn's last, with
 *     tools off and the wrap-up system message (`loopAnswerMessage`);
 *   - different tool calls interspersed: the streak starts over;
 *   - same tool name but different args: no loop.
 * A tool result never carries the nudge's words; the wrap-up is a system
 * message, the same as the Python agent's (fixture `agent_loop/
 * tool_round_budget.json`, `one_detector`).
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, handoff } from '../../../src/core/decorators.js';
import { TOOL_LOOP, loopAnswerMessage } from '../../../src/core/tool-budget.js';
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
  const toolsOffered: boolean[] = [];
  const toolChoices: string[] = [];

  class SequenceLLM extends Skill {
    @handoff({ name: 'seq-llm' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      const messages = context.get<AgenticMessage[]>('_agentic_messages');
      if (messages) capturedConversations.push([...messages]);
      toolsOffered.push((context.get<unknown[]>('_agentic_tools') ?? []).length > 0);
      toolChoices.push(String(context.get<string>('_agentic_tool_choice') ?? ''));

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
    toolsOffered,
    toolChoices,
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

  it("3 identical calls with the same result: every result is the tool's own, and the next call is the last, tools off", async () => {
    const searchSkill = new SearchSkill();
    const args = '{"query":"newww"}';
    const { skill: llm, getCapturedConversations, getCallCount, toolsOffered, toolChoices } = createSequenceLLM([
      { toolCalls: [{ id: 'c1', name: 'search', arguments: args }] },
      { toolCalls: [{ id: 'c2', name: 'search', arguments: args }] },
      { toolCalls: [{ id: 'c3', name: 'search', arguments: args }] },
      { text: 'I could not find it.' },
    ]);

    const agent = new BaseAgent({ skills: [llm, searchSkill], maxToolIterations: 6 });
    const events = await collectEvents(agent.processUAMP(buildInputEvents('search loop')));

    expect(searchSkill.callLog).toEqual(['newww', 'newww', 'newww']);
    const convos = getCapturedConversations();
    const lastConvo = convos[convos.length - 1];
    // The old detector rewrote the third of these; all three are the tool's own now.
    expect(lastConvo.filter(m => m.role === 'tool').map(m => String(m.content))).toEqual([
      'Results for: newww', 'Results for: newww', 'Results for: newww',
    ]);
    // The fourth call is the turn's last: the wrap-up system message last,
    // the tool choice `none`, and the tool definitions still listed so the
    // provider's cached prefix survives (a call the model makes is not run).
    expect(getCallCount()).toBe(4);
    expect(toolsOffered).toEqual([true, true, true, true]);
    expect(toolChoices).toEqual(['auto', 'auto', 'auto', 'none']);
    expect(lastConvo[lastConvo.length - 1]).toEqual({ role: 'system', content: loopAnswerMessage('search') });
    const done = events.find(e => e.type === 'response.done') as unknown as { response: Record<string, unknown> };
    expect(done.response.finish_reason).toBe(TOOL_LOOP);
    expect(done.response.finish_tool).toBe('search');
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
