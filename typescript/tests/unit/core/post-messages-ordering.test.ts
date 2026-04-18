/**
 * Verifies StructuredToolResult._post_messages are appended AFTER the
 * role:'tool' result row, never before. This protects Anthropic and OpenAI
 * which require a tool_use to be immediately followed by a tool_result in the
 * next message — the same invariant that broke "ask sonnet" with a 400
 * "tool_use ids were found without tool_result blocks immediately after".
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, handoff } from '../../../src/core/decorators.js';
import type { Context, AgenticMessage, StructuredToolResult } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDoneEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import type { ContentItem } from '../../../src/uamp/types.js';

function createConversationAwareLLM(responses: Array<{
  text?: string;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
}>) {
  let callIndex = 0;
  const captured: AgenticMessage[][] = [];

  class ConversationLLM extends Skill {
    @handoff({ name: 'conversation-llm' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      const messages = context.get<AgenticMessage[]>('_agentic_messages');
      if (messages) captured.push(messages.map(m => ({ ...m })));
      const response = responses[callIndex] ?? { text: 'default' };
      callIndex++;
      const responseId = generateEventId();
      yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
      const output: ContentItem[] = [];
      if (response.text) output.push({ type: 'text', text: response.text });
      if (response.toolCalls) {
        for (const tc of response.toolCalls) output.push({ type: 'tool_call', tool_call: tc });
      }
      yield createResponseDoneEvent(responseId, output);
    }
  }

  return { skill: new ConversationLLM(), getCaptured: () => captured };
}

class FakeReadContentSkill extends Skill {
  @tool({ description: 'Mimics read_content: returns text + an _inline_for_llm post_message.' })
  async fake_read(params: { id: string }, _ctx: Context): Promise<StructuredToolResult> {
    return {
      text: `Content ${params.id} (file) loaded into your context.`,
      _post_messages: [{
        role: 'user' as const,
        content: `[Loaded content for analysis: ${params.id} (file)]`,
        content_items: [{ type: 'file', content_id: params.id, file: '/api/content/' + params.id }],
        _inline_for_llm: true,
      }],
    };
  }
}

function buildInputEvents(text: string): ClientEvent[] {
  return [createSessionCreateEvent({ modalities: ['text'] }), createInputTextEvent(text), createResponseCreateEvent()];
}

async function drain(gen: AsyncGenerator<ServerEvent>): Promise<void> {
  for await (const _ of gen) { /* drain */ }
}

describe('StructuredToolResult._post_messages ordering', () => {
  it('appends _post_messages AFTER the role:"tool" result row, not before', async () => {
    const { skill: llm, getCaptured } = createConversationAwareLLM([
      { toolCalls: [{ id: 'call_1', name: 'fake_read', arguments: '{"id":"abc-123"}' }] },
      { text: 'done' },
    ]);
    const agent = new BaseAgent({ skills: [llm, new FakeReadContentSkill()] });

    await drain(agent.processUAMP(buildInputEvents('read it')));

    const captured = getCaptured();
    expect(captured.length).toBeGreaterThanOrEqual(2);
    const second = captured[1];

    // Locate the assistant turn that issued the fake_read call_1.
    const asstIdx = second.findIndex(m => m.role === 'assistant'
      && Array.isArray(m.tool_calls)
      && m.tool_calls.some(tc => tc.id === 'call_1'));
    expect(asstIdx).toBeGreaterThanOrEqual(0);

    // The very next message MUST be the role:'tool' result for call_1 — NOT
    // the _inline_for_llm user follow-up (that would break Anthropic).
    const next = second[asstIdx + 1];
    expect(next).toBeDefined();
    expect(next.role).toBe('tool');
    expect(next.tool_call_id).toBe('call_1');

    // The follow-up _inline_for_llm user message must come AFTER the tool result.
    const inlineIdx = second.findIndex((m, i) =>
      i > asstIdx + 1
      && m.role === 'user'
      && (m as { _inline_for_llm?: boolean })._inline_for_llm === true,
    );
    expect(inlineIdx).toBeGreaterThan(asstIdx + 1);
  });

  it('does not duplicate _post_messages when the iteration is replayed', async () => {
    const { skill: llm, getCaptured } = createConversationAwareLLM([
      { toolCalls: [{ id: 'call_1', name: 'fake_read', arguments: '{"id":"abc-123"}' }] },
      { text: 'final' },
    ]);
    const agent = new BaseAgent({ skills: [llm, new FakeReadContentSkill()] });
    await drain(agent.processUAMP(buildInputEvents('read it')));
    const second = getCaptured()[1];
    const inlineCount = second.filter(m =>
      m.role === 'user' && (m as { _inline_for_llm?: boolean })._inline_for_llm === true,
    ).length;
    expect(inlineCount).toBe(1);
  });
});
