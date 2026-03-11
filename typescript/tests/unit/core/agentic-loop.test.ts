/**
 * Agentic Loop Tests
 *
 * Tests the server-side tool execution loop in BaseAgent.processUAMP():
 * - Internal tools are executed server-side and results fed back to the LLM
 * - External tools are returned to the client
 * - Mixed internal/external tool calls handled correctly
 * - Max iteration limit enforced
 * - Hook integration (before_tool, after_tool)
 * - Scope enforcement on tools
 */

import { describe, it, expect, vi } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, hook, handoff } from '../../../src/core/decorators.js';
import type { Context, HookData, HookResult, AgenticMessage } from '../../../src/core/types.js';
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

// ============================================================================
// Test Helpers: Mock LLM Skills
// ============================================================================

/**
 * A mock LLM skill that returns tool calls, then a final text response.
 * The `responses` array controls what the LLM returns on each iteration.
 */
function createMockLLM(responses: Array<{
  text?: string;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
}>) {
  let callIndex = 0;

  class MockLLM extends Skill {
    @handoff({ name: 'mock-llm' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      const response = responses[callIndex] ?? { text: 'default response' };
      callIndex++;

      const responseId = generateEventId();
      yield {
        type: 'response.created',
        event_id: generateEventId(),
        response_id: responseId,
      } as ServerEvent;

      const output: ContentItem[] = [];

      if (response.text) {
        yield createResponseDeltaEvent(responseId, {
          type: 'text',
          text: response.text,
        });
        output.push({ type: 'text', text: response.text });
      }

      if (response.toolCalls) {
        for (const tc of response.toolCalls) {
          yield createResponseDeltaEvent(responseId, {
            type: 'tool_call',
            tool_call: tc,
          });
          output.push({
            type: 'tool_call',
            tool_call: tc,
          });
        }
      }

      yield createResponseDoneEvent(responseId, output);
    }
  }

  return { skill: new MockLLM(), getCallCount: () => callIndex };
}

/**
 * A mock LLM that reads _agentic_messages from context to verify
 * the conversation state includes tool results.
 */
function createConversationAwareLLM(responses: Array<{
  text?: string;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
}>) {
  let callIndex = 0;
  const capturedConversations: AgenticMessage[][] = [];

  class ConversationLLM extends Skill {
    @handoff({ name: 'conversation-llm' })
    async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
      const messages = context.get<AgenticMessage[]>('_agentic_messages');
      if (messages) {
        capturedConversations.push([...messages]);
      }

      const response = responses[callIndex] ?? { text: 'default' };
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
          output.push({ type: 'tool_call', tool_call: tc });
        }
      }

      yield createResponseDoneEvent(responseId, output);
    }
  }

  return {
    skill: new ConversationLLM(),
    getCallCount: () => callIndex,
    getCapturedConversations: () => capturedConversations,
  };
}

class MathToolsSkill extends Skill {
  @tool({ description: 'Add two numbers' })
  async add(params: { a: number; b: number }, _c: Context): Promise<number> {
    return params.a + params.b;
  }

  @tool({ description: 'Multiply two numbers' })
  async multiply(params: { a: number; b: number }, _c: Context): Promise<number> {
    return params.a * params.b;
  }
}

function buildInputEvents(text: string, instructions?: string): ClientEvent[] {
  return [
    createSessionCreateEvent({
      modalities: ['text'],
      instructions,
    }),
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

// ============================================================================
// Tests
// ============================================================================

describe('Agentic Loop', () => {
  describe('no tool calls (passthrough)', () => {
    it('yields response directly when LLM returns only text', async () => {
      const { skill } = createMockLLM([{ text: 'Hello world' }]);
      const agent = new BaseAgent({ skills: [skill] });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Hi'))
      );

      const types = events.map(e => e.type);
      expect(types).toContain('response.created');
      expect(types).toContain('response.delta');
      expect(types).toContain('response.done');

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      const textOutput = doneEvent.response.output.find(
        (o: ContentItem) => o.type === 'text'
      );
      expect(textOutput?.text).toBe('Hello world');
    });
  });

  describe('internal tool execution', () => {
    it('executes internal tools and re-calls LLM with results', async () => {
      const { skill: llm, getCallCount } = createMockLLM([
        {
          toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":2,"b":3}' }],
        },
        { text: 'The sum is 5' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('What is 2+3?'))
      );

      expect(getCallCount()).toBe(2);

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      expect(doneEvent).toBeDefined();
      const textOutput = doneEvent.response.output.find(
        (o: ContentItem) => o.type === 'text'
      );
      expect(textOutput?.text).toBe('The sum is 5');
    });

    it('handles multiple tool calls in one iteration', async () => {
      const { skill: llm, getCallCount } = createMockLLM([
        {
          toolCalls: [
            { id: 'call_1', name: 'add', arguments: '{"a":2,"b":3}' },
            { id: 'call_2', name: 'multiply', arguments: '{"a":4,"b":5}' },
          ],
        },
        { text: '2+3=5 and 4*5=20' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Add and multiply'))
      );

      expect(getCallCount()).toBe(2);

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      expect(doneEvent.response.output.find((o: ContentItem) => o.type === 'text')?.text)
        .toBe('2+3=5 and 4*5=20');
    });

    it('chains multiple tool iterations', async () => {
      const { skill: llm, getCallCount } = createMockLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":2}' }] },
        { toolCalls: [{ id: 'call_2', name: 'multiply', arguments: '{"a":3,"b":4}' }] },
        { text: 'Done: 3 and 12' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Chain'))
      );

      expect(getCallCount()).toBe(3);

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      expect(doneEvent.response.output.find((o: ContentItem) => o.type === 'text')?.text)
        .toBe('Done: 3 and 12');
    });
  });

  describe('conversation context propagation', () => {
    it('passes tool calls and results in _agentic_messages on second iteration', async () => {
      const { skill: llm, getCapturedConversations } = createConversationAwareLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":10,"b":20}' }] },
        { text: 'Answer: 30' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      await collectEvents(
        agent.processUAMP(buildInputEvents('What is 10+20?', 'Be helpful'))
      );

      const conversations = getCapturedConversations();
      expect(conversations.length).toBe(2);

      // First call: system + user messages
      const firstConvo = conversations[0];
      expect(firstConvo[0]).toEqual({ role: 'system', content: 'Be helpful' });
      expect(firstConvo[1]).toEqual({ role: 'user', content: 'What is 10+20?' });

      // Second call: system + user + assistant (with tool_calls) + tool result
      const secondConvo = conversations[1];
      expect(secondConvo.length).toBe(4);
      expect(secondConvo[0].role).toBe('system');
      expect(secondConvo[1].role).toBe('user');
      expect(secondConvo[2].role).toBe('assistant');
      expect(secondConvo[2].tool_calls).toHaveLength(1);
      expect(secondConvo[2].tool_calls![0].function.name).toBe('add');
      expect(secondConvo[3].role).toBe('tool');
      expect(secondConvo[3].tool_call_id).toBe('call_1');
      expect(secondConvo[3].content).toBe('30');
    });
  });

  describe('external tool passthrough', () => {
    it('returns tool calls for unregistered tools to the client', async () => {
      const { skill: llm } = createMockLLM([
        {
          toolCalls: [
            { id: 'call_ext', name: 'browser_click', arguments: '{"selector":".btn"}' },
          ],
        },
      ]);

      const agent = new BaseAgent({ skills: [llm] });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Click the button'))
      );

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      expect(doneEvent).toBeDefined();

      const toolCallOutput = doneEvent.response.output.filter(
        (o: ContentItem) => o.type === 'tool_call'
      );
      expect(toolCallOutput).toHaveLength(1);
      expect(toolCallOutput[0].tool_call?.name).toBe('browser_click');
    });
  });

  describe('mixed internal and external tools', () => {
    it('executes internal tools and returns external tools to client', async () => {
      const { skill: llm, getCallCount } = createMockLLM([
        {
          toolCalls: [
            { id: 'call_int', name: 'add', arguments: '{"a":1,"b":2}' },
            { id: 'call_ext', name: 'external_search', arguments: '{"q":"test"}' },
          ],
        },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Search and add'))
      );

      // Only 1 LLM call -- loop breaks on external tools
      expect(getCallCount()).toBe(1);

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };

      const toolCallOutputs = doneEvent.response.output.filter(
        (o: ContentItem) => o.type === 'tool_call'
      );
      // Only external tool should be in the output
      expect(toolCallOutputs).toHaveLength(1);
      expect(toolCallOutputs[0].tool_call?.name).toBe('external_search');
    });
  });

  describe('overridden tools', () => {
    it('treats overridden tools as external', async () => {
      const { skill: llm, getCallCount } = createMockLLM([
        {
          toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":2}' }],
        },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      agent.overrideTool('add');

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Add'))
      );

      // Loop should NOT execute 'add' internally
      expect(getCallCount()).toBe(1);

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      const toolCallOutputs = doneEvent.response.output.filter(
        (o: ContentItem) => o.type === 'tool_call'
      );
      expect(toolCallOutputs).toHaveLength(1);
      expect(toolCallOutputs[0].tool_call?.name).toBe('add');
    });
  });

  describe('max iterations', () => {
    it('stops after maxToolIterations and yields error', async () => {
      // LLM always returns a tool call -- infinite loop
      const infiniteResponses = Array.from({ length: 20 }, (_, i) => ({
        toolCalls: [{ id: `call_${i}`, name: 'add', arguments: '{"a":1,"b":1}' }],
      }));

      const { skill: llm, getCallCount } = createMockLLM(infiniteResponses);
      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
        maxToolIterations: 3,
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Loop forever'))
      );

      expect(getCallCount()).toBe(3);

      const errorEvent = events.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect((errorEvent as unknown as { error: { code: string } }).error.code).toBe(
        'max_iterations'
      );
    });

    it('defaults to 10 max iterations', () => {
      const agent = new BaseAgent();
      expect((agent as unknown as { maxToolIterations: number }).maxToolIterations).toBe(10);
    });
  });

  describe('tool execution errors', () => {
    it('handles tool parse errors gracefully', async () => {
      const { skill: llm } = createConversationAwareLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{invalid json' }] },
        { text: 'Handled error' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Bad args'))
      );

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      expect(doneEvent.response.output.find((o: ContentItem) => o.type === 'text')?.text)
        .toBe('Handled error');
    });

    it('handles tool execution errors gracefully', async () => {
      class FailingSkill extends Skill {
        @tool({ description: 'Always fails' })
        async failTool(_p: Record<string, unknown>, _c: Context): Promise<never> {
          throw new Error('Tool crashed');
        }
      }

      const { skill: llm, getCapturedConversations } = createConversationAwareLLM([
        { toolCalls: [{ id: 'call_1', name: 'failTool', arguments: '{}' }] },
        { text: 'Recovered from error' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new FailingSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Fail'))
      );

      const conversations = getCapturedConversations();
      const secondConvo = conversations[1];
      const toolResult = secondConvo.find(m => m.role === 'tool');
      expect(toolResult?.content).toContain('Tool execution error');
      expect(toolResult?.content).toContain('Tool crashed');

      const doneEvent = events.find(e => e.type === 'response.done') as unknown as {
        response: { output: ContentItem[] };
      };
      expect(doneEvent.response.output.find((o: ContentItem) => o.type === 'text')?.text)
        .toBe('Recovered from error');
    });
  });

  describe('hook integration', () => {
    it('runs before_tool and after_tool hooks during internal tool execution', async () => {
      const hookCalls: string[] = [];

      class HookSkill extends Skill {
        @hook({ lifecycle: 'before_tool' })
        async beforeTool(data: HookData, _c: Context): Promise<void> {
          hookCalls.push(`before:${data.tool_name}`);
        }

        @hook({ lifecycle: 'after_tool' })
        async afterTool(data: HookData, _c: Context): Promise<void> {
          hookCalls.push(`after:${data.tool_name}`);
        }
      }

      const { skill: llm } = createMockLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":2}' }] },
        { text: 'Done' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill(), new HookSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Add')));

      expect(hookCalls).toContain('before:add');
      expect(hookCalls).toContain('after:add');
    });

    it('before_tool hook can abort tool execution', async () => {
      class BlockingHookSkill extends Skill {
        @hook({ lifecycle: 'before_tool' })
        async blockTool(_data: HookData, _c: Context): Promise<HookResult> {
          return { abort: true, abort_reason: 'Blocked by policy' };
        }
      }

      const { skill: llm, getCapturedConversations } = createConversationAwareLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":2}' }] },
        { text: 'Tool was blocked' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill(), new BlockingHookSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Try add')));

      const secondConvo = getCapturedConversations()[1];
      const toolResult = secondConvo.find(m => m.role === 'tool');
      expect(toolResult?.content).toBe('Blocked by policy');
    });
  });

  describe('scope enforcement', () => {
    it('scope-restricted tools fail gracefully in the agentic loop', async () => {
      class ScopedSkill extends Skill {
        @tool({ description: 'Admin only', scopes: ['admin'] })
        async adminAction(_p: Record<string, unknown>, _c: Context): Promise<string> {
          return 'admin result';
        }
      }

      const { skill: llm, getCapturedConversations } = createConversationAwareLLM([
        { toolCalls: [{ id: 'call_1', name: 'adminAction', arguments: '{}' }] },
        { text: 'Permission denied' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new ScopedSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Do admin')));

      const secondConvo = getCapturedConversations()[1];
      const toolResult = secondConvo.find(m => m.role === 'tool');
      expect(toolResult?.content).toContain('Insufficient permissions');
    });
  });

  describe('before/after_llm_call and on_chunk hooks', () => {
    it('fires before_llm_call and after_llm_call on each loop iteration', async () => {
      const hookCalls: string[] = [];

      class LLMHookSkill extends Skill {
        @hook({ lifecycle: 'before_llm_call' })
        async beforeLLM(data: HookData, _c: Context): Promise<void> {
          hookCalls.push(`before_llm:${data.iteration}`);
        }

        @hook({ lifecycle: 'after_llm_call' })
        async afterLLM(data: HookData, _c: Context): Promise<void> {
          hookCalls.push(`after_llm:${data.iteration}`);
        }
      }

      const { skill: llm } = createMockLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":2}' }] },
        { text: 'Done' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill(), new LLMHookSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Add')));

      expect(hookCalls).toContain('before_llm:1');
      expect(hookCalls).toContain('after_llm:1');
      expect(hookCalls).toContain('before_llm:2');
      expect(hookCalls).toContain('after_llm:2');
    });

    it('fires on_chunk for each streaming delta', async () => {
      const chunks: unknown[] = [];

      class ChunkHookSkill extends Skill {
        @hook({ lifecycle: 'on_chunk' })
        async onChunk(data: HookData, _c: Context): Promise<void> {
          chunks.push(data.chunk);
        }
      }

      const { skill: llm } = createMockLLM([{ text: 'Hello world' }]);
      const agent = new BaseAgent({
        skills: [llm, new ChunkHookSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Hi')));

      expect(chunks.length).toBeGreaterThan(0);
      const textChunks = chunks.filter((c: any) => c?.type === 'text');
      expect(textChunks.length).toBeGreaterThan(0);
    });

    it('fires before_toolcall and after_toolcall (Python-compatible)', async () => {
      const hookCalls: string[] = [];

      class PythonCompatHookSkill extends Skill {
        @hook({ lifecycle: 'before_toolcall' })
        async beforeToolcall(data: HookData, _c: Context): Promise<void> {
          hookCalls.push(`before_toolcall:${data.tool_name}`);
        }

        @hook({ lifecycle: 'after_toolcall' })
        async afterToolcall(data: HookData, _c: Context): Promise<void> {
          hookCalls.push(`after_toolcall:${data.tool_name}`);
        }
      }

      const { skill: llm } = createMockLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":2}' }] },
        { text: 'Done' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill(), new PythonCompatHookSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Add')));

      expect(hookCalls).toContain('before_toolcall:add');
      expect(hookCalls).toContain('after_toolcall:add');
    });

    it('fires on_connection and finalize_connection lifecycle hooks', async () => {
      const hookCalls: string[] = [];

      class LifecycleHookSkill extends Skill {
        @hook({ lifecycle: 'on_connection' })
        async onConnection(_data: HookData, _c: Context): Promise<void> {
          hookCalls.push('on_connection');
        }

        @hook({ lifecycle: 'finalize_connection' })
        async finalizeConnection(_data: HookData, _c: Context): Promise<void> {
          hookCalls.push('finalize_connection');
        }
      }

      const { skill: llm } = createMockLLM([{ text: 'Hello' }]);
      const agent = new BaseAgent({
        skills: [llm, new LifecycleHookSkill()],
      });

      await collectEvents(agent.processUAMP(buildInputEvents('Hi')));

      expect(hookCalls).toEqual(['on_connection', 'finalize_connection']);
    });
  });

  describe('streaming behavior', () => {
    it('yields intermediate text deltas from tool-call iterations', async () => {
      const { skill: llm } = createMockLLM([
        {
          text: 'Let me calculate...',
          toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":5,"b":5}' }],
        },
        { text: 'The answer is 10' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const events = await collectEvents(
        agent.processUAMP(buildInputEvents('Calculate'))
      );

      const deltas = events
        .filter(e => e.type === 'response.delta')
        .map(e => {
          const delta = (e as unknown as { delta: { type: string; text?: string } }).delta;
          return delta.type === 'text' ? delta.text : null;
        })
        .filter(Boolean);

      expect(deltas).toContain('Let me calculate...');
      expect(deltas).toContain('The answer is 10');
    });
  });

  describe('run() and runStreaming() integration', () => {
    it('run() returns final text after agentic loop completes', async () => {
      const { skill: llm } = createMockLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":7,"b":8}' }] },
        { text: '7+8=15' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const response = await agent.run([{ role: 'user', content: 'Add 7+8' }]);

      expect(response.content).toContain('15');
    });

    it('runStreaming() yields deltas from the full agentic loop', async () => {
      const { skill: llm } = createMockLLM([
        { toolCalls: [{ id: 'call_1', name: 'add', arguments: '{"a":1,"b":1}' }] },
        { text: 'Result: 2' },
      ]);

      const agent = new BaseAgent({
        skills: [llm, new MathToolsSkill()],
      });

      const chunks: string[] = [];
      for await (const chunk of agent.runStreaming([{ role: 'user', content: 'Add' }])) {
        if (chunk.type === 'delta' && chunk.delta) {
          chunks.push(chunk.delta);
        }
      }

      expect(chunks).toContain('Result: 2');
    });
  });
});
