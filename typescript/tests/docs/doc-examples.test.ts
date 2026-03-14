/**
 * Runnable tests that mirror the highest-value TypeScript doc examples.
 *
 * Each test references the doc file/section it validates.
 * No real API keys or LLM calls -- tests validate SDK wiring only.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { tool, hook, handoff } from '../../src/core/decorators.js';
import type { Context } from '../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../src/uamp/events.js';

// ---------------------------------------------------------------------------
// Mock LLM handoff -- echoes input text back
// ---------------------------------------------------------------------------

class EchoLLM extends Skill {
  @handoff({ name: 'echo-llm' })
  async *processUAMP(
    events: ClientEvent[],
    _ctx: Context,
  ): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text') {
        texts.push((e as { type: string; text: string }).text);
      }
    }
    const response = texts.join(' | ');
    yield createResponseDeltaEvent('r1', { type: 'text', text: response });
    yield createResponseDoneEvent(
      'r1',
      [{ type: 'text', text: response }],
      'completed',
      { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    );
  }
}

// ---------------------------------------------------------------------------
// Validates: quickstart.md -- BaseAgent with skills array
// ---------------------------------------------------------------------------

describe('Quickstart agent creation', () => {
  it('creates an agent with name, model, and skills array', () => {
    const agent = new BaseAgent({
      name: 'quickstart-agent',
      instructions: 'You are a helpful assistant.',
      model: 'openai/gpt-4o',
      skills: [new EchoLLM()],
    });

    expect(agent.name).toBe('quickstart-agent');
    expect(agent.model).toBe('openai/gpt-4o');
  });

  it('runs and returns a response using the echo LLM', async () => {
    const agent = new BaseAgent({
      name: 'echo-agent',
      skills: [new EchoLLM()],
    });
    const response = await agent.run([{ role: 'user', content: 'Hello' }]);
    expect(response.content).toBe('Hello');
    expect(response.usage?.total_tokens).toBe(15);
  });
});

// ---------------------------------------------------------------------------
// Validates: api/typescript.md -- Skill with @tool decorator
// ---------------------------------------------------------------------------

describe('Skill with @tool', () => {
  class MathSkill extends Skill {
    @tool({
      description: 'Add two numbers',
      parameters: {
        type: 'object',
        properties: {
          a: { type: 'number' },
          b: { type: 'number' },
        },
        required: ['a', 'b'],
      },
    })
    async add(params: { a: number; b: number }, _ctx: Context) {
      return params.a + params.b;
    }
  }

  it('collects tool metadata from decorated methods', () => {
    const skill = new MathSkill();
    expect(skill.tools.length).toBeGreaterThanOrEqual(1);
    const addTool = skill.tools.find((t) => t.name === 'add');
    expect(addTool).toBeDefined();
    expect(addTool!.description).toBe('Add two numbers');
  });
});

// ---------------------------------------------------------------------------
// Validates: agent/lifecycle.md -- @hook decorator
// ---------------------------------------------------------------------------

describe('Skill with @hook', () => {
  class LoggingSkill extends Skill {
    public connected = false;
    public finalized = false;

    @hook({ lifecycle: 'on_connection', priority: 1 })
    async onConnection(_data: unknown, _ctx: Context) {
      this.connected = true;
      return {};
    }

    @hook({ lifecycle: 'finalize_connection', priority: 99 })
    async onFinalize(_data: unknown, _ctx: Context) {
      this.finalized = true;
      return {};
    }
  }

  it('collects hook metadata from decorated methods', () => {
    const skill = new LoggingSkill();
    const lifecycles = skill.hooks.map((h) => h.lifecycle);
    expect(lifecycles).toContain('on_connection');
    expect(lifecycles).toContain('finalize_connection');
  });
});

// ---------------------------------------------------------------------------
// Validates: UAMP transport -- agent responds to session.create + input.text
// ---------------------------------------------------------------------------

describe('UAMP event processing', () => {
  it('processUAMP yields response.delta and response.done', async () => {
    const agent = new BaseAgent({
      name: 'uamp-agent',
      skills: [new EchoLLM()],
    });

    const events: ServerEvent[] = [];
    for await (const event of agent.processUAMP([
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('hello world'),
      createResponseCreateEvent(),
    ])) {
      events.push(event);
    }

    const types = events.map((e) => e.type);
    expect(types).toContain('response.delta');
    expect(types).toContain('response.done');

    const delta = events.find((e) => e.type === 'response.delta');
    expect((delta as Record<string, unknown>).delta).toEqual({
      type: 'text',
      text: 'hello world',
    });
  });
});

// ---------------------------------------------------------------------------
// Validates: server/ -- createFetchHandler
// ---------------------------------------------------------------------------

describe('Server handler', () => {
  it('createFetchHandler returns a function', async () => {
    const { createFetchHandler } = await import(
      '../../src/server/handler.js'
    );
    const agent = new BaseAgent({
      name: 'handler-agent',
      skills: [new EchoLLM()],
    });
    const handler = createFetchHandler(agent);
    expect(typeof handler).toBe('function');
  });
});
