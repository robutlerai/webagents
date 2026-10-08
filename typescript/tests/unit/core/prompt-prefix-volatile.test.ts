/**
 * Stable and volatile skill prompts (`Prompt.volatile`).
 *
 * A provider's prompt cache matches a request's leading bytes exactly, so
 * the base system message has to be the same from one request to the next
 * and from one caller to the next. A prompt whose text depends on who is
 * calling or on the turn is therefore rendered into a system message of its
 * own, right behind the base message, and never into the base message.
 *
 * Pinned here:
 *   - stable prompts land in the base system message, in priority order;
 *   - volatile prompts land in ONE second system message, in priority
 *     order, and the base message does not contain them;
 *   - two callers whose volatile prompts differ get a byte-identical base
 *     message;
 *   - the flag travels through the `@prompt` decorator, not only through
 *     `registerPrompt`;
 *   - with no volatile text there is no second system message;
 *   - a volatile prompt filtered out by scope leaves nothing behind.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, prompt } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';

const BASE = 'You are the probe.';

class Prompts extends Skill {
  constructor(private readonly withVolatile: boolean) {
    super({ name: 'prompts' });
    this.registerPrompt({ name: 'rules', priority: 10, scope: 'all', handler: () => 'RULES' });
    this.registerPrompt({ name: 'guide', priority: 5, scope: 'all', handler: () => 'GUIDE' });
    if (withVolatile) {
      this.registerPrompt({
        name: 'caller',
        priority: 4,
        scope: 'all',
        volatile: true,
        handler: (ctx: Context) => `CALLER ${ctx.auth?.user_id ?? 'anonymous'}`,
      });
      this.registerPrompt({ name: 'ownerOnly', priority: 6, scope: 'owner', volatile: true, handler: () => 'OWNER-RULES' });
    }
  }
}

class Decorated extends Skill {
  @prompt({ name: 'decoratedVolatile', priority: 7, volatile: true })
  decoratedVolatile(ctx: Context): string {
    return `DECORATED ${ctx.auth?.user_id ?? 'anonymous'}`;
  }

  @prompt({ name: 'decoratedStable', priority: 8 })
  decoratedStable(): string {
    return 'DECORATED-STABLE';
  }
}

/** Stands in for the model: keeps the leading system messages of each run, keyed by caller. */
class CaptureLLM extends Skill {
  readonly systems = new Map<string, string[]>();

  @handoff({ name: 'capture-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const messages = (ctx.get('_agentic_messages') as Array<{ role: string; content?: unknown }>) ?? [];
    const leading: string[] = [];
    for (const m of messages) {
      if (m.role !== 'system') break;
      leading.push(String(m.content ?? ''));
    }
    this.systems.set(String(ctx.auth?.user_id ?? 'anonymous'), leading);
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

const hi = [{ role: 'user' as const, content: 'hi' }];

describe('stable and volatile prompts', () => {
  it('stable prompts join the base message; volatile ones go into the system message behind it', async () => {
    const capture = new CaptureLLM();
    const agent = new BaseAgent({ name: 'probe', instructions: BASE, skills: [new Prompts(true), capture] });
    await agent.run(hi, { userId: 'ada', auth: { scope: 'user' } });
    const leading = capture.systems.get('ada')!;
    expect(leading).toHaveLength(2);
    expect(leading[0]).toBe(`${BASE}\n\nGUIDE\n\nRULES`);
    expect(leading[1]).toBe('CALLER ada');
  });

  it('two callers share a byte-identical base message', async () => {
    const capture = new CaptureLLM();
    const agent = new BaseAgent({ name: 'probe', instructions: BASE, skills: [new Prompts(true), capture] });
    await agent.run(hi, { userId: 'ada', auth: { scope: 'user' } });
    await agent.run(hi, { userId: 'bob', auth: { scope: 'user' } });
    const ada = capture.systems.get('ada')!;
    const bob = capture.systems.get('bob')!;
    expect(ada[0]).toBe(bob[0]);
    expect(ada[1]).toBe('CALLER ada');
    expect(bob[1]).toBe('CALLER bob');
  });

  it('an owner gets the owner-scoped volatile prompt in the same second message, after the earlier one', async () => {
    const capture = new CaptureLLM();
    const agent = new BaseAgent({ name: 'probe', instructions: BASE, skills: [new Prompts(true), capture] });
    await agent.run(hi, { userId: 'owner', auth: { scope: 'owner', scopes: ['owner'] } });
    const leading = capture.systems.get('owner')!;
    expect(leading[0]).toBe(`${BASE}\n\nGUIDE\n\nRULES`);
    expect(leading[1]).toBe('CALLER owner\n\nOWNER-RULES');
  });

  it('with no volatile text there is no second system message', async () => {
    const capture = new CaptureLLM();
    const agent = new BaseAgent({ name: 'probe', instructions: BASE, skills: [new Prompts(false), capture] });
    await agent.run(hi, { userId: 'ada', auth: { scope: 'user' } });
    expect(capture.systems.get('ada')).toEqual([`${BASE}\n\nGUIDE\n\nRULES`]);
  });

  it('the decorator carries the flag', async () => {
    const capture = new CaptureLLM();
    const agent = new BaseAgent({ name: 'probe', instructions: BASE, skills: [new Decorated(), capture] });
    await agent.run(hi, { userId: 'ada', auth: { scope: 'user' } });
    const leading = capture.systems.get('ada')!;
    expect(leading[0]).toBe(`${BASE}\n\nDECORATED-STABLE`);
    expect(leading[1]).toBe('DECORATED ada');
  });

  it('with no base instructions the stable prompts become the base and the volatile ones follow', async () => {
    const capture = new CaptureLLM();
    const agent = new BaseAgent({ name: 'probe', skills: [new Prompts(true), capture] });
    await agent.run(hi, { userId: 'ada', auth: { scope: 'user' } });
    expect(capture.systems.get('ada')).toEqual(['GUIDE\n\nRULES', 'CALLER ada']);
  });
});
