/**
 * Owner-scoped skill prompts reach the owner, and only the owner (2026-09-24).
 *
 * THE DEFECT THIS PINS. `_enhanceInstructionsWithPrompts` called
 * `_executePrompts()` with no scope, so every run was filtered at `all`. An
 * `@prompt({ scope: 'owner' })` block (`secretsGuide`, `selfEditGuide`)
 * reached nobody, the owner included, while the owner-gated tools it explains
 * were offered to the owner: the tools without the instructions for them. The
 * fix reads the tier off the RUN context with the tool gate's own `hasScope`.
 *
 * The cases: each tier on `run()`, including owner scope in the shape the
 * portal's `executeAgent` overlays it (`scope` plus `scopes`); the same on
 * `runStreaming()`; one instance serving an owner and then a stranger (what
 * would catch the enhanced instructions being cached on the instance); and
 * two overlapping runs of one instance, each seeing its own tier.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, prompt } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';

const PUBLIC = 'PUBLIC-GUIDE';
const OWNER = 'OWNER-GUIDE';
const ADMIN = 'ADMIN-GUIDE';

class GuideSkill extends Skill {
  /** When set, every run waits here inside `_executePrompts` until released. */
  meet?: () => Promise<void>;

  @prompt({ name: 'publicGuide', scope: 'all' })
  async publicGuide(): Promise<string> {
    if (this.meet) await this.meet();
    return PUBLIC;
  }

  @prompt({ name: 'ownerGuide', scope: 'owner' })
  ownerGuide(): string {
    return OWNER;
  }

  @prompt({ name: 'adminGuide', scope: 'admin' })
  adminGuide(): string {
    return ADMIN;
  }
}

/** Stands in for the model: records each run's system message under the run's caller. */
class CaptureLLM extends Skill {
  readonly systems = new Map<string, string>();

  @handoff({ name: 'capture-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const messages = (ctx.get('_agentic_messages') as Array<{ role: string; content?: unknown }>) ?? [];
    const system = messages.find((m) => m.role === 'system');
    this.systems.set(String(ctx.auth?.user_id ?? 'anonymous'), typeof system?.content === 'string' ? system.content : '');
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

/** Resolves for everyone once `n` callers have arrived, so runs are provably in flight together. */
function meeting(n: number): () => Promise<void> {
  let arrived = 0;
  let open!: () => void;
  const all = new Promise<void>((resolve) => { open = resolve; });
  return async () => {
    arrived += 1;
    if (arrived >= n) open();
    await all;
  };
}

function build(): { agent: BaseAgent; guides: GuideSkill; capture: CaptureLLM } {
  const guides = new GuideSkill();
  const capture = new CaptureLLM();
  const agent = new BaseAgent({ name: 'probe', skills: [guides, capture] });
  return { agent, guides, capture };
}

const hi = [{ role: 'user' as const, content: 'hi' }];

describe('owner-scoped prompts', () => {
  it('an anonymous run gets the public prompts only', async () => {
    const { agent, capture } = build();
    await agent.run(hi);
    const system = capture.systems.get('anonymous') ?? '';
    expect(system).toContain(PUBLIC);
    expect(system).not.toContain(OWNER);
    expect(system).not.toContain(ADMIN);
  });

  it('a signed-in caller who is not the owner does not get owner prompts', async () => {
    const { agent, capture } = build();
    await agent.run(hi, { userId: 'stranger', auth: { scope: 'user' } });
    const system = capture.systems.get('stranger') ?? '';
    expect(system).toContain(PUBLIC);
    expect(system).not.toContain(OWNER);
  });

  it('the owner gets owner prompts (the defect: they reached nobody)', async () => {
    const { agent, capture } = build();
    await agent.run(hi, { userId: 'owner', auth: { scope: 'owner' } });
    const system = capture.systems.get('owner') ?? '';
    expect(system).toContain(PUBLIC);
    expect(system).toContain(OWNER);
    expect(system).not.toContain(ADMIN);
  });

  it('owner scope carried in `scopes`, the way the portal overlays it, counts', async () => {
    const { agent, capture } = build();
    await agent.run(hi, { userId: 'owner', auth: { scopes: ['owner'] } });
    expect(capture.systems.get('owner') ?? '').toContain(OWNER);
  });

  it('an admin run sees owner and admin prompts, as the tier order says', async () => {
    const { agent, capture } = build();
    await agent.run(hi, { userId: 'admin', auth: { scope: 'admin' } });
    const system = capture.systems.get('admin') ?? '';
    expect(system).toContain(OWNER);
    expect(system).toContain(ADMIN);
  });

  it('runStreaming() applies the same tiers', async () => {
    const { agent, capture } = build();
    for await (const _ of agent.runStreaming(hi, { userId: 'owner', auth: { scope: 'owner' } })) {
      // drain
    }
    for await (const _ of agent.runStreaming(hi, { userId: 'stranger' })) {
      // drain
    }
    expect(capture.systems.get('owner') ?? '').toContain(OWNER);
    expect(capture.systems.get('stranger') ?? '').not.toContain(OWNER);
  });

  it('one instance serving the owner and then a stranger does not carry the owner tier over', async () => {
    const { agent, capture } = build();
    await agent.run(hi, { userId: 'owner', auth: { scope: 'owner' } });
    await agent.run(hi, { userId: 'stranger' });
    await agent.run(hi, { userId: 'owner-again', auth: { scope: 'owner' } });
    expect(capture.systems.get('owner') ?? '').toContain(OWNER);
    expect(capture.systems.get('stranger') ?? '').not.toContain(OWNER);
    expect(capture.systems.get('owner-again') ?? '').toContain(OWNER);
  });

  it('two overlapping runs of one instance each get their own tier', async () => {
    const { agent, guides, capture } = build();
    guides.meet = meeting(2);
    await Promise.all([
      agent.run(hi, { userId: 'owner', auth: { scope: 'owner' } }),
      agent.run(hi, { userId: 'stranger' }),
    ]);
    expect(capture.systems.get('owner') ?? '').toContain(OWNER);
    expect(capture.systems.get('stranger') ?? '').toContain(PUBLIC);
    expect(capture.systems.get('stranger') ?? '').not.toContain(OWNER);
  });
});
