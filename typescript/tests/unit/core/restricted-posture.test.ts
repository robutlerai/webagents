/**
 * The restricted tool posture is enforced (2026-09-24, portal security log S-030).
 *
 * THE DEFECT THIS PINS. The portal computes a posture for every turn someone
 * other than the agent's owner causes (`lib/turns/posture.ts`) and stamps it
 * on the run as `ctx.metadata.turn.posture`. Nothing read it, so a
 * `restricted` turn, a stranger's message waking the agent while its owner
 * was silent, ran with every tool the agent had.
 *
 * The rule under test, at every place a tool is offered or run:
 *   - no posture, or `full`: every tool, exactly as before;
 *   - `restricted`: only tools that are, or whose skill is, declared `allow`
 *     (skills default to `deny`, so an unclassified skill is closed);
 *   - anything else (`classify`, an unknown value): no tool at all.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, tool } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent, SessionCreateEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';

/** An unclassified skill: denied in a restricted turn by default. */
class SpendSkill extends Skill {
  calls = 0;
  @tool({ name: 'spend', description: 'spends money' })
  async spend(): Promise<string> { this.calls++; return 'spent'; }
  @tool({ name: 'peek', description: 'read-only, opted in', restrictedPosture: 'allow' })
  async peek(): Promise<string> { return 'peeked'; }
}

/** A skill declared safe for restricted turns. */
class ClockSkill extends Skill {
  static restrictedPostureDefault = 'allow' as const;
  @tool({ name: 'clock', description: 'reads the clock' })
  async clock(): Promise<string> { return 'noon'; }
  @tool({ name: 'set_alarm', description: 'writes, opted out', restrictedPosture: 'deny' })
  async setAlarm(): Promise<string> { return 'set'; }
}

/** Records the tool names the run offered the model. */
class CaptureLLM extends Skill {
  static restrictedPostureDefault = 'allow' as const;
  offered: string[] = [];
  @handoff({ name: 'capture-llm' })
  async *processUAMP(events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
    const create = events.find((e) => e.type === 'session.create') as SessionCreateEvent | undefined;
    const tools = (create?.session?.tools ?? []) as Array<{ function?: { name?: string } }>;
    this.offered = tools.map((t) => t.function?.name ?? '').filter(Boolean).sort();
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

function build() {
  const spend = new SpendSkill();
  const clock = new ClockSkill();
  const capture = new CaptureLLM();
  const agent = new BaseAgent({ name: 'probe', skills: [spend, clock, capture] });
  return { agent, spend, clock, capture };
}

const hi = [{ role: 'user' as const, content: 'hi' }];
const at = (posture: unknown) => ({ metadata: { turn: { id: null, posture } } });

describe('restricted posture: what the model is offered', () => {
  it('no posture offers every tool (every owner chat, every legacy path)', async () => {
    const { agent, capture } = build();
    await agent.run(hi);
    expect(capture.offered).toEqual(['clock', 'peek', 'set_alarm', 'spend']);
  });

  it('full offers every tool', async () => {
    const { agent, capture } = build();
    await agent.run(hi, at('full'));
    expect(capture.offered).toEqual(['clock', 'peek', 'set_alarm', 'spend']);
  });

  it('restricted offers only what is declared allowed, by skill or by tool', async () => {
    const { agent, capture } = build();
    await agent.run(hi, at('restricted'));
    // clock: its skill allows; peek: the tool opts in inside a denied skill;
    // set_alarm: the tool opts out inside an allowed skill; spend: unclassified.
    expect(capture.offered).toEqual(['clock', 'peek']);
  });

  it('an unknown posture (the design\'s classify, or a typo) offers nothing', async () => {
    const { agent, capture } = build();
    await agent.run(hi, at('classify'));
    expect(capture.offered).toEqual([]);
  });

  it('a host that classifies a skill after adding it is heard (read at check time)', async () => {
    const { agent, spend, capture } = build();
    spend.restrictedPosture = 'allow';
    await agent.run(hi, at('restricted'));
    expect(capture.offered).toContain('spend');
  });

  it('config overrides the class default', () => {
    expect(new SpendSkill({ restrictedPosture: 'allow' }).restrictedPosture).toBe('allow');
    expect(new ClockSkill({ restrictedPosture: 'deny' }).restrictedPosture).toBe('deny');
    expect(new SpendSkill().restrictedPosture).toBe('deny');
    expect(new ClockSkill().restrictedPosture).toBe('allow');
  });
});

describe('restricted posture: what can run', () => {
  it('a denied tool is refused at execution, and its handler never runs', async () => {
    const { agent, spend } = build();
    await expect(agent.runTool('spend', {}, at('restricted'))).rejects.toThrow(/not available in this turn/);
    expect(spend.calls).toBe(0);
  });

  it('an allowed tool runs', async () => {
    const { agent } = build();
    await expect(agent.runTool('clock', {}, at('restricted'))).resolves.toBe('noon');
    await expect(agent.runTool('peek', {}, at('restricted'))).resolves.toBe('peeked');
  });

  it('the same denied tool runs at full posture', async () => {
    const { agent, spend } = build();
    await expect(agent.runTool('spend', {}, at('full'))).resolves.toBe('spent');
    expect(spend.calls).toBe(1);
  });

  it('the agent\'s own built-ins are closed in a restricted turn (they read and write the owner\'s content)', async () => {
    const { agent, capture } = build();
    await agent.run(hi, at('restricted'));
    for (const builtIn of ['present', 'read_content', 'save_content']) {
      expect(capture.offered).not.toContain(builtIn);
    }
  });
});
