/**
 * `addSkill` registers a decorated `@prompt` once (2026-09-26, the
 * new-developer e2e run): `Skill.initialize()` collects a skill's decorated
 * prompts into `skill.prompts`, so a skill added after it was initialised had
 * every `@prompt` registered twice, from the property and from the decorator
 * scan, and the model read the guide twice. The Python agent's registry has
 * no such double path; this pins the TypeScript side.
 */

import { describe, expect, it } from 'vitest';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { prompt } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';

class Guided extends Skill {
  @prompt({ priority: 50, name: 'guide', scope: 'all' })
  guide(_ctx: Context): string {
    return 'THE ONE GUIDE';
  }
}

function registered(agent: BaseAgent): string[] {
  return (agent as unknown as { promptRegistry: { name: string }[] }).promptRegistry.map((p) => p.name);
}

describe('a decorated prompt is registered once', () => {
  it('when the skill is added before it is initialised', async () => {
    const agent = new BaseAgent({ name: 'a', instructions: 'x', skills: [new Guided()] });
    expect(registered(agent).filter((n) => n === 'guide')).toHaveLength(1);
    await agent.initialize();
    expect(registered(agent).filter((n) => n === 'guide')).toHaveLength(1);
  });

  it('when the skill was initialised first, so its prompts property already carries the prompt', async () => {
    const skill = new Guided();
    await skill.initialize();
    expect(skill.prompts.map((p) => p.name)).toEqual(['guide']);
    const agent = new BaseAgent({ name: 'a', instructions: 'x', skills: [] });
    agent.addSkill(skill);
    expect(registered(agent).filter((n) => n === 'guide')).toHaveLength(1);
    // And the guide reaches the system prompt once.
    const text = await (agent as unknown as { buildSystemPrompt?: (ctx: unknown) => Promise<string> }).buildSystemPrompt?.({});
    if (typeof text === 'string') expect(text.split('THE ONE GUIDE').length - 1).toBe(1);
  });
});
