/**
 * Tools a skill registers DURING `initialize()` must reach the agent.
 *
 * WHY THIS EXISTS (2026-09-16). `addSkill` copies `skill.tools` into the
 * agent's registry at the moment the skill is added, and `registerTool` only
 * pushes to the skill's own array: it holds no reference to the agent and
 * nothing re-read it. `agent.initialize()` runs every skill's `initialize()`
 * AFTER all of that, so any tool registered there landed on the skill and never
 * on the agent.
 *
 * The case that exposed it was `OpenAPISkill`, which registers its meta tool in
 * the constructor and one tool per DISCOVERED operation in `initialize()` —
 * discovery needs a network fetch, so it cannot happen any earlier. An agent
 * given an OpenAPI integration therefore ended up able to LIST its operations
 * and unable to CALL any of them: measured as `10 skill-tools, 9 agent-tools`
 * on a live agent, the missing one being the only real operation. The model
 * read the name out of the meta tool's own output, called it, and the turn
 * ended silently, because a name the agent does not know is handed back to the
 * client as an external tool call and nothing on the other side runs it. No
 * error, no result, an empty reply.
 *
 * That silence is why this is a test and not a comment: every visible signal
 * said the integration was wired up.
 */

import { describe, it, expect, vi } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import type { Tool } from '../../../src/core/types.js';

/* `enabled` is load-bearing: `Skill.tools` filters on it, so a tool without it
   is invisible to the skill's own getter and therefore to the agent too. */
const mkTool = (name: string): Tool => ({
  name,
  description: `tool ${name}`,
  parameters: { type: 'object', properties: {} },
  enabled: true,
  execute: async () => 'ok',
});

/** Registers one tool up front and one only once it has initialized. */
class LateSkill extends Skill {
  constructor(private readonly early: string, private readonly late: string) {
    super({});
    (this as unknown as { registerTool(t: Tool): void }).registerTool(mkTool(early));
  }

  async initialize(): Promise<void> {
    (this as unknown as { registerTool(t: Tool): void }).registerTool(mkTool(this.late));
  }
}

describe('a skill that registers tools while initializing', () => {
  it('lands them on the AGENT, not only on the skill', async () => {
    const agent = new BaseAgent({ name: 'test', instructions: 'x' });
    const skill = new LateSkill('meta_tool', 'server__operation');
    agent.addSkill(skill);

    // Before initialize: only what the constructor registered.
    expect(agent.getToolDefinitions().map((t) => t.function.name)).toContain('meta_tool');
    expect(agent.getToolDefinitions().map((t) => t.function.name)).not.toContain('server__operation');

    await agent.initialize();

    /* THE ASSERTION THAT WAS FAILING. The skill's own list had both all along;
       the agent's had one. `getToolDefinitions` is what the model is SHOWN
       (`{ type: 'function', function: { name, ... } }`), so a tool missing
       here is a tool the model can read about and cannot call. */
    expect(skill.tools.map((t) => t.name).sort()).toEqual(['meta_tool', 'server__operation']);
    expect(agent.getToolDefinitions().map((t) => t.function.name).sort()).toEqual(['meta_tool', 'server__operation']);
  });

  it('does not warn about a tool it is merely re-syncing', async () => {
    /* The re-sync runs over every tool the skill has, including the ones
       already registered. Warning on those would print a false collision for
       each one on every boot, which is how a real collision stops being
       noticed. */
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const agent = new BaseAgent({ name: 'test', instructions: 'x' });
      agent.addSkill(new LateSkill('meta_tool', 'server__operation'));
      await agent.initialize();
      expect(warn.mock.calls.filter((c) => String(c[0]).includes('already registered'))).toEqual([]);
    } finally { warn.mockRestore(); }
  });

  it('still warns when two different skills claim one name', async () => {
    /* The identity check is what makes the re-sync silent; it must not make a
       genuine clash silent too. */
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const agent = new BaseAgent({ name: 'test', instructions: 'x' });
      agent.addSkill(new LateSkill('shared_name', 'a__op'));
      agent.addSkill(new LateSkill('shared_name', 'b__op'));
      await agent.initialize();
      expect(warn.mock.calls.some((c) => String(c[0]).includes('"shared_name" already registered'))).toBe(true);
    } finally { warn.mockRestore(); }
  });
});
