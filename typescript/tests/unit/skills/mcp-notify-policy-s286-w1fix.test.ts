/**
 * S-286 (2026-09-26): the `notify` MCP tool policy. TypeScript honours it
 * through a host-provided `policyHook` — a tool with policy `notify` runs only
 * after the hook approves. Python has no approval channel, so it refuses a
 * server whose `toolPolicies` names `notify` (its twin,
 * `tests/agents/skills/test_mcp_notify_policy_s286_w1fix.py`). This asymmetry
 * is recorded in the shared fixture `python/tests/fixtures/mcp_tool/
 * config_shapes.json` (`tool_policies`), which both suites read.
 *
 * This file pins the TypeScript side: the config loads a `notify` server
 * (not refused, unlike Python), and the gate fires when a host set a
 * `policyHook` — a denied decision withholds the call, an approval lets it
 * through, and `notify` with no hook is a no-op (a bare agent file wires
 * none).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { mcpServersFromConfig } from '../../../src/skills/mcp/config';
import { MCPSkill, NOTIFY_NO_HOOK_REFUSAL, notifyNoHookRefusal } from '../../../src/skills/mcp/skill';
import type { Context } from '../../../src/core/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool');
const SHAPES = JSON.parse(readFileSync(path.join(FIXTURES, 'config_shapes.json'), 'utf8')) as {
  tool_policies: {
    config: Record<string, unknown>;
    typescript: { loads: string[] };
    python: { loads: string[]; rejected: { name: string; reason: string }[] };
    no_hook: { refusal: string };
  };
};
const POLICIES = SHAPES.tool_policies;

const CONTEXT = { auth: { authenticated: true, scope: 'owner' }, get: () => undefined } as unknown as Context;

/**
 * Register one dynamic tool for `server`/`tool` under `config`, with a fake
 * session injected, and return its handler and a live view of the call count.
 */
function toolWith(config: Record<string, unknown>, server = 'srv', tool = 'act') {
  const skill = new MCPSkill({}) as unknown as {
    sessions: Map<string, unknown>;
    serverConfigs: Map<string, unknown>;
    _registerDynamicTool: (qualified: string, def: unknown, server: string) => void;
    tools: { name: string; handler: (p: Record<string, unknown>, c: Context) => Promise<unknown> }[];
  };
  const state = { calls: 0 };
  skill.sessions.set(server, {
    async callTool() {
      state.calls += 1;
      return { content: [{ type: 'text', text: 'ran' }] };
    },
  });
  skill.serverConfigs.set(server, config);
  skill._registerDynamicTool(`${server}__${tool}`, { name: tool, description: 'x', inputSchema: {} }, server);
  const handler = skill.tools.find((t) => t.name === `${server}__${tool}`)!.handler;
  return { handler, state };
}

describe('S-286: TypeScript loads a notify server (Python refuses it)', () => {
  it('keeps both servers where Python would reject the notify one', () => {
    const resolution = mcpServersFromConfig(POLICIES.config);
    expect(resolution.servers.map((s) => s.name).sort()).toEqual([...POLICIES.typescript.loads].sort());
    expect(resolution.rejected).toEqual([]);
    // The recorded asymmetry: Python drops `risky`, TypeScript keeps it.
    expect(POLICIES.python.loads).not.toContain('risky');
    expect(POLICIES.python.rejected[0].name).toBe('risky');
  });
});

describe('S-286: TypeScript honours notify through the host policyHook', () => {
  it('withholds the call when the hook does not approve', async () => {
    const seen: unknown[] = [];
    const { handler, state } = toolWith({
      toolPolicies: { act: 'notify' },
      policyHook: async (info: unknown) => {
        seen.push(info);
        return 'rejected';
      },
    });
    const out = await handler({ x: 1 }, CONTEXT);
    expect(out).toBe('Error: User declined to approve srv__act.');
    expect(state.calls).toBe(0);
    expect((seen[0] as { qualifiedName: string }).qualifiedName).toBe('srv__act');
  });

  it('runs the tool when the hook approves', async () => {
    const { handler, state } = toolWith({
      toolPolicies: { act: 'notify' },
      policyHook: async () => 'approved',
    });
    const out = await handler({}, CONTEXT);
    expect(out).toBe('ran');
    expect(state.calls).toBe(1);
  });

  it('is REFUSED, not run, when notify is set but no hook is provided (a bare agent file; S-286 addendum)', async () => {
    const { handler, state } = toolWith({ toolPolicies: { act: 'notify' } });
    const out = await handler({}, CONTEXT);
    expect(out).toBe(POLICIES.no_hook.refusal.replace('{name}', 'srv__act'));
    expect(out).toBe(notifyNoHookRefusal('srv__act'));
    expect(state.calls).toBe(0);
  });

  it('the refusal sentence is the fixture’s', () => {
    expect(NOTIFY_NO_HOOK_REFUSAL).toBe(POLICIES.no_hook.refusal);
  });
});
