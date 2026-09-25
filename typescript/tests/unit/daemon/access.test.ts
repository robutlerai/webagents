/**
 * The daemon honours an agent file's `access:` block (ADR-0045), as `serve`
 * and the Python daemon do: an agent built from a file gets the access skill,
 * a refusal is its own status rather than a 500, and a malformed block keeps
 * the agent from being served at all. Until 2026-09-25 the daemon resolved
 * skills with its own short list and never read the block, so a file that
 * kept callers out under `serve` let everyone in under the daemon.
 */

import { describe, it, expect, vi } from 'vitest';
import path from 'node:path';

import { WebAgentsDaemon } from '../../../src/daemon/server';
import type { AgentDefinition } from '../../../src/daemon/watcher';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

function definition(access: unknown): AgentDefinition {
  const dir = tempDir('daemon-access-');
  return {
    name: 'gated',
    instructions: 'Gated.',
    // A model skill, never called: the refusal comes first.
    skills: ['openai', 'rest'],
    skillEntries: ['openai', 'rest'],
    access,
    filePath: path.join(dir, 'AGENT.md'),
    content: '',
  };
}

async function build(daemon: WebAgentsDaemon, def: AgentDefinition) {
  return (daemon as unknown as { buildAgent(d: AgentDefinition): Promise<unknown> }).buildAgent(def);
}

describe('the daemon and the access block', () => {
  it('refuses a caller the block keeps out, with 403', async () => {
    const daemon = new WebAgentsDaemon({ port: 0, watch: false, cron: false });
    const agent = await build(daemon, definition({ default: 'none' }));
    expect(agent).not.toBeNull();
    daemon.registry.registerLocal(agent as never);
    const res = await daemon.app.fetch(
      new Request('http://localhost/agents/gated/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: 'Bearer made-up' },
        body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
      }),
    );
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({ error: { code: 'forbidden', message: 'This agent does not accept requests from this caller.' } });
  });

  it('does not serve an agent whose block is malformed, and says why', async () => {
    const daemon = new WebAgentsDaemon({ port: 0, watch: false, cron: false });
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const def = definition({ groupz: {} });
    expect(await build(daemon, def)).toBeNull();
    expect(errors).toHaveBeenCalledWith(
      `[daemon] ${def.filePath}: access: unknown key "groupz". It takes deny, groups, default, instructions and tools. The agent is not served.`,
    );
    errors.mockRestore();
  });

  it('loads the skills serve loads, rest among them', async () => {
    const daemon = new WebAgentsDaemon({ port: 0, watch: false, cron: false });
    const agent = (await build(daemon, definition(undefined))) as { toolRegistry: Map<string, unknown> };
    expect(agent.toolRegistry.has('rest_request')).toBe(true);
  });
});
