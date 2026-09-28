/**
 * The host-reaching tools are owner-only unless the agent file opens them
 * (S-248, 2026-09-26).
 *
 * `ShellSkill.runCommand` and the six `FilesystemSkill` tools declared no
 * scopes, so every caller an agent answered could run commands and read and
 * write files as the developer's user. They are `audience: 'owner'` now, and
 * an `access: tools:` block replaces that scope with the named group's. The
 * Python suite pins the same in
 * `tests/skills/local/test_host_tools_owner_only_s248.py`.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';
import { callerScopes, scopeAllows } from '../../../src/core/scopes';
import { applyAccessTools } from '../../../src/access/install';
import { parseAccess } from '../../../src/access/policy';
import { ShellSkill } from '../../../src/skills/shell/skill';
import { FilesystemSkill } from '../../../src/skills/filesystem/skill';

const FILESYSTEM_TOOLS = ['list_directory', 'read_file', 'write_file', 'glob', 'search_file_content', 'replace'];
const HOST_TOOLS = ['runCommand', ...FILESYSTEM_TOOLS];

/** Records which tools each turn was offered. */
class RecordingLLM extends Skill {
  readonly offered: string[][] = [];

  @handoff({ name: 'recording-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const tools = ((ctx.get('_agentic_tools') as Array<{ function?: { name?: string }; name?: string }>) ?? [])
      .map((t) => t.function?.name ?? t.name ?? '')
      .sort();
    this.offered.push(tools);
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };
const USER = { authenticated: true, scope: 'user', user_id: 'u-1', provider: 'portal' };

describe('shell and filesystem tools declare the owner scope', () => {
  it('every host tool is audience owner', () => {
    const shell = new ShellSkill();
    const filesystem = new FilesystemSkill();
    expect(shell.tools.map((t) => t.name)).toEqual(['runCommand']);
    expect(filesystem.tools.map((t) => t.name).sort()).toEqual([...FILESYSTEM_TOOLS].sort());
    for (const tool of [...shell.tools, ...filesystem.tools]) {
      expect(tool.scopes, tool.name).toEqual(['owner']);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: true, scope: 'owner' as never }))).toBe(true);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: true, scope: 'admin' as never }))).toBe(true);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: true, scope: 'user' as never }))).toBe(false);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: false }))).toBe(false);
    }
  });
});

describe('through the agent', () => {
  function build(): { agent: BaseAgent; llm: RecordingLLM; shell: ShellSkill; filesystem: FilesystemSkill } {
    const shell = new ShellSkill();
    const filesystem = new FilesystemSkill();
    const llm = new RecordingLLM();
    const agent = new BaseAgent({ name: 'host', instructions: 'Host.', skills: [shell, filesystem, llm] });
    return { agent, llm, shell, filesystem };
  }

  it('offers them to the owner and to nobody else', async () => {
    const { agent, llm } = build();
    await agent.run([{ role: 'user', content: 'hi' }], { auth: OWNER });
    await agent.run([{ role: 'user', content: 'hi' }], { auth: USER });
    await agent.run([{ role: 'user', content: 'hi' }], {});
    const [owner, user, anonymous] = llm.offered;
    for (const name of HOST_TOOLS) {
      expect(owner, name).toContain(name);
      expect(user, name).not.toContain(name);
      expect(anonymous, name).not.toContain(name);
    }
  });

  it('refuses a stranger who names the tool anyway', async () => {
    const { agent } = build();
    await expect(agent.runTool('runCommand', { command: 'echo hi' }, { auth: USER })).rejects.toThrow();
    await expect(agent.runTool('read_file', { path: 'x' }, {})).rejects.toThrow();
  });

  it('an access block hands them to a group; the owner keeps them, the rest do not', async () => {
    const { agent, llm, shell, filesystem } = build();
    const policy = parseAccess({ groups: { friends: [] }, tools: { friends: ['shell', 'filesystem'] } });
    applyAccessTools(policy, new Map([['shell', shell], ['filesystem', filesystem]]));
    for (const tool of [...shell.tools, ...filesystem.tools]) expect(tool.scopes, tool.name).toEqual(['group:friends']);
    await agent.run([{ role: 'user', content: 'hi' }], { auth: { ...USER, scopes: ['group:friends'] } });
    await agent.run([{ role: 'user', content: 'hi' }], { auth: OWNER });
    await agent.run([{ role: 'user', content: 'hi' }], { auth: USER });
    const [friend, owner, user] = llm.offered;
    for (const name of HOST_TOOLS) {
      expect(friend, name).toContain(name);
      expect(owner, name).toContain(name);
      expect(user, name).not.toContain(name);
    }
  });
});
