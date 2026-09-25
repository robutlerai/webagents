/**
 * The one scope rule (ADR-0045 section 5, 2026-09-25), from the table both SDKs
 * run, and where the TypeScript agent applies it: the tools offered to the
 * model, the tool that would run, handoff choice and prompts.
 *
 * Before this the tool gate required EVERY listed scope (Python: any one), an
 * admin failed an `owner` tool while passing an `owner` prompt, and a prompt
 * with an unknown scope (a `group:` prompt among them) reached everyone.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { createContext } from '../../../src/core/context';
import { callerScopes, scopeAllows } from '../../../src/core/scopes';
import { handoff, prompt, tool } from '../../../src/core/decorators';
import type { AuthInfo, Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/scopes/scope_allows.json'), 'utf8'),
) as { cases: Array<{ required: string | string[] | null; caller: string[]; allowed: boolean }> };

describe('the shared table', () => {
  for (const c of TABLE.cases) {
    it(`${JSON.stringify(c.required)} <- ${JSON.stringify(c.caller)}`, () => {
      expect(scopeAllows(c.required, c.caller)).toBe(c.allowed);
    });
  }
});

describe('callerScopes', () => {
  it('anonymous holds nothing', () => {
    expect([...callerScopes({ authenticated: false })]).toEqual([]);
  });

  it('the list, the tier, and user when authenticated', () => {
    const held = callerScopes({ authenticated: true, scope: 'owner' as AuthInfo['scope'], scopes: ['group:friends'] });
    expect([...held].sort()).toEqual(['group:friends', 'owner', 'user']);
  });

  it('an admin now passes an owner check, as the prompt tier always let it', () => {
    const ctx = createContext({ auth: { authenticated: true, scope: 'admin' as AuthInfo['scope'] } });
    expect(ctx.hasScope('owner')).toBe(true);
    expect(ctx.hasScope('group:anything')).toBe(true);
  });
});

const FRIENDS = 'FRIENDS-GUIDE';
const ODD = 'ODD-GUIDE';

class Guarded extends Skill {
  @tool({ name: 'friends_tool', description: 'For friends.', scopes: ['group:friends'] })
  async friendsTool(): Promise<string> {
    return 'friends';
  }

  @tool({ name: 'kin_tool', description: 'For friends or family.', scopes: ['group:friends', 'group:family'] })
  async kinTool(): Promise<string> {
    return 'kin';
  }

  @tool({ name: 'odd_tool', description: 'A scope nobody grants.', scopes: ['project'] })
  async oddTool(): Promise<string> {
    return 'odd';
  }

  @prompt({ name: 'friendsGuide', scope: 'group:friends' })
  friendsGuide(): string {
    return FRIENDS;
  }

  @prompt({ name: 'oddGuide', scope: 'project' })
  oddGuide(): string {
    return ODD;
  }
}

/** Stands in for the model: records, per caller, the system text and the tool names it was offered. */
class Capture extends Skill {
  readonly seen = new Map<string, { system: string; tools: string[] }>();

  @handoff({ name: 'capture-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const messages = (ctx.get('_agentic_messages') as Array<{ role: string; content?: unknown }>) ?? [];
    const system = messages.find((m) => m.role === 'system');
    const tools = ((ctx.get('_agentic_tools') as Array<{ function?: { name?: string }; name?: string }>) ?? [])
      .map((t) => t.function?.name ?? t.name ?? '');
    this.seen.set(String(ctx.auth?.user_id ?? 'anonymous'), {
      system: typeof system?.content === 'string' ? system.content : '',
      tools,
    });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

const hi = [{ role: 'user' as const, content: 'hi' }];

async function seenBy(auth: Partial<AuthInfo> | undefined, who: string): Promise<{ system: string; tools: string[] }> {
  const capture = new Capture();
  const agent = new BaseAgent({ name: 'probe', instructions: 'Probe.', skills: [new Guarded(), capture] });
  await agent.run(hi, auth ? { userId: who, auth: auth as AuthInfo } : undefined);
  return capture.seen.get(auth ? who : 'anonymous') ?? { system: '', tools: [] };
}

describe('the agent applies it', () => {
  it('a group tool and a group prompt are the group\'s', async () => {
    const stranger = await seenBy({ scope: 'user' as AuthInfo['scope'] }, 'stranger');
    expect(stranger.tools).not.toContain('friends_tool');
    expect(stranger.system).not.toContain(FRIENDS);

    const friend = await seenBy({ scope: 'user' as AuthInfo['scope'], scopes: ['group:friends'] }, 'friend');
    expect(friend.tools).toContain('friends_tool');
    expect(friend.system).toContain(FRIENDS);

    const owner = await seenBy({ scope: 'owner' as AuthInfo['scope'] }, 'owner');
    expect(owner.tools).toContain('friends_tool');
    expect(owner.system).toContain(FRIENDS);
  });

  it('a declared list is any-of', async () => {
    const family = await seenBy({ scope: 'user' as AuthInfo['scope'], scopes: ['group:family'] }, 'family');
    expect(family.tools).toContain('kin_tool');
    expect(family.tools).not.toContain('friends_tool');
  });

  it('an unknown scope fails closed, for tools and for prompts', async () => {
    for (const tier of [undefined, 'user', 'owner', 'admin']) {
      const seen = await seenBy(tier ? { scope: tier as AuthInfo['scope'] } : undefined, `t-${tier}`);
      expect(seen.tools).not.toContain('odd_tool');
      expect(seen.system).not.toContain(ODD);
    }
  });

  it('the tool that would run is checked the same way', async () => {
    const agent = new BaseAgent({ name: 'probe', skills: [new Guarded()] });
    await expect(agent.executeTool('friends_tool', {})).rejects.toThrow(/Insufficient permissions/);
  });
});
