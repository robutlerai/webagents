/**
 * The session skill, served (2026-09-25): an agent keeps each verified
 * caller's conversation when the request names it (`src/skills/session/skill.ts`,
 * the Python skill's twin). Pinned through the skill's own hook, over a
 * context shaped like a served turn's: the owner's conversation is kept where
 * the chat keeps theirs, another caller's in a namespace of their own, nothing
 * for an anonymous caller or a request that names no session, and two callers
 * naming the same id get two conversations. The skill has no tools: the
 * scratchpad it replaced was keyed on an id the caller chose (S-262).
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { SessionSkill } from '../../../../src/skills/session/skill';
import { callerSessionsDir, sessionsDir } from '../../../../src/cli/sessions';
import { tempDirs } from '../../../helpers/cli';

const tempDir = tempDirs();
const SESSION = '0f1e2d3c-4b5a-4968-8776-655443322110';

let home: string | undefined;
let profile: string | undefined;
let folder = '';

beforeEach(() => {
  home = process.env.HOME;
  profile = process.env.WEBAGENTS_PROFILE;
  process.env.HOME = tempDir('wa-session-home-');
  delete process.env.WEBAGENTS_PROFILE;
  folder = tempDir('wa-session-agent-');
});

afterEach(() => {
  if (home === undefined) delete process.env.HOME;
  else process.env.HOME = home;
  if (profile !== undefined) process.env.WEBAGENTS_PROFILE = profile;
});

const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ALICE = { authenticated: true, provider: 'platform', scopes: [], user_id: 'alice' };
const BOB = { authenticated: true, provider: 'platform', scopes: [], user_id: 'bob' };
const NOBODY = { authenticated: false };

async function serveTurn(skill: SessionSkill, auth: unknown, metadata: Record<string, unknown>, reply = 'Step one: the date.') {
  const context = { auth, metadata } as never;
  const request = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Plan the launch.' },
  ];
  await skill.keepConversation({ messages: request as never, response: reply }, context);
}

const skillFor = () => new SessionSkill({ agentDir: folder, agentName: 'helper' });
const kept = (dir: string) => JSON.parse(fs.readFileSync(path.join(dir, `${SESSION}.json`), 'utf-8'));

describe('SessionSkill (served)', () => {
  it("keeps the owner's conversation where the chat keeps theirs, owner-only", async () => {
    await serveTurn(skillFor(), OWNER, { session_id: SESSION });

    const dir = sessionsDir(folder, 'helper');
    const data = kept(dir);
    expect(data.messages).toEqual([
      { role: 'user', content: 'Plan the launch.' },
      { role: 'assistant', content: 'Step one: the date.' },
    ]);
    expect(data.agent_name).toBe('helper');
    expect(data.metadata.sdk).toBe('typescript');
    expect(fs.statSync(path.join(dir, `${SESSION}.json`)).mode & 0o777).toBe(0o600);
  });

  it('gives each caller a namespace of their own', async () => {
    const skill = skillFor();
    await serveTurn(skill, ALICE, { session_id: SESSION }, 'for alice');
    await serveTurn(skill, BOB, { session_id: SESSION }, 'for bob');

    expect(kept(callerSessionsDir(folder, 'helper', 'user:alice')).messages.at(-1).content).toBe('for alice');
    expect(kept(callerSessionsDir(folder, 'helper', 'user:bob')).messages.at(-1).content).toBe('for bob');
    expect(fs.existsSync(path.join(sessionsDir(folder, 'helper'), `${SESSION}.json`))).toBe(false);
  });

  it('keeps nothing for an anonymous caller, without a session, or for an id that is a path', async () => {
    const skill = skillFor();
    await serveTurn(skill, NOBODY, { session_id: SESSION });
    await serveTurn(skill, OWNER, {});
    await serveTurn(skill, OWNER, { session_id: '../../escape' });

    expect(fs.existsSync(sessionsDir(folder, 'helper'))).toBe(false);
  });

  it('the next turn replaces the conversation and keeps when it began', async () => {
    const skill = skillFor();
    await serveTurn(skill, OWNER, { session_id: SESSION });
    const first = kept(sessionsDir(folder, 'helper'));
    await serveTurn(skill, OWNER, { session_id: SESSION }, 'Step two.');
    const second = kept(sessionsDir(folder, 'helper'));

    expect(second.created_at).toBe(first.created_at);
    expect(second.messages.at(-1).content).toBe('Step two.');
  });

  it('has no tools and refuses a backend it does not know, naming the fix', () => {
    expect(skillFor().tools).toEqual([]);
    expect(() => new SessionSkill({ backend: 'cloud' })).toThrow('backend must be "local" or "robutler"');
  });
});
