/**
 * The access block, end to end through the TypeScript server (ADR-0045); the
 * same cases as Python's tests/access/test_access_skill.py.
 *
 * A caller is placed by what it can PROVE: a Web Bot Auth signature this agent
 * verifies (the key set here is a stub), or the tier something trusted already
 * set (the local chat's owner). The model sees only the tools and prompts its
 * group may use, and a signature that does not verify is a 401, never
 * "anonymous".
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { writeFileSync } from 'node:fs';
import path from 'node:path';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';
import { RestSkill } from '../../../src/skills/rest/skill';
import { accessSkillFor, applyAccessTools } from '../../../src/access/install';
import { parseKeySet, type Discovery, type KeySetOutcome } from '../../../src/crypto/web-bot-auth-verify';
import { signMessage } from '../../../src/crypto/http-signature';
import { AgentIdentity } from '../../../src/crypto/identity';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const FRIEND = 'https://bot.acme.com/agents/scout';
const STRANGER = 'https://stranger.example/agents/x';
const SPAMMER = 'https://spam.example/agents/x';

class RecordingLLM extends Skill {
  readonly seen: Array<{ system: string; tools: string[] }> = [];

  @handoff({ name: 'recording-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const messages = (ctx.get('_agentic_messages') as Array<{ role: string; content?: unknown }>) ?? [];
    const system = messages.filter((m) => m.role === 'system').map((m) => String(m.content ?? '')).join('\n');
    const tools = ((ctx.get('_agentic_tools') as Array<{ function?: { name?: string }; name?: string }>) ?? [])
      .map((t) => t.function?.name ?? t.name ?? '')
      .sort();
    this.seen.push({ system, tools });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

const identities = new Map<string, AgentIdentity>();
beforeAll(async () => {
  for (const issuer of [FRIEND, STRANGER, SPAMMER]) {
    const identity = new AgentIdentity({ agentId: 'x', issuer });
    await identity.initialize();
    identities.set(issuer, identity);
  }
});

class StubKeySets {
  async get(discovery: Discovery): Promise<KeySetOutcome> {
    const identity = identities.get(discovery.principal);
    const parsed = await parseKeySet(identity ? identity.getJwks() : { keys: [] }, { wellKnownDirectory: false });
    return parsed.ok ? { ok: true, keys: parsed.keys, ttlS: 300 } : { ok: false, code: 'key_set_invalid', reason: parsed.reason };
  }
}

const ACCESS = {
  deny: ['agent:https://spam.example/**'],
  groups: { friends: ['agent:https://*.acme.com/**'] },
  instructions: { friends: 'FRIENDS.md' },
  tools: { friends: ['rest'] },
};

function build(access: Record<string, unknown>): { handler: (r: Request) => Promise<Response>; llm: RecordingLLM; agent: BaseAgent } {
  const dir = tempDir('access-');
  writeFileSync(path.join(dir, 'FRIENDS.md'), 'FRIENDS-ONLY GUIDANCE');
  const rest = new RestSkill({});
  const llm = new RecordingLLM();
  const { skill, policy } = accessSkillFor(access, path.join(dir, 'AGENT.md'));
  skill.publicUrl = 'https://agent.example';
  skill.keySets = new StubKeySets();
  const agent = new BaseAgent({ name: 'mini', instructions: 'Mini.', skills: [rest, llm, skill] });
  applyAccessTools(policy, new Map([['rest', rest]]));
  return { handler: createFetchHandler(agent as never, { basePath: '/mini' }), llm, agent };
}

const BODY = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] });

async function signed(issuer: string, body: string, host = 'agent.example'): Promise<Record<string, string>> {
  const bytes = new TextEncoder().encode(body);
  const out = await signMessage(identities.get(issuer)!, { method: 'POST', url: `https://${host}/mini/chat/completions`, body: bytes });
  return { host, 'content-type': 'application/json', ...(out.headers as unknown as Record<string, string>) };
}

function post(handler: (r: Request) => Promise<Response>, headers: Record<string, string>, body: string): Promise<Response> {
  return handler(new Request('https://agent.example/mini/chat/completions', { method: 'POST', headers, body }));
}

describe('the access block through the server', () => {
  it('a friend gets the friends tools and instructions', async () => {
    const { handler, llm } = build(ACCESS);
    const res = await post(handler, await signed(FRIEND, BODY), BODY);
    expect(res.status, await res.clone().text()).toBe(200);
    const seen = llm.seen.at(-1)!;
    expect(seen.tools).toContain('rest_request');
    expect(seen.system).toContain('FRIENDS-ONLY GUIDANCE');
    expect(seen.system).toContain(
      `This turn is from the agent ${FRIEND}, which proved it with a Web Bot Auth signature. Groups: friends.`,
    );
    expect(seen.system).toContain('## Calling web APIs');
  });

  it('a stranger gets the default group and nothing granted', async () => {
    const { handler, llm } = build(ACCESS);
    const res = await post(handler, await signed(STRANGER, BODY), BODY);
    expect(res.status).toBe(200);
    const seen = llm.seen.at(-1)!;
    expect(seen.tools).not.toContain('rest_request');
    expect(seen.system).not.toContain('FRIENDS-ONLY GUIDANCE');
    expect(seen.system).not.toContain('## Calling web APIs');
    expect(seen.system).toContain('Groups: everyone.');
  });

  it('deny is a 403', async () => {
    const { handler, llm } = build(ACCESS);
    const res = await post(handler, await signed(SPAMMER, BODY), BODY);
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({ error: { code: 'forbidden', message: 'This agent does not accept requests from this caller.' } });
    expect(llm.seen).toEqual([]);
  });

  it('default none keeps out a caller in no group, and a bearer it cannot verify', async () => {
    const { handler } = build({ ...ACCESS, default: 'none' });
    expect((await post(handler, await signed(STRANGER, BODY), BODY)).status).toBe(403);
    expect((await post(handler, { authorization: 'Bearer made-up', 'content-type': 'application/json' }, BODY)).status).toBe(403);
    expect((await post(handler, await signed(FRIEND, BODY), BODY)).status).toBe(200);
  });

  it('a signature that does not verify is a 401, not anonymous', async () => {
    const { handler, llm } = build(ACCESS);
    const headers = await signed(FRIEND, BODY);
    const other = JSON.stringify({ messages: [{ role: 'user', content: 'something else' }] });
    const res = await post(handler, headers, other);
    expect(res.status).toBe(401);
    expect(((await res.json()) as { error: { code: string } }).error.code).toBe('content_digest_mismatch');
    expect(llm.seen).toEqual([]);
  });

  it('a signature for another host is a 401', async () => {
    const { handler } = build(ACCESS);
    const res = await post(handler, await signed(FRIEND, BODY, 'other.example'), BODY);
    expect(res.status).toBe(401);
    expect(((await res.json()) as { error: { code: string } }).error.code).toBe('signature_authority_mismatch');
  });

  it('the local owner gets everything', async () => {
    const { agent, llm } = build(ACCESS);
    await agent.run([{ role: 'user', content: 'hi' }], { auth: { authenticated: true, scope: 'owner', provider: 'local' } });
    const seen = llm.seen.at(-1)!;
    expect(seen.tools).toContain('rest_request');
    expect(seen.system).toContain('FRIENDS-ONLY GUIDANCE');
    expect(seen.system).toContain("This turn is from this agent's owner.");
  });
});

describe('the loader', () => {
  it('an unknown tool name is refused with the Python sentence', () => {
    const { policy } = accessSkillFor({ groups: { friends: [] }, tools: { friends: ['nope'] } }, undefined);
    expect(() => applyAccessTools(policy, new Map([['rest', new RestSkill({})]]))).toThrow(
      'access.tools.friends: "nope" is not a skill in this agent file or one of its tools.',
    );
  });

  it('a missing instructions file is refused with the Python sentence', () => {
    const dir = tempDir('access-');
    expect(() => accessSkillFor({ instructions: { everyone: 'GONE.md' } }, path.join(dir, 'AGENT.md'))).toThrow(
      'access.instructions.everyone: GONE.md was not found next to the agent file.',
    );
  });
});

describe('service token principals (S-240)', () => {
  // A platform service token names its sender to the block only through the
  // platform's signed claim, and only when the token is addressed to this
  // agent's own URL. The body's sender names no one. Python:
  // tests/access/test_access_skill.py::TestServiceTokenPrincipals.
  async function groupsOf(auth: Record<string, unknown>): Promise<string[]> {
    const { ContextImpl } = await import('../../../src/core/context');
    const { skill } = accessSkillFor({ groups: { family: ['user:user-alice'] } }, undefined);
    const context = new ContextImpl();
    context.setAuth({ authenticated: true, scope: 'user' as never, scopes: ['platform'], provider: 'service_token', ...auth });
    await (skill as unknown as { admit(d: unknown, c: unknown): Promise<void> }).admit({}, context);
    return (context.auth.scopes ?? []).filter((s) => s.startsWith('group:'));
  }

  it('places the signed sender of a token addressed here', async () => {
    const claims = { sub: 'service:robutler-router', sender: { id: 'user-alice', username: 'alice' } };
    expect(await groupsOf({ claims, audienceVerified: true })).toEqual(['group:family']);
  });

  it('places no one from a token not addressed here', async () => {
    const claims = { sub: 'service:robutler-router', sender: { id: 'user-alice' } };
    expect(await groupsOf({ claims, audienceVerified: false })).toEqual(['group:everyone']);
  });

  it('places no one from the body sender', async () => {
    const claims = { sub: 'service:robutler-router' };
    expect(await groupsOf({ claims, audienceVerified: true, user_id: 'user-alice' })).toEqual(['group:everyone']);
  });
});
