/**
 * Trust-gated groups end to end through the TypeScript server (plan item 2.5,
 * 2026-09-26); the same cases as Python's
 * tests/access/test_access_skill_trust_w2trust.py.
 *
 * The platform is a stub on the skill (`trust`): the access skill must ask it
 * about the VERIFIED calling agent (the one the Web Bot Auth signature
 * proved), on the topic the block names, and place the caller by the answer.
 * When the stub throws, the platform is "unreachable" and the trust-gated
 * group is not joined: the caller keeps its other groups or the default, and
 * `default: none` refuses it.
 */

import { describe, it, expect, beforeAll, vi } from 'vitest';
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
import type { TrustScoreSource } from '../../../src/skills/access/skill';
import { parseKeySet, type Discovery, type KeySetOutcome } from '../../../src/crypto/web-bot-auth-verify';
import { signMessage } from '../../../src/crypto/http-signature';
import { AgentIdentity } from '../../../src/crypto/identity';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();

const FRIEND = 'https://bot.acme.com/agents/scout';
const STRANGER = 'https://stranger.example/agents/x';

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
  for (const issuer of [FRIEND, STRANGER]) {
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

/** A platform that answers per agent and topic, records what it was asked, or throws. */
class StubPlatform implements TrustScoreSource {
  readonly asked: Array<[string, string | undefined]> = [];
  constructor(private readonly answers: Record<string, number | null> | 'unreachable') {}
  async lookup(agent: string, topic?: string) {
    this.asked.push([agent, topic]);
    if (this.answers === 'unreachable') throw new Error('ECONNREFUSED');
    const score = this.answers[`${agent}|${topic ?? ''}`];
    if (score === undefined) throw new Error('not found');
    return topic ? { score: 0.1, topic: { query: topic, score } } : { score: score ?? 0 };
  }
}

const ACCESS = {
  groups: {
    friends: ['agent:https://*.acme.com/**'],
    billing: { trust: { min: 0.6, topic: 'billing' } },
  },
  tools: { billing: ['rest'] },
};

function build(access: Record<string, unknown>, platform: TrustScoreSource) {
  const dir = tempDir('access-trust-');
  const rest = new RestSkill({});
  const llm = new RecordingLLM();
  const { skill, policy } = accessSkillFor(access, path.join(dir, 'AGENT.md'));
  skill.publicUrl = 'https://agent.example';
  skill.keySets = new StubKeySets();
  skill.trust = platform;
  const agent = new BaseAgent({ name: 'mini', instructions: 'Mini.', skills: [rest, llm, skill] });
  applyAccessTools(policy, new Map([['rest', rest]]));
  return { handler: createFetchHandler(agent as never, { basePath: '/mini' }), llm };
}

const BODY = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] });

async function signed(issuer: string, body: string): Promise<Record<string, string>> {
  const out = await signMessage(identities.get(issuer)!, { method: 'POST', url: 'https://agent.example/mini/chat/completions', body: new TextEncoder().encode(body) });
  return { host: 'agent.example', 'content-type': 'application/json', ...(out.headers as unknown as Record<string, string>) };
}

function post(handler: (r: Request) => Promise<Response>, headers: Record<string, string>, body: string): Promise<Response> {
  return handler(new Request('https://agent.example/mini/chat/completions', { method: 'POST', headers, body }));
}

describe('trust-gated groups through the server', () => {
  it('a verified agent the platform scores above the threshold joins the group and gets its tools', async () => {
    const platform = new StubPlatform({ [`${FRIEND}|billing`]: 0.7 });
    const { handler, llm } = build(ACCESS, platform);
    const res = await post(handler, await signed(FRIEND, BODY), BODY);
    expect(res.status, await res.clone().text()).toBe(200);
    const seen = llm.seen.at(-1)!;
    expect(seen.tools).toContain('rest_request');
    expect(seen.system).toContain('Groups: friends, billing.');
    // Asked about the agent the SIGNATURE proved, on the block's topic, and nothing else.
    expect(platform.asked).toEqual([[FRIEND, 'billing']]);
  });

  it('below the threshold the caller keeps only its identity groups', async () => {
    const platform = new StubPlatform({ [`${FRIEND}|billing`]: 0.59 });
    const { handler, llm } = build(ACCESS, platform);
    expect((await post(handler, await signed(FRIEND, BODY), BODY)).status).toBe(200);
    const seen = llm.seen.at(-1)!;
    expect(seen.tools).not.toContain('rest_request');
    expect(seen.system).toContain('Groups: friends.');
  });

  it('the platform unreachable fails closed: no trust group, the rest untouched', async () => {
    const platform = new StubPlatform('unreachable');
    const { handler, llm } = build(ACCESS, platform);
    expect((await post(handler, await signed(FRIEND, BODY), BODY)).status).toBe(200);
    expect(llm.seen.at(-1)!.tools).not.toContain('rest_request');
    expect(llm.seen.at(-1)!.system).toContain('Groups: friends.');
    expect(platform.asked).toEqual([[FRIEND, 'billing']]);
  });

  it('a topic the platform could not score (null) does not admit', async () => {
    const platform = new StubPlatform({ [`${FRIEND}|billing`]: null });
    const { handler, llm } = build(ACCESS, platform);
    expect((await post(handler, await signed(FRIEND, BODY), BODY)).status).toBe(200);
    expect(llm.seen.at(-1)!.system).toContain('Groups: friends.');
  });

  it('with default none and only a trust group, an unreachable platform refuses the caller', async () => {
    const closed = { groups: { billing: { trust: { min: 0.6, topic: 'billing' } } }, default: 'none' };
    const refused = build(closed, new StubPlatform('unreachable'));
    expect((await post(refused.handler, await signed(STRANGER, BODY), BODY)).status).toBe(403);
    expect(refused.llm.seen).toEqual([]);
    const admitted = build(closed, new StubPlatform({ [`${STRANGER}|billing`]: 0.8 }));
    expect((await post(admitted.handler, await signed(STRANGER, BODY), BODY)).status).toBe(200);
    expect(admitted.llm.seen.at(-1)!.system).toContain('Groups: billing.');
  });

  it('a caller with no verified agent is never looked up', async () => {
    const platform = new StubPlatform({ [`${FRIEND}|billing`]: 0.9 });
    const { handler, llm } = build(ACCESS, platform);
    // A bearer this agent cannot verify names no one (the existing access suite's anonymous shape).
    expect((await post(handler, { authorization: 'Bearer made-up', 'content-type': 'application/json' }, BODY)).status).toBe(200);
    expect(llm.seen.at(-1)!.system).toContain('Groups: everyone.');
    expect(platform.asked).toEqual([]);
  });

  it('a block without trust groups asks the platform nothing', async () => {
    const platform = new StubPlatform({ [`${FRIEND}|billing`]: 0.9 });
    const { handler } = build({ groups: { friends: ['agent:https://*.acme.com/**'] } }, platform);
    expect((await post(handler, await signed(FRIEND, BODY), BODY)).status).toBe(200);
    expect(platform.asked).toEqual([]);
  });

  it('the lookup is made once per topic key, not once per group', async () => {
    const platform = new StubPlatform({ [`${FRIEND}|billing`]: 0.9, [`${FRIEND}|`]: 0.5 });
    const spy = vi.spyOn(platform, 'lookup');
    const access = {
      groups: {
        a: { trust: { min: 0.1, topic: 'billing' } },
        b: { trust: { min: 0.2, topic: 'billing' } },
        c: { members: ['agent:https://*.acme.com/**'], trust: { min: 0.4 } },
      },
    };
    const { handler, llm } = build(access, platform);
    expect((await post(handler, await signed(FRIEND, BODY), BODY)).status).toBe(200);
    expect(spy).toHaveBeenCalledTimes(2);
    expect(llm.seen.at(-1)!.system).toContain('Groups: a, b, c.');
  });
});
