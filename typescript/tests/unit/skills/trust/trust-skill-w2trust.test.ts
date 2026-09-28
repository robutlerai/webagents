/**
 * The trust skill (plan items 2.5 and 2.7, 2026-09-26): the `trust` tool both
 * SDKs offer (`python/tests/fixtures/trust/trust_tool_definition.json`; Python
 * tests/trustflow/test_trust_skill_w2trust.py), what it answers, how it
 * fails, and that an agent file can name it.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../../src/core/agent';
import { TrustSkill } from '../../../../src/skills/trust/trust-skill';
import { resolvableSkillNames, resolveSkillsByName } from '../../../../src/skills/resolve';
import { AgentIdentity } from '../../../../src/crypto/identity';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/trust/trust_tool_definition.json'), 'utf8'));
const RECORD_FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/trust/trustflow_record.json'), 'utf8'));
const PORTAL = 'https://portal.test';

function response(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

const originalFetch = globalThis.fetch;
const savedEnv: Record<string, string | undefined> = {};
beforeEach(() => {
  for (const name of ['WEBAGENTS_API_KEY', 'WEBAGENTS_AGENT_TOKEN', 'ROBUTLER_API_URL', 'ROBUTLER_INTERNAL_API_URL']) {
    savedEnv[name] = process.env[name];
    delete process.env[name];
  }
  globalThis.fetch = vi.fn(async () => response(404, {})) as typeof fetch;
});
afterEach(() => {
  for (const [name, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  globalThis.fetch = originalFetch;
});

describe('the trust tool both SDKs share', () => {
  it('offers the definition in the shared fixture', () => {
    const agent = new BaseAgent({ name: 't', instructions: 'x', skills: [new TrustSkill({ portalUrl: PORTAL, apiKey: 'k' }) as never] });
    const def = agent.getToolDefinitions().find((d) => d.function.name === 'trust');
    expect(def).toEqual(FIXTURE.definition);
  });

  it('answers the platform record unchanged', async () => {
    globalThis.fetch = vi.fn(async () => response(200, FIXTURE.lookup_response)) as typeof fetch;
    const skill = new TrustSkill({ portalUrl: PORTAL, apiKey: 'k' });
    expect(await skill.trust({ agent: '@scout', topic: 'billing' })).toEqual(FIXTURE.lookup_response);
    const url = (globalThis.fetch as unknown as { mock: { calls: [string][] } }).mock.calls[0][0];
    expect(url).toBe(`${PORTAL}/api/trust/lookup?agent=%40scout&topic=billing`);
  });

  it('says the fixture sentence when the platform has no answer', async () => {
    globalThis.fetch = vi.fn(async () => response(404, { error: 'Agent not found' })) as typeof fetch;
    const skill = new TrustSkill({ portalUrl: PORTAL, apiKey: 'k' });
    expect(await skill.trust({ agent: '@nobody' })).toEqual({ error: FIXTURE.messages.not_found.replace('{agent}', '@nobody') });
    globalThis.fetch = vi.fn(async () => { throw new TypeError('fetch failed'); }) as typeof fetch;
    expect(await skill.trust({ agent: '@nobody' })).toEqual({ error: FIXTURE.messages.unreachable.replace('{agent}', '@nobody') });
  });

  it('with neither identity nor key it says so, and dials nothing', async () => {
    const skill = new TrustSkill({ portalUrl: PORTAL });
    const agent = new BaseAgent({ name: 'lonely', instructions: 'x', skills: [skill as never] });
    await agent.initialize();
    expect(skill.credential().kind).toBe('none');
    expect(await skill.trust({ agent: '@scout' })).toEqual({ error: FIXTURE.no_credential });
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it('needs an agent to look up', async () => {
    const skill = new TrustSkill({ portalUrl: PORTAL, apiKey: 'k' });
    expect(await skill.trust({ agent: '  ' })).toEqual({ error: 'agent is required: the agent URL, @username or platform id.' });
  });

  it("uses the agent's identity to sign when serve() attached one", async () => {
    const identity = new AgentIdentity({ agentId: 'scout', issuer: 'https://agents.example.com/agents/scout' });
    await identity.initialize();
    const skill = new TrustSkill({ portalUrl: PORTAL });
    const agent = new BaseAgent({ name: 'scout', instructions: 'x', skills: [skill as never] });
    agent.identity = identity;
    expect(skill.credential()).toEqual({ kind: 'signature', identity });
  });
});

describe('the own record', () => {
  it('fetches this agent record by its identity URL, and verifies a record from anyone', async () => {
    const identity = new AgentIdentity({ agentId: 'scout', issuer: 'https://agents.example.com/agents/scout' });
    await identity.initialize();
    const answer = { record: RECORD_FIXTURE.records.valid, payload: RECORD_FIXTURE.payload, jwks_url: `${PORTAL}/.well-known/jwks.json` };
    const seen: string[] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      seen.push(input instanceof Request ? input.url : String(input));
      return response(200, answer);
    }) as typeof fetch;
    const skill = new TrustSkill({ portalUrl: PORTAL, identity });
    expect(await skill.record()).toEqual(answer);
    expect(seen[0]).toBe(`${PORTAL}/api/trust/record?agent=${encodeURIComponent('https://agents.example.com/agents/scout')}`);
    const verified = await skill.verify(answer.record, { keys: RECORD_FIXTURE.jwks.keys, issuer: RECORD_FIXTURE.issuer, now: RECORD_FIXTURE.now });
    expect(verified.ok).toBe(true);
  });

  it('without an identity there is no own record to fetch', async () => {
    const skill = new TrustSkill({ portalUrl: PORTAL, apiKey: 'k' });
    await expect(skill.record()).rejects.toMatchObject({ code: 'no_credential' });
  });
});

describe('the agent file entry', () => {
  it('`- trust` resolves to the skill', async () => {
    expect(resolvableSkillNames()).toContain('trust');
    const resolved = await resolveSkillsByName([{ trust: { portalUrl: PORTAL, apiKey: 'k' } }]);
    expect(resolved.unknown).toEqual([]);
    expect(resolved.failed).toEqual([]);
    expect(resolved.byName.get('trust')).toBeInstanceOf(TrustSkill);
  });
});
