/**
 * The TrustFlow record in the A2A card (plan item 2.7, 2026-09-26): an
 * `a2a` entry with `trust_record: true` fetches this agent's signed record
 * from the platform, as the agent, and serves it as the card extension both
 * SDKs write; the card signature covers it, and a peer verifies both (Python
 * tests/trustflow/test_a2a_trust_record_w2trust.py).
 */

import { describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../src/core/agent';
import { AgentIdentity } from '../../../src/crypto/identity';
import { A2ATransportSkill } from '../../../src/skills/transport/a2a/skill';
import { verifyAgentCard } from '../../../src/skills/transport/a2a/card';
import { trustRecordFromCard, verifyTrustRecord } from '../../../src/trustflow/trust-record';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/trust/trustflow_record.json'), 'utf8'));
const PRINCIPAL = 'https://agents.example.com/agents/scout';
const ANSWER = { record: FIXTURE.records.valid, payload: FIXTURE.payload, jwks_url: `${FIXTURE.issuer}/.well-known/jwks.json` };

async function build(config: Record<string, unknown>, record: () => Promise<typeof ANSWER>) {
  const skill = new A2ATransportSkill(config);
  const agent = new BaseAgent({ name: 'scout', description: 'Scout', skills: [skill] });
  const identity = new AgentIdentity({ agentId: 'scout', issuer: PRINCIPAL });
  await identity.initialize();
  agent.identity = identity;
  const lookup = { record: vi.fn(record) };
  skill.trustLookup = lookup;
  return { skill, agent, identity, lookup };
}

describe('trust_record in the card', () => {
  it('carries the platform record as the extension, under the card signature', async () => {
    const { skill, agent, identity, lookup } = await build({ trust_record: true }, async () => ANSWER);
    const card = (await skill.buildCard(agent)) as Record<string, unknown> & { capabilities: { extensions: unknown[] }; signatures: unknown[] };
    expect(card.capabilities.extensions).toEqual(FIXTURE.extension.card_with_record.capabilities.extensions);
    expect(lookup.record).toHaveBeenCalledWith(PRINCIPAL);
    const cardCheck = await verifyAgentCard(card, { keys: identity.getJwks().keys as JsonWebKey[] });
    expect(cardCheck.ok).toBe(true);
    const record = trustRecordFromCard(card)!;
    const recordCheck = await verifyTrustRecord(record, { keys: FIXTURE.jwks.keys, issuer: FIXTURE.issuer, now: FIXTURE.now, subject: { url: PRINCIPAL } });
    expect(recordCheck.ok).toBe(true);
    // A tampered extension breaks the card signature.
    const tampered = { ...card, capabilities: { ...card.capabilities, extensions: [{ ...(card.capabilities.extensions[0] as object), params: { record: FIXTURE.records.tampered } }] } };
    expect((await verifyAgentCard(tampered, { keys: identity.getJwks().keys as JsonWebKey[] })).ok).toBe(false);
  });

  it('holds the record: one platform call for many cards', async () => {
    // The clock is pinned to the fixture's `now` (2026-09-21 14:15Z). The skill holds
    // a record until an hour before its `exp`, and the fixture's valid record
    // expires 2026-09-28 14:13:20Z. Against the real clock this test began
    // failing at 13:13:20Z that day, with two platform calls instead of one:
    // a date trap, not a regression. Only `Date` is faked; key generation and
    // timers stay real.
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(FIXTURE.now * 1000);
    try {
      const { skill, agent, lookup } = await build({ trust_record: true }, async () => ANSWER);
      await skill.buildCard(agent);
      await skill.buildCard(agent);
      expect(lookup.record).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('serves the card without the record when the platform cannot answer, and does not retry per request', async () => {
    const { skill, agent, lookup } = await build({ trust_record: true }, async () => { throw new Error('unreachable'); });
    const card = (await skill.buildCard(agent)) as { capabilities: { extensions?: unknown[] } };
    expect(card.capabilities.extensions).toBeUndefined();
    await skill.buildCard(agent);
    expect(lookup.record).toHaveBeenCalledTimes(1);
  });

  it('asks the platform nothing unless the entry says trust_record: true', async () => {
    const { skill, agent, lookup } = await build({}, async () => ANSWER);
    const card = (await skill.buildCard(agent)) as { capabilities: { extensions?: unknown[] } };
    expect(card.capabilities.extensions).toBeUndefined();
    expect(lookup.record).not.toHaveBeenCalled();
    expect(skill.settings.trust_record).toBe(false);
  });
});
