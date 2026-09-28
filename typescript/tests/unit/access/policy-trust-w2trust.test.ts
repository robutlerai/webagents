/**
 * Trust-gated `access:` groups (plan item 2.5, 2026-09-26): the decision over
 * the platform's evidence, from the table both SDKs run
 * (`python/tests/fixtures/access/policy.json`, `trust_cases`; Python
 * tests/access/test_policy_trust_w2trust.py). The platform is never dialled
 * here: `trust` IS what the access skill learned, or null when it could not.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  decide,
  meetsTrust,
  parseAccess,
  trustKey,
  trustRequirements,
  verifiedAgentOf,
  type TrustEvidence,
} from '../../../src/access/policy';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/access/policy.json'), 'utf8')) as {
  policies: Record<string, unknown>;
  trust_cases: Array<{
    name: string;
    policy?: string;
    principals: string[];
    tier?: string;
    trust?: TrustEvidence | null;
    expect: { allow: boolean; groups?: string[] };
  }>;
  trust_requirements: Record<string, Array<{ min: number; topic: string | null }>>;
};
const POLICIES = Object.fromEntries(Object.entries(TABLE.policies).map(([name, raw]) => [name, parseAccess(raw)]));

describe('the decision over trust evidence (shared table)', () => {
  for (const c of TABLE.trust_cases) {
    it(c.name, () => {
      const decision = decide(POLICIES[c.policy ?? 'open'], c.principals, c.tier, c.trust);
      expect(decision.allow).toBe(c.expect.allow);
      if (decision.allow) expect(decision.groups).toEqual(c.expect.groups);
    });
  }
});

describe('what a block asks the platform for', () => {
  for (const [name, expected] of Object.entries(TABLE.trust_requirements)) {
    it(`${name}: one requirement per topic key`, () => {
      expect(trustRequirements(POLICIES[name])).toEqual(expected);
    });
  }

  it('the overall score is filed under the empty key', () => {
    expect(trustKey(null)).toBe('');
    expect(trustKey(undefined)).toBe('');
    expect(trustKey('billing')).toBe('billing');
  });

  it('the verified agent is the first agent principal, whatever else the caller proved', () => {
    expect(verifiedAgentOf(['key:x', 'agent:https://a.example/x', 'agent:https://b.example/y'])).toBe('https://a.example/x');
    expect(verifiedAgentOf(['user:@alice', 'key:x'])).toBeNull();
  });

  it('meetsTrust fails closed on every missing piece', () => {
    const requirement = { min: 0.5, topic: null };
    const principals = ['agent:https://a.example/x'];
    expect(meetsTrust(requirement, principals, { agent: 'https://a.example/x', scores: { '': 0.5 } })).toBe(true);
    expect(meetsTrust(requirement, principals, { agent: 'https://a.example/x', scores: { '': 0.49 } })).toBe(false);
    expect(meetsTrust(requirement, principals, { agent: 'https://a.example/x', scores: { billing: 0.9 } })).toBe(false);
    expect(meetsTrust(requirement, principals, { agent: 'https://a.example/x', scores: { '': Number.NaN } })).toBe(false);
    expect(meetsTrust(requirement, principals, { agent: 'https://other.example/x', scores: { '': 1 } })).toBe(false);
    expect(meetsTrust(requirement, principals, null)).toBe(false);
    expect(meetsTrust(requirement, principals, undefined)).toBe(false);
    expect(meetsTrust(requirement, ['user:@alice'], { agent: 'https://a.example/x', scores: { '': 1 } })).toBe(false);
  });

  it('a block with no trust group asks for nothing', () => {
    expect(trustRequirements(parseAccess({}))).toEqual([]);
    expect(trustRequirements(parseAccess({ groups: { a: { members: ['user:@x'] } } }))).toEqual([]);
  });
});
