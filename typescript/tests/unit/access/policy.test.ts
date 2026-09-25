/**
 * The access block's decision and its refusals of a malformed block (ADR-0045),
 * from the table both SDKs run (`python/tests/fixtures/access/policy.json`;
 * Python tests/access/test_policy.py).
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AccessConfigError, decide, parseAccess } from '../../../src/access/policy';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/access/policy.json'), 'utf8')) as {
  policies: Record<string, unknown>;
  cases: Array<{ name: string; policy?: string; principals: string[]; tier?: string; expect: { allow: boolean; groups?: string[] } }>;
  bad: Array<{ access: unknown; error: string }>;
};
const POLICIES = Object.fromEntries(Object.entries(TABLE.policies).map(([name, raw]) => [name, parseAccess(raw)]));

describe('the decision', () => {
  for (const c of TABLE.cases) {
    it(c.name, () => {
      const decision = decide(POLICIES[c.policy ?? 'open'], c.principals, c.tier);
      expect(decision.allow).toBe(c.expect.allow);
      if (decision.allow) expect(decision.groups).toEqual(c.expect.groups);
    });
  }
});

describe('a malformed block', () => {
  for (const c of TABLE.bad) {
    it(c.error.slice(0, 60), () => {
      expect(() => parseAccess(c.access)).toThrow(AccessConfigError);
      expect(() => parseAccess(c.access)).toThrow(c.error);
    });
  }

  it('an empty block lets everyone in as everyone', () => {
    expect(decide(parseAccess({}), []).groups).toEqual(['everyone']);
  });
});
