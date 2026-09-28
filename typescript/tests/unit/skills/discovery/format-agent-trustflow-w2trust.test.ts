/**
 * Discovery results carry the TrustFlow score (plan item 2.5, 2026-09-26):
 * `formatAgent` adds `trustflow` and `trustflow_for_query` from the platform's
 * row, as the shared cases pin them
 * (`python/tests/fixtures/trust/trust_tool_definition.json`,
 * `discovery_agent_fields`; Python tests/agents/skills/test_format_agent_trustflow_w2trust.py).
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SEARCH_DESCRIPTION, formatAgent } from '../../../../src/skills/discovery/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/trust/trust_tool_definition.json'), 'utf8'));

describe('an agent result carries its TrustFlow score', () => {
  for (const c of FIXTURE.discovery_agent_fields.cases as Array<{ name: string; raw: Record<string, unknown>; expect: Record<string, unknown> }>) {
    it(c.name, () => {
      expect(formatAgent(c.raw)).toEqual(c.expect);
    });
  }

  it('the search description tells the model what the score is', () => {
    expect(SEARCH_DESCRIPTION).toContain('TrustFlow score (trustflow, 0 to 1, computed by the platform');
  });
});
