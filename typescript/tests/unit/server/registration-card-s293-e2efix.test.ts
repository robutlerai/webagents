/**
 * S-293 (2026-09-26): the registration card carries the agent's
 * `description:` and never its instructions. The TypeScript card always did;
 * the Python card put the instructions first and its listings carried the
 * whole system prompt to any caller who could reach the port. Both suites
 * read `python/tests/fixtures/registration/card.json` (Python:
 * `tests/server/test_registration_card_s293_e2efix.py`), so the two cards
 * cannot drift apart again on this.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildAgentCard } from '../../../src/server/card';
import type { IAgent } from '../../../src/core/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/registration/card.json'), 'utf8')) as {
  agent: { name: string; description: string; instructions: string };
  principal: string;
  card: Record<string, unknown>;
  never_carries: string;
};

function agentLike(description: string | undefined): IAgent {
  return {
    name: FIXTURE.agent.name,
    description,
    instructions: FIXTURE.agent.instructions,
    getToolDefinitions: () => [],
  } as unknown as IAgent;
}

describe('S-293: the registration card never carries the instructions', () => {
  it('carries the description and the self-naming fields the fixture pins', () => {
    const card = buildAgentCard(agentLike(FIXTURE.agent.description), { principal: FIXTURE.principal, signs: true });
    for (const [key, expected] of Object.entries(FIXTURE.card)) {
      expect(card[key as keyof typeof card], key).toEqual(expected);
    }
    expect(JSON.stringify(card)).not.toContain(FIXTURE.never_carries);
  });

  it('an agent with no description gets none, not its instructions', () => {
    const card = buildAgentCard(agentLike(undefined), { principal: FIXTURE.principal, signs: true });
    expect(card.description).toBeUndefined();
    expect(JSON.stringify(card)).not.toContain(FIXTURE.never_carries);
  });
});
