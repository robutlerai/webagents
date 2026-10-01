/**
 * The memory skill against the fixture both SDKs run
 * (`python/tests/fixtures/memory_tool/definition.json`; Python:
 * `python/tests/agents/skills/test_memory_w2mem.py`): the four tool
 * definitions, what a `- memory: {...}` entry accepts, the namespace a caller
 * gets, where the local tier keeps it, the id an entry has everywhere, the
 * token estimate and the transcript (the agent's compaction counts the same
 * way, `core/context-compaction.ts`), and the frozen notes.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../../src/core/agent';
import { MemorySkill, parseMemoryConfig } from '../../../../src/skills/memory/skill';
import { estimateTokens, transcriptOf, type CompactMessage as CompactableMessage } from '../../../../src/core/context-compaction';
import { callerKey, entryIdFor, isValidKey, keyRefusal, localDirOf, namespaceOf, readableNamespaces, writableNamespaces } from '../../../../src/skills/memory/namespace';
import { renderNotes } from '../../../../src/skills/memory/notes';
import { resolveSkillsByName } from '../../../../src/skills/resolve';
import { tempDirs } from '../../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/memory_tool/definition.json'), 'utf8'),
);
const tempDir = tempDirs();

interface NamespaceCase {
  case: string;
  tier: string | null;
  principals: string[] | null;
  user_id: string | null;
  namespace: string | null;
}

/** The caller in this SDK's terms: no `principals` when no access block ran. */
function authOf(c: NamespaceCase): Record<string, unknown> {
  const auth: Record<string, unknown> = { authenticated: true, provider: 'platform', scopes: c.tier ? [c.tier] : [], user_id: c.user_id ?? undefined };
  if (c.principals !== null) auth.principals = c.principals;
  if (!c.tier && !c.principals && !c.user_id) auth.authenticated = false;
  return auth;
}

describe('the memory skill (shared fixture)', () => {
  it('offers the four definitions both SDKs share', () => {
    const agent = new BaseAgent({ name: 'm', instructions: 'x', skills: [new MemorySkill({ agentDir: tempDir('mem-defs-') })] });
    const defs = agent.getToolDefinitions().filter((d) => d.function.name.startsWith('memory_'));
    expect(defs).toEqual(FIXTURE.definitions);
  });

  it('checks the entry as the Python loader does', () => {
    for (const shape of FIXTURE.config_shapes as Array<Record<string, unknown>>) {
      if (shape.error) {
        expect(() => parseMemoryConfig(shape.config as Record<string, unknown>), String(shape.case)).toThrow(String(shape.error));
      } else {
        const parsed = parseMemoryConfig(shape.config as Record<string, unknown>);
        expect({ local: parsed.local, portal: parsed.portal }, String(shape.case)).toEqual({ local: shape.local, portal: shape.portal });
      }
    }
    const defaults = parseMemoryConfig({});
    expect(defaults.notesBudget).toBe(FIXTURE.defaults.notes_budget);
    expect(defaults.compaction).toEqual(FIXTURE.defaults.compaction);
  });

  it('is named `memory` in an agent file, and a wrong entry is a skill that failed with the sentence', async () => {
    const dir = tempDir('mem-resolve-');
    const { skills, unknown, failed } = await resolveSkillsByName(['memory'], { agentDir: dir });
    expect(unknown).toEqual([]);
    expect(failed).toEqual([]);
    const skill = skills[0] as unknown as MemorySkill;
    expect(skill.tiers).toEqual({ local: true, portal: false });
    expect(skill.memoryRoot).toBe(path.join(dir, '.webagents', 'memory'));

    const wrong = await resolveSkillsByName([{ memory: { portal: true, cloud: true } }], { agentDir: dir });
    expect(wrong.failed).toEqual([{ name: 'memory', reason: 'memory: unknown key "cloud". It takes local, portal, notes_budget and compaction.' }]);
  });

  it('derives the namespace from the verified caller, and nothing else', () => {
    for (const c of FIXTURE.namespaces as NamespaceCase[]) {
      expect(namespaceOf(authOf(c) as never), c.case).toBe(c.namespace);
    }
    expect(namespaceOf(undefined)).toBeNull();
  });

  it('reads and writes only what the namespace allows', () => {
    for (const v of FIXTURE.visibility as Array<{ namespace: string | null; reads: string[] | '*'; writes: string[] }>) {
      const reads = readableNamespaces(v.namespace);
      expect(reads === null ? '*' : reads).toEqual(v.reads);
      expect(writableNamespaces(v.namespace)).toEqual(v.writes);
    }
  });

  it('keeps each namespace where the fixture says, hashing a caller as the session skill does', () => {
    for (const c of FIXTURE.local_dirs as Array<{ namespace: string; dir: string }>) {
      expect(localDirOf(c.namespace), c.namespace).toBe(c.dir);
    }
    expect(callerKey('user:4c3ad5e6-510d-486d-af50-426ba16b6b6c')).toBe('9691dcadd7ab4f5c3c4e8120264e66e4');
  });

  it('gives an entry the same id in every tier', () => {
    for (const c of FIXTURE.entry_ids.cases as Array<{ namespace: string; key: string; id: string }>) {
      expect(entryIdFor(FIXTURE.entry_ids.store, c.namespace, c.key)).toBe(c.id);
    }
  });

  it('accepts a slug as a key and refuses the rest with the sentence', () => {
    for (const key of FIXTURE.keys.valid as string[]) expect(isValidKey(key), key).toBe(true);
    for (const key of FIXTURE.keys.invalid as string[]) {
      expect(isValidKey(key), key).toBe(false);
      expect(keyRefusal(key)).toBe(String(FIXTURE.keys.refused).replace('{key}', key));
    }
  });

  it('estimates tokens as the fixture counts them', () => {
    for (const c of FIXTURE.tokens.cases as Array<{ messages: CompactableMessage[]; tokens: number }>) {
      expect(estimateTokens(c.messages)).toBe(c.tokens);
    }
  });

  it('renders the older turns for the summarizing model one line each', () => {
    for (const c of FIXTURE.transcript.cases as Array<{ messages: CompactableMessage[]; text: string }>) {
      expect(transcriptOf(c.messages)).toBe(c.text);
    }
  });

  it('renders the frozen notes within the budget, counting the rest', () => {
    for (const c of FIXTURE.notes.cases as Array<{ case: string; budget: number; sections: never; text: string }>) {
      expect(renderNotes(c.sections, c.budget), c.case).toBe(c.text);
    }
  });
});
