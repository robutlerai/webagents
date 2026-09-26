/**
 * `webagents skills add` and `skills remove` (2026-09-25), the same in both
 * CLIs. The cases are `python/tests/fixtures/cli/skills_edit.json`, which the
 * Python suite runs too (`tests/cli/test_skills_edit.py`): the editor, byte for
 * byte, and the command's words, exit codes and choice of file.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { editSkillList, SkillListError, skillsCommand } from '../../../src/cli/skills-edit';

const HERE = path.dirname(fileURLToPath(import.meta.url));

interface EditCase {
  case: string;
  file: string;
  text: string;
  action: 'add' | 'remove';
  names: string[];
  error?: string;
  result?: string;
  changed?: boolean;
  added?: string[];
  already?: string[];
  removed?: string[];
  absent?: string[];
  skills?: string[];
}

interface CommandCase {
  case: string;
  files: Record<string, string>;
  args: string[];
  agent?: string;
  facts: { keys: string[]; signed_in: boolean };
  out: string[];
  err: string[];
  exit: number;
  after?: Record<string, string>;
}

const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/skills_edit.json'), 'utf8'),
) as { edits: EditCase[]; commands: CommandCase[] };

describe('the skills list editor', () => {
  it.each(FIXTURE.edits)('$case', (c) => {
    if (c.error) {
      let caught: unknown;
      try {
        editSkillList(c.text, c.action, c.names, c.file);
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(SkillListError);
      expect((caught as SkillListError).problem).toBe(c.error);
      return;
    }
    const edit = editSkillList(c.text, c.action, c.names, c.file);
    expect(edit.text).toBe(c.result);
    expect(edit.changed).toBe(c.changed);
    expect([edit.added, edit.already, edit.removed, edit.absent, edit.skills]).toEqual([
      c.added,
      c.already,
      c.removed,
      c.absent,
      c.skills,
    ]);
  });
});

describe('webagents skills add|remove', () => {
  let folder: string;
  let profile: string | undefined;

  beforeEach(() => {
    folder = mkdtempSync(path.join(tmpdir(), 'skills-edit-'));
    // The hints name commands as they must be typed, with `--profile` under one.
    profile = process.env.WEBAGENTS_PROFILE;
    delete process.env.WEBAGENTS_PROFILE;
  });

  afterEach(() => {
    rmSync(folder, { recursive: true, force: true });
    if (profile !== undefined) process.env.WEBAGENTS_PROFILE = profile;
  });

  it.each(FIXTURE.commands)('$case', async (c) => {
    for (const [name, text] of Object.entries(c.files)) writeFileSync(path.join(folder, name), text);
    const keys = new Set(c.facts.keys);
    const out: string[] = [];
    const err: string[] = [];

    const code = await skillsCommand(
      c.args[0] as 'add' | 'remove',
      c.args.slice(1),
      {
        agent: c.agent,
        folder,
        facts: async () => ({ hasKey: (variable) => keys.has(variable), signedIn: c.facts.signed_in }),
      },
      { out: (line) => out.push(line), err: (line) => err.push(line) },
    );

    expect(code).toBe(c.exit);
    expect(out.length ? out.join('\n').split('\n') : []).toEqual(c.out);
    expect(err.length ? err.join('\n').split('\n') : []).toEqual(c.err);
    for (const [name, text] of Object.entries(c.files)) {
      expect(readFileSync(path.join(folder, name), 'utf8'), name).toBe(c.after?.[name] ?? text);
    }
  });
});
