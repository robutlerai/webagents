/**
 * `webagents skills add` and `skills remove` (2026-09-25), the same in both
 * CLIs. The cases are `python/tests/fixtures/cli/skills_edit.json`, which the
 * Python suite runs too (`tests/cli/test_skills_edit.py`): the editor, byte for
 * byte, and the command's words, exit codes and choice of file.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { editSkillList, planSkills, SkillListError, skillsCommand } from '../../../src/cli/skills-edit';

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

type FileSpec = string | { link: string };

interface CommandCase {
  case: string;
  files: Record<string, FileSpec>;
  args: string[];
  agent?: string;
  facts: { keys: string[]; signed_in: boolean };
  out: string[];
  err: string[];
  exit: number;
  after?: Record<string, string>;
}

interface PlanCase {
  case: string;
  files: Record<string, FileSpec>;
  args: string[];
  plan: {
    file: string | null;
    errors: string[];
    add: string[];
    remove: string[];
    skills_after: string[];
    installed_removals: string[];
    sources: string[];
    changed: boolean;
  };
}

const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/skills_edit.json'), 'utf8'),
) as { edits: EditCase[]; commands: CommandCase[]; plans: PlanCase[] };

/** Write a case's files into `folder`, making a symbolic link for a `{link}` value (S-290). */
function layout(folder: string, files: Record<string, FileSpec>): void {
  for (const [name, spec] of Object.entries(files)) {
    const full = path.join(folder, name);
    mkdirSync(path.dirname(full), { recursive: true });
    if (typeof spec === 'string') writeFileSync(full, spec);
    else symlinkSync(spec.link, full);
  }
}

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
    layout(folder, c.files);
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
    for (const [name, spec] of Object.entries(c.files)) {
      // A link's own bytes are its target's; the target is checked as its own entry.
      if (typeof spec !== 'string') continue;
      expect(readFileSync(path.join(folder, name), 'utf8'), name).toBe(c.after?.[name] ?? spec);
    }
  });
});

describe('planSkills (the checks, no write)', () => {
  let folder: string;
  beforeEach(() => {
    folder = mkdtempSync(path.join(tmpdir(), 'skills-plan-'));
  });
  afterEach(() => rmSync(folder, { recursive: true, force: true }));

  it.each(FIXTURE.plans)('$case', (c) => {
    layout(folder, c.files);
    const plan = planSkills(c.args[0] as 'add' | 'remove', c.args.slice(1), { folder });
    expect(plan.errors).toEqual(c.plan.errors);
    expect(plan.add).toEqual(c.plan.add);
    expect(plan.remove).toEqual(c.plan.remove);
    expect(plan.skillsAfter).toEqual(c.plan.skills_after);
    expect(plan.installedRemovals).toEqual(c.plan.installed_removals);
    expect(plan.sources).toEqual(c.plan.sources);
    expect(plan.changed).toBe(c.plan.changed);
    expect(plan.file === null ? null : path.basename(plan.file)).toBe(c.plan.file);
  });
});
