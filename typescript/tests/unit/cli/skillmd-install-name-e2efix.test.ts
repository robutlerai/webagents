/**
 * The installed SKILL.md folder is named after the skill's own validated
 * `name:` (2026-09-26, the new-developer e2e run): `skills add` named it after
 * the repository, so a skill declared `greeter` in a repository
 * `greeter-skill` landed as `.agents/skills/greeter-skill` and doctor warned
 * about its own name on every load. And doctor's fix line names the folders
 * (it stopped at "named"). Both are pinned by `install_name` and `doctor` in
 * `python/tests/fixtures/skillmd/skillmd.json`, which the Python suite reads
 * too (`tests/cli/test_skillmd_install_name_e2efix.py`).
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { discoverSkills, doctorReport } from '../../../src/skills/skillmd/skillmd-loader';
import { locateSkills, parseSource, repoName } from '../../../src/skills/skillmd/skillmd-install';
import { skillsCommand } from '../../../src/cli/skills-edit';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/skillmd/skillmd.json'), 'utf8')) as {
  install_name: { cases: { name: string; repo: string; path: string; declared: string | null; installed: string }[] };
  doctor: { some: string; warning: string; fix: string; fix_unnamed: string };
};
const tempDir = tempDirs();
const fill = (template: string, values: Record<string, string | number>): string =>
  template.replace(/\{(\w+)\}/g, (_m, key: string) => String(values[key] ?? `{${key}}`));

/** A local source folder named `repo` holding one SKILL.md at `relative`, declaring `declared` (or nothing). */
function sourceFolder(repo: string, relative: string, declared: string | null): string {
  const root = path.join(tempDir('wa-skillmd-name-'), repo);
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const front = declared === null ? '' : `name: ${declared}\n`;
  fs.writeFileSync(file, `---\n${front}description: Greets the person by name.\n---\n\n# Greeter\n\nGreet them.\n`);
  return root;
}

function agentFolder(): string {
  const folder = path.join(tempDir('wa-skillmd-name-agent-'), 'agent');
  fs.mkdirSync(folder);
  fs.writeFileSync(path.join(folder, 'AGENT.md'), '---\nname: bot\nskills:\n  - openai\n---\n\nBody.\n');
  return folder;
}

describe('the installed folder is named after the validated name:', () => {
  it.each(FIXTURE.install_name.cases)('$name', ({ repo, path: relative, declared, installed }) => {
    const root = sourceFolder(repo, relative, declared);
    const source = parseSource(root);
    const { located, skipped } = locateSkills(root, source.subpath, repoName(source));
    expect(skipped).toEqual([]);
    expect(located.map((c) => c.name)).toEqual([installed]);
  });

  it('installs under that name, and doctor then has nothing to warn about', async () => {
    const root = sourceFolder('greeter-skill', 'SKILL.md', 'greeter');
    const folder = agentFolder();
    const out: string[] = [];
    const err: string[] = [];
    const code = await skillsCommand('add', [root], { folder, tty: false, yes: true }, { out: (l) => out.push(l), err: (l) => err.push(l) });
    expect(code, err.join('\n')).toBe(0);
    expect(fs.existsSync(path.join(folder, '.agents', 'skills', 'greeter', 'SKILL.md'))).toBe(true);
    expect(fs.existsSync(path.join(folder, '.agents', 'skills', 'greeter-skill'))).toBe(false);
    expect(doctorReport(discoverSkills(folder))).toEqual({ status: 'ok', detail: fill(FIXTURE.doctor.some, { count: 1, s: '', names: 'greeter' }) });
  });
});

describe('doctor names the folders to fix', () => {
  it('lists the skipped folders and the skills that warned, and the fix is the fixture sentence', () => {
    const folder = agentFolder();
    const skills = path.join(folder, '.agents', 'skills');
    fs.mkdirSync(path.join(skills, 'broken'), { recursive: true });
    fs.writeFileSync(path.join(skills, 'broken', 'SKILL.md'), '---\nname: broken\n---\n');
    fs.mkdirSync(path.join(skills, 'greeter-skill'), { recursive: true });
    fs.writeFileSync(path.join(skills, 'greeter-skill', 'SKILL.md'), '---\nname: greeter\ndescription: Greets.\n---\nGreet.\n');
    const report = doctorReport(discoverSkills(folder));
    expect(report.status).toBe('warn');
    expect(report.fix).toBe(fill(FIXTURE.doctor.fix, { names: 'broken, greeter-skill' }));
    expect(report.fix).not.toMatch(/named$/);
  });
});
