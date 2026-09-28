/**
 * Loading SKILL.md skills (gap-closure plan item 1.4, 2026-09-26), against
 * the shared fixture `python/tests/fixtures/skillmd/skillmd.json`, which the
 * Python suite reads too (`tests/skills/local/test_skillmd_loader.py`): the
 * parser's cases (CRLF, a closing fence with no newline, a BOM, an unquoted
 * colon, metadata values as strings, the three spellings of allowed-tools,
 * other clients' keys kept, the two reasons a skill is skipped), discovery
 * under `.agents/skills` and through `agent_skills:`, and the words the
 * model sees.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  AUTO_SKILLS_DIR,
  CATALOG_PREAMBLE,
  EXPLICIT_KEY,
  KNOWN_KEYS,
  SKILL_FILE_NAMES,
  SKIPPED_DIRS,
  activationText,
  bundledFiles,
  catalogText,
  discoverSkills,
  loadSkillDir,
  parseSkillMd,
  type SkillMd,
} from '../../../src/skills/skillmd/skillmd-loader';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/skillmd');
const FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'skillmd.json'), 'utf8'));
const REPO = path.join(FIXTURES, FIXTURE.sample.repo);
const tempDir = tempDirs();

interface ParseCase {
  case: string;
  dir: string;
  text: string;
  skipped?: string;
  reason?: string;
  reason_starts_with?: string;
  description?: string;
  allowed_tools?: string[];
  metadata?: Record<string, string>;
  extra?: Record<string, unknown>;
  warnings?: string[];
  body?: string;
}

describe('the fixture is the contract', () => {
  it('pins the layout', () => {
    const layout = FIXTURE.layout;
    expect(AUTO_SKILLS_DIR).toBe(layout.auto_dir.split('/').join(path.sep));
    expect(EXPLICIT_KEY).toBe(layout.explicit_key);
    expect([...SKILL_FILE_NAMES]).toEqual(layout.skill_file_names);
    expect([...SKIPPED_DIRS].sort()).toEqual([...layout.skipped_dirs].sort());
    expect([...KNOWN_KEYS]).toEqual(FIXTURE.frontmatter.known_keys);
    expect(CATALOG_PREAMBLE).toBe(FIXTURE.catalog.preamble);
  });
});

describe('the parser', () => {
  it.each(FIXTURE.parse_cases as ParseCase[])('$case', (c) => {
    const parsed = parseSkillMd(c.text, c.dir);
    if (c.skipped) {
      expect('problem' in parsed).toBe(true);
      const problem = parsed as { problem: string; reason: string };
      expect(problem.problem).toBe(c.skipped);
      if (c.reason_starts_with) expect(problem.reason.startsWith(c.reason_starts_with), problem.reason).toBe(true);
      else expect(problem.reason).toBe(c.reason);
      return;
    }
    expect('problem' in parsed).toBe(false);
    const skill = parsed as SkillMd;
    expect(skill.name).toBe(c.dir);
    expect(skill.description).toBe(c.description);
    expect(skill.allowedTools).toEqual(c.allowed_tools);
    expect(skill.metadata).toEqual(c.metadata);
    expect(skill.extra).toEqual(c.extra);
    expect(skill.warnings).toEqual(c.warnings);
    expect(skill.body).toBe(c.body);
  });

  it('keeps allowed-tools a hint and never a grant', () => {
    const parsed = parseSkillMd('---\ndescription: d\nallowed-tools: Bash(*) Read Write\n---\n', 'x') as SkillMd;
    expect(parsed.allowedTools).toEqual(['Bash(*)', 'Read', 'Write']);
    expect('scope' in parsed).toBe(false);
    expect('scopes' in parsed).toBe(false);
  });
});

describe('the sample repository', () => {
  it('loads the skills and reports the skipped one', () => {
    const sample = FIXTURE.sample;
    const found = discoverSkills(REPO, sample.explicit);
    expect(found.skills.map((s) => s.name)).toEqual(sample.skills);
    expect(found.skipped.map((s) => [s.name, s.problem, s.reason])).toEqual(
      (sample.skipped as Array<{ name: string; problem: string; reason: string }>).map((s) => [s.name, s.problem, s.reason]),
    );
    expect(found.warnings).toEqual([]);
    const pdf = found.skills[0];
    expect(pdf.source).toBe('explicit');
    expect(pdf.allowedTools).toEqual(sample.pdf.allowed_tools);
    expect(pdf.metadata).toEqual(sample.pdf.metadata);
    expect(pdf.license).toBe(sample.pdf.license);
    expect(pdf.warnings).toEqual(sample.pdf.warnings);
    expect(pdf.body).toBe(sample.pdf.body);
    expect(pdf.location).toBe(fs.realpathSync(path.join(REPO, 'skills', 'pdf', 'SKILL.md')));
    expect(pdf.directory).toBe(fs.realpathSync(path.join(REPO, 'skills', 'pdf')));
  });

  it('renders the catalog', () => {
    const found = discoverSkills(REPO, FIXTURE.sample.explicit);
    const byName = new Map(found.skills.map((s) => [s.name, s]));
    const expected = (FIXTURE.sample.catalog as string)
      .replace('{pdf_location}', byName.get('pdf')!.location)
      .replace('{xlsx_location}', byName.get('xlsx')!.location);
    expect(catalogText(found.skills)).toBe(expected);
  });

  it('renders the activation, listing files without reading them', () => {
    const found = discoverSkills(REPO, FIXTURE.sample.explicit);
    const byName = new Map(found.skills.map((s) => [s.name, s]));
    expect(bundledFiles(byName.get('pdf')!.directory)).toEqual(FIXTURE.sample.pdf.files);
    const text = activationText(byName.get('pdf')!);
    expect(text).toBe((FIXTURE.sample.activation as string).replace('{pdf_dir}', byName.get('pdf')!.directory));
    // The substitution stays text: nothing ran `date`.
    expect(text).toContain('!`date`');
    expect(activationText(byName.get('xlsx')!)).toContain(FIXTURE.sample.activation_xlsx_files);
  });

  it('escapes the catalog as XML', () => {
    const escaped = FIXTURE.catalog.escaped;
    const skill: SkillMd = {
      name: escaped.name,
      description: escaped.description,
      location: escaped.location,
      directory: '/x/a-b',
      body: '',
      allowedTools: [],
      metadata: {},
      extra: {},
      warnings: [],
      source: 'auto',
    };
    expect(catalogText([skill])).toBe(escaped.text);
  });

  it('has no catalog without skills', () => {
    expect(catalogText([])).toBe(FIXTURE.catalog.empty);
  });
});

describe('discovery', () => {
  function skill(folder: string, name: string, description = 'Does things.'): string {
    const directory = path.join(folder, name);
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, 'SKILL.md'), `---\nname: ${name}\ndescription: ${description}\n---\nBody of ${name}.\n`);
    return directory;
  }

  it('finds .agents/skills on its own', () => {
    const base = tempDir('wa-skillmd-');
    skill(path.join(base, '.agents', 'skills'), 'alpha');
    skill(path.join(base, '.agents', 'skills'), 'beta');
    const found = discoverSkills(base);
    expect(found.skills.map((s) => s.name)).toEqual(['alpha', 'beta']);
    expect(found.skills.every((s) => s.source === 'auto')).toBe(true);
  });

  it('accepts skill.md in lower case', () => {
    const base = tempDir('wa-skillmd-');
    const directory = path.join(base, '.agents', 'skills', 'low');
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, 'skill.md'), '---\ndescription: d\n---\n');
    expect(discoverSkills(base).skills.map((s) => s.name)).toEqual(['low']);
  });

  it('reads an explicit entry as one skill or as a folder of skills', () => {
    const base = tempDir('wa-skillmd-');
    skill(path.join(base, 'one'), 'one');
    const shared = path.join(base, 'shared');
    skill(shared, 's1');
    skill(shared, 's2');
    const found = discoverSkills(base, ['one/one', 'shared', path.join(shared, 's1')]);
    expect(found.skills.map((s) => s.name)).toEqual(['one', 's1', 's2']);
    expect(found.skills.every((s) => s.source === 'explicit')).toBe(true);
    expect(found.warnings).toHaveLength(1);
    expect(found.warnings[0]).toContain('s1');
    expect(found.warnings[0]).toContain('shadowed');
  });

  it('lets an explicit entry win over a discovered one', () => {
    const base = tempDir('wa-skillmd-');
    skill(path.join(base, '.agents', 'skills'), 'dup', 'auto one');
    skill(path.join(base, 'elsewhere'), 'dup', 'explicit one');
    const found = discoverSkills(base, ['elsewhere/dup']);
    expect(found.skills.map((s) => s.name)).toEqual(['dup']);
    expect(found.skills[0].description).toBe('explicit one');
    expect(found.warnings).toHaveLength(1);
    expect(found.warnings[0]).toContain('shadowed');
  });

  it('reports a missing or empty explicit entry instead of throwing', () => {
    const base = tempDir('wa-skillmd-');
    fs.mkdirSync(path.join(base, 'empty'));
    const found = discoverSkills(base, ['nowhere', 'empty']);
    expect(found.skills).toEqual([]);
    expect(found.skipped.map((s) => s.problem)).toEqual(['not_found', 'not_found']);
    expect(found.skipped[0].reason).toBe(`${EXPLICIT_KEY}: nowhere is not a folder`);
    expect(found.skipped[1].reason).toBe(`${EXPLICIT_KEY}: empty holds no SKILL.md and no folder with one`);
  });

  it('reports a malformed skill and loads the rest', () => {
    const base = tempDir('wa-skillmd-');
    skill(path.join(base, '.agents', 'skills'), 'good');
    const broken = path.join(base, '.agents', 'skills', 'broken');
    fs.mkdirSync(broken, { recursive: true });
    fs.writeFileSync(path.join(broken, 'SKILL.md'), '---\nname: broken\n---\nNo description.\n');
    const found = discoverSkills(base);
    expect(found.skills.map((s) => s.name)).toEqual(['good']);
    expect(found.skipped.map((s) => [s.name, s.problem])).toEqual([['broken', 'no_description']]);
  });

  it('never treats .git or node_modules as skills or bundled files', () => {
    const base = tempDir('wa-skillmd-');
    const directory = skill(path.join(base, '.agents', 'skills'), 'tidy');
    fs.mkdirSync(path.join(directory, '.git'));
    fs.writeFileSync(path.join(directory, '.git', 'config'), 'x');
    fs.mkdirSync(path.join(directory, 'node_modules', 'm'), { recursive: true });
    fs.writeFileSync(path.join(directory, 'node_modules', 'm', 'index.js'), 'x');
    fs.mkdirSync(path.join(directory, 'scripts'));
    fs.writeFileSync(path.join(directory, 'scripts', 'run.sh'), 'echo hi\n');
    fs.symlinkSync('/etc/hosts', path.join(directory, 'escape'));
    for (const name of ['.git', 'node_modules']) {
      fs.mkdirSync(path.join(base, '.agents', 'skills', name), { recursive: true });
      fs.writeFileSync(path.join(base, '.agents', 'skills', name, 'SKILL.md'), '---\ndescription: d\n---\n');
    }
    const found = discoverSkills(base);
    expect(found.skills.map((s) => s.name)).toEqual(['tidy']);
    expect(bundledFiles(directory)).toEqual(['scripts/run.sh']);
  });

  it('names a loaded folder after itself', () => {
    const base = tempDir('wa-skillmd-');
    const directory = skill(base, 'named');
    const loaded = loadSkillDir(directory);
    expect('problem' in loaded).toBe(false);
    expect((loaded as SkillMd).name).toBe('named');
    expect((loaded as SkillMd).location).toBe(fs.realpathSync(path.join(directory, 'SKILL.md')));
    expect('problem' in loadSkillDir(base)).toBe(true);
  });
});
