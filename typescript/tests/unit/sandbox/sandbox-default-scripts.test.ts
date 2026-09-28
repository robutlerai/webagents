/**
 * SKILL.md scripts are confined commands too (the sandbox-default lane,
 * 2026-09-27): an agent that runs them is not one that "cannot run
 * commands", so `doctor` and `/sandbox` say the state its scripts run under
 * (`SkillMdSkill.scriptState`). The Python twin is
 * `python/tests/sandbox/test_sandbox_default_scripts.py`; the words come
 * from `python/tests/fixtures/sandbox/srt.json` (`status`).
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SkillMdSkill } from '../../../src/skills/skillmd/skillmd-skill';
import { discoverSkills } from '../../../src/skills/skillmd/skillmd-loader';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const SKILLMD = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'skillmd', 'skillmd.json'), 'utf8'));
const SRT = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'sandbox', 'srt.json'), 'utf8'));
const REPO = path.join(FIXTURES, 'skillmd', SKILLMD.sample.repo);

function skillFor(sandbox?: unknown): SkillMdSkill {
  const found = discoverSkills(REPO, SKILLMD.sample.explicit);
  return new SkillMdSkill({ skills: found.skills, skipped: found.skipped, warnings: found.warnings, agentDir: REPO, sandbox: sandbox as never });
}

describe('the state SKILL.md scripts run under', () => {
  it('is strict by default, the agent file preset when declared, off for the opt-out, and empty without skills', () => {
    expect(skillFor().scriptState()).toBe('strict (default)');
    expect(skillFor({ preset: 'development', network: ['example.com'] }).scriptState()).toBe(SRT.status.agent_file.replace('{preset}', 'development'));
    expect(skillFor(false).scriptState()).toBe(SRT.status.off_agent_file);
    expect(skillFor('off').scriptState()).toBe(SRT.status.off_agent_file);
    expect(skillFor({ preset: 'stirct' }).scriptState()).toBe('invalid (agent file)');
    expect(new SkillMdSkill({ skills: [], agentDir: REPO }).scriptState()).toBe('');
  });
});
