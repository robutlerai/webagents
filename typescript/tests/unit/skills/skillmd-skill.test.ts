/**
 * The SKILL.md skill an agent carries (gap-closure plan item 1.4,
 * 2026-09-26), against the shared fixture
 * `python/tests/fixtures/skillmd/skillmd.json`, which the Python suite reads
 * too (`tests/skills/local/test_skillmd_skill.py`): the three tools and
 * their words, the catalog in the prompt for callers who may activate,
 * activation once per conversation, reading confined to the skill's folder,
 * and the owner-only default that `access: tools:` opens. Scripts run under
 * real srt in `tests/unit/sandbox/skillmd-scripts.test.ts`.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { applyAccessTools } from '../../../src/access/install';
import { parseAccess } from '../../../src/access/policy';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff } from '../../../src/core/decorators';
import { callerScopes, scopeAllows } from '../../../src/core/scopes';
import type { Context, ISkill } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent } from '../../../src/uamp/events';
import { discoverSkills } from '../../../src/skills/skillmd/skillmd-loader';
import {
  DEFAULT_TIMEOUT,
  INTERPRETERS,
  MAX_TIMEOUT,
  READ_MAX_BYTES,
  SKILL_KEY,
  SkillMdSkill,
  TOOL_NAMES,
} from '../../../src/skills/skillmd/skillmd-skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/skillmd');
const FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'skillmd.json'), 'utf8'));
const REPO = path.join(FIXTURES, FIXTURE.sample.repo);
const tempDir = tempDirs();

const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };
const USER = { authenticated: true, scope: 'user', user_id: 'u-1', provider: 'portal' };

function skillFor(agentDir = REPO, sandbox?: Record<string, unknown> | null): SkillMdSkill {
  const found = discoverSkills(agentDir, FIXTURE.sample.explicit);
  return new SkillMdSkill({ skills: found.skills, skipped: found.skipped, warnings: found.warnings, agentDir, sandbox });
}

/** A run context as a tool handler sees it: the caller, and the conversation so far. */
function contextOf(auth: unknown, messages: unknown[] = []): Context {
  return {
    auth,
    get: (key: string) => (key === '_agentic_messages' ? messages : undefined),
  } as unknown as Context;
}

function tool(skill: SkillMdSkill, name: string) {
  const found = skill.tools.find((t) => t.name === name);
  if (!found) throw new Error(`no tool ${name}`);
  return found;
}

/** Records the system message and the tools each turn was offered. */
class RecordingLLM extends Skill {
  readonly system: string[] = [];
  readonly offered: string[][] = [];

  @handoff({ name: 'recording-llm' })
  async *processUAMP(_events: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
    const messages = (ctx.get('_agentic_messages') as Array<{ role: string; content: string | null }>) ?? [];
    this.system.push(messages.filter((m) => m.role === 'system').map((m) => m.content ?? '').join('\n---\n'));
    const tools = ((ctx.get('_agentic_tools') as Array<{ function?: { name?: string }; name?: string }>) ?? [])
      .map((t) => t.function?.name ?? t.name ?? '')
      .sort();
    this.offered.push(tools);
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

describe('the tools match the fixture', () => {
  it('pins the names, scope and limits', () => {
    expect([...TOOL_NAMES]).toEqual(FIXTURE.tools.names);
    expect({ ...INTERPRETERS }).toEqual(FIXTURE.scripts.interpreters);
    expect(DEFAULT_TIMEOUT).toBe(FIXTURE.scripts.default_timeout);
    expect(MAX_TIMEOUT).toBe(FIXTURE.scripts.max_timeout);
    expect(READ_MAX_BYTES).toBe(FIXTURE.read.max_bytes);
    expect(SKILL_KEY).toBe(FIXTURE.layout.skill_key);
  });

  it('declares them word for word, owner-only', () => {
    const skill = skillFor();
    expect(skill.tools.map((t) => t.name)).toEqual(FIXTURE.tools.names);
    for (const name of FIXTURE.tools.names as string[]) {
      const found = tool(skill, name);
      expect(found.description, name).toBe(FIXTURE.tools[name].description);
      expect(found.parameters, name).toEqual(FIXTURE.tools[name].parameters);
      expect(found.scopes, name).toEqual([FIXTURE.tools.scope]);
      expect(scopeAllows(found.scopes, callerScopes(OWNER as never))).toBe(true);
      expect(scopeAllows(found.scopes, callerScopes({ authenticated: true, scope: 'admin' } as never))).toBe(true);
      expect(scopeAllows(found.scopes, callerScopes(USER as never))).toBe(false);
      expect(scopeAllows(found.scopes, callerScopes({ authenticated: false } as never))).toBe(false);
    }
  });

  it('is offered to the owner and to nobody else, and the catalog goes with it', async () => {
    const skill = skillFor();
    const llm = new RecordingLLM();
    const agent = new BaseAgent({ name: 'host', instructions: 'Host.', skills: [skill, llm] });
    await agent.run([{ role: 'user', content: 'hi' }], { auth: OWNER });
    await agent.run([{ role: 'user', content: 'hi' }], { auth: USER });
    const [owner, user] = llm.offered;
    for (const name of FIXTURE.tools.names as string[]) {
      expect(owner, name).toContain(name);
      expect(user, name).not.toContain(name);
    }
    const byName = new Map(found().map((s) => [s.name, s]));
    const expected = (FIXTURE.sample.catalog as string)
      .replace('{pdf_location}', byName.get('pdf')!.location)
      .replace('{xlsx_location}', byName.get('xlsx')!.location);
    expect(llm.system[0]).toContain(expected);
    // Once, not twice (a decorated prompt would be registered twice by addSkill).
    expect(llm.system[0].split(FIXTURE.catalog.preamble).length).toBe(2);
    expect(llm.system[1]).not.toContain(FIXTURE.catalog.preamble);
  });

  it('an access block opens the skill by its name, or a tool by its own', () => {
    let skill = skillFor();
    let byName = new Map<string, ISkill>([[SKILL_KEY, skill as unknown as ISkill]]);
    applyAccessTools(parseAccess({ groups: { friends: [] }, tools: { friends: [SKILL_KEY] } }), byName);
    for (const name of FIXTURE.tools.names as string[]) {
      expect(tool(skill, name).scopes, name).toEqual(['group:friends']);
      expect(scopeAllows(tool(skill, name).scopes, ['group:friends'])).toBe(true);
      expect(scopeAllows(tool(skill, name).scopes, ['owner'])).toBe(true);
      expect(scopeAllows(tool(skill, name).scopes, ['user'])).toBe(false);
    }
    skill = skillFor();
    byName = new Map<string, ISkill>([[SKILL_KEY, skill as unknown as ISkill]]);
    applyAccessTools(parseAccess({ groups: { readers: [] }, tools: { readers: ['activate_skill', 'read_skill_file'] } }), byName);
    expect(tool(skill, 'activate_skill').scopes).toEqual(['group:readers']);
    expect(tool(skill, 'read_skill_file').scopes).toEqual(['group:readers']);
    expect(tool(skill, 'run_skill_script').scopes).toEqual(['owner']);
  });
});

function found() {
  return discoverSkills(REPO, FIXTURE.sample.explicit).skills;
}

describe('the catalog', () => {
  it('shows for the owner and not for a caller who cannot activate', () => {
    const skill = skillFor();
    expect(skill.skillsCatalog(contextOf(USER))).toBe('');
    expect(skill.skillsCatalog(contextOf(OWNER)).startsWith(FIXTURE.catalog.preamble)).toBe(true);
    // Outside any run there is no caller: the process itself.
    expect(skill.skillsCatalog(undefined).startsWith(FIXTURE.catalog.preamble)).toBe(true);
  });

  it('shows for a group the block opens', () => {
    const skill = skillFor();
    const byName = new Map<string, ISkill>([[SKILL_KEY, skill as unknown as ISkill]]);
    applyAccessTools(parseAccess({ groups: { friends: [] }, tools: { friends: [SKILL_KEY] } }), byName);
    expect(skill.skillsCatalog(contextOf({ ...USER, scopes: ['group:friends'] })).startsWith(FIXTURE.catalog.preamble)).toBe(true);
    expect(skill.skillsCatalog(contextOf({ ...USER, scopes: ['group:others'] }))).toBe('');
  });

  it('is empty without skills', () => {
    const skill = new SkillMdSkill({ skills: [], agentDir: tempDir('wa-skillmd-empty-') });
    expect(skill.skillsCatalog(contextOf(OWNER))).toBe(FIXTURE.catalog.empty);
  });
});

describe('activation', () => {
  it('returns the body and lists the files', async () => {
    const skill = skillFor();
    const pdf = skill.skills.get('pdf')!;
    const expected = (FIXTURE.sample.activation as string).replace('{pdf_dir}', pdf.directory);
    expect(await tool(skill, 'activate_skill').handler({ name: 'pdf' }, contextOf(OWNER))).toBe(expected);
  });

  it('never injects twice in one conversation', async () => {
    const skill = skillFor();
    const activate = tool(skill, 'activate_skill').handler;
    const first = (await activate({ name: 'pdf' }, contextOf(OWNER))) as string;
    const conversation = [
      { role: 'user', content: 'hi' },
      { role: 'tool', tool_call_id: 'c1', content: first },
    ];
    expect(await activate({ name: 'pdf' }, contextOf(OWNER, conversation))).toBe(
      (FIXTURE.activation.already_active as string).replace('{name}', 'pdf'),
    );
    // Another skill is not affected, and a new conversation starts over.
    const xlsx = (await activate({ name: 'xlsx' }, contextOf(OWNER, conversation))) as string;
    expect(xlsx.startsWith((FIXTURE.activation.marker as string).replace('{name}', 'xlsx'))).toBe(true);
    expect(await activate({ name: 'pdf' }, contextOf(OWNER, [{ role: 'user', content: 'new' }]))).toBe(first);
  });

  it('counts a tool result carried as text parts', async () => {
    const skill = skillFor();
    const activate = tool(skill, 'activate_skill').handler;
    const first = (await activate({ name: 'pdf' }, contextOf(OWNER))) as string;
    const conversation = [{ role: 'tool', content: [{ type: 'text', text: first }] }];
    expect(await activate({ name: 'pdf' }, contextOf(OWNER, conversation))).toBe(
      (FIXTURE.activation.already_active as string).replace('{name}', 'pdf'),
    );
  });

  it('names an unknown skill', async () => {
    const skill = skillFor();
    const expected = (FIXTURE.activation.unknown as string).replace('{name}', 'nope').replace('{available}', 'pdf, xlsx');
    expect(await tool(skill, 'activate_skill').handler({ name: 'nope' }, contextOf(OWNER))).toBe(expected);
    const empty = new SkillMdSkill({ skills: [], agentDir: REPO });
    expect(await tool(empty, 'activate_skill').handler({ name: 'nope' }, contextOf(OWNER))).toBe(
      (FIXTURE.activation.unknown as string).replace('{name}', 'nope').replace('{available}', FIXTURE.activation.unknown_none),
    );
  });
});

describe('reading files', () => {
  it('reads a bundled file', async () => {
    const skill = skillFor();
    const read = tool(skill, 'read_skill_file').handler;
    expect(await read({ skill: 'pdf', path: 'forms.md' }, contextOf(OWNER))).toBe(fs.readFileSync(path.join(REPO, 'skills', 'pdf', 'forms.md'), 'utf8'));
    expect(((await read({ skill: 'pdf', path: 'scripts/fill_form.py' }, contextOf(OWNER))) as string).startsWith('"""Fixture script')).toBe(true);
  });

  it('is confined to the skill folder', async () => {
    const skill = skillFor();
    const read = tool(skill, 'read_skill_file').handler;
    const refusal = FIXTURE.read.refusals.outside as string;
    for (const p of ['../xlsx/SKILL.md', '/etc/hosts', 'scripts', 'missing.md', '../../README.md']) {
      expect(await read({ skill: 'pdf', path: p }, contextOf(OWNER)), p).toBe(refusal.replace('{path}', p));
    }
  });

  it('refuses a symbolic link out of the folder', async () => {
    const base = tempDir('wa-skillmd-link-');
    const directory = path.join(base, '.agents', 'skills', 'linky');
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, 'SKILL.md'), '---\ndescription: d\n---\n');
    fs.writeFileSync(path.join(base, 'secret.txt'), 'TOP-SECRET');
    fs.symlinkSync(path.join(base, 'secret.txt'), path.join(directory, 'escape.txt'));
    const skill = new SkillMdSkill({ skills: discoverSkills(base).skills, agentDir: base });
    expect(await tool(skill, 'read_skill_file').handler({ skill: 'linky', path: 'escape.txt' }, contextOf(OWNER))).toBe(
      (FIXTURE.read.refusals.outside as string).replace('{path}', 'escape.txt'),
    );
  });

  it('refuses binary and large files', async () => {
    const base = tempDir('wa-skillmd-big-');
    const directory = path.join(base, '.agents', 'skills', 'big');
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, 'SKILL.md'), '---\ndescription: d\n---\n');
    fs.writeFileSync(path.join(directory, 'blob.bin'), Buffer.from([0, 1, 2]));
    fs.writeFileSync(path.join(directory, 'huge.txt'), Buffer.alloc(READ_MAX_BYTES + 1, 'x'));
    const skill = new SkillMdSkill({ skills: discoverSkills(base).skills, agentDir: base });
    const read = tool(skill, 'read_skill_file').handler;
    expect(await read({ skill: 'big', path: 'blob.bin' }, contextOf(OWNER))).toBe((FIXTURE.read.refusals.binary as string).replace('{path}', 'blob.bin'));
    expect(await read({ skill: 'big', path: 'huge.txt' }, contextOf(OWNER))).toBe((FIXTURE.read.refusals.large as string).replace('{path}', 'huge.txt'));
  });
});

describe('script refusals decided before the sandbox is asked', () => {
  it('outside the folder, and not runnable', async () => {
    const skill = skillFor();
    const run = tool(skill, 'run_skill_script').handler;
    const refusals = FIXTURE.scripts.refusals;
    expect(await run({ skill: 'pdf', script: '../xlsx/SKILL.md' }, contextOf(OWNER))).toBe(refusals.outside.replace('{script}', '../xlsx/SKILL.md'));
    expect(await run({ skill: 'pdf', script: 'scripts/none.py' }, contextOf(OWNER))).toBe(refusals.outside.replace('{script}', 'scripts/none.py'));
    expect(await run({ skill: 'pdf', script: 'forms.md' }, contextOf(OWNER))).toBe(refusals.not_runnable.replace('{script}', 'forms.md'));
  });

  it('refuses unrestricted', async () => {
    const skill = skillFor(REPO, { preset: 'unrestricted' });
    expect(await tool(skill, 'run_skill_script').handler({ skill: 'pdf', script: 'scripts/fill_form.py' }, contextOf(OWNER))).toBe(FIXTURE.scripts.refusals.unrestricted);
  });

  it('refuses an invalid declaration', async () => {
    const skill = skillFor(REPO, { preset: 'stirct' });
    const out = (await tool(skill, 'run_skill_script').handler({ skill: 'pdf', script: 'scripts/fill_form.py' }, contextOf(OWNER))) as string;
    expect(out.startsWith((FIXTURE.scripts.refusals.invalid_declaration as string).split('{message}')[0])).toBe(true);
    expect(out).toContain('stirct');
  });

  it('keeps the skill folder read-only and readable in the policy', () => {
    const base = tempDir('wa-skillmd-policy-');
    const pdf = found()[0];
    const undeclared = skillFor(base, null).scriptPolicy(pdf);
    expect(typeof undeclared).not.toBe('string');
    const strict = undeclared as Exclude<typeof undeclared, string>;
    expect(strict.preset).toBe(FIXTURE.scripts.policy.undeclared.preset);
    expect(strict.scopedReads && strict.readRoots.includes(pdf.directory)).toBe(true);
    expect(strict.readOnly).toContain(pdf.directory);
    expect(strict.writeRoots).not.toContain(fs.realpathSync(base));
    expect(strict.networkDomains).toEqual([]);
    const declared = skillFor(base, { preset: 'development', network: ['example.com'] }).scriptPolicy(pdf) as Exclude<typeof undeclared, string>;
    expect(declared.preset).toBe('development');
    expect(declared.writeRoots).toContain(fs.realpathSync(base));
    expect(declared.readOnly).toContain(pdf.directory);
    expect(declared.networkDomains).toEqual(['example.com']);
  });

  it('quotes every argument of the command', () => {
    const skill = skillFor();
    const target = path.join(REPO, 'skills', 'pdf', 'scripts', 'fill_form.py');
    expect(skill.scriptCommand(target, ['a b', '$HOME'])).toBe(`python3 ${target} 'a b' '$HOME'`);
  });
});
