/**
 * `webagents init --template tool-agent` ships an `access:` block that keeps
 * shell and filesystem owner-only (S-248, 2026-09-26), byte for byte what the
 * Python CLI writes (`python/tests/fixtures/cli/init_templates.json`), and
 * the file it writes loads: the block parses, and it scopes both skills'
 * tools to the `trusted` group, which nobody but the owner is in until the
 * owner names someone.
 */

import { describe, it, expect } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tempDirs, TSX_CLI, CLI_SOURCE } from '../../helpers/cli';
import { parseAgentMarkdown } from '../../../src/agents/index';
import { accessSkillFor, applyAccessTools } from '../../../src/access/install';
import { ShellSkill } from '../../../src/skills/shell/skill';
import { FilesystemSkill } from '../../../src/skills/filesystem/skill';
import { callerScopes, scopeAllows } from '../../../src/core/scopes';

const tempDir = tempDirs();
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/init_templates.json'), 'utf8'),
) as {
  templates: Record<string, { description: string; skills: string[]; sandbox?: string[]; access: string[] }>;
  restores_open_tools: { access: string[] };
  agent_md: Record<string, string>;
  with_model: { name: string; template: string; model: string; agent_md: string };
  no_model: { comment: string[]; runs_on_without_a_key: string };
  with_key: { env: Record<string, string>; model: string; name: string; template: string; agent_md: string };
  init_line: { cases: Array<{ model: string; keyed: boolean; signed_in: boolean; line: string }> };
};

function runInit(name: string, template: string): { dir: string; status: number | null; stderr: string } {
  const cwd = tempDir('wa-init-s248-');
  const home = tempDir('wa-init-s248-home-');
  const base = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file' } as Record<string, string | undefined>;
  // No provider key: `init` then writes Robutler's choice (B3).
  for (const key of ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY']) delete base[key];
  delete base.WEBAGENTS_DEBUG;
  delete base.WEBAGENTS_TOKEN;
  const out = spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, 'init', name, '-t', template], {
    cwd,
    env: base as NodeJS.ProcessEnv,
    encoding: 'utf-8',
    timeout: 60_000,
  });
  return { dir: path.join(cwd, name), status: out.status, stderr: out.stderr };
}

function expectedAgentMd(name: string, template: string): string {
  const t = FIXTURE.templates[template];
  // The file's description is the template's, and the tool-agent template
  // ships a sandbox block before its access block (2026-09-26). With no
  // provider key here it names no model and no provider skill, and says what
  // runs (2026-09-28, `no_model`).
  const text = [
    '---',
    `name: ${name}`,
    `description: ${t.description}`,
    ...FIXTURE.no_model.comment,
    ...(t.skills.length ? ['skills:', ...t.skills.map((s) => `  - ${s}`)] : []),
    ...(t.sandbox ?? []),
    ...t.access,
    '---',
    '',
    `# ${name}`,
    '',
    'You are a helpful assistant.',
    '',
  ].join('\n');
  expect(text).toBe(FIXTURE.agent_md[template].replace(/\{name\}/g, name));
  return text;
}

describe('agentMarkdown (spec 3.3)', () => {
  it('renders each template byte for byte, the renderer init and /agent new share', async () => {
    const { agentMarkdown } = await import('../../../src/cli/init-templates');
    for (const [template, text] of Object.entries(FIXTURE.agent_md)) {
      if (template === 'about') continue;
      expect(agentMarkdown('demo', template)).toBe(text.replace(/\{name\}/g, 'demo'));
    }
  });

  it('names the provider the chat runs on when a model is passed', async () => {
    const { agentMarkdown } = await import('../../../src/cli/init-templates');
    const wm = FIXTURE.with_model;
    expect(agentMarkdown(wm.name, wm.template, wm.model)).toBe(wm.agent_md);
  });

  it("init with a key names that provider's model; with none, Robutler's choice (B3)", async () => {
    const { agentMarkdown, initModel } = await import('../../../src/cli/init-templates');
    const wk = FIXTURE.with_key;
    expect(await initModel(wk.env)).toBe(wk.model);
    expect(agentMarkdown(wk.name, wk.template, wk.model)).toBe(wk.agent_md);
    expect(await initModel({})).toBeUndefined();
  });

  it('the last line names the way in only when one is needed', async () => {
    const { initLine } = await import('../../../src/cli/init-templates');
    const saved = process.env.WEBAGENTS_PROFILE;
    delete process.env.WEBAGENTS_PROFILE;
    try {
      for (const c of FIXTURE.init_line.cases) expect(initLine(c.model, c.keyed, c.signed_in)).toBe(c.line);
    } finally {
      if (saved !== undefined) process.env.WEBAGENTS_PROFILE = saved;
    }
  });
});

describe('init --template tool-agent (S-248)', () => {
  it('writes the file the shared fixture describes, access block included', () => {
    const { dir, status, stderr } = runInit('tools', 'tool-agent');
    expect(status, stderr).toBe(0);
    expect(fs.readFileSync(path.join(dir, 'AGENT.md'), 'utf8')).toBe(expectedAgentMd('tools', 'tool-agent'));
  });

  it('the chatbot template carries no access block', () => {
    const { dir, status, stderr } = runInit('chat', 'chatbot');
    expect(status, stderr).toBe(0);
    expect(fs.readFileSync(path.join(dir, 'AGENT.md'), 'utf8')).toBe(expectedAgentMd('chat', 'chatbot'));
    expect(FIXTURE.templates.chatbot.access).toEqual([]);
  });

  it('the written block loads and scopes shell and filesystem to the trusted group, which keeps them owner-only', () => {
    const parsed = parseAgentMarkdown(expectedAgentMd('tools', 'tool-agent'), '/x/AGENT.md');
    expect(parsed.skills).toEqual(['filesystem', 'shell']);
    expect(parsed.model).toBeUndefined();
    expect(parsed.access).toEqual({ groups: { trusted: [] }, tools: { trusted: ['filesystem', 'shell'] } });

    const shell = new ShellSkill();
    const filesystem = new FilesystemSkill();
    const { policy } = accessSkillFor(parsed.access, undefined);
    applyAccessTools(policy, new Map([['filesystem', filesystem], ['shell', shell]]));
    for (const tool of [...shell.tools, ...filesystem.tools]) {
      expect(tool.scopes, tool.name).toEqual(['group:trusted']);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: true, scope: 'owner' as never }))).toBe(true);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: true, scope: 'user' as never }))).toBe(false);
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: false }))).toBe(false);
      // The group the owner would name callers in: a member gets the tools.
      expect(scopeAllows(tool.scopes, callerScopes({ authenticated: true, scopes: ['group:trusted'] }))).toBe(true);
    }
  });

  it('the block that restores the old open tools grants them to the default group, anonymous callers included', () => {
    const text = ['---', 'name: open', 'skills:', '  - filesystem', '  - shell', ...FIXTURE.restores_open_tools.access, '---', 'Open.', ''].join('\n');
    const parsed = parseAgentMarkdown(text, '/x/AGENT.md');
    const shell = new ShellSkill();
    const { policy } = accessSkillFor(parsed.access, undefined);
    applyAccessTools(policy, new Map([['filesystem', new FilesystemSkill()], ['shell', shell]]));
    const run = shell.tools.find((t) => t.name === 'runCommand')!;
    expect(run.scopes).toEqual(['group:everyone']);
    // The access skill places every caller the block names no other group for in `everyone`.
    expect(scopeAllows(run.scopes, callerScopes({ authenticated: false, scopes: ['group:everyone'] }))).toBe(true);
  });
});
