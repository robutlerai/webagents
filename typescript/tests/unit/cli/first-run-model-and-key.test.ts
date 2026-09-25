/**
 * The TypeScript first run, as tests (2026-09-24).
 *
 * Found by walking the onboarding path with the CLI packed and installed from
 * this tree, against a stand-in OpenAI endpoint that logged what it received:
 *
 *   1. The scaffold says `model: openai/gpt-4o-mini` and the request said
 *      `gpt-4o`. The LLM skill was built with no config and fell back to its
 *      own hardcoded model, in `serve` and in the REPL alike, so a newcomer was
 *      billed at gpt-4o rates by a file that says gpt-4o-mini.
 *   2. Without `OPENAI_API_KEY`, `webagents -p` died with a Node stack trace
 *      from inside the first model call.
 *   3. The loop's trace lines landed on stdout ahead of `-p`'s answer, so
 *      `--output-format json` could not be piped into anything.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import { InteractiveREPL } from '../../../src/cli/app';
import { agentTrace, setAgentTrace } from '../../../src/core/trace';
import { missingProviderKey, modelForProvider } from '../../../src/skills/llm/providers';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs, TSX_CLI } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

function modelOfSkill(skill: unknown): string {
  return (skill as { getCapabilities(): { id: string } }).getCapabilities().id;
}

describe('the model in the agent file reaches the LLM skill', () => {
  it('passes a matching provider/model through', async () => {
    const { skills } = await resolveSkillsByName(['openai'], { model: 'openai/gpt-4o-mini' });
    // THE BUG: this was 'gpt-4o', the skill's own default.
    expect(modelOfSkill(skills[0])).toBe('openai/gpt-4o-mini');
  });

  it('passes a bare model id through', async () => {
    const { skills } = await resolveSkillsByName(['openai'], { model: 'gpt-4.1-mini' });
    expect(modelOfSkill(skills[0])).toBe('gpt-4.1-mini');
  });

  it("does not force another provider's model onto a skill", () => {
    expect(modelForProvider('openai', 'anthropic/claude-sonnet-4')).toBeUndefined();
    expect(modelForProvider('anthropic', 'claude/claude-sonnet-4')).toBe('claude/claude-sonnet-4');
  });

  it('leaves the default alone when the file names no model', async () => {
    const { skills } = await resolveSkillsByName(['openai']);
    expect(modelOfSkill(skills[0])).toBe('gpt-4o');
  });
});

describe('the key the agent needs', () => {
  it("names the declared provider's variable when it is missing", () => {
    expect(missingProviderKey(['openai'], {})?.envVar).toBe('OPENAI_API_KEY');
    expect(missingProviderKey(['filesystem', 'claude'], {})?.envVar).toBe('ANTHROPIC_API_KEY');
  });

  it('is satisfied when the variable is set', () => {
    expect(missingProviderKey(['openai'], { OPENAI_API_KEY: 'x' })).toBeUndefined();
  });

  it('falls back to OpenAI when no LLM skill is declared', () => {
    expect(missingProviderKey([], {})?.envVar).toBe('OPENAI_API_KEY');
  });

  it('asks nothing of a provider that needs no key', () => {
    expect(missingProviderKey(['webllm'], {})).toBeUndefined();
  });
});

describe('the REPL on a scaffold-shaped project', () => {
  const cwd = process.cwd();
  // The chat reads stored keys and the sign-in at start (model-access.ts), so
  // it runs under a throwaway HOME and the file backend: never the keychain.
  const ISOLATED = ['OPENAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE'];
  const saved: Record<string, string | undefined> = {};
  beforeEach(() => {
    for (const name of ISOLATED) saved[name] = process.env[name];
    delete process.env.WEBAGENTS_TOKEN;
    delete process.env.WEBAGENTS_PROFILE;
    process.env.HOME = tempDir('wa-first-run-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  });
  afterEach(() => {
    process.chdir(cwd);
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  });

  function scaffold(): string {
    const dir = tempDir('wa-first-run-');
    fs.writeFileSync(
      path.join(dir, 'AGENT.md'),
      '---\nname: first-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\n\nYou help.\n',
    );
    return dir;
  }

  it('runs the model the file declares', async () => {
    process.chdir(scaffold());
    process.env.OPENAI_API_KEY = 'dummy-not-a-key';
    const repl = new InteractiveREPL({});
    await repl.initialize();
    const agent = (repl as unknown as { agent: { skills: { name?: string }[] } }).agent;
    // The agent also carries built-in skills; the LLM one is named for its provider.
    const llm = agent.skills.filter((skill) => skill.name === 'openai');
    expect(llm.map(modelOfSkill)).toEqual(['openai/gpt-4o-mini']);
    expect(repl.modelProblem).toBeUndefined();
  });

  it('reports the missing key instead of failing on the first message', async () => {
    process.chdir(scaffold());
    delete process.env.OPENAI_API_KEY;
    const repl = new InteractiveREPL({});
    await repl.initialize();
    expect(repl.modelProblem).toContain('OPENAI_API_KEY is not set');
  });
});

describe('the agent loop trace', () => {
  afterEach(() => setAgentTrace({ enabled: true, sink: (line) => console.log(line) }));

  it('stays on by default, for the hosts that read it', () => {
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    agentTrace('[agent] iteration 1/50');
    expect(log).toHaveBeenCalledWith('[agent] iteration 1/50');
    log.mockRestore();
  });

  it('is silent when the host turns it off', () => {
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    setAgentTrace({ enabled: false });
    agentTrace('[agent] iteration 1/50');
    expect(log).not.toHaveBeenCalled();
    log.mockRestore();
  });

  it('goes where the host sends it', () => {
    const lines: string[] = [];
    setAgentTrace({ enabled: true, sink: (line) => lines.push(line) });
    agentTrace('[agent] iteration 1/50');
    expect(lines).toEqual(['[agent] iteration 1/50']);
  });

  it('is the only way the loop writes a trace line', () => {
    // A new bare console.log in the loop would reach the CLI's stdout again.
    const source = fs.readFileSync(path.resolve(__dirname, '../../../src/core/agent.ts'), 'utf-8');
    const offenders = source
      .split('\n')
      .map((line, i) => ({ line: line.replace(/\/\/.*$/, ''), n: i + 1 }))
      .filter(({ line }) => /console\.log\(/.test(line))
      .map(({ n }) => `agent.ts:${n}`);
    expect(offenders).toEqual([]);
  });
});

describe('the CLI itself, spawned', () => {
  const CLI = path.resolve(__dirname, '../../../src/cli/index.ts');
  // A clean HOME and no provider key, so the machine running the suite does
  // not leak into what a first-time user would see.
  function runCli(args: string[], cwd: string, env: Record<string, string> = {}) {
    const home = tempDir('wa-first-run-home-');
    // The file backend: a throwaway HOME must not reach the keychain.
    const base = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', ...env } as Record<string, string | undefined>;
    delete base.OPENAI_API_KEY;
    delete base.WEBAGENTS_DEBUG;
    delete base.WEBAGENTS_TOKEN;
    return spawnSync(process.execPath, [TSX_CLI, CLI, ...args], {
      cwd,
      env: base as NodeJS.ProcessEnv,
      encoding: 'utf-8',
      timeout: 60_000,
    });
  }

  it('lists only the templates init can make', () => {
    const out = runCli(['templates', 'list'], os.tmpdir());
    expect(out.status).toBe(0);
    expect(out.stdout).toContain('chatbot');
    expect(out.stdout).toContain('tool-agent');
    // THE BUG: these were listed, and init made the same scaffold for all of them.
    expect(out.stdout).not.toContain('rag-agent');
    expect(out.stdout).not.toContain('mcp-agent');
  });

  it('refuses a template it cannot make, and creates nothing', () => {
    const dir = tempDir('wa-first-run-init-');
    const out = runCli(['init', 'x', '--template', 'rag-agent'], dir);
    expect(out.status).toBe(1);
    expect(out.stderr).toContain("Unknown template 'rag-agent'");
    expect(fs.existsSync(path.join(dir, 'x'))).toBe(false);
  });

  it('answers a missing key with one line and exit 1, not a stack trace', () => {
    const dir = tempDir('wa-first-run-p-');
    fs.writeFileSync(
      path.join(dir, 'AGENT.md'),
      '---\nname: first-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\n\nYou help.\n',
    );
    const out = runCli(['-p', 'hello'], dir);
    expect(out.status).toBe(1);
    expect(out.stderr).toContain('OPENAI_API_KEY is not set');
    expect(out.stderr).not.toMatch(/\n\s+at /);
    expect(out.stdout).toBe('');
  });
});

describe('S-227: the trace carries lengths, not content', () => {
  const saved = process.env.LOG_LOOP_DEBUG;
  afterEach(() => {
    if (saved === undefined) delete process.env.LOG_LOOP_DEBUG;
    else process.env.LOG_LOOP_DEBUG = saved;
  });

  it('keeps reply text and tool arguments out of the loop trace unless asked', () => {
    const source = fs.readFileSync(path.resolve(__dirname, '../../../src/core/agent.ts'), 'utf-8');
    // THE BUG: `text=${...}` and `args=${tc.arguments...}` were unconditional.
    // Each now sits behind traceContent(), and nowhere else interpolates them.
    expect(source).toMatch(/traceContent\(\)\s*\?\s*`args=\$\{tc\.arguments/);
    expect(source).toMatch(/traceContent\(\)\s*\?\s*`text=\$\{JSON\.stringify/);
    expect(source.match(/args=\$\{tc\.arguments/g)).toHaveLength(1);
    expect(source.match(/`text=\$\{JSON\.stringify\(\(delta/g)).toHaveLength(1);
  });

  it('is off unless LOG_LOOP_DEBUG=1', async () => {
    const { traceContent } = await import('../../../src/core/trace');
    delete process.env.LOG_LOOP_DEBUG;
    expect(traceContent()).toBe(false);
    process.env.LOG_LOOP_DEBUG = '1';
    expect(traceContent()).toBe(true);
  });
});
