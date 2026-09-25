/**
 * The REPL runs the model the agent file declares (2026-09-24).
 *
 * The constructor defaulted `config.model` to 'gpt-4o' BEFORE the file was
 * read, and the load step did `config.model ?? declaredModel`, which is never
 * nullish after that, so an AGENT.md's `model:` never applied in the TypeScript
 * REPL. Intended order, per the code's own comment: an explicit `--model`, then
 * the file, then the REPL default. Found by the onboarding docs audit.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { InteractiveREPL } from '../../../src/cli/app';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const cwd = process.cwd();
// The chat reads stored keys and the sign-in at start (model-access.ts), so it
// runs under a throwaway HOME and the file backend: never the keychain.
const ISOLATED = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE'];
const saved: Record<string, string | undefined> = {};
beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-repl-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
});
afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
});

function projectWithModel(model: string): string {
  const dir = tempDir('wa-repl-');
  fs.writeFileSync(
    path.join(dir, 'AGENT.md'),
    `---\nname: helper\nmodel: ${model}\nskills: []\n---\n\nYou help.\n`,
  );
  return dir;
}

function modelOf(repl: InteractiveREPL): unknown {
  return (repl as unknown as { config: { model?: string } }).config.model;
}

describe('which model the REPL runs', () => {
  it("uses the agent file's model when no --model is given", async () => {
    process.chdir(projectWithModel('anthropic/claude-sonnet-4'));
    const repl = new InteractiveREPL({});
    await repl.initialize();
    // THE BUG: this was always 'gpt-4o'.
    expect(modelOf(repl)).toBe('anthropic/claude-sonnet-4');
  });

  it('lets an explicit --model win over the file', async () => {
    process.chdir(projectWithModel('anthropic/claude-sonnet-4'));
    const repl = new InteractiveREPL({ model: 'openai/gpt-4o-mini' });
    await repl.initialize();
    expect(modelOf(repl)).toBe('openai/gpt-4o-mini');
  });
});

describe('/model in a session', () => {
  it('switches away from the model the file declares, and stays switched', async () => {
    process.chdir(projectWithModel('openai/gpt-4o-mini'));
    // A key, so both models can run here; a switch to one that cannot is refused.
    process.env.OPENAI_API_KEY = 'sk-test';
    const repl = new InteractiveREPL({});
    await repl.initialize();
    const commands = (repl as unknown as {
      commands: Map<string, { handler: (args: string) => Promise<void> }>;
    }).commands;
    await commands.get('model')!.handler('openai/gpt-4.1-mini');
    // THE BUG: initialize() re-applied the file's model, so this read
    // 'openai/gpt-4o-mini' again while the REPL said it had switched.
    expect(modelOf(repl)).toBe('openai/gpt-4.1-mini');
    const agent = (repl as unknown as { agent: { skills: { name?: string; getCapabilities?: () => { id: string } }[] } }).agent;
    const llm = agent.skills.find((skill) => skill.name === 'openai');
    expect(llm?.getCapabilities?.().id).toBe('openai/gpt-4.1-mini');
  });
});
