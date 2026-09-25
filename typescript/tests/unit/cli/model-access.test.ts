/**
 * Which model the TypeScript chat runs on, and what it offers when there is
 * none (2026-09-24, `src/cli/model-access.ts`, `src/cli/app.ts`).
 *
 * The chat attached `OpenAISkill` to every agent that named no LLM skill, so
 * with no OPENAI_API_KEY the first reply was "OpenAI API key not configured":
 * for a person signed in to Robutler, which serves models, and for one with
 * an Anthropic key exported. The card said "Exit, export it, then start
 * again". Pinned here: the decision, the agent the chat builds from it (the
 * proxy paid by the signed-in person), keys kept in the store both CLIs read,
 * and the offer to sign in or take a key when nothing works.
 *
 * Every test runs under a throwaway HOME with the FILE secrets backend, so
 * nothing here reads or writes the machine's keychain or its real login.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const H = vi.hoisted(() => ({
  lines: [] as Array<string | null>,
  secrets: [] as string[],
  login: vi.fn(),
}));

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => (H.lines.length ? H.lines.shift()! : null)),
  promptSecret: vi.fn(async () => H.secrets.shift() ?? ''),
}));
vi.mock('../../../src/cli/browser-login.js', () => ({ browserLogin: H.login }));

import { InteractiveREPL } from '../../../src/cli/app';
import {
  describeAccess,
  platformLlmUrl,
  resolveModelAccess,
  unavailableMessage,
} from '../../../src/cli/model-access';
import { readStoredProviderKeys, storeProviderKey } from '../../../src/cli/provider-keys';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const YES = () => true;
const NO = () => false;
const KEY_VARS = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY'];
const ISOLATED = [
  ...KEY_VARS,
  'HOME',
  'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN',
  'WEBAGENTS_PROFILE',
  'ROBUTLER_API_URL',
  'ROBUTLER_LLM_PROXY_URL',
];

const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-model-access-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'https://portal.example';
  H.lines = [];
  H.secrets = [];
  H.login.mockReset();
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

function project(frontMatter: string): string {
  const dir = tempDir('wa-model-access-');
  fs.writeFileSync(path.join(dir, 'AGENT.md'), `---\nname: helper\n${frontMatter}---\n\nYou help.\n`);
  return dir;
}

type Inside = {
  agent: { skills: Array<{ name?: string; getCapabilities?: () => { id: string } }> };
  modelLabel(): string;
  keyCandidates(): Array<{ id: string }>;
  offerModelAccess(): Promise<void>;
  handleInput(line: string): Promise<void>;
  welcomeInfo(): { model?: string; warnings: string[] };
  commands: Map<string, { handler: (args: string) => Promise<void> }>;
};
const inside = (repl: InteractiveREPL) => repl as unknown as Inside;

function llmSkills(repl: InteractiveREPL) {
  const names = new Set(['openai', 'anthropic', 'google', 'xai', 'llm-proxy']);
  return inside(repl).agent.skills.filter((skill) => skill.name && names.has(skill.name));
}

function proxyOf(repl: InteractiveREPL) {
  const skill = llmSkills(repl).find((s) => s.name === 'llm-proxy') as unknown as
    | { proxyUrl: string; modelConfig: { model?: string; platformToken?: () => Promise<string | undefined> } }
    | undefined;
  return skill;
}

async function chatIn(dir: string, config = {}): Promise<InteractiveREPL> {
  process.chdir(dir);
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  const repl = new InteractiveREPL(config);
  await repl.initialize();
  return repl;
}

describe('the decision', () => {
  it('runs the declared model directly with its key', async () => {
    const access = await resolveModelAccess('openai/gpt-4o', { signedIn: NO, env: { OPENAI_API_KEY: 'sk' } });
    expect([access.kind, access.model]).toEqual(['direct', 'openai/gpt-4o']);
  });

  it('runs the SAME model through Robutler when its key is missing and the person is signed in', async () => {
    const access = await resolveModelAccess('anthropic/claude-x', { signedIn: YES, env: {} });
    expect([access.kind, access.model, access.reason]).toEqual(['proxy', 'anthropic/claude-x', 'ANTHROPIC_API_KEY is not set']);
    expect(describeAccess(access)).toBe('anthropic/claude-x via Robutler');
  });

  it('with neither, is none, and names both ways out', async () => {
    const access = await resolveModelAccess('openai/gpt-4o', { signedIn: NO, env: {} });
    expect(access.kind).toBe('none');
    const message = unavailableMessage(access);
    expect(message).toContain('webagents login');
    expect(message).toContain('set OPENAI_API_KEY');
  });

  it("takes whichever provider has a key when no model is named (not OpenAI's only)", async () => {
    const access = await resolveModelAccess(undefined, { signedIn: YES, env: { XAI_API_KEY: 'k' } });
    expect(access.kind).toBe('direct');
    expect(access.model).toMatch(/^xai\//);
  });

  it("is Robutler's own choice with no model, no key and a sign-in", async () => {
    const access = await resolveModelAccess(undefined, { signedIn: YES, env: {} });
    expect([access.kind, access.model]).toEqual(['proxy', 'auto/balanced']);
  });

  it('lists every key it could use when there is nothing', async () => {
    const message = unavailableMessage(await resolveModelAccess(undefined, { signedIn: NO, env: {} }));
    for (const name of ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY']) {
      expect(message).toContain(name);
    }
  });

  it.each([
    ['auto/fast', 'auto/fast'],
    ['proxy/openai/gpt-4o', 'openai/gpt-4o'],
    ['robutler/anthropic/claude-x', 'anthropic/claude-x'],
    ['proxy/', 'auto/balanced'],
  ])('sends %s, which only Robutler serves, as %s', async (declared, sent) => {
    const access = await resolveModelAccess(declared, { signedIn: YES, env: { OPENAI_API_KEY: 'sk' } });
    expect([access.kind, access.model]).toEqual(['proxy', sent]);
    const none = await resolveModelAccess(declared, { signedIn: NO, env: { OPENAI_API_KEY: 'sk' } });
    expect(none.kind).toBe('none');
    // A key would not help, so none is offered.
    expect(unavailableMessage(none)).not.toContain('OPENAI_API_KEY');
  });

  it('sends a provider this SDK has no client for through Robutler rather than calling nothing', async () => {
    const env = { MISTRAL_API_KEY: 'm' };
    const signedIn = await resolveModelAccess('mistral/mistral-large', { signedIn: YES, env });
    expect([signedIn.kind, signedIn.model]).toEqual(['proxy', 'mistral/mistral-large']);
    const signedOut = await resolveModelAccess('mistral/mistral-large', { signedIn: NO, env });
    expect(signedOut.kind).toBe('none');
    expect(signedOut.reason).toContain('no built-in client for mistral');
  });

  it('runs Fireworks with its key, as the Python SDK does (2026-09-25)', async () => {
    const access = await resolveModelAccess('fireworks/deepseek-v3p2', { signedIn: NO, env: { FIREWORKS_API_KEY: 'fw' } });
    expect([access.kind, access.model]).toEqual(['direct', 'fireworks/deepseek-v3p2']);
  });

  it('does not ask about the sign-in when a key already answers', async () => {
    const never = () => {
      throw new Error('asked for a sign-in that a key already answers');
    };
    expect((await resolveModelAccess('openai/gpt-4o', { signedIn: never, env: { OPENAI_API_KEY: 'sk' } })).kind).toBe('direct');
  });

  it("finds the platform's /llm socket from the platform URL", () => {
    expect(platformLlmUrl('https://robutler.ai/')).toBe('wss://robutler.ai/llm');
    expect(platformLlmUrl('http://localhost:3000')).toBe('ws://localhost:3000/llm');
    process.env.ROBUTLER_LLM_PROXY_URL = 'ws://elsewhere/llm';
    expect(platformLlmUrl('https://robutler.ai')).toBe('ws://elsewhere/llm');
  });
});

describe('the proxy loader keeps the model', () => {
  it("passes another provider's model through, since the proxy serves them all", async () => {
    const { skills } = await resolveSkillsByName(['proxy'], {
      model: 'openai/gpt-4o',
      proxy: { proxyUrl: 'wss://portal.example/llm', platformToken: 'jwt' },
    });
    const skill = skills[0] as unknown as { proxyUrl: string; modelConfig: { model?: string; platformToken?: unknown } };
    expect(skill.modelConfig.model).toBe('openai/gpt-4o');
    expect(skill.proxyUrl).toBe('wss://portal.example/llm');
    expect(skill.modelConfig.platformToken).toBe('jwt');
  });

  it('strips the `proxy/` route prefix, which the platform does not', async () => {
    const { skills } = await resolveSkillsByName(['proxy'], { model: 'proxy/anthropic/claude-x' });
    expect((skills[0] as unknown as { modelConfig: { model?: string } }).modelConfig.model).toBe('anthropic/claude-x');
  });
});

describe('the agent the chat builds', () => {
  it('has no model, and says how to get one, with no key and no sign-in', async () => {
    const repl = await chatIn(project('skills: []\n'));
    expect(llmSkills(repl)).toEqual([]);
    expect(repl.modelProblem).toContain('webagents login');
    expect(inside(repl).welcomeInfo().warnings).toEqual([repl.modelProblem]);
    // THE OLD CARD: "Exit, export it, then start again."
    expect(repl.modelProblem).not.toContain('Exit');
  });

  it("runs on Robutler's models, paid by the signed-in person, with no key", async () => {
    process.env.WEBAGENTS_TOKEN = 'jwt.login';
    const repl = await chatIn(project('skills: []\n'));
    expect(repl.modelProblem).toBeUndefined();
    const proxy = proxyOf(repl)!;
    expect(proxy.proxyUrl).toBe('wss://portal.example/llm');
    expect(proxy.modelConfig.model).toBe('auto/balanced');
    // Read per request, so a /login mid-session reaches the built agent.
    expect(await proxy.modelConfig.platformToken!()).toBe('jwt.login');
    expect(inside(repl).modelLabel()).toBe('auto/balanced via Robutler');
    expect(llmSkills(repl).map((s) => s.name)).toEqual(['llm-proxy']);
  });

  it('runs the file\'s model through Robutler when only its key is missing', async () => {
    process.env.WEBAGENTS_TOKEN = 'jwt.login';
    const repl = await chatIn(project('model: anthropic/claude-x\nskills: []\n'));
    expect(proxyOf(repl)!.modelConfig.model).toBe('anthropic/claude-x');
  });

  it('uses an Anthropic key when that is the key there is', async () => {
    process.env.ANTHROPIC_API_KEY = 'sk-ant-test';
    const repl = await chatIn(project('skills: []\n'));
    expect(llmSkills(repl).map((s) => s.name)).toEqual(['anthropic']);
    expect(repl.modelProblem).toBeUndefined();
  });

  it('uses a key stored with `secrets set` (by either CLI), and never over an exported one', async () => {
    await storeProviderKey('OPENAI_API_KEY', 'sk-stored');
    const repl = await chatIn(project('skills: []\n'));
    const [openai] = llmSkills(repl) as unknown as Array<{ name: string; modelConfig: { apiKey?: string } }>;
    expect(openai.name).toBe('openai');
    expect(openai.modelConfig.apiKey).toBe('sk-stored');

    process.env.OPENAI_API_KEY = 'sk-exported';
    expect(await readStoredProviderKeys()).toEqual({});
  });

  it('keeps a stored key out of the environment, which the shell skill hands to every command', async () => {
    // S-220: an unsandboxed command inherits the whole environment.
    await storeProviderKey('ANTHROPIC_API_KEY', 'sk-ant-stored');
    const repl = await chatIn(project('model: anthropic/claude-x\nskills:\n  - anthropic\n  - shell\n'));
    expect(repl.modelProblem).toBeUndefined();
    expect(process.env.ANTHROPIC_API_KEY).toBeUndefined();
    const shell = inside(repl).agent.skills.find((s) => s.name === 'ShellSkill') as unknown as {
      runCommand(p: { command: string }, c: unknown): Promise<string>;
    };
    expect(await shell.runCommand({ command: 'echo "[$ANTHROPIC_API_KEY]"' }, {})).toContain('[]');
  });

  it("keeps an agent's own LLM skill when its key is here", async () => {
    process.env.WEBAGENTS_TOKEN = 'jwt.login';
    process.env.OPENAI_API_KEY = 'sk-test';
    const repl = await chatIn(project('model: openai/gpt-4o-mini\nskills:\n  - openai\n'));
    // Signed in too, but the agent chose OpenAI and can run on it.
    expect(proxyOf(repl)).toBeUndefined();
    expect(llmSkills(repl).map((s) => s.getCapabilities?.().id)).toEqual(['openai/gpt-4o-mini']);
  });

  it("runs the scaffold's model through Robutler when its key is missing and the person is signed in", async () => {
    // `init` writes `skills: [openai]`; a first run without the key is exactly this.
    process.env.WEBAGENTS_TOKEN = 'jwt.login';
    const repl = await chatIn(project('model: openai/gpt-4o-mini\nskills:\n  - openai\n'));
    expect(repl.modelProblem).toBeUndefined();
    expect(llmSkills(repl).map((s) => s.name)).toEqual(['llm-proxy']);
    expect(proxyOf(repl)!.modelConfig.model).toBe('openai/gpt-4o-mini');
  });

  it('offers both ways out for a named provider without its key or a sign-in', async () => {
    const repl = await chatIn(project('model: openai/gpt-4o-mini\nskills:\n  - openai\n'));
    expect(repl.modelProblem).toContain('OPENAI_API_KEY is not set');
    expect(repl.modelProblem).toContain('webagents login');
    expect(inside(repl).keyCandidates().map((p) => p.id)).toEqual(['openai']);
    // The OpenAI skill would only fail; it is not left in the agent.
    expect(llmSkills(repl)).toEqual([]);
  });

  it('gives a declared proxy skill the sign-in, and asks for one when there is none', async () => {
    const signedOut = await chatIn(project('skills:\n  - proxy\n'));
    expect(signedOut.modelProblem).toContain('webagents login');
    process.env.WEBAGENTS_TOKEN = 'jwt.login';
    const signedIn = await chatIn(project('model: openai/gpt-4o\nskills:\n  - proxy\n'));
    expect(signedIn.modelProblem).toBeUndefined();
    expect(proxyOf(signedIn)!.modelConfig.model).toBe('openai/gpt-4o');
  });

  it('does not send a message it has no model for', async () => {
    const repl = await chatIn(project('skills: []\n'));
    const run = vi.fn();
    (inside(repl).agent as unknown as { runStreaming: unknown }).runStreaming = run;
    await inside(repl).handleInput('hello');
    expect(run).not.toHaveBeenCalled();
  });

  it('does not switch /model to one it cannot run', async () => {
    process.env.OPENAI_API_KEY = 'sk-test';
    const repl = await chatIn(project('model: openai/gpt-4o-mini\nskills: []\n'));
    await inside(repl).commands.get('model')!.handler('auto/fast');
    expect(repl.modelProblem).toBeUndefined();
    expect(llmSkills(repl).map((s) => s.getCapabilities?.().id)).toEqual(['openai/gpt-4o-mini']);
  });
});

describe("what a failed turn on Robutler's models says to do", () => {
  // The full table both SDKs run is tests/unit/cli/failures.test.ts; this
  // pins that the chat signed in to Robutler applies it.
  it.each([
    ['Not enough credits to start a model call. Add credits in Robutler, or use your own provider key.', 'Not enough credits', 'Add credits'],
    ['session.create requires X-Payment-Token, or the Bearer token `webagents login` stores as Authorization, in session.extensions', 'did not accept your sign-in', '/login'],
    ['Invalid or expired payment token', 'did not accept your sign-in', '/login'],
    ['session.create requires X-Payment-Token in session.extensions', 'does not run models for a CLI sign-in', 'webagents secrets set OPENAI_API_KEY'],
  ])('%s', async (message, headline, hint) => {
    process.env.WEBAGENTS_TOKEN = 'jwt.login';
    const repl = await chatIn(project('skills: []\n'));
    const explained = repl.explainFailure(message);
    expect(explained.headline).toContain(headline);
    expect(explained.hint).toContain(hint);
  });
});

describe('the offer when there is no model', () => {
  it('signs in through the browser, keeps the token, and runs on Robutler', async () => {
    const repl = await chatIn(project('skills: []\n'));
    H.lines = ['1'];
    H.login.mockResolvedValue({ token: 'jwt.fresh', username: 'dev' });
    await inside(repl).offerModelAccess();
    expect(H.login).toHaveBeenCalledWith('https://portal.example', expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(repl.modelProblem).toBeUndefined();
    expect(await proxyOf(repl)!.modelConfig.platformToken!()).toBe('jwt.fresh');
    // Kept in the profile's store (the file backend here), for the next start.
    const { getToken } = await import('../../../src/cli/credentials');
    expect(await getToken()).toBe('jwt.fresh');
  });

  it('takes a key with echo off, keeps it for next time, and runs on that provider', async () => {
    const repl = await chatIn(project('skills: []\n'));
    H.lines = ['2', '2']; // "Enter a provider key", then anthropic (second in the list)
    H.secrets = ['sk-ant-entered'];
    await inside(repl).offerModelAccess();
    const [anthropic] = llmSkills(repl) as unknown as Array<{ name: string; modelConfig: { apiKey?: string } }>;
    expect([anthropic.name, anthropic.modelConfig.apiKey]).toEqual(['anthropic', 'sk-ant-entered']);
    // To the model client only, and kept for next time.
    expect(process.env.ANTHROPIC_API_KEY).toBeUndefined();
    expect(await readStoredProviderKeys()).toEqual({ ANTHROPIC_API_KEY: 'sk-ant-entered' });
  });

  it('asks for the one key an agent that chose its provider needs', async () => {
    const repl = await chatIn(project('model: openai/gpt-4o-mini\nskills:\n  - openai\n'));
    H.lines = ['2']; // "Enter OPENAI_API_KEY": no provider question for this agent
    H.secrets = ['sk-entered'];
    await inside(repl).offerModelAccess();
    expect(H.login).not.toHaveBeenCalled();
    expect(repl.modelProblem).toBeUndefined();
    expect(llmSkills(repl).map((s) => s.getCapabilities?.().id)).toEqual(['openai/gpt-4o-mini']);
  });

  it('carries on without a model when asked to, or on Ctrl+D', async () => {
    const repl = await chatIn(project('skills: []\n'));
    H.lines = ['3'];
    await inside(repl).offerModelAccess();
    H.lines = [null];
    await inside(repl).offerModelAccess();
    expect(H.login).not.toHaveBeenCalled();
    expect(repl.modelProblem).toContain('webagents login');
  });

  it('says so when the sign-in is cancelled, and keeps the chat', async () => {
    const repl = await chatIn(project('skills: []\n'));
    H.login.mockRejectedValue(new Error('Sign-in was cancelled in the browser.'));
    await inside(repl).commands.get('login')!.handler('');
    const printed = (console.log as unknown as { mock: { calls: unknown[][] } }).mock.calls.flat().join('\n');
    expect(printed).toContain('Sign-in was cancelled in the browser.');
    expect(repl.modelProblem).toContain('webagents login');
  });
});
