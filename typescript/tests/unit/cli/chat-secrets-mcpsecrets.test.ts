/**
 * The chat's `/secrets` (S-292, 2026-09-26), pinned by `chat` in
 * `python/tests/fixtures/cli/secrets.json` (the Python suite runs the same,
 * `tests/cli/test_chat_secrets_mcpsecrets.py`): it lists the names in the
 * store an MCP server's `${secret:NAME}` reads and how to add one; `/secrets
 * set NAME` takes the value at a hidden local prompt, like `/keys set`, and
 * the value goes to the store and nowhere else: not to the model, not to
 * the environment, not to the screen. Scratch HOME, file backend, never the
 * keychain.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const H = vi.hoisted(() => ({ secrets: [] as Array<string | null>, prompts: [] as string[] }));
vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async (prompt: string) => {
    H.prompts.push(prompt);
    return H.secrets.length ? H.secrets.shift()! : '';
  }),
  promptSecretOrPipe: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = (JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/secrets.json'), 'utf8')) as { chat: Record<string, string> & { where: Record<string, string>; stored_where: Record<string, string> } }).chat;
const VALUE = 'dummy-github-token-not-a-real-credential';

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS', 'GITHUB_TOKEN',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let project = '';
let printed: string[] = [];

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-chat-secrets-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '200';
  project = tempDir('wa-chat-secrets-project-');
  process.chdir(project);
  fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: helper\n---\nHelp.\n');
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  H.secrets = [];
  H.prompts = [];
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

type Inside = { handleInput(line: string): Promise<void>; messages: unknown[]; modelProblem: string | undefined };

async function chat(): Promise<Inside> {
  const repl = new InteractiveREPL({ interactive: true });
  await repl.initialize();
  return repl as unknown as Inside;
}

async function say(repl: Inside, line: string): Promise<string> {
  const before = printed.length;
  await repl.handleInput(line);
  return printed.slice(before).join('\n');
}

async function stored(name: string): Promise<string | null> {
  const { providerKeyStore } = await import('../../../src/cli/provider-keys');
  return (await providerKeyStore()).get(name);
}

describe('/secrets', () => {
  it('lists none and says how to add one', async () => {
    const out = await say(await chat(), '/secrets');
    expect(out).toContain(FIXTURE.heading);
    expect(out).toContain(FIXTURE.none);
    expect(out).toContain(FIXTURE.hint);
  });

  it('set takes the value at a hidden prompt and keeps it out of everything else', async () => {
    const repl = await chat();
    H.secrets = [VALUE];
    const out = await say(repl, '/secrets set GITHUB_TOKEN');
    expect(H.prompts).toHaveLength(1);
    expect(H.prompts[0]).toContain(FIXTURE.prompt.replace('{name}', 'GITHUB_TOKEN').trim());
    expect(out).toContain(FIXTURE.stored.replace('{name}', 'GITHUB_TOKEN').replace('{where}', FIXTURE.stored_where.file));
    expect(out).toContain(FIXTURE.stored_detail.replace('{name}', 'GITHUB_TOKEN'));
    expect(out).not.toContain(VALUE);
    // Not in the conversation, and not in the environment the shell tool inherits.
    expect(repl.messages).toEqual([]);
    expect(process.env.GITHUB_TOKEN).toBeUndefined();
    expect(await stored('GITHUB_TOKEN')).toBe(VALUE);

    const listing = await say(repl, '/secrets');
    expect(listing).toContain('GITHUB_TOKEN');
    expect(listing).toContain(FIXTURE.where.file);
    expect(listing).not.toContain(VALUE);

    expect(await say(repl, '/secrets remove GITHUB_TOKEN')).toContain(FIXTURE.removed.replace('{name}', 'GITHUB_TOKEN'));
    expect(await say(repl, '/secrets remove GITHUB_TOKEN')).toContain(FIXTURE.not_stored.replace('{name}', 'GITHUB_TOKEN'));
    expect(await stored('GITHUB_TOKEN')).toBeNull();
  });

  it('nothing entered stores nothing, and a cancelled prompt too', async () => {
    const repl = await chat();
    H.secrets = ['   '];
    expect(await say(repl, '/secrets set GITHUB_TOKEN')).toContain(FIXTURE.nothing_entered);
    H.secrets = [null];
    expect(await say(repl, '/secrets set GITHUB_TOKEN')).toContain(FIXTURE.nothing_entered);
    expect(await stored('GITHUB_TOKEN')).toBeNull();
  });

  it('a bad name and a bad verb are refused', async () => {
    const repl = await chat();
    expect(await say(repl, '/secrets set bad-name')).toContain(FIXTURE.bad_name.replace('{name}', 'bad-name'));
    expect(await say(repl, '/secrets bogus')).toContain(FIXTURE.usage_error);
    expect(await say(repl, '/secrets set')).toContain(FIXTURE.usage_error);
  });

  it('a provider key stored this way reaches the model as /keys set would', async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n');
    const repl = await chat();
    expect(repl.modelProblem).toBeDefined();
    H.secrets = ['sk-dummy-provider-key'];
    await say(repl, '/secrets set OPENAI_API_KEY');
    expect(repl.modelProblem).toBeUndefined();
    expect(process.env.OPENAI_API_KEY).toBeUndefined();
  });
});
