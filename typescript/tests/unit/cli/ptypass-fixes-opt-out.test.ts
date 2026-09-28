/**
 * The opt-out is said in one wording, and as a notice in the chat (the
 * ptypass-fixes lane, 2026-09-27, brief item 12; fixture
 * `python/tests/fixtures/sandbox/srt.json` `unrestricted`,
 * `status.warnings`). The Python twin is
 * `python/tests/cli/test_ptypass_fixes_opt_out.py`.
 *
 * WHY. The real-terminal PTY pass saw the `sandbox: off` load line printed
 * raw above the welcome card, wrapped mid-word ("Use `deve" / "lopment`"),
 * and saying "Use development or strict" while `/sandbox` said "Remove
 * `sandbox: off`".
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { CHAT_WORDS, fill } from '../../../src/cli/chat-words';
import { NO_SANDBOX_WARNING, UNRESTRICTED_WARNING } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS', 'WEBAGENTS_NO_SANDBOX',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let printed: string[] = [];
let warned: string[] = [];
const OFF = '---\nname: helper\nskills:\n  - openai\n  - shell\nsandbox: off\n---\nHelp.\n';

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-optout-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '60';
  const project = tempDir('wa-optout-project-');
  process.chdir(project);
  fs.writeFileSync(path.join(project, 'AGENT.md'), OFF);
  printed = [];
  warned = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => printed.push(args.map(String).join(' ')));
  vi.spyOn(console, 'warn').mockImplementation((...args: unknown[]) => warned.push(args.map(String).join(' ')));
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

describe('one wording', () => {
  it('the load line is /sandbox\'s headline and fix', () => {
    expect(UNRESTRICTED_WARNING).toBe(`Sandbox: ${fill('sandboxOff', { state: 'off (agent file)' })} ${CHAT_WORDS.sandboxOffFix}`);
    expect(NO_SANDBOX_WARNING).toBe(`Sandbox: ${fill('sandboxOff', { state: 'off (--no-sandbox)' })} ${CHAT_WORDS.sandboxFlagFix}`);
    expect(UNRESTRICTED_WARNING).not.toContain('Use `development` or `strict`');
  });
});

describe('where it is said', () => {
  it('in the chat, as /sandbox\'s notice, wrapped at words, and not as the raw line', async () => {
    const repl = new InteractiveREPL({ interactive: true }) as unknown as { chatting: boolean; initialize(): Promise<void> };
    repl.chatting = true;
    await repl.initialize();
    const text = printed.join('\n');
    expect(text.replace(/\s+/g, ' ')).toContain('▲ Sandbox: off (agent file): commands are not confined');
    expect(text.replace(/\s+/g, ' ')).toContain(CHAT_WORDS.sandboxOffFix);
    expect(warned.join('\n')).not.toContain(UNRESTRICTED_WARNING);
    for (const line of text.split('\n')) expect(line.length, line).toBeLessThanOrEqual(60);
  });

  it('outside the chat, on stderr as before', async () => {
    const repl = new InteractiveREPL({}) as unknown as { initialize(): Promise<void> };
    await repl.initialize();
    expect(warned).toContain(UNRESTRICTED_WARNING);
    expect(printed.join('\n')).not.toContain('Sandbox: off');
  });
});
