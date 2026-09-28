/**
 * The chat's commands that read or change the agent file (2026-09-26,
 * interactive-mode spec section 3), against the shared fixture
 * `python/tests/fixtures/cli/chat_edits.json`, which the Python suite reads
 * too (`tests/cli/test_chat_edits_interactive.py`): the words both chats say,
 * the loader's refusals, and each command driven through `handleInput` with
 * the prompts mocked, as `chat-commands.test.ts` does.
 *
 * Everything runs under a throwaway HOME, the FILE secrets backend, no key
 * (unless a case sets one) and no sign-in, at NO_COLOR and COLUMNS=100.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const H = vi.hoisted(() => ({ lines: [] as Array<string | null>, prompts: [] as string[] }));
vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async (prompt: string) => {
    H.prompts.push(prompt);
    return H.lines.length ? H.lines.shift()! : null;
  }),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { CHAT_WORDS } from '../../../src/cli/chat-words';
import { AgentFileError, parseAgentMarkdown, readAgentFile } from '../../../src/agents/index';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_edits.json'), 'utf8'),
) as {
  words: Record<string, string>;
  loader: {
    not_yaml: string;
    not_yaml_at: string;
    unknown_key: string;
    cron_string: string;
    linked: string;
    known_keys: string;
    cases: Array<Record<string, unknown>>;
  };
  cases: Array<Record<string, unknown>>;
  offers: Array<{ case: string; files: Record<string, string>; answers: Array<string | null>; printed: string[] }>;
};

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'XAI_API_KEY', 'HOME',
  'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL',
  'NO_COLOR', 'COLUMNS', 'VISUAL', 'EDITOR', 'WEBAGENTS_SRT_CLI', 'WEBAGENTS_SRT_NODE',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let home = '';
let project = '';
let printed: string[] = [];

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  home = tempDir('wa-edits-home-');
  process.env.HOME = home;
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '240';
  project = tempDir('wa-edits-project-');
  process.chdir(project);
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => printed.push(args.map(String).join(' ')));
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => printed.push(args.map(String).join(' ')));
  H.lines = [];
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

function writeFiles(folder: string, files: Record<string, string>): void {
  for (const [name, text] of Object.entries(files ?? {})) {
    const full = path.join(folder, name);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, text);
  }
}

/** A tiny editor script the chat launches: `append` adds a line, `fail` exits 3. */
function editorScript(kind: string): string {
  const file = path.join(home, `editor-${kind}.sh`);
  const body = kind === 'append' ? `#!/bin/sh\nprintf 'More.\\n' >> "$1"\n` : `#!/bin/sh\nexit 3\n`;
  fs.writeFileSync(file, body, { mode: 0o755 });
  return file;
}

function printedLines(): string[] {
  return printed.flatMap((chunk) => chunk.split('\n')).map((l) => l.replace(/\s+$/, '')).filter((l) => l.trim());
}

/**
 * Assert each `expected` line appears, in order, as a whitespace-collapsed
 * substring of the printed output. The lines are joined first, so a notice
 * the terminal wrapped over several rows still matches.
 */
function expectInOrder(actual: string[], expected: string[], subs: Record<string, string>): void {
  const hay = actual.join(' ').replace(/\s+/g, ' ');
  let at = 0;
  for (const raw of expected) {
    const wanted = fillSubs(raw, subs).replace(/\s+/g, ' ').trim();
    const found = hay.indexOf(wanted, at);
    expect(found, `"${wanted}" not found in order in:\n${actual.join('\n')}`).toBeGreaterThanOrEqual(0);
    at = found + wanted.length;
  }
}

function fillSubs(text: string, subs: Record<string, string>): string {
  return text.replace(/\{(\w+)\}/g, (m, key: string) => (key in subs ? subs[key] : m));
}

describe('the words match the shared fixture', () => {
  it('every sentence template is equal in both chats', () => {
    expect(CHAT_WORDS).toEqual(FIXTURE.words);
  });
});

describe('the loader', () => {
  it.each(FIXTURE.loader.cases)('$case', (c: Record<string, unknown>) => {
    const file = path.join(project, c.file as string);
    if (c.link) {
      fs.writeFileSync(path.join(project, c.link as string), c.text as string);
      fs.symlinkSync(c.link as string, file);
    } else {
      fs.writeFileSync(file, c.text as string);
    }
    if (c.error) {
      const known = FIXTURE.loader.known_keys;
      let expected: string;
      if (c.error === 'linked') expected = fillSubs(FIXTURE.loader.linked, { path: file });
      else if (c.error === 'not_yaml') expected = fillSubs(FIXTURE.loader.not_yaml, { path: file });
      else if (c.error === 'not_yaml_at') expected = fillSubs(FIXTURE.loader.not_yaml_at, { path: file });
      else if (c.error === 'cron_string') expected = fillSubs(FIXTURE.loader.cron_string, { path: file });
      else if (c.error === 'unknown_key') expected = fillSubs(FIXTURE.loader.unknown_key, { path: file, key: c.key as string, near: c.near as string, known });
      else expected = fillSubs(FIXTURE.loader.unknown_key_no_match as unknown as string ?? '', { path: file, key: c.key as string, known });
      let caught: unknown;
      try {
        const text = readAgentFile(file);
        parseAgentMarkdown(text, file);
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(AgentFileError);
      if (expected.includes('{line}')) {
        // The position is each SDK's parser's (`not_yaml_at_about`): the shape is pinned, the numbers are not.
        const shape = new RegExp(`^${expected.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace('\\{line\\}', '\\d+').replace('\\{column\\}', '\\d+')}$`);
        expect((caught as Error).message).toMatch(shape);
        return;
      }
      expect((caught as Error).message).toBe(expected);
      return;
    }
    const parsed = parseAgentMarkdown(readAgentFile(file), file);
    if (c.name) expect(parsed.name).toBe(c.name);
    if (c.skills) expect(parsed.skills).toEqual(c.skills);
    if (c.instructions) expect(parsed.instructions).toBe(c.instructions);
  });
});

describe('the chat cases', () => {
  it.each(FIXTURE.cases)('$case', async (c: Record<string, unknown>) => {
    const folder = c.cwd === 'home' ? home : project;
    if (c.cwd === 'home') process.chdir(home);
    for (const [k, v] of Object.entries((c.env as Record<string, string>) ?? {})) process.env[k] = v;
    writeFiles(folder, (c.files as Record<string, string>) ?? {});
    if (c.editor === 'append' || c.editor === 'fail') process.env.EDITOR = editorScript(c.editor as string);

    const config: { interactive: boolean; model?: string; agentFile?: string | null } = { interactive: Boolean(c.interactive) };
    if (c.model) config.model = c.model as string;
    if (c.agent === 'robutler') config.agentFile = null;
    const repl = new InteractiveREPL(config);
    await repl.initialize();
    const inside = repl as unknown as { handleInput(line: string): Promise<void> };

    // Changes made after the chat loaded the file.
    writeFiles(folder, (c.then_write as Record<string, string>) ?? {});
    for (const [name, target] of Object.entries((c.then_link as Record<string, string>) ?? {})) {
      const full = path.join(folder, name);
      fs.rmSync(full, { force: true });
      fs.symlinkSync(target, full);
    }

    H.lines = [...((c.answers as string[]) ?? [])];
    H.prompts = [];
    const before = printed.length;
    for (const line of (c.typed as string[]) ?? []) await inside.handleInput(line);

    const subs = { folder: process.cwd(), known: FIXTURE.loader.known_keys };
    expectInOrder(printedLines().slice(0), (c.printed as string[]) ?? [], subs);
    void before;

    for (const [name, expected] of Object.entries((c.files_after as Record<string, string | null>) ?? {})) {
      const full = path.join(folder, name);
      if (expected === null) {
        expect(fs.existsSync(full), `${name} should not exist`).toBe(false);
      } else {
        expect(fs.readFileSync(full, 'utf8'), name).toBe(expected);
      }
    }
    if (c.lock_after) {
      const lock = JSON.parse(fs.readFileSync(path.join(folder, '.webagents', 'skills.lock'), 'utf8')) as { skills?: Record<string, unknown> };
      expect(Object.keys(lock.skills ?? {}).sort()).toEqual(c.lock_after);
    }
  });
});

describe('the first-run offer', () => {
  // A null answer is Ctrl+C or Ctrl+D at "Choose 1-3": the offer is
  // cancelled and said so, and the chat goes on (2026-09-26, the e2e run,
  // where the Python chat echoed ^C and hung).
  it.each(FIXTURE.offers)('$case', async (c) => {
    writeFiles(project, c.files);
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    expect((repl as unknown as { modelProblem?: string }).modelProblem).toBeTruthy();
    H.lines = [...c.answers];
    const before = printed.length;
    await (repl as unknown as { offerModelAccess(): Promise<void> }).offerModelAccess();
    const lines = [...H.prompts, ...printed.slice(before)].flatMap((chunk) => chunk.split('\n')).map((l) => l.replace(/\s+$/, ''));
    expectInOrder(lines, c.printed, {});
  });
});
