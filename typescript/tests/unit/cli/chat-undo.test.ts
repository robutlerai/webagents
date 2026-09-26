/**
 * `/undo` and `/rewind` in the chat (2026-09-25, `src/cli/checkpoints.ts`),
 * the Python chat's cases (`tests/cli/test_chat.py`): the snapshot before a
 * message to an agent that can change files, what `/undo` shows and asks, a
 * yes putting back what the message changed and a no leaving it, nothing to
 * undo said so, the home folder refused, and `/rewind` listing the folder's
 * snapshots and putting one back.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const H = vi.hoisted(() => ({ answers: [] as string[] }));
vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => (H.answers.length ? H.answers.shift()! : null)),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const FILE_AGENT = '---\nname: helper\nskills: [filesystem]\n---\nHelp.\n';

const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
const tty = process.stdin.isTTY;
let project = '';
let printed: string[] = [];

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-undo-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '100';
  project = tempDir('wa-undo-project-');
  process.chdir(project);
  printed = [];
  H.answers = [];
  Object.defineProperty(process.stdin, 'isTTY', { value: true, configurable: true });
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  process.chdir(cwd);
  Object.defineProperty(process.stdin, 'isTTY', { value: tty, configurable: true });
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

type Inside = {
  handleInput(line: string): Promise<void>;
  snapshotBeforeTurn(message: string): void;
  turnSnapshots: string[];
};

async function chat(folder = project): Promise<Inside> {
  fs.writeFileSync(path.join(folder, 'AGENT.md'), FILE_AGENT);
  const repl = new InteractiveREPL({});
  await repl.initialize();
  return repl as unknown as Inside;
}

async function say(repl: Inside, line: string): Promise<string> {
  const before = printed.length;
  await repl.handleInput(line);
  return printed.slice(before).join('\n');
}

const file = (name: string) => path.join(project, name);

describe('/undo and /rewind', () => {
  it('/undo puts back what the last message changed', async () => {
    const repl = await chat();
    fs.writeFileSync(file('plan.md'), 'first\n');
    repl.snapshotBeforeTurn('rewrite the plan');
    fs.writeFileSync(file('plan.md'), 'rewritten by the agent\n');
    fs.writeFileSync(file('new.md'), 'made by the agent\n');

    H.answers.push('y');
    const out = await say(repl, '/undo');

    expect(out).toContain("Undo your last message's changes to this folder:");
    expect(out).toContain('restore  plan.md');
    expect(out).toContain('remove   new.md');
    expect(out).toContain('Put back 1 file and removed 1 file made since.');
    expect(fs.readFileSync(file('plan.md'), 'utf8')).toBe('first\n');
    expect(fs.existsSync(file('new.md'))).toBe(false);
    expect(await say(repl, '/undo')).toContain('Nothing to undo in this conversation.');
  });

  it('/undo asks first, and a no leaves the folder alone', async () => {
    const repl = await chat();
    fs.writeFileSync(file('plan.md'), 'first\n');
    repl.snapshotBeforeTurn('rewrite the plan');
    fs.writeFileSync(file('plan.md'), 'rewritten\n');

    H.answers.push('n');
    expect(await say(repl, '/undo')).toContain('Left as it is.');
    expect(fs.readFileSync(file('plan.md'), 'utf8')).toBe('rewritten\n');
  });

  it('/undo when the last message changed nothing', async () => {
    const repl = await chat();
    repl.snapshotBeforeTurn('just asking');
    expect(await say(repl, '/undo')).toContain('Nothing to undo: the folder is as it was before your last message.');
  });

  it('/undo is off in the home folder', async () => {
    const home = process.env.HOME!;
    process.chdir(home);
    const repl = await chat(home);
    repl.snapshotBeforeTurn('anything');
    expect(repl.turnSnapshots).toEqual([]);
    expect(await say(repl, '/undo')).toContain(
      '/undo is off in your home folder and above: start the chat in a project folder to use it.',
    );
  });

  it('/rewind lists the snapshots and puts one back', async () => {
    const repl = await chat();
    expect(await say(repl, '/rewind')).toContain('No snapshots of this folder yet.');
    fs.writeFileSync(file('plan.md'), 'first\n');
    repl.snapshotBeforeTurn('plan the launch');
    fs.writeFileSync(file('plan.md'), 'second\n');

    const listing = await say(repl, '/rewind');
    expect(listing).toContain('Snapshots of this folder');
    expect(listing).toContain('before "plan the launch"');
    expect(listing).toContain('Put the folder back with /rewind <number>.');
    expect(await say(repl, '/rewind 9')).toContain('There is no snapshot 9.');

    H.answers.push('yes');
    const out = await say(repl, '/rewind 1');
    expect(out).toContain('Put the folder back as it was just now (before "plan the launch"):');
    expect(out).toContain('Put back 1 file.');
    expect(fs.readFileSync(file('plan.md'), 'utf8')).toBe('first\n');
    expect(await say(repl, '/rewind')).toContain('before /rewind');
  });
});
