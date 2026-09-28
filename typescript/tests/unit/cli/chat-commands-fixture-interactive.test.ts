/**
 * The chat's command table, help and refusals, against the shared fixture
 * (`python/tests/fixtures/cli/chat_commands.json`), which the Python suite
 * reads too (`tests/cli/test_chat_commands_fixture_interactive.py`). This
 * replaces the old source-parsing parity test: the fixture is the reference
 * both chats are held to (2026-09-26, interactive-mode spec 3.8, 3.9).
 *
 * Everything runs under a throwaway HOME with the FILE secrets backend, no
 * key and no sign-in, at NO_COLOR and COLUMNS=100.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const H = vi.hoisted(() => ({ lines: [] as Array<string | null> }));
vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => (H.lines.length ? H.lines.shift()! : null)),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { CHAT_COMMANDS, CHAT_KEYS, MOVED_COMMANDS, OUTSIDE_THE_CHAT } from '../../../src/cli/chat-commands';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_commands.json'), 'utf8'),
) as {
  commands: Array<{ name: string; usage: string; description: string; group: string; details: string[] }>;
  keys: Array<[string, string]>;
  groups: Array<[string, string]>;
  moved: Record<string, string>;
  outside: string;
  tip: string;
  help: string[];
  help_command: Record<string, string[]>;
  refusals: Array<{ typed: string; lines: string[] }>;
};

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
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
  process.env.HOME = tempDir('wa-cmds-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '240';
  project = tempDir('wa-cmds-project-');
  process.chdir(project);
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  H.lines = [];
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

interface Inside {
  handleInput(line: string): Promise<void>;
  sayNewAgentTip(): void;
}

async function chat(agentText?: string): Promise<Inside> {
  if (agentText) fs.writeFileSync(path.join(project, 'AGENT.md'), agentText);
  const repl = new InteractiveREPL({ interactive: true });
  await repl.initialize();
  return repl as unknown as Inside;
}

async function say(repl: Inside, line: string): Promise<string[]> {
  const before = printed.length;
  await repl.handleInput(line);
  // Split every printed chunk into its own line, dropping blank ones.
  return printed.slice(before).flatMap((chunk) => chunk.split('\n')).map((l) => l.replace(/\s+$/, '')).filter((l) => l.trim());
}

describe('the command table matches the shared fixture', () => {
  it('names, usage, descriptions, groups and details', () => {
    expect(CHAT_COMMANDS.map((c) => ({ name: c.name, usage: c.usage, description: c.description, group: c.group, details: [...c.details] }))).toEqual(FIXTURE.commands);
  });

  it('the keys', () => {
    expect(CHAT_KEYS.map((k) => [k[0], k[1]])).toEqual(FIXTURE.keys);
  });

  it('the moved names and the "outside the chat" line', () => {
    expect(MOVED_COMMANDS).toEqual(FIXTURE.moved);
    expect(OUTSIDE_THE_CHAT).toBe(FIXTURE.outside);
  });
});

describe('/help', () => {
  it('prints every group, its commands, the keys and the outside line', async () => {
    const out = await say(await chat('---\nname: helper\n---\nHelp.\n'), '/help');
    expect(out).toEqual(FIXTURE.help.filter((l) => l.trim()));
  });

  it.each(Object.entries(FIXTURE.help_command))('/help %s shows its forms', async (command, lines) => {
    const out = await say(await chat('---\nname: helper\n---\nHelp.\n'), `/help ${command}`);
    // The terminal wraps and re-spaces a notice, so match the collapsed text.
    const hay = out.join(' ').replace(/\s+/g, ' ');
    for (const line of lines) {
      const wanted = line.replace(/^[✦✗▲✓] /, '').replace(/\s+/g, ' ').trim();
      expect(hay, wanted).toContain(wanted);
    }
  });
});

describe('refusals and did-you-mean', () => {
  it.each(FIXTURE.refusals)('$typed', async ({ typed, lines }) => {
    const out = await say(await chat('---\nname: helper\n---\nHelp.\n'), typed);
    for (const line of lines) {
      const wanted = line.replace(/\s+/g, ' ').trim();
      expect(out.some((l) => l.replace(/\s+/g, ' ').includes(wanted))).toBe(true);
    }
  });
});

describe('the tip', () => {
  it('shows on the built-in agent in a folder with no agent file', async () => {
    const repl = await chat(); // built-in
    const before = printed.length;
    repl.sayNewAgentTip();
    const out = printed.slice(before).join('\n');
    expect(out).toContain(FIXTURE.tip);
  });

  it('does not show when the folder has an agent file', async () => {
    const repl = await chat('---\nname: helper\n---\nHelp.\n');
    const before = printed.length;
    repl.sayNewAgentTip();
    expect(printed.slice(before).join('\n')).not.toContain(FIXTURE.tip);
  });
});
