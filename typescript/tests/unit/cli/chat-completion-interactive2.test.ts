/**
 * Argument completion in the chat's box (interactive-mode spec 3.8,
 * 2026-09-26): after `/<command> ` the menu offers the next word's values,
 * enter and tab insert one and never run the command, and the menu closes
 * when nothing more is offered. The commands that complete, and the first
 * values each offers, are pinned by `python/tests/fixtures/cli/chat_commands.json`
 * (`completion`), which `tests/cli/test_chat_completion_interactive2.py` runs too.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { COMPLETED_COMMANDS } from '../../../src/cli/chat-commands';
import { InputEditor, type Command } from '../../../src/cli/ui/input';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_commands.json'), 'utf8'),
) as { completion: { commands: string[]; first_values: Record<string, string[]> } };

const NOW = 1_000_000;
const key = (name: string) => ({ name });

function editor(commands: Command[]): InputEditor {
  return new InputEditor([], commands);
}

function type(ed: InputEditor, text: string): void {
  for (const ch of text) ed.handleKey(ch, { sequence: ch }, NOW);
}

const COMMANDS: Command[] = [
  { name: 'help', description: 'Show the commands and keys' },
  {
    name: 'agent',
    description: 'List this folder',
    complete: (args) => {
      const [verb, ...rest] = args.split(/\s+/).filter(Boolean);
      if (verb === 'edit') return rest.length ? [] : [{ value: 'helper', description: 'A helper' }];
      if (verb) return [];
      return [
        { value: 'helper', description: 'A helper' },
        { value: 'robutler', description: 'The general assistant' },
        { value: 'new', description: 'make one here' },
        { value: 'edit', description: 'open its file' },
      ];
    },
  },
  {
    name: 'skills',
    description: 'Skills',
    complete: (args) => {
      const [verb, ...rest] = args.split(/\s+/).filter(Boolean);
      if (verb === 'add') return ['todo', 'shell', 'memory'].filter((n) => !rest.includes(n)).map((value) => ({ value, description: '' }));
      if (verb) return [];
      return [{ value: 'add', description: '' }, { value: 'remove', description: '' }];
    },
  },
  { name: 'exit', description: 'Leave' },
];

describe('the menu after a command', () => {
  it('stays open with the values the command offers, ranked by prefix then substring', () => {
    const ed = editor(COMMANDS);
    type(ed, '/agent ');
    expect(ed.menu().map((m) => m.name)).toEqual(['helper', 'robutler', 'new', 'edit']);
    expect(ed.menu().every((m) => m.argument)).toBe(true);
    // By prefix first, then by substring, in the order offered.
    type(ed, 'e');
    expect(ed.menu().map((m) => m.name)).toEqual(['edit', 'helper', 'robutler', 'new']);
    type(ed, 'd');
    expect(ed.menu().map((m) => m.name)).toEqual(['edit']);
  });

  it('enter inserts the highlighted value with a space, and never runs the command', () => {
    const ed = editor(COMMANDS);
    type(ed, '/agent he');
    const action = ed.handleKey(undefined, key('return'), NOW);
    expect(action).toEqual({ kind: 'render' });
    expect(ed.value).toBe('/agent helper ');
    // Nothing more is offered after a name: the menu is closed, and enter now sends.
    expect(ed.menu()).toEqual([]);
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/agent helper ' });
  });

  it('tab inserts too, and a second level follows the first', () => {
    const ed = editor(COMMANDS);
    type(ed, '/skills ');
    expect(ed.menu().map((m) => m.name)).toEqual(['add', 'remove']);
    ed.handleKey(undefined, key('tab'), NOW);
    expect(ed.value).toBe('/skills add ');
    expect(ed.menu().map((m) => m.name)).toEqual(['todo', 'shell', 'memory']);
    type(ed, 'me');
    ed.handleKey(undefined, key('return'), NOW);
    expect(ed.value).toBe('/skills add memory ');
    // A list command keeps offering the names not yet typed; esc closes its menu, then enter sends.
    expect(ed.menu().map((m) => m.name)).toEqual(['todo', 'shell']);
    ed.handleKey(undefined, key('escape'), NOW);
    expect(ed.menu()).toEqual([]);
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/skills add memory ' });
  });

  it('closes after a command that completes nothing', () => {
    const ed = editor(COMMANDS);
    type(ed, '/help ');
    expect(ed.menu()).toEqual([]);
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/help ' });
  });

  it('the command menu itself still completes names and runs on enter', () => {
    const ed = editor(COMMANDS);
    type(ed, '/ex');
    expect(ed.menu().map((m) => m.name)).toEqual(['exit']);
    expect(ed.menu()[0].argument).toBeUndefined();
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/exit' });
  });
});

describe("the chat's own completers", () => {
  const tempDir = tempDirs();
  const ISOLATED = ['HOME', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_BACKEND', 'ROBUTLER_API_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'WEBAGENTS_TOKEN'];
  const saved: Record<string, string | undefined> = {};
  const cwd = process.cwd();

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-complete-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
    process.chdir(tempDir('wa-complete-project-'));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    process.chdir(cwd);
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
    vi.restoreAllMocks();
  });

  it('the commands that complete are the fixture list', () => {
    expect([...COMPLETED_COMMANDS]).toEqual(FIXTURE.completion.commands);
  });

  it('offers the first-level values the fixture pins, plus the agents here', async () => {
    fs.writeFileSync(path.join(process.cwd(), 'AGENT.md'), '---\nname: helper\ndescription: A helper\nskills:\n  - todo\n---\nHelp.\n');
    const { InteractiveREPL } = await import('../../../src/cli/app');
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as {
      completions(): Record<string, (args: string) => Array<{ value: string }>>;
      refreshCompletionData(): Promise<void>;
    };
    await inside.refreshCompletionData();
    const completions = inside.completions();
    expect(Object.keys(completions).sort()).toEqual([...FIXTURE.completion.commands].sort());
    for (const [command, values] of Object.entries(FIXTURE.completion.first_values)) {
      const offered = completions[command]('').map((c) => c.value);
      for (const value of values) expect(offered, command).toContain(value);
    }
    expect(completions.agent('').map((c) => c.value)).toContain('helper');
    expect(completions.agent('').map((c) => c.value)).toContain('robutler');
    expect(completions.agent('edit').map((c) => c.value)).toEqual(['helper']);
    expect(completions.agent('helper')).toEqual([]);
    expect(completions.skills('add').map((c) => c.value)).toContain('todo');
    expect(completions.skills('add todo').map((c) => c.value)).not.toContain('todo');
    expect(completions.skills('remove').map((c) => c.value)).toEqual(['todo']);
    expect(completions.keys('set OPENAI_API_KEY')).toEqual([]);
    expect(completions.help('').map((c) => c.value)).toContain('status');
    expect(completions.keys('set').map((c) => c.value)).toContain('OPENAI_API_KEY');
    expect(completions.cron('run')).toEqual([]);
    expect(completions.memory('forget')).toEqual([]);
  });
});
