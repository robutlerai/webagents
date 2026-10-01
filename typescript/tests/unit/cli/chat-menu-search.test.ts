/**
 * Search and pickers in the chat's menu (2026-09-30, the owner: "/resume and
 * other commands/subcommands should have search/filter on typing and up/down
 * arrow selection for the filtered options"), the same in the Python box
 * (`python/tests/cli/test_chat_menu_search.py`).
 *
 * Pinned here:
 *   - how rows are found (`rankRows`), and how a completer splits what is
 *     typed (`argumentWords`, `restAfter`), by the fixture's `menu_search`;
 *   - in the box: a query of several words; return on a row that completes
 *     the command sends the line with the text that row inserts, return on
 *     any other row inserts it, tab always inserts; return on `/resume` opens
 *     its list;
 *   - in the chat: `/resume` lists this folder's conversations numbered as the
 *     command numbers them, each inserting the start of its id, `delete` last
 *     and `/resume delete` choosing among the same; an id that is all digits
 *     still finds its conversation; `/rewind` lists the snapshots; `/model`
 *     the models the agent may switch to; `/help` finds a command by its
 *     description; a key to set runs, a key to remove is inserted.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { COMPLETED_COMMANDS, MODEL_TIERS, PICKER_COMMANDS } from '../../../src/cli/chat-commands';
import { takeSnapshot } from '../../../src/cli/checkpoints';
import { newSessionId, saveSession } from '../../../src/cli/sessions';
import {
  COMMAND_SEARCH_MIN,
  InputEditor,
  argumentWords,
  rankRows,
  restAfter,
  type Command,
  type Slot,
  type SlotRow,
} from '../../../src/cli/ui/input';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_commands.json'), 'utf8')) as {
  completion: { commands: string[]; pickers: string[]; model_tiers: string[] };
  menu_search: {
    command_search_min: number;
    cases: Array<{ about: string; rows: string[][]; query: string; commands: boolean; expected: string[] }>;
    argument_words: Array<{ args: string; before: string[]; partial: string }>;
    rest_after: Array<{ args: string; words: number; rest: string }>;
  };
};
const SEARCH = FIXTURE.menu_search;
const NOW = 1_000_000;
const key = (name: string) => ({ name });

function type(ed: InputEditor, text: string): void {
  for (const ch of text) ed.handleKey(ch, { sequence: ch }, NOW);
}

describe('finding rows', () => {
  for (const c of SEARCH.cases) {
    it(c.about, () => {
      const rows = c.rows.map(([value, description, search]) => ({ value, description, ...(search ? { search } : {}) }));
      expect(rankRows(rows, c.query, c.commands).map((r) => r.value)).toEqual(c.expected);
    });
  }

  it('splits what is typed as the fixture says', () => {
    expect(COMMAND_SEARCH_MIN).toBe(SEARCH.command_search_min);
    for (const c of SEARCH.argument_words) expect(argumentWords(c.args), c.args).toEqual({ before: c.before, partial: c.partial });
    for (const c of SEARCH.rest_after) expect(restAfter(c.args, c.words), c.args).toBe(c.rest);
  });

  it('has the constants the fixture pins', () => {
    expect([...COMPLETED_COMMANDS]).toEqual(FIXTURE.completion.commands);
    expect([...PICKER_COMMANDS]).toEqual(FIXTURE.completion.pickers);
    expect([...MODEL_TIERS]).toEqual(FIXTURE.completion.model_tiers);
  });
});

const CONVERSATIONS: SlotRow[] = [
  { value: '1', description: '2h ago · 14 messages · plan the launch', search: '5b1f2c9a-0000', runs: true, insert: '5b1f2c9a' },
  { value: '2', description: 'yesterday · 3 messages · fix the budget sheet', search: '0c3d4e5f-0000', runs: true, insert: '0c3d4e5f' },
];

function commands(conversations: SlotRow[] = CONVERSATIONS, picker = true): Command[] {
  return [
    {
      name: 'resume',
      description: 'Continue an earlier conversation, or delete one',
      picker,
      complete: (args) => {
        const words = args.split(/\s+/).filter(Boolean);
        if (words[0] === 'delete' && (words.length > 1 || /\s$/.test(args))) return { rows: conversations, query: restAfter(args, 1) };
        return { rows: [...conversations, { value: 'delete', description: 'delete an earlier conversation' }], query: restAfter(args, 0) };
      },
    },
    {
      name: 'mcp',
      description: 'The MCP servers',
      complete: (args) => {
        const { before, partial } = argumentWords(args);
        if (before.length === 1 && before[0] === 'remove') return { rows: [{ value: 'sqlite', description: "this agent's server" }], query: partial };
        return before.length ? null : { rows: [{ value: 'remove', description: 'take one out' }], query: partial };
      },
    },
    { name: 'login', description: 'Sign in to Robutler' },
  ];
}

describe('the box', () => {
  it('narrows the list with a query of several words', () => {
    const ed = new InputEditor([], commands());
    type(ed, '/resume ');
    expect(ed.menu().map((m) => m.name)).toEqual(['1', '2', 'delete']);
    type(ed, 'launch');
    expect(ed.menu().map((m) => m.name)).toEqual(['1']);
    ed.set('/resume the bud');
    expect(ed.menu().map((m) => m.name)).toEqual(['2']);
    ed.set('/resume nothing like it');
    expect(ed.menu()).toEqual([]);
    // The command list finds a command by a word of its description.
    ed.set('/sign');
    expect(ed.menu().map((m) => m.name)).toEqual(['login']);
  });

  it('sends the line with what a row inserts when the row completes the command', () => {
    let ed = new InputEditor([], commands());
    ed.set('/resume launch pl');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/resume 5b1f2c9a' });
    ed = new InputEditor([], commands());
    ed.set('/resume ');
    ed.handleKey(undefined, key('down'), NOW);
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/resume 0c3d4e5f' });
    ed = new InputEditor([], commands());
    ed.set('/resume delete bud');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/resume delete 0c3d4e5f' });
    // The number itself, typed out, is sent as typed.
    ed = new InputEditor([], commands());
    ed.set('/resume 2');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/resume 2' });
  });

  it('inserts any other row, and tab always inserts', () => {
    let ed = new InputEditor([], commands());
    ed.set('/resume del');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'render' });
    expect(ed.value).toBe('/resume delete ');
    ed = new InputEditor([], commands());
    ed.set('/mcp remove sq');
    ed.handleKey(undefined, key('return'), NOW);
    expect(ed.value).toBe('/mcp remove sqlite ');
    ed = new InputEditor([], commands());
    ed.set('/resume launch pl');
    ed.handleKey(undefined, key('tab'), NOW);
    expect(ed.value).toBe('/resume 5b1f2c9a ');
  });

  it('opens a picker on return when there is one to choose', () => {
    let ed = new InputEditor([], commands());
    type(ed, '/res');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'render' });
    expect(ed.value).toBe('/resume ');
    ed = new InputEditor([], commands(CONVERSATIONS, false));
    type(ed, '/res');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/resume' });
    // No conversation to choose: the command runs, and says so.
    ed = new InputEditor([], commands([]));
    type(ed, '/res');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/resume' });
  });
});

describe("the chat's lists", () => {
  const tempDir = tempDirs();
  const ISOLATED = ['HOME', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_BACKEND', 'ROBUTLER_API_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'WEBAGENTS_TOKEN'];
  const saved: Record<string, string | undefined> = {};
  const cwd = process.cwd();
  let project = '';

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-picker-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
    project = tempDir('wa-picker-project-');
    process.chdir(project);
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    process.chdir(cwd);
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
    vi.restoreAllMocks();
  });

  const HELPER = '---\nname: helper\nskills:\n  - openai\nmodel: openai/gpt-4o-mini\n---\nHelp.\n';

  type Inside = {
    completions(): Record<string, (args: string) => Slot | null>;
    refreshCompletionData(): Promise<void>;
    conversationEntries(platform: unknown[]): Array<{ id?: string }>;
    sessionDir(): string;
  };

  async function chat(agentFile: string): Promise<Inside> {
    fs.writeFileSync(path.join(project, 'AGENT.md'), agentFile);
    const { InteractiveREPL } = await import('../../../src/cli/app');
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as Inside;
    await inside.refreshCompletionData();
    return inside;
  }

  function kept(inside: Inside, firstMessage: string, sessionId = newSessionId()): string {
    saveSession(inside.sessionDir(), {
      session_id: sessionId,
      agent_name: 'helper',
      created_at: '',
      updated_at: '',
      messages: [
        { role: 'user', content: firstMessage },
        { role: 'assistant', content: 'ok' },
      ],
      metadata: {},
      input_tokens: 0,
      output_tokens: 0,
    });
    return sessionId;
  }

  it('lists the conversations as /resume numbers them, each inserting its id', async () => {
    const inside = await chat(HELPER);
    const older = kept(inside, 'fix the budget sheet');
    await new Promise((resolve) => setTimeout(resolve, 5));
    const newer = kept(inside, 'plan the launch');
    await inside.refreshCompletionData();
    const slot = inside.completions().resume(' ')!;
    expect(slot.rows.map((r) => r.value)).toEqual(['1', '2', 'delete']);
    expect(slot.rows[0].description.endsWith('· 2 messages · plan the launch')).toBe(true);
    expect([slot.rows[0].runs, slot.rows[0].insert, slot.rows[1].insert]).toEqual([true, newer.slice(0, 8), older.slice(0, 8)]);
    expect(slot.rows[2].runs).toBeUndefined();
    expect(inside.completions().resume(' delete ')!.rows.map((r) => r.value)).toEqual(['1', '2']);
  });

  it('continues a conversation whose id starts with digits only, from what its row inserts', async () => {
    const inside = await chat(HELPER);
    const digits = '12345678-0000-4000-8000-000000000000';
    kept(inside, 'plan the launch', digits);
    await inside.refreshCompletionData();
    expect(inside.completions().resume(' ')!.rows[0].insert).toBe('12345678');
    const repl = inside as unknown as { handleInput(line: string): Promise<void>; messages: Array<{ content?: unknown }>; sessionId: string };
    await repl.handleInput('/resume 12345678');
    expect(repl.sessionId).toBe(digits);
    expect(repl.messages[0]?.content).toBe('plan the launch');
  });

  it('takes a short number as a position, never the start of an id', async () => {
    // The gate run of 2026-09-30 caught this: `/resume 9` with one conversation
    // continued it when its random id happened to start with 9.
    const inside = await chat(HELPER);
    const nine = '9aaaaaaa-0000-4000-8000-000000000000';
    kept(inside, 'plan the launch', nine);
    const repl = inside as unknown as { handleInput(line: string): Promise<void>; sessionId: string };
    await repl.handleInput('/resume 9');
    expect(repl.sessionId).not.toBe(nine);
    await repl.handleInput('/resume 9aaa');
    expect(repl.sessionId).toBe(nine);
  });

  it('offers no conversation and no delete in a new folder', async () => {
    const inside = await chat(HELPER);
    expect(inside.completions().resume(' ')!.rows).toEqual([]);
    expect(inside.completions().resume(' delete ')).toBeNull();
  });

  it('lists the snapshots for /rewind', async () => {
    const inside = await chat(HELPER);
    fs.writeFileSync(path.join(project, 'notes.txt'), 'one');
    takeSnapshot(project, 'before: plan the launch');
    await inside.refreshCompletionData();
    const rows = inside.completions().rewind(' ')!.rows;
    expect(rows.map((r) => r.value)).toEqual(['1']);
    expect(rows[0].runs).toBe(true);
    expect(rows[0].description.endsWith('· before: plan the launch')).toBe(true);
  });

  it("offers the declared provider's models, the one in use first", async () => {
    process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
    const inside = await chat(HELPER);
    const slot = inside.completions().model(' ')!;
    expect(slot.rows[0]).toEqual({ value: 'openai/gpt-4o-mini', description: 'in use', runs: true });
    expect(slot.rows.every((r) => r.value.startsWith('openai/'))).toBe(true);
    const found = inside.completions().model(' 4.1')!;
    expect(rankRows(found.rows, found.query).map((r) => r.value)).toEqual(['openai/gpt-4.1']);
  });

  it('offers the tiers and every known model without a provider skill', async () => {
    const inside = await chat('---\nname: helper\nskills:\n  - todo\n---\nHelp.\n');
    const values = inside.completions().model(' ')!.rows.map((r) => r.value);
    expect(values.slice(values.indexOf('auto/fastest'), values.indexOf('auto/fastest') + 3)).toEqual([...MODEL_TIERS]);
    expect(values).toContain('anthropic/claude-sonnet-4-6');
    expect(values).toContain('openai/gpt-4o-mini');
  });

  it('finds a command for /help by its description, and runs only a key to set', async () => {
    const inside = await chat(HELPER);
    const help = inside.completions().help(' sign in')!;
    expect(rankRows(help.rows, help.query).map((r) => r.value)).toEqual(['login']);
    expect(inside.completions().keys(' set ')!.rows.every((r) => r.runs)).toBe(true);
    expect(inside.completions().keys(' remove ')!.rows.some((r) => r.runs)).toBe(false);
  });
});
