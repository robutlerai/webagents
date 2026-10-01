/**
 * Suggestions from history in the TypeScript box (2026-09-29): the Python box
 * showed the rest of an earlier line faint after what is typed, and this box
 * showed none. The rule is prompt_toolkit's `AutoSuggestFromHistory`: the
 * newest history line that starts with the last line typed; taken by tab, →,
 * ctrl+e or ctrl+f, alt+f for one word; none while the command menu is open or
 * when the cursor is not at the end. The Python twin is
 * `python/tests/cli/test_prompt_box_tab_suggestion.py`.
 */

import { describe, expect, it } from 'vitest';

import { stripAnsi } from '../../../src/cli/ui/ansi';
import { InputEditor, layoutPrompt } from '../../../src/cli/ui/input';
import { themeFor } from '../../../src/cli/ui/theme';

const NOW = 1_000_000;
const COMMANDS = [{ name: 'mcp', description: 'The MCP servers this agent uses' }] as never;
const colour = themeFor({ isTTY: true }, {}, { depth: 16777216, animate: false });

function key(ed: InputEditor, name: string, mods: { ctrl?: boolean; meta?: boolean } = {}): void {
  ed.handleKey(undefined, { name, sequence: '', ...mods } as never, NOW);
}

function type(ed: InputEditor, text: string): void {
  for (const ch of text) ed.handleKey(ch, { sequence: ch } as never, NOW);
}

/** History oldest first, as the chat hands it to the box. */
function typed(history: string[], text: string): InputEditor {
  const ed = new InputEditor(history, COMMANDS);
  type(ed, text);
  return ed;
}

describe('suggestions from history', () => {
  it('suggest the rest of the newest line that starts with what is typed', () => {
    expect(typed(['what dog breeds are there?', 'what dog food'], 'what dog').suggestion()).toBe(' food');
    expect(typed(['what dog breeds are there?'], 'what dog').suggestion()).toBe(' breeds are there?');
    expect(typed(['hello'], 'what').suggestion()).toBe('');
    expect(typed(['hello'], '   ').suggestion()).toBe('');
  });

  it.each([
    ['tab', {}],
    ['right', {}],
    ['e', { ctrl: true }],
    ['f', { ctrl: true }],
  ] as const)('is taken by %s %j', (name, mods) => {
    const ed = typed(['what dog breeds are there?'], 'what dog');
    key(ed, name, mods);
    expect(ed.value).toBe('what dog breeds are there?');
  });

  it('gives one word to alt+f', () => {
    const ed = typed(['git commit -m fix'], 'git ');
    key(ed, 'f', { meta: true });
    expect(ed.value).toBe('git commit ');
  });

  it('is none while the command menu is open, and none mid-text', () => {
    expect(typed(['/mcp list'], '/mc').suggestion()).toBe('');
    const ed = typed(['what dog breeds are there?'], 'what dog');
    key(ed, 'left');
    expect(ed.suggestion()).toBe('');
    key(ed, 'right');
    expect(ed.value).toBe('what dog');
  });

  it('is drawn after the text in the box', () => {
    const ed = typed(['what dog breeds are there?'], 'what dog');
    const frame = layoutPrompt(colour, ed, 80, { left: [] }, 'Message', NOW);
    expect(stripAnsi(frame.box[1])).toContain('❯ what dog breeds are there?');
    // The cursor stays after what was typed.
    expect(frame.cursorCol).toBe(4 + 'what dog'.length);
  });
});
