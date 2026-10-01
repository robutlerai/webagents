/**
 * A line from history keeps the command menu closed (2026-09-29, the owner:
 * "up/down history stops when there is /command because menu grabs the focus").
 * A recalled `/mcp` must not open the menu, so ↑ and ↓ keep walking the
 * history; typing or moving the cursor opens it again, and a menu closed with
 * esc stays closed when only the cursor moves. The Python twin is
 * `python/tests/cli/test_prompt_box_history_recall.py`.
 */

import { describe, expect, it } from 'vitest';

import { InputEditor } from '../../../src/cli/ui/input';

const NOW = 1_000_000;
const COMMANDS = [
  { name: 'mcp', description: 'The MCP servers this agent uses' },
  { name: 'help', description: 'Show the commands and keys' },
] as never;

function key(ed: InputEditor, name: string): void {
  ed.handleKey(undefined, { name, sequence: '' } as never, NOW);
}

function type(ed: InputEditor, text: string): void {
  for (const ch of text) ed.handleKey(ch, { sequence: ch } as never, NOW);
}

describe('a line from history keeps the menu closed', () => {
  it('lets ↑ and ↓ walk past a /command', () => {
    const ed = new InputEditor(['hello', '/mcp', 'first'], COMMANDS);
    key(ed, 'up');
    expect(ed.value).toBe('first');
    key(ed, 'up');
    expect(ed.value).toBe('/mcp');
    expect(ed.menu()).toEqual([]);
    key(ed, 'up');
    expect(ed.value).toBe('hello');
    key(ed, 'down');
    key(ed, 'down');
    expect(ed.value).toBe('first');
  });

  it('opens the menu again when the cursor moves', () => {
    const ed = new InputEditor(['/mcp'], COMMANDS);
    key(ed, 'up');
    expect(ed.menu()).toEqual([]);
    key(ed, 'left');
    expect(ed.menu().map((m) => m.name)).toContain('mcp');
  });

  it('opens the menu again when the line is edited', () => {
    const ed = new InputEditor(['/he'], COMMANDS);
    key(ed, 'up');
    expect(ed.menu()).toEqual([]);
    type(ed, 'l');
    expect(ed.menu().map((m) => m.name)).toEqual(['help']);
  });

  it('keeps a menu closed with esc closed when only the cursor moves', () => {
    const ed = new InputEditor([], COMMANDS);
    type(ed, '/mc');
    expect(ed.menu().length).toBeGreaterThan(0);
    key(ed, 'escape');
    expect(ed.menu()).toEqual([]);
    key(ed, 'left');
    expect(ed.menu()).toEqual([]);
  });
});
