/**
 * The smaller CLI items from the 2026-09-27 real-model pass (the chat-fixes
 * lane; the Python twin is `tests/cli/test_chat_fixes_small_items.py`):
 *
 *  - return on a fully typed argument sends the line (`/agent edit` offered
 *    `edit` and put a space after it instead);
 *  - a key hint names the model's provider (`providerKeyFor`).
 */

import { describe, expect, it } from 'vitest';

import { providerKeyFor } from '../../../src/cli/failures';
import { InputEditor, type Command } from '../../../src/cli/ui/input';

const NOW = 1_000_000;
const key = (name: string) => ({ name });

function type(ed: InputEditor, text: string): void {
  for (const ch of text) ed.handleKey(ch, { sequence: ch }, NOW);
}

const COMMANDS: Command[] = [
  {
    name: 'agent',
    description: 'List this folder',
    complete: (args) => {
      const [verb, ...rest] = args.split(/\s+/).filter(Boolean);
      if (verb === 'edit') return rest.length ? [] : [{ value: 'helper', description: 'A helper' }];
      if (verb) return [];
      return [
        { value: 'helper', description: 'A helper' },
        { value: 'new', description: 'make one here' },
        { value: 'edit', description: 'open its file' },
      ];
    },
  },
];

describe('return while the menu offers the word already typed', () => {
  it('sends a fully typed /agent edit', () => {
    const ed = new InputEditor([], COMMANDS);
    type(ed, '/agent edit');
    expect(ed.menu().map((m) => m.name)).toEqual(['edit']);
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/agent edit' });
  });

  it('still inserts a value that is only partly typed', () => {
    const ed = new InputEditor([], COMMANDS);
    type(ed, '/agent ed');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'render' });
    expect(ed.value).toBe('/agent edit ');
  });

  it('matches the word without regard to case', () => {
    const ed = new InputEditor([], COMMANDS);
    type(ed, '/agent EDIT');
    expect(ed.handleKey(undefined, key('return'), NOW)).toEqual({ kind: 'submit', text: '/agent EDIT' });
  });
});

describe('the key a hint names', () => {
  it("is the model's provider's, and OPENAI_API_KEY when the model names none this SDK can call", () => {
    expect(providerKeyFor('google/gemini-2.5-flash')).toBe('GOOGLE_API_KEY');
    expect(providerKeyFor('anthropic/claude-sonnet-4-5')).toBe('ANTHROPIC_API_KEY');
    expect(providerKeyFor('xai/grok-4')).toBe('XAI_API_KEY');
    expect(providerKeyFor('auto/balanced')).toBe('OPENAI_API_KEY');
    expect(providerKeyFor('proxy/gpt-4o')).toBe('OPENAI_API_KEY');
    expect(providerKeyFor('bedrock/anthropic.claude')).toBe('OPENAI_API_KEY');
    expect(providerKeyFor(undefined)).toBe('OPENAI_API_KEY');
  });
});
