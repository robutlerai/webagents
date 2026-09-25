/**
 * How the TypeScript chat draws a turn (2026-09-24, `src/cli/render.ts`).
 *
 * The chat used to write `Assistant: ` and then the text deltas raw: markdown
 * arrived as its source, and every other chunk was dropped, so tool calls,
 * their results and model errors never appeared (a rejected key answered with
 * nothing at all). Found by driving the real chat through a pseudo-terminal
 * against a stand-in model; each case below is one thing it got wrong or one
 * thing the replacement has to keep right. Layout under test: a two-column
 * gutter carrying ✦ (a block of text), ● (a tool call), ✗ (an error).
 */

import { describe, expect, it } from 'vitest';

import { InteractiveREPL } from '../../../src/cli/app';
import {
  MarkdownLines,
  TurnPrinter,
  streamJsonEvent,
  toolFailed,
  toolResultSummary,
  visibleWidth,
  wrapStyled,
} from '../../../src/cli/render';
import { stripAnsi } from '../../../src/cli/ui/ansi';
import { themeFor } from '../../../src/cli/ui/theme';
import type { StreamChunk } from '../../../src/core/types';

/** A terminal that records what was written. No `start()`, so no live region. */
function screen(columns = 60) {
  let written = '';
  const out = {
    isTTY: true,
    columns,
    rows: 24,
    write(chunk: string) {
      written += chunk;
      return true;
    },
  } as unknown as NodeJS.WriteStream;
  return { out, text: () => written };
}

function draw(chunks: StreamChunk[], options: { columns?: number; interrupted?: boolean; live?: boolean } = {}) {
  const { out, text } = screen(options.columns);
  const printer = new TurnPrinter({ out, color: false, live: options.live ?? true });
  for (const chunk of chunks) printer.feed(chunk);
  printer.finish({ interrupted: options.interrupted });
  return { output: text(), printer };
}

const call = (id: string, name: string, args: string): StreamChunk => ({
  type: 'tool_call',
  tool_call: { id, name, arguments: args },
});
const result = (id: string, text: string, isError = false): StreamChunk => ({
  type: 'tool_result',
  tool_result: { call_id: id, result: text, is_error: isError },
});
const delta = (text: string): StreamChunk => ({ type: 'delta', delta: text });

describe('a turn in the TypeScript chat', () => {
  it('shows the tool call and its result, in the order they happened', () => {
    const { output } = draw([
      delta('Let me look first.\n'),
      call('c1', 'list_directory', '{"path": "."}'),
      result('c1', 'AGENT.md\nnotes.txt'),
      delta('There are two files.\n'),
    ]);
    const first = output.indexOf('Let me look first.');
    const tool = output.indexOf('● list_directory(.)');
    const summary = output.indexOf('⎿  2 entries: AGENT.md, notes.txt');
    const after = output.indexOf('There are two files.');
    // THE BUG: the call and its result were not drawn at all.
    expect(first).toBeGreaterThanOrEqual(0);
    expect(tool).toBeGreaterThan(first);
    expect(summary).toBeGreaterThan(tool);
    expect(after).toBeGreaterThan(summary);
  });

  it('draws a call announced twice once', () => {
    const { output } = draw([
      call('c1', 'read_file', '{"path": "a.txt"}'),
      call('c1', 'read_file', '{"path": "a.txt"}'),
      result('c1', 'one\ntwo'),
      call('c1', 'read_file', '{"path": "a.txt"}'),
    ]);
    expect(output.match(/● read_file/g)).toHaveLength(1);
    expect(output).toContain('⎿  Read 2 lines');
  });

  it('shows a failure the tool reported as text as a failure', () => {
    // The file tools RETURN "File not found: x"; there is no error flag.
    expect(toolFailed(false, 'File not found: missing.txt')).toBe(true);
    expect(toolResultSummary('read_file', 'File not found: missing.txt', true)).toBe('File not found: missing.txt');
    expect(toolFailed(false, 'hello\nworld')).toBe(false);
    expect(toolFailed(true, 'anything')).toBe(true);
  });

  it('shows a model error instead of an empty answer', () => {
    const { output, printer } = draw([{ type: 'error', error: new Error('OpenAI API returned 401: invalid key\n{...}') }]);
    // THE BUG: the error chunk was dropped and nothing was printed.
    expect(output).toContain('✗ OpenAI API returned 401: invalid key');
    expect(output).not.toContain('{...}');
    expect(printer.failed).toBe(true);
  });

  it('puts the chat\'s advice under an error', () => {
    const { out, text } = screen();
    const printer = new TurnPrinter({ out, color: false, live: true, errorHint: () => 'Check OPENAI_API_KEY.' });
    printer.feed({ type: 'error', error: new Error('OpenAI API returned 401') });
    printer.finish();
    expect(text()).toContain('✗ OpenAI API returned 401\n  ⎿  Check OPENAI_API_KEY.');
  });

  it('styles markdown instead of printing its source', () => {
    const { output } = draw([delta('## Title\n\nSome **bold** and `code`.\n\n- one\n- two\n')]);
    // The block's first line carries the ✦; the rest sit in the gutter's shadow.
    expect(output).toContain('✦ Title');
    expect(output).not.toContain('##');
    expect(output).toContain('  Some bold and code.');
    expect(output).toContain('  • one');
    expect(output).not.toContain('**');
  });

  it('aligns a table once its last row has arrived', () => {
    const { output } = draw([
      delta('| File | What |\n| --- | --- |\n'),
      delta('| `AGENT.md` | the agent |\n| notes.txt | scratch |\n\nDone.\n'),
    ]);
    const lines = output.split('\n');
    const header = lines.find((line) => line.includes('File'));
    const row = lines.find((line) => line.includes('AGENT.md'));
    // Rounded borders, columns as wide as their widest cell (9 and 9).
    expect(header).toBe('  │ File      │ What      │');
    expect(row).toBe('  │ AGENT.md  │ the agent │');
    // The block's ✦ sits on its first line, the top border.
    expect(lines).toContain(`✦ ╭${'─'.repeat(11)}┬${'─'.repeat(11)}╮`);
    expect(lines).toContain(`  ├${'─'.repeat(11)}┼${'─'.repeat(11)}┤`);
    expect(output).not.toContain('|');
  });

  it('draws a table that ends the answer', () => {
    const { output } = draw([delta('| a | b |\n| - | - |\n| 1 | 2 |')]);
    expect(output).toContain('│ 1 │ 2 │');
  });

  it('wraps at word boundaries, with a hanging indent under a bullet', () => {
    const { output } = draw([delta(`- ${'word '.repeat(20).trim()}\n`)], { columns: 30 });
    const lines = output.trimEnd().split('\n');
    expect(lines.length).toBeGreaterThan(1);
    for (const line of lines) expect(line.length).toBeLessThan(30);
    expect(lines[0].startsWith('✦ • word')).toBe(true);
    expect(lines[1].startsWith('    word')).toBe(true);
    expect(wrapStyled('aaa bbb ccc', 8)).toEqual(['aaa bbb', 'ccc']);
  });

  it('keeps a code block as written, and with nothing on its lines a copy would pick up', () => {
    const md = new MarkdownLines(themeFor({ isTTY: true }, {}, { depth: 0 }), () => 40);
    const out = ['```python', 'def f():', '', '    return 1', '```'].flatMap((line) => md.render(line));
    // Rules above and below; the code itself with one space of padding and
    // no border character on any line (a `│` is what a copy would carry).
    expect(out[0].startsWith('── python ──')).toBe(true);
    expect(out.slice(1, 4)).toEqual([' def f():', ' ', '     return 1']);
    expect(out[4]).toBe('─'.repeat(40));
  });

  it('draws a code block as a shaded band with syntax colour where the terminal has colour', () => {
    const md = new MarkdownLines(themeFor({ isTTY: true }, {}, { depth: 16777216, background: '#101010' }), () => 40);
    const out = ['```python', 'def f():', '    return "x"', '```'].flatMap((line) => md.render(line));
    expect(stripAnsi(out[0])).toBe('▄'.repeat(40));
    expect(stripAnsi(out[out.length - 1])).toBe('▀'.repeat(40));
    expect(stripAnsi(out[1])).toMatch(/^ def f\(\):\s+python $/);
    expect(out[1]).toContain('\x1b[48;2;');
    for (const line of out.slice(1, -1)) expect(visibleWidth(line)).toBe(40);
  });

  it('never doubles a blank line and never ends on one', () => {
    const { output } = draw([
      delta('One.\n\n\n\nTwo.\n\n'),
      call('c1', 'list_directory', '{}'),
      result('c1', ''),
      delta('\n\nThree.\n\n\n'),
    ]);
    expect(output).not.toMatch(/\n\n\n/);
    expect(output.endsWith('Three.\n')).toBe(true);
  });

  it('marks what was cut off when the turn is interrupted', () => {
    const { output } = draw([delta('Half a sent'), call('c1', 'run', '{"command": "sleep 60"}')], { interrupted: true });
    expect(output).toContain('Half a sent');
    expect(output).toContain('● run(sleep 60)');
    expect(output).toContain('⎿  interrupted');
    expect(output.trimEnd().endsWith('⎿  Interrupted')).toBe(true);
  });

  it('sends each update as one write, so the status line never blinks out', () => {
    const writes: string[] = [];
    const out = { isTTY: true, columns: 60, rows: 24, write: (chunk: string) => writes.push(chunk) > 0 };
    const printer = new TurnPrinter({ out: out as unknown as NodeJS.WriteStream, color: false });
    printer.start();
    const before = writes.length;
    printer.feed(delta('First line.\nSecond li'));
    // Erase the live region, print the finished line, draw the region again:
    // three writes before, which a terminal could paint separately.
    expect(writes.length - before).toBe(1);
    expect(writes[writes.length - 1].startsWith('\x1b[?2026h')).toBe(true);
    expect(writes[writes.length - 1].endsWith('\x1b[?2026l')).toBe(true);
    expect(writes[writes.length - 1]).toContain('First line.');
    expect(writes[writes.length - 1]).toContain('Writing…');
    printer.finish();
    expect(writes.join('')).toContain('Second li');
  });

  it('writes the text as it arrives when the output is not a terminal', () => {
    const { output } = draw(
      [delta('## Title\nSome **bold**'), call('c1', 'list_directory', '{"path": "."}'), result('c1', 'a\nb')],
      { live: false },
    );
    // A pipe gets the markdown itself, and the tools as plain lines.
    expect(output).toContain('## Title\nSome **bold**\n[list_directory] .\n  -> 2 entries: a, b');
  });
});

describe('-p --output-format stream-json', () => {
  it('carries tool calls, their results and errors, not just text', () => {
    expect(streamJsonEvent(call('c1', 'list_directory', '{}'))).toEqual({
      type: 'tool_call',
      tool_call: { id: 'c1', name: 'list_directory', arguments: '{}' },
    });
    expect(streamJsonEvent(result('c1', 'ok'))).toEqual({
      type: 'tool_result',
      tool_result: { call_id: 'c1', result: 'ok', is_error: false },
    });
    const error = Object.assign(new Error('bad key'), { code: 'openai_error' });
    expect(streamJsonEvent({ type: 'error', error })).toEqual({
      type: 'error',
      error: { message: 'bad key', code: 'openai_error' },
    });
    // An Error object serialises to {}; the line must say what went wrong.
    expect(JSON.stringify(streamJsonEvent({ type: 'error', error: new Error('x') }))).toContain('"message":"x"');
  });
});

describe('the conversation the chat keeps', () => {
  function replWith(chunks: StreamChunk[]) {
    const repl = new InteractiveREPL({});
    (repl as unknown as { agent: unknown }).agent = {
      async *runStreaming() {
        for (const chunk of chunks) yield chunk;
      },
    };
    return repl;
  }
  const history = (repl: InteractiveREPL) => (repl as unknown as { messages: unknown[] }).messages;

  it('reports a model error instead of answering with nothing', async () => {
    const repl = replWith([{ type: 'error', error: new Error('OpenAI API returned 401') }]);
    const deltas: string[] = [];
    // THE BUG: this resolved with an empty answer and no error.
    await expect(
      (async () => {
        for await (const text of repl.sendMessageStreaming('hello')) deltas.push(text);
      })(),
    ).rejects.toThrow('OpenAI API returned 401');
    // A turn that failed before saying anything leaves no trace.
    expect(history(repl)).toEqual([]);
  });

  it('keeps the message and the answer of a finished turn', async () => {
    const repl = replWith([delta('Hi '), delta('there.'), { type: 'done', response: { content: 'Hi there.' } }]);
    const events: string[] = [];
    for await (const chunk of repl.streamTurn('hello')) events.push(chunk.type);
    expect(events).toEqual(['delta', 'delta', 'done']);
    expect(history(repl)).toEqual([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'Hi there.' },
    ]);
  });
});
