/**
 * The chat's terminal UI kit (2026-09-24, `src/cli/ui/`): colour depth,
 * width, the input box's editor and layout, syntax colour, the background
 * query, the welcome card, the moving parts.
 *
 * The chat was redesigned after its owner called the first rework bland; the
 * input box replaced Node's readline, so its keys are pinned here the way a
 * person uses them. Each `it` is a behaviour someone would notice if it broke.
 */

import { describe, expect, it } from 'vitest';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';

import {
  compactNumber,
  detectColorDepth,
  gradientAt,
  stripAnsi,
  truncate,
  truncateStart,
  visibleWidth,
  wrapStyled,
} from '../../../src/cli/ui/ansi';
import { welcomeCard, wordmark } from '../../../src/cli/ui/banner';
import { highlightLine } from '../../../src/cli/ui/highlight';
import { InputEditor, layoutPrompt, promptBox, sentMessage, type Key } from '../../../src/cli/ui/input';
import { STAR_FRAMES, shimmer, sparkAt } from '../../../src/cli/ui/motion';
import { parseBackgroundReply, parseCursorReply } from '../../../src/cli/ui/terminal';
import { themeFor } from '../../../src/cli/ui/theme';

const plain = themeFor({ isTTY: true }, {}, { depth: 0, animate: false });
const colour = themeFor({ isTTY: true }, {}, { depth: 16777216, animate: true, background: '#101010' });

describe('what the terminal can show', () => {
  it('reads colour depth from the environment, NO_COLOR first', () => {
    expect(detectColorDepth({ NO_COLOR: '1', COLORTERM: 'truecolor' })).toBe(0);
    expect(detectColorDepth({ FORCE_COLOR: '0' })).toBe(0);
    expect(detectColorDepth({ FORCE_COLOR: '3' }, false)).toBe(16777216);
    expect(detectColorDepth({ COLORTERM: 'truecolor' })).toBe(16777216);
    expect(detectColorDepth({ TERM_PROGRAM: 'iTerm.app' })).toBe(16777216);
    expect(detectColorDepth({ TERM: 'xterm-256color' })).toBe(256);
    expect(detectColorDepth({ TERM: 'xterm' })).toBe(16);
    // A pipe gets no colour at all.
    expect(detectColorDepth({ COLORTERM: 'truecolor' }, false)).toBe(0);
  });

  it('degrades one palette to 256 and 16 colours', () => {
    const at256 = themeFor({ isTTY: true }, {}, { depth: 256 });
    const at16 = themeFor({ isTTY: true }, {}, { depth: 16 });
    expect(at256.paint.fg('#ff6b6b', 'x')).toMatch(/\x1b\[38;5;\d+mx/);
    expect(at16.paint.fg('#ff6b6b', 'x')).toMatch(/\x1b\[9[0-7]mx|\x1b\[3[0-7]mx/);
    // No band below 256 colours: a basic background is too blunt.
    expect(at16.paint.bg('#262a36', 'x')).toBe('x');
  });

  it('reads the background from the terminal, and says so only when it answered', () => {
    // The colour alone is not the end: the DA1 reply follows it, and must be read too.
    expect(parseBackgroundReply('\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\')).toEqual({ background: '#1e1e1e', complete: false });
    expect(parseBackgroundReply('\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\\x1b[?62;22c')).toEqual({ background: '#1e1e1e', complete: true });
    expect(parseBackgroundReply('\x1b]11;rgb:ff/ff/ff\x07\x1b[?62;22c').background).toBe('#ffffff');
    // DA1 alone: the terminal ignores OSC 11. Done, with nothing.
    expect(parseBackgroundReply('\x1b[?1;2c')).toEqual({ background: null, complete: true });
    expect(parseBackgroundReply('\x1b]11;rgb:1e').complete).toBe(false);
  });

  it('mixes the bands from the real background, dark or light', () => {
    const dark = themeFor({ isTTY: true }, {}, { depth: 16777216, background: '#000000' });
    const light = themeFor({ isTTY: true }, {}, { depth: 16777216, background: '#ffffff' });
    expect(dark.light).toBe(false);
    expect(light.light).toBe(true);
    expect(dark.palette.surface).toBe('#1f1f1f');
    expect(light.palette.surface).toBe('#f0f0f0');
  });
});

describe('width', () => {
  it('counts columns, not characters or escapes', () => {
    expect(visibleWidth('\x1b[1mhello\x1b[22m')).toBe(5);
    expect(visibleWidth('日本')).toBe(4);
    expect(visibleWidth('\x1b]8;;https://x.test\x1b\\link\x1b]8;;\x1b\\')).toBe(4);
  });

  it('cuts from the end, or keeps the end of a path', () => {
    expect(truncate('abcdefgh', 5)).toBe('abcd…');
    expect(truncateStart('/home/me/projects/demo', 10)).toBe('…ects/demo');
  });

  it('wraps words, and breaks a word wider than the line', () => {
    expect(wrapStyled('aaa bbb ccc', 8)).toEqual(['aaa bbb', 'ccc']);
    for (const line of wrapStyled('x'.repeat(25), 10)) expect(visibleWidth(line)).toBeLessThanOrEqual(10);
  });

  it('keeps token counts short', () => {
    expect(compactNumber(950)).toBe('950');
    expect(compactNumber(1234)).toBe('1.2k');
    expect(compactNumber(48_000)).toBe('48k');
  });
});

describe('the input box', () => {
  const key = (name: string, extra: Partial<Key> = {}): Key => ({ name, ...extra });
  const type = (editor: InputEditor, text: string) => {
    for (const ch of text) editor.handleKey(ch, { sequence: ch }, 0);
  };
  const commands = [
    { name: 'help', description: 'Show available commands' },
    { name: 'model', description: 'Change or show current model' },
    { name: 'history', description: 'Show conversation history' },
  ];

  it('edits like a shell line', () => {
    const editor = new InputEditor([], commands);
    type(editor, 'hello world');
    editor.handleKey(undefined, key('w', { ctrl: true }), 0);
    expect(editor.value).toBe('hello ');
    editor.handleKey(undefined, key('a', { ctrl: true }), 0);
    type(editor, '> ');
    expect(editor.value).toBe('> hello ');
    editor.handleKey(undefined, key('k', { ctrl: true }), 0);
    expect(editor.value).toBe('> ');
    editor.handleKey(undefined, key('backspace'), 0);
    expect(editor.value).toBe('>');
  });

  it('sends on enter, and adds a line on alt+enter or a trailing backslash', () => {
    const editor = new InputEditor([], commands);
    type(editor, 'one');
    expect(editor.handleKey(undefined, key('return', { meta: true }), 0)).toEqual({ kind: 'render' });
    type(editor, 'two\\');
    editor.handleKey(undefined, key('return'), 0);
    type(editor, 'three');
    expect(editor.handleKey(undefined, key('return'), 0)).toEqual({ kind: 'submit', text: 'one\ntwo\nthree' });
  });

  it('takes a paste as text: its newlines do not send it', () => {
    const editor = new InputEditor([], commands);
    editor.handleKey(undefined, key('paste-start'), 0);
    editor.handleKey('line 1', { sequence: 'line 1' }, 0);
    expect(editor.handleKey('\r', key('return'), 0)).toEqual({ kind: 'render' });
    editor.handleKey('line 2', { sequence: 'line 2' }, 0);
    editor.handleKey(undefined, key('paste-end'), 0);
    expect(editor.value).toBe('line 1\nline 2');
  });

  it('walks the history and gives back what was being typed', () => {
    const editor = new InputEditor(['first', 'second'], commands);
    type(editor, 'draft');
    editor.handleKey(undefined, key('up'), 0);
    expect(editor.value).toBe('second');
    editor.handleKey(undefined, key('up'), 0);
    expect(editor.value).toBe('first');
    editor.handleKey(undefined, key('down'), 0);
    editor.handleKey(undefined, key('down'), 0);
    expect(editor.value).toBe('draft');
  });

  it('offers commands on /, narrows as you type, and runs the highlighted one', () => {
    const editor = new InputEditor([], commands);
    type(editor, '/');
    expect(editor.menu().map((c) => c.name)).toEqual(['help', 'model', 'history']);
    type(editor, 'h');
    expect(editor.menu().map((c) => c.name)).toEqual(['help', 'history']);
    editor.handleKey(undefined, key('down'), 0);
    expect(editor.handleKey(undefined, key('return'), 0)).toEqual({ kind: 'submit', text: '/history' });
  });

  it('completes with tab, then leaves room for arguments', () => {
    const editor = new InputEditor([], commands);
    type(editor, '/mo');
    editor.handleKey(undefined, key('tab'), 0);
    expect(editor.value).toBe('/model ');
    expect(editor.menu()).toEqual([]);
  });

  it('closes the menu on esc, and clears the box on a second esc', () => {
    const editor = new InputEditor([], commands);
    type(editor, '/he');
    editor.handleKey(undefined, key('escape'), 1000);
    expect(editor.menu()).toEqual([]);
    editor.handleKey(undefined, key('escape'), 1100);
    expect(editor.value).toBe('/he');
    editor.handleKey(undefined, key('escape'), 1200);
    expect(editor.value).toBe('');
  });

  it('clears on ctrl+c, and leaves only on a second ctrl+c at an empty box', () => {
    const editor = new InputEditor([], commands);
    type(editor, 'text');
    expect(editor.handleKey(undefined, key('c', { ctrl: true }), 0)).toEqual({ kind: 'render' });
    expect(editor.value).toBe('');
    expect(editor.handleKey(undefined, key('c', { ctrl: true }), 5000)).toEqual({ kind: 'render' });
    expect(editor.exitArmed(5100)).toBe(true);
    expect(editor.handleKey(undefined, key('c', { ctrl: true }), 5500)).toEqual({ kind: 'exit' });
    // Too late for a second: it arms again instead.
    const late = new InputEditor([], commands);
    late.handleKey(undefined, key('c', { ctrl: true }), 0);
    expect(late.handleKey(undefined, key('c', { ctrl: true }), 9000)).toEqual({ kind: 'render' });
  });

  it('leaves on ctrl+d only when the box is empty', () => {
    const editor = new InputEditor([], commands);
    type(editor, 'ab');
    editor.handleKey(undefined, key('left'), 0);
    expect(editor.handleKey(undefined, key('d', { ctrl: true }), 0)).toEqual({ kind: 'render' });
    expect(editor.value).toBe('a');
    editor.handleKey(undefined, key('backspace'), 0);
    expect(editor.handleKey(undefined, key('d', { ctrl: true }), 0)).toEqual({ kind: 'exit' });
  });

  it('lays out a box whose every line fits, with the cursor after the text', () => {
    const editor = new InputEditor([], commands);
    type(editor, 'hello');
    const frame = layoutPrompt(plain, editor, 40, { left: ['helper', 'openai/gpt-4o', '~/a/very/long/folder/name'] }, 'Say something', 0);
    for (const line of frame.lines) expect(visibleWidth(line)).toBeLessThanOrEqual(39);
    expect(frame.lines[1]).toBe('│ ❯ hello                             │');
    expect(frame.cursorRow).toBe(1);
    expect(frame.cursorCol).toBe(4 + 5);
    // The footer drops whole items from the end rather than cutting one.
    expect(frame.lines[3]).toContain('helper');
    expect(frame.lines[3]).not.toContain('…');
  });

  it('wraps a long message inside the box and follows the cursor', () => {
    const editor = new InputEditor([], commands);
    type(editor, 'x'.repeat(50));
    const frame = layoutPrompt(plain, editor, 30, { left: [] }, '', 0);
    // 30 columns: the box is 29 wide, 23 of them for text.
    expect(frame.lines.filter((line) => line.startsWith('│')).length).toBe(3);
    expect(frame.cursorRow).toBe(3);
    expect(frame.cursorCol).toBe(4 + 4);
  });

  it('shows the placeholder only while the box is empty, and the menu under it', () => {
    const editor = new InputEditor([], commands);
    expect(stripAnsi(layoutPrompt(plain, editor, 60, { left: [] }, 'Message helper', 0).lines[1])).toContain('Message helper');
    type(editor, '/');
    const frame = layoutPrompt(plain, editor, 60, { left: [] }, 'Message helper', 0);
    expect(frame.lines.some((line) => line.includes('❯ /help'))).toBe(true);
    expect(frame.lines.some((line) => line.includes('/model'))).toBe(true);
  });

  it('keeps a sent message in the conversation, marked as the person\'s', () => {
    expect(sentMessage(plain, 40, 'hi\nthere')).toEqual([' ❯ hi', '   there']);
  });
});

describe('syntax colour', () => {
  it('colours keywords, strings, comments and numbers', () => {
    const line = highlightLine(colour, 'python', 'def f(x): return "a" + 1  # note', {});
    expect(stripAnsi(line)).toBe('def f(x): return "a" + 1  # note');
    const keyword = colour.paint.fg(colour.palette.code.keyword, 'def');
    expect(line).toContain(keyword);
    expect(line).toContain(colour.paint.fg(colour.palette.code.string, '"a"'));
    expect(line).toContain(colour.paint.fg(colour.palette.code.number, '1'));
  });

  it('carries a block comment across lines', () => {
    const state = {};
    highlightLine(colour, 'ts', 'const a = 1; /* starts', state);
    const next = highlightLine(colour, 'ts', 'still a comment */ const b = 2;', state);
    expect(next.startsWith(colour.paint.italic(colour.paint.fg(colour.palette.code.comment, 'still a comment */')))).toBe(true);
  });

  it('tells a JSON key from a value, and leaves unknown languages alone', () => {
    const json = highlightLine(colour, 'json', '{"name": "x"}', {});
    expect(json).toContain(colour.paint.fg(colour.palette.code.property, '"name"'));
    expect(json).toContain(colour.paint.fg(colour.palette.code.string, '"x"'));
    expect(stripAnsi(highlightLine(colour, 'brainfuck', '+[-->-[>>+>-----<<]<--<---]>-.', {}))).toBe('+[-->-[>>+>-----<<]<--<---]>-.');
  });
});

describe('the welcome screen and the moving parts', () => {
  it('draws a card whose lines are all the same width', () => {
    const lines = welcomeCard(colour, 100, {
      agent: 'helper',
      description: 'A tool agent',
      model: 'openai/gpt-4o-mini',
      tools: ['read_file', 'write_file', 'list_directory', 'glob', 'search_file_content', 'replace'],
      folder: '/a/very/long/path/that/keeps/going/and/going/until/it/cannot/fit/in/the/card/anymore/work',
      warnings: ['OPENAI_API_KEY is not set.'],
      version: '0.3.6',
    });
    const box = lines.slice(0, -1);
    for (const line of box) expect(visibleWidth(line)).toBe(80);
    expect(stripAnsi(lines[0])).toContain('webagents 0.3.6');
    // The folder keeps its end: that is where you are.
    expect(stripAnsi(lines.find((line) => line.includes('folder')) ?? '')).toContain('/anymore/work');
  });

  it('uses the small wordmark on a narrow terminal', () => {
    expect(wordmark(plain, 120)).toHaveLength(6);
    expect(wordmark(plain, 60)).toHaveLength(3);
  });

  it('runs the star forward and back, and leaves text plain without colour', () => {
    expect(STAR_FRAMES.join('')).toBe('·✢✳✶✻✽✻✶✳✢');
    expect(shimmer(plain, 'Writing…', '#a78bfa', 500)).toBe('Writing…');
    expect(stripAnsi(shimmer(colour, 'Writing…', '#a78bfa', 500))).toBe('Writing…');
    expect(gradientAt(['#000000', '#ffffff'], 0.5)).toBe('#808080');
  });

  it('twinkles the same way on every redraw of the same moment', () => {
    const row = (now: number) => Array.from({ length: 60 }, (_, c) => sparkAt(colour, c, now)).join('');
    expect(row(1234)).toBe(row(1234));
    expect(stripAnsi(row(1234)).trim().length).toBeGreaterThanOrEqual(0);
  });
});

describe('the box goes back down when the menu closes (2026-09-25)', () => {
  it('reads the cursor row and hands back typing that came first', () => {
    expect(parseCursorReply('\x1b[20;1R\x1b[?62;22c')).toEqual({ row: 20, complete: true, rest: '' });
    expect(parseCursorReply('he\x1b[7;3Rllo\x1b[?1;2c')).toEqual({ row: 7, complete: true, rest: 'hello' });
    expect(parseCursorReply('\x1b[?62;22c')).toEqual({ row: null, complete: true, rest: '' });
    expect(parseCursorReply('\x1b[20;1R').complete).toBe(false);
  });

  /** A terminal of `rows` rows whose cursor is on `cursorRow`, answering the position query. */
  function terminal(rows: number, cursorRow: number) {
    const input = Object.assign(new PassThrough(), {
      isTTY: true,
      isRaw: false,
      setRawMode(on: boolean) {
        input.isRaw = on;
        return input;
      },
    });
    const writes: string[] = [];
    const output = Object.assign(new EventEmitter(), {
      isTTY: true,
      rows,
      columns: 80,
      write(chunk: string) {
        writes.push(chunk);
        if (chunk.includes('\x1b[6n')) setImmediate(() => input.write(`\x1b[${cursorRow};1R\x1b[?62;22c`));
        return true;
      },
    });
    return { input: input as unknown as NodeJS.ReadStream, output: output as unknown as NodeJS.WriteStream, writes };
  }

  const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  it('scrolls back down by the rows the menu lifted it, as far as there is room', async () => {
    const { input, output, writes } = terminal(20, 20);
    const commands = ['help', 'new', 'clear', 'resume', 'model', 'agent'].map((name) => ({ name, description: name }));
    const result = promptBox({ theme: plain, commands, history: [], placeholder: 'Say something', footer: () => ({ left: [] }), input, output });
    await pause(20);
    // On the last row, the four-row box scrolls the terminal by 3; the menu
    // (six rows in the footer's place) by 5 more; closing it takes back 5.
    input.write('/');
    await pause(20);
    const before = writes.length;
    input.write('\x1b');
    await pause(700); // a lone esc is told from a sequence after a timeout
    const after = writes.slice(before).join('');
    expect(after).toContain(`\x1b7\x1b[1;1H${'\x1bM'.repeat(5)}\x1b8\x1b[5B`);
    input.write('\x03'); // clear the "/"
    await pause(20);
    input.write('\x04'); // and leave
    await expect(result).resolves.toEqual({ kind: 'exit' });
  });

  it('does not move a box that the menu never lifted', async () => {
    const { input, output, writes } = terminal(40, 5);
    const commands = ['help', 'new'].map((name) => ({ name, description: name }));
    const result = promptBox({ theme: plain, commands, history: [], placeholder: 'Say something', footer: () => ({ left: [] }), input, output });
    await pause(20);
    input.write('/');
    await pause(20);
    input.write('\x1b');
    await pause(700);
    expect(writes.join('')).not.toContain('\x1bM');
    input.write('\x03');
    await pause(20);
    input.write('\x04');
    await expect(result).resolves.toEqual({ kind: 'exit' });
  });
});
