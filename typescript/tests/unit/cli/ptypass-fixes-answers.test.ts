/**
 * What the chat leaves on the screen after a mid-turn question (the
 * ptypass-fixes lane, 2026-09-27; fixture
 * `python/tests/fixtures/cli/ptypass_fixes_answers.json`). The Python twin is
 * `python/tests/cli/test_ptypass_fixes_answers.py`.
 *
 * WHY. The real-terminal PTY pass found the Python chat's answered questions
 * erased by its live display, and here the notice after `always` ("Added
 * example.com to network.hosts in AGENT.md.") printed under the resumed live
 * frame, where the next redraw erased it (transcript `08`). Both chats now
 * end the exchange with one line saying what was decided, and every notice
 * during a turn goes above the live region.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { InteractiveREPL, controlPath } from '../../../src/cli/app';
import { TurnPrinter } from '../../../src/cli/render';
import { ANSWER_WORDS } from '../../../src/cli/chat-words';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/ptypass_fixes_answers.json'), 'utf8'));

type Private = {
  turnPause?: { pause(): void; resume(): void };
  aboveTurn(show: () => void): void;
  answerLine(template: string, values: Record<string, string>): void;
};

afterEach(() => vi.restoreAllMocks());

describe('the answer lines', () => {
  it('are the fixture\'s', () => {
    expect(ANSWER_WORDS).toEqual({ control: FIXTURE.control, host: FIXTURE.host });
  });

  it('fill the path or the host, indented under the question', () => {
    const repl = new InteractiveREPL({}) as unknown as Private;
    const printed: string[] = [];
    vi.spyOn(console, 'log').mockImplementation((line?: unknown) => void printed.push(String(line)));
    repl.answerLine(ANSWER_WORDS.host.once, { host: 'example.com' });
    repl.answerLine(ANSWER_WORDS.control.no, { path: 'AGENT.md' });
    const plain = printed.map((line) => line.replace(/\x1b\[[0-9;]*m/g, ''));
    expect(plain).toEqual([`${FIXTURE.indent}✓ Allowed example.com for this command.`, `${FIXTURE.indent}✗ Declined: AGENT.md was not changed.`]);
  });
});

describe('a notice during a turn', () => {
  it('goes above the live region: taken down, printed, put back', () => {
    const repl = new InteractiveREPL({}) as unknown as Private;
    const order: string[] = [];
    repl.turnPause = { pause: () => order.push('pause'), resume: () => order.push('resume') };
    repl.aboveTurn(() => order.push('notice'));
    expect(order).toEqual(['pause', 'notice', 'resume']);
    repl.turnPause = undefined;
    repl.aboveTurn(() => order.push('outside a turn'));
    expect(order[order.length - 1]).toBe('outside a turn');
  });
});

describe('brief item 13', () => {
  it('names a control file as this folder knows it, not by an absolute path wrapped mid-name', () => {
    const folder = path.resolve('/tmp/ptf-agent');
    expect(controlPath(path.join(folder, 'AGENT.md'), folder)).toBe('AGENT.md');
    expect(controlPath('AGENT.md', folder)).toBe('AGENT.md');
    expect(controlPath(path.join(folder, '.agents', 'skills', 'x', 'SKILL.md'), folder)).toBe(path.join('.agents', 'skills', 'x', 'SKILL.md'));
    expect(controlPath('/etc/hosts', folder)).toBe('/etc/hosts');
  });

  it('takes the time spent answering out of a running tool\'s timer', () => {
    const out = { isTTY: true, columns: 80, rows: 24, write: () => true } as unknown as NodeJS.WriteStream;
    const printer = new TurnPrinter({ out, color: false, live: true });
    let now = 1_000_000;
    vi.spyOn(Date, 'now').mockImplementation(() => now);
    printer.feed({ type: 'tool_call', tool_call: { id: 'c1', name: 'write_file', arguments: '{}' } });
    const tool = (printer as unknown as { tools: Array<{ id: string; startedAt: number }> }).tools.find((t) => t.id === 'c1')!;
    const before = tool.startedAt;
    printer.suspend();
    now += 60_000;
    printer.resume();
    printer.finish();
    expect(tool.startedAt).toBe(before + 60_000);
  });
});
