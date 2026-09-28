/**
 * `webagents cron list` exits non-zero when a file the daemon would refuse
 * sits in the folder (2026-09-26, the new-developer e2e run): the old string
 * `cron:` form was said on stderr and the command still exited 0 after "No
 * schedules", as if the file were fine. Pinned by `list.refused_exit` in
 * `python/tests/fixtures/cli/cron.json` and the refusal sentence in
 * `python/tests/fixtures/daemon/cron.json`, which the Python suite reads too
 * (`tests/cli/test_cron_list_refused_e2efix.py`).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LIST_REFUSED_EXIT, cronListAction } from '../../../src/cli/cron-action';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const LIST = (JSON.parse(readFileSync(path.join(FIXTURES, 'cli', 'cron.json'), 'utf8')) as {
  list: { refused_exit: number; none: string; none_refused: string; refused_line: string };
}).list;
const DAEMON = JSON.parse(readFileSync(path.join(FIXTURES, 'daemon', 'cron.json'), 'utf8')) as { string_form_refused: string };
const tempDir = tempDirs();

describe('cron list on the old string form', () => {
  it('says the refusal, lists nothing, and exits non-zero', async () => {
    expect(LIST_REFUSED_EXIT).toBe(LIST.refused_exit);
    const folder = tempDir('wa-cron-refused-');
    const file = path.join(folder, 'AGENT.md');
    writeFileSync(file, '---\nname: old\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\ncron: "0 9 * * 1-5"\n---\nBody\n');
    const out: string[] = [];
    const err: string[] = [];
    const code = await cronListAction({ watch: folder }, { log: (l) => out.push(l), error: (l) => err.push(l) });
    expect(code).toBe(LIST.refused_exit);
    expect(err).toEqual([LIST.refused_line.replace('{file}', file).replace('{reason}', DAEMON.string_form_refused)]);
    expect(out).toEqual([LIST.none_refused.replace('{folder}', folder)]);
  });

  it('exits 0 when every file loads, with or without schedules', async () => {
    const folder = tempDir('wa-cron-fine-');
    writeFileSync(path.join(folder, 'AGENT.md'), '---\nname: fine\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n');
    const err: string[] = [];
    expect(await cronListAction({ watch: folder }, { log: () => {}, error: (l) => err.push(l) })).toBe(0);
    expect(err).toEqual([]);
  });

  it('--json carries the count of refused files', async () => {
    const folder = tempDir('wa-cron-refused-json-');
    writeFileSync(path.join(folder, 'AGENT.md'), '---\nname: old\ncron: "0 9 * * 1-5"\n---\nBody\n');
    const emitted: unknown[] = [];
    const code = await cronListAction({ watch: folder }, { json: true, emit: (d) => emitted.push(d), error: () => {} });
    expect(code).toBe(LIST.refused_exit);
    expect(emitted).toEqual([{ schedules: [], refused: 1 }]);
  });
});
