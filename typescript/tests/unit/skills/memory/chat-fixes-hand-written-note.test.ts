/**
 * A memory note written by hand is loaded (2026-09-27, the chat-fixes lane),
 * against the shared fixture
 * `python/tests/fixtures/memory_tool/chat_fixes_hand_written_note.json`, which
 * the Python suite runs too (`test_chat_fixes_hand_written_note.py`). The docs
 * said the files could be edited by hand; a plain `.md` with no front matter
 * was ignored without a word.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LocalMemoryStore, parseEntryFile } from '../../../../src/skills/memory/local-store';
import { tempDirs } from '../../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/memory_tool/chat_fixes_hand_written_note.json'), 'utf8'),
) as {
  cases: Array<{ name: string; text: string; namespace: string; key: string; entry: { namespace: string; key: string; content: string; source: string } | null }>;
};
const tempDir = tempDirs();

describe('a note written by hand (the cases both stores read)', () => {
  it.each(FIXTURE.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const entry = parseEntryFile(c.text, { namespace: c.namespace, key: c.key, store: 'helper' });
    if (c.entry === null) {
      expect(entry).toBeNull();
      return;
    }
    expect(entry).not.toBeNull();
    expect({ namespace: entry!.namespace, key: entry!.key, content: entry!.content, source: entry!.source }).toEqual(c.entry);
  });

  it('a note dropped into owner/ is found by the store', async () => {
    const root = path.join(tempDir('mem-hand-'), 'memory');
    fs.mkdirSync(path.join(root, 'owner'), { recursive: true });
    fs.writeFileSync(path.join(root, 'owner', 'preferences.md'), 'Prefers short answers.\n');
    const store = new LocalMemoryStore({ root, store: 'helper', plainIndex: true });
    await store.open();
    try {
      const found = await store.get('owner', 'preferences');
      expect(found).toMatchObject({ content: 'Prefers short answers.', source: 'owner' });
      expect((await store.search('short', ['owner'])).map((e) => e.key)).toEqual(['preferences']);
    } finally {
      await store.close();
    }
  });
});
