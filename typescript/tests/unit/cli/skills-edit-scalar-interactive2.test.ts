/**
 * `editFrontMatterScalar` (the chat's `/model --save`, interactive-mode spec
 * 3.5, 2026-09-26), against the shared fixture
 * `python/tests/fixtures/cli/skills_edit.json` (`scalar_edits`), which the
 * Python suite runs too (`tests/cli/test_skills_edit_scalar_interactive2.py`):
 * the exact bytes after each edit, and the sentence each refusal carries.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { editFrontMatterScalar } from '../../../src/cli/skills-edit';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/skills_edit.json'), 'utf8'),
) as {
  scalar_edits: Array<{
    case: string;
    file: string;
    key: string;
    value: string;
    before: string;
    after?: string;
    previous: string | null;
    unchanged?: boolean;
    error?: string;
  }>;
};

describe('editFrontMatterScalar', () => {
  it.each(FIXTURE.scalar_edits)('$case', (c) => {
    const file = path.join('/some/folder', c.file);
    if (c.error) {
      expect(() => editFrontMatterScalar(c.before, c.key, c.value, file)).toThrow(c.error.replace('{file}', c.file));
      return;
    }
    const edit = editFrontMatterScalar(c.before, c.key, c.value, file);
    expect(edit.previous).toBe(c.previous ?? undefined);
    if (c.unchanged) {
      expect(edit.changed).toBe(false);
      expect(edit.text).toBe(c.before);
      return;
    }
    expect(edit.changed).toBe(true);
    expect(edit.text).toBe(c.after);
  });
});
