/**
 * What a failed turn says, in the chat and in `-p` (2026-09-25): the table the
 * Python CLI runs too (`python/tests/cli/test_failure_presentation.py`), so a
 * refusal reads the same in both, whichever words each SDK raised it with.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { presentFailure } from '../../../src/cli/failures';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/failure_presentation.json'), 'utf8'),
) as {
  proxy_url: string;
  cases: Array<{ name: string; proxy: boolean; message: string; headline: string; hint: string | null; code: string | null }>;
};

describe('what a failed turn says (the table both SDKs run)', () => {
  it.each(TABLE.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const explained = presentFailure(c.message, c.proxy ? { proxyUrl: TABLE.proxy_url } : {});
    expect(explained).toEqual({
      headline: c.headline,
      ...(c.hint === null ? {} : { hint: c.hint }),
      ...(c.code === null ? {} : { code: c.code }),
    });
  });

  it("gives a provider's error the chat's own advice", () => {
    expect(presentFailure('Error code: 401 - invalid api key', { genericHint: () => 'Check OPENAI_API_KEY.' })).toEqual({
      headline: 'Error code: 401 - invalid api key',
      hint: 'Check OPENAI_API_KEY.',
    });
  });
});
