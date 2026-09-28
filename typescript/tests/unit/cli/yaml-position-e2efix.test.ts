/**
 * Invalid front matter is refused with the line and column (2026-09-26, the
 * new-developer e2e run: "is not valid YAML" and nothing else). The sentence
 * is `loader.not_yaml_at` in `python/tests/fixtures/cli/chat_edits.json`,
 * which the Python loader prints too (`tests/cli/test_yaml_position_e2efix.py`);
 * the numbers are the FILE's line (the opening fence is line 1) and a 1-based
 * column, from each SDK's own parser.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseAgentMarkdown } from '../../../src/agents/index';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LOADER = (JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_edits.json'), 'utf8')) as { loader: { not_yaml: string; not_yaml_at: string } }).loader;

/** The fixture template as a regex, capturing the line and the column. */
function pattern(template: string, file: string): RegExp {
  const escaped = template.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace('\\{path\\}', file.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).replace('\\{line\\}', '(\\d+)').replace('\\{column\\}', '(\\d+)');
  return new RegExp(`^${escaped}$`);
}

describe('invalid front matter names the line and column', () => {
  it('a file whose third line is not YAML', () => {
    const file = '/x/AGENT.md';
    const text = '---\nname: a\nskills: [\n  - openai\n---\nBody\n';
    let message = '';
    try {
      parseAgentMarkdown(text, file);
    } catch (error) {
      message = (error as Error).message;
    }
    const match = pattern(LOADER.not_yaml_at, file).exec(message);
    expect(match, message).not.toBeNull();
    const line = Number(match![1]);
    const column = Number(match![2]);
    // Inside the front matter of the file: after the opening fence, before the closing one.
    expect(line).toBeGreaterThanOrEqual(2);
    expect(line).toBeLessThanOrEqual(4);
    expect(column).toBeGreaterThanOrEqual(1);
  });

  it('the plain sentence stays for a front matter that is YAML but not a mapping', () => {
    let message = '';
    try {
      parseAgentMarkdown('---\n- a\n- b\n---\nBody\n', '/x/AGENT.md');
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).toBe(LOADER.not_yaml.replace('{path}', '/x/AGENT.md'));
  });
});
