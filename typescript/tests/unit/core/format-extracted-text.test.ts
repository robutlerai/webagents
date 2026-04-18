/**
 * Tests for formatExtractedText (read_content text-extraction formatter).
 *
 * Covers the scenarios in the "Fix delegate attachment access" plan:
 * - default head/tail summary
 * - view_range slicing
 * - regex search with default ±2 context
 * - regex search with metachars (^class \w+)
 * - asymmetric before/after windows
 * - pagination via offset+limit and accurate total count
 * - configurable limit, server-side cap at 200
 * - validation of view_range/search precedence
 * - overlap merging vs disjoint windows
 * - invalid regex error
 */

import { describe, it, expect } from 'vitest';
import { formatExtractedText } from '../../../src/core/agent.js';

const buildText = (n: number, fn?: (i: number) => string) =>
  Array.from({ length: n }, (_, i) => (fn ? fn(i + 1) : `line ${i + 1}`)).join('\n');

const makeOpts = (text: string, overrides: Partial<Parameters<typeof formatExtractedText>[0]> = {}) => ({
  filename: 'unicorn.html',
  text,
  totalLines: text === '' ? 0 : text.split('\n').length,
  byteSize: Buffer.byteLength(text, 'utf-8'),
  ...overrides,
});

describe('formatExtractedText', () => {
  describe('default summary', () => {
    it('returns the whole file when small (<= head+tail)', () => {
      const text = buildText(50);
      const out = formatExtractedText(makeOpts(text));
      expect(out).toContain('unicorn.html (50 lines, ');
      expect(out).toContain('1\tline 1');
      expect(out).toContain('50\tline 50');
      expect(out).not.toContain('elided');
    });

    it('returns first 200 + last 50 with elision marker for large files', () => {
      const text = buildText(500);
      const out = formatExtractedText(makeOpts(text));
      expect(out).toContain('showing 1-200 + tail 50');
      expect(out).toContain('1\tline 1');
      expect(out).toContain('200\tline 200');
      expect(out).toContain('451\tline 451');
      expect(out).toContain('500\tline 500');
      expect(out).toContain('250 lines elided');
      expect(out).not.toContain('201\tline 201');
    });
  });

  describe('view_range', () => {
    it('returns exactly the requested range with 1-indexed numbers', () => {
      const text = buildText(100);
      const out = formatExtractedText(makeOpts(text, { viewRange: [10, 20] }));
      expect(out).toContain('showing 10-20');
      expect(out.split('\n').filter((l) => /^\s*\d+\t/.test(l))).toHaveLength(11);
      expect(out).toContain('10\tline 10');
      expect(out).toContain('20\tline 20');
      expect(out).not.toContain('21\tline 21');
    });

    it('clamps to file bounds', () => {
      const text = buildText(10);
      const out = formatExtractedText(makeOpts(text, { viewRange: [5, 9999] }));
      expect(out).toContain('showing 5-10');
    });
  });

  describe('search (default ±2 context)', () => {
    it('returns hit lines with default 2 lines of leading and trailing context', () => {
      const text = buildText(20, (i) => (i === 10 ? 'foo here' : `line ${i}`));
      const out = formatExtractedText(makeOpts(text, { search: 'foo' }));
      expect(out).toContain('search=/foo/gm, hits 1-1 of 1');
      expect(out).toContain('before=2, after=2');
      expect(out).toContain(' 8\tline 8');
      expect(out).toContain(' 9\tline 9');
      expect(out).toContain('10:foo here');
      expect(out).toContain('11\tline 11');
      expect(out).toContain('12\tline 12');
      expect(out).not.toContain(' 7\tline 7');
      expect(out).not.toContain('13\tline 13');
    });

    it('treats search as regex (metachars)', () => {
      const lines = [
        'import x from y',
        'class Foo extends Bar {',
        '  doStuff() {}',
        '}',
        'class Baz {',
        '}',
      ];
      const text = lines.join('\n');
      const out = formatExtractedText(makeOpts(text, { search: '^class\\s+\\w+', before: 0, after: 0 }));
      expect(out).toContain('hits 1-2 of 2');
      expect(out).toContain('class Foo extends Bar');
      expect(out).toContain('class Baz');
    });
  });

  describe('search context configuration', () => {
    it('asymmetric before=0 after=10 acts as log-tail', () => {
      const text = buildText(50);
      const out = formatExtractedText(makeOpts(text, { search: '^line 20$', before: 0, after: 5 }));
      expect(out).toContain('hits 1-1 of 1');
      expect(out).toContain('20:line 20');
      expect(out).toContain('21\tline 21');
      expect(out).toContain('25\tline 25');
      expect(out).not.toContain('19\tline 19');
      expect(out).not.toContain('26\tline 26');
    });

    it('clamps before/after to [0, 20]', () => {
      const text = buildText(200);
      const out = formatExtractedText(makeOpts(text, { search: '^line 100$', before: 9999, after: 9999 }));
      expect(out).toContain('before=20, after=20');
      expect(out).toContain('80\tline 80');
      expect(out).toContain('100:line 100');
      expect(out).toContain('120\tline 120');
      expect(out).not.toContain('79\tline 79');
      expect(out).not.toContain('121\tline 121');
    });
  });

  describe('pagination', () => {
    it('default limit=30 returns 30 hits with footer pointing at next offset', () => {
      const text = Array.from({ length: 100 }, (_, i) => `match ${i + 1}`).join('\n');
      const out = formatExtractedText(makeOpts(text, { search: '^match', before: 0, after: 0 }));
      expect(out).toContain('hits 1-30 of 100');
      expect(out).toContain('70 more hits');
      expect(out).toContain('offset=30');
    });

    it('next page with offset=30 returns the next 30 hits', () => {
      const text = Array.from({ length: 100 }, (_, i) => `match ${i + 1}`).join('\n');
      const out = formatExtractedText(makeOpts(text, { search: '^match', offset: 30, before: 0, after: 0 }));
      expect(out).toContain('hits 31-60 of 100');
      expect(out).toContain('40 more hits');
      expect(out).toContain('offset=60');
    });

    it('configurable limit returns more hits per page', () => {
      const text = Array.from({ length: 100 }, (_, i) => `match ${i + 1}`).join('\n');
      const out = formatExtractedText(makeOpts(text, { search: '^match', limit: 100, before: 0, after: 0 }));
      expect(out).toContain('hits 1-100 of 100');
      expect(out).not.toContain('more hits');
    });

    it('clamps limit > 200 down to 200', () => {
      const text = Array.from({ length: 250 }, (_, i) => `match ${i + 1}`).join('\n');
      const out = formatExtractedText(makeOpts(text, { search: '^match', limit: 5000, before: 0, after: 0 }));
      expect(out).toContain('hits 1-200 of 250');
      expect(out).toContain('limit=200');
      expect(out).toContain('50 more hits');
      expect(out).toContain('offset=200');
    });

    it('total hit count is consistent across limit values', () => {
      const text = Array.from({ length: 75 }, (_, i) => `match ${i + 1}`).join('\n');
      const a = formatExtractedText(makeOpts(text, { search: '^match', limit: 30 }));
      const b = formatExtractedText(makeOpts(text, { search: '^match', limit: 100 }));
      expect(a).toContain('of 75');
      expect(b).toContain('of 75');
    });
  });

  describe('window merging', () => {
    it('merges adjacent hits with overlapping context (no -- separator, no duplicated lines)', () => {
      // Hits at line 10 and line 12 with default before=2/after=2 → windows
      // 8-12 and 10-14 → merged into 8-14.
      const text = buildText(20, (i) => (i === 10 || i === 12 ? `match ${i}` : `line ${i}`));
      const out = formatExtractedText(makeOpts(text, { search: '^match' }));
      expect(out).toContain('hits 1-2 of 2');
      expect(out).not.toContain('\n--\n');
      const block = out.split('\n').filter((l) => /^[\s\d]+[\t:]/.test(l));
      // Should include lines 8..14, single contiguous block (7 lines).
      expect(block).toHaveLength(7);
      expect(out).toContain(' 8\tline 8');
      expect(out).toContain('10:match 10');
      expect(out).toContain('11\tline 11');
      expect(out).toContain('12:match 12');
      expect(out).toContain('14\tline 14');
    });

    it('separates disjoint windows with --', () => {
      const text = buildText(40, (i) => (i === 5 || i === 30 ? `match ${i}` : `line ${i}`));
      const out = formatExtractedText(makeOpts(text, { search: '^match', before: 1, after: 1 }));
      expect(out).toContain('--');
    });
  });

  describe('precedence and validation', () => {
    it('search wins over view_range and notes view_range ignored in header', () => {
      const text = buildText(50, (i) => (i === 25 ? 'needle here' : `line ${i}`));
      const out = formatExtractedText(makeOpts(text, { search: 'needle', viewRange: [1, 5] }));
      expect(out).toContain('view_range ignored');
      expect(out).toContain('25:needle here');
    });

    it('returns Invalid regex error for malformed pattern', () => {
      const text = buildText(10);
      const out = formatExtractedText(makeOpts(text, { search: 'foo[bar' }));
      expect(out).toMatch(/^Invalid regex:/);
      expect(out).toContain('Pattern: "foo[bar"');
    });

    it('reports zero hits with a hint to broaden the pattern', () => {
      const text = buildText(10);
      const out = formatExtractedText(makeOpts(text, { search: 'definitely-not-here' }));
      expect(out).toContain('0 hits');
      expect(out).toContain('No matches');
    });
  });
});
