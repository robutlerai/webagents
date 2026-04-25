import { describe, expect, it } from 'vitest';

import {
  CONTENT_PATH_RE,
  UUID_RE,
  extractContentId,
} from '../../../../src/skills/messaging/shared/content-id';

const VALID_UUID = '11111111-2222-4333-8444-555555555555';

describe('shared/content-id', () => {
  it('UUID_RE accepts canonical lowercase v4 UUIDs', () => {
    expect(UUID_RE.test(VALID_UUID)).toBe(true);
  });

  it('UUID_RE accepts mixed case', () => {
    expect(UUID_RE.test(VALID_UUID.toUpperCase())).toBe(true);
  });

  it('UUID_RE rejects partial / malformed UUIDs', () => {
    expect(UUID_RE.test('not-a-uuid')).toBe(false);
    expect(UUID_RE.test('11111111-2222-4333-8444')).toBe(false);
    expect(UUID_RE.test('  ')).toBe(false);
  });

  it('CONTENT_PATH_RE captures the UUID embedded in /api/content/<uuid>', () => {
    const m = `/api/content/${VALID_UUID}`.match(CONTENT_PATH_RE);
    expect(m?.[1]).toBe(VALID_UUID);
  });

  it('CONTENT_PATH_RE matches when the path appears inside a longer URL', () => {
    const url = `https://portal.test/api/content/${VALID_UUID}?signed=1`;
    const m = url.match(CONTENT_PATH_RE);
    expect(m?.[1]).toBe(VALID_UUID);
  });

  describe('extractContentId', () => {
    it('returns the explicit contentId when valid', () => {
      expect(extractContentId({ contentId: VALID_UUID })).toBe(VALID_UUID);
    });

    it('trims surrounding whitespace from contentId', () => {
      expect(extractContentId({ contentId: `  ${VALID_UUID}  ` })).toBe(VALID_UUID);
    });

    it('falls back to extracting from a relative /api/content URL', () => {
      expect(extractContentId({ url: `/api/content/${VALID_UUID}` })).toBe(VALID_UUID);
    });

    it('falls back to extracting from an absolute URL with the path', () => {
      expect(
        extractContentId({ url: `https://portal.test/api/content/${VALID_UUID}` }),
      ).toBe(VALID_UUID);
    });

    it('prefers explicit valid contentId over URL', () => {
      const other = '99999999-aaaa-4bbb-8ccc-ddddeeeefff0';
      expect(
        extractContentId({ contentId: VALID_UUID, url: `/api/content/${other}` }),
      ).toBe(VALID_UUID);
    });

    it('returns null for absolute external URLs without a content path', () => {
      expect(extractContentId({ url: 'https://example.com/foo.jpg' })).toBeNull();
    });

    it('returns null when both inputs are missing', () => {
      expect(extractContentId({})).toBeNull();
    });

    it('returns null when contentId is malformed and no URL is provided', () => {
      expect(extractContentId({ contentId: 'not-a-uuid' })).toBeNull();
    });
  });
});
