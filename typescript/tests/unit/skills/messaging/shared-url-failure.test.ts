import { describe, expect, it } from 'vitest';

import {
  URL_FETCH_FAILURE_PATTERNS,
  isUrlFetchFailure,
} from '../../../../src/skills/messaging/shared/url-failure';

describe('shared/url-failure', () => {
  it('exports a non-empty list of platform-agnostic patterns', () => {
    expect(URL_FETCH_FAILURE_PATTERNS.length).toBeGreaterThan(0);
  });

  describe('isUrlFetchFailure', () => {
    it('returns false for undefined / empty messages', () => {
      expect(isUrlFetchFailure(undefined)).toBe(false);
      expect(isUrlFetchFailure('')).toBe(false);
    });

    it('matches Telegram URL-fetch error wording', () => {
      expect(
        isUrlFetchFailure('Bad Request: failed to get HTTP URL content'),
      ).toBe(true);
      expect(
        isUrlFetchFailure('Bad Request: wrong type of the web page content'),
      ).toBe(true);
      expect(
        isUrlFetchFailure('Bad Request: wrong remote file identifier'),
      ).toBe(true);
    });

    it('matches generic transport-level failures', () => {
      expect(isUrlFetchFailure('TypeError: fetch failed')).toBe(true);
      expect(isUrlFetchFailure('connect ECONNRESET 1.2.3.4:443')).toBe(true);
      expect(isUrlFetchFailure('getaddrinfo ENOTFOUND example.com')).toBe(true);
      expect(isUrlFetchFailure('the request timed out')).toBe(true);
    });

    it('does not match unrelated error messages', () => {
      expect(isUrlFetchFailure('Forbidden: bot was blocked by the user')).toBe(false);
      expect(isUrlFetchFailure('Unauthorized')).toBe(false);
    });

    it('honours extra patterns from the caller', () => {
      const extra = [/failed to download/i, /unable to fetch media/i];
      expect(isUrlFetchFailure('Failed to download the media', extra)).toBe(true);
      expect(isUrlFetchFailure('We were unable to fetch media', extra)).toBe(true);
      expect(isUrlFetchFailure('Failed to download the media')).toBe(false);
    });
  });
});
