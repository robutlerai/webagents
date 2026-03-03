/**
 * WebSearchSkill Unit Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WebSearchSkill } from '../../../../src/skills/browser/search.js';

describe('WebSearchSkill', () => {
  let skill: WebSearchSkill;

  beforeEach(() => {
    skill = new WebSearchSkill();
  });

  describe('initialization', () => {
    it('should have correct id and description', () => {
      expect(skill.id).toBe('web-search');
      expect(skill.description).toBe('Search the web using DuckDuckGo, Google, or Bing');
    });

    it('should default to duckduckgo engine', () => {
      const defaultSkill = new WebSearchSkill();
      expect(defaultSkill['searchConfig'].defaultEngine).toBe('duckduckgo');
    });

    it('should accept custom configuration', () => {
      const customSkill = new WebSearchSkill({
        defaultEngine: 'google',
        maxResults: 20,
        timeout: 5000,
        googleApiKey: 'test-key',
        googleCseId: 'test-cse',
      });
      expect(customSkill['searchConfig'].defaultEngine).toBe('google');
      expect(customSkill['searchConfig'].maxResults).toBe(20);
      expect(customSkill['searchConfig'].timeout).toBe(5000);
    });
  });

  describe('searchDuckDuckGo', () => {
    it('should return search results', async () => {
      // Mock fetch
      const mockHtml = `
        <html>
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Example Site</a>
          <a class="result__snippet">This is an example snippet</a>
        </html>
      `;
      
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(mockHtml),
      });

      const result = await skill.searchDuckDuckGo('test query', 10);
      
      expect(result.query).toBe('test query');
      expect(result.error).toBeUndefined();
    });

    it('should handle network errors gracefully', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const result = await skill.searchDuckDuckGo('test query', 10);
      
      expect(result.query).toBe('test query');
      expect(result.results).toEqual([]);
      expect(result.error).toBe('Network error');
    });

    it('should handle non-ok responses', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
      });

      const result = await skill.searchDuckDuckGo('test query', 10);
      
      expect(result.error).toBe('HTTP 429');
    });
  });

  describe('searchGoogle', () => {
    it('should require API key and CSE ID', async () => {
      const result = await skill.searchGoogle('test query', 10);
      
      expect(result.error).toBe('Google API key and CSE ID required');
    });

    it('should make API call with credentials', async () => {
      const googleSkill = new WebSearchSkill({
        googleApiKey: 'test-api-key',
        googleCseId: 'test-cse-id',
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          items: [
            { title: 'Result 1', link: 'https://example.com/1', snippet: 'Snippet 1' },
            { title: 'Result 2', link: 'https://example.com/2', snippet: 'Snippet 2' },
          ],
          searchInformation: { totalResults: '100' },
        }),
      });

      const result = await googleSkill.searchGoogle('test query', 10);
      
      expect(result.query).toBe('test query');
      expect(result.results).toHaveLength(2);
      expect(result.results[0].title).toBe('Result 1');
      expect(result.results[0].source).toBe('google');
      expect(result.totalResults).toBe(100);
    });
  });

  describe('searchBing', () => {
    it('should require API key', async () => {
      const result = await skill.searchBing('test query', 10);
      
      expect(result.error).toBe('Bing API key required');
    });

    it('should make API call with credentials', async () => {
      const bingSkill = new WebSearchSkill({
        bingApiKey: 'test-bing-key',
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          webPages: {
            value: [
              { name: 'Bing Result', url: 'https://example.com', snippet: 'Bing snippet' },
            ],
            totalEstimatedMatches: 500,
          },
        }),
      });

      const result = await bingSkill.searchBing('test query', 10);
      
      expect(result.results).toHaveLength(1);
      expect(result.results[0].title).toBe('Bing Result');
      expect(result.results[0].source).toBe('bing');
      expect(result.totalResults).toBe(500);
    });
  });

  describe('fetchPage', () => {
    it('should extract title and content from HTML', async () => {
      const mockHtml = `
        <html>
          <head><title>Test Page Title</title></head>
          <body>
            <script>console.log('hidden');</script>
            <style>.hidden { display: none; }</style>
            <p>This is the main content.</p>
            <noscript>No JS content</noscript>
          </body>
        </html>
      `;

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(mockHtml),
      });

      const result = await skill.fetchPage('https://example.com', 5000);
      
      expect(result.title).toBe('Test Page Title');
      expect(result.content).toContain('This is the main content');
      expect(result.content).not.toContain('console.log');
      expect(result.content).not.toContain('No JS content');
      expect(result.url).toBe('https://example.com');
    });

    it('should truncate long content', async () => {
      const longContent = 'a'.repeat(10000);
      const mockHtml = `<html><body>${longContent}</body></html>`;

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(mockHtml),
      });

      const result = await skill.fetchPage('https://example.com', 100);
      
      expect(result.content.length).toBeLessThanOrEqual(103); // 100 + '...'
      expect(result.content.endsWith('...')).toBe(true);
    });

    it('should handle fetch errors', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Connection refused'));

      const result = await skill.fetchPage('https://example.com', 5000);
      
      expect(result.error).toBe('Connection refused');
      expect(result.title).toBe('');
      expect(result.content).toBe('');
    });
  });

  describe('search (unified)', () => {
    it('should use default engine', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve('<html></html>'),
      });

      await skill.search('test', 5);
      
      // Should call DuckDuckGo by default
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('html.duckduckgo.com'),
        expect.any(Object)
      );
    });

    it('should respect configured default engine', async () => {
      const googleSkill = new WebSearchSkill({
        defaultEngine: 'google',
        googleApiKey: 'key',
        googleCseId: 'cse',
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ items: [] }),
      });

      await googleSkill.search('test', 5);
      
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('googleapis.com'),
        expect.any(Object)
      );
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });
});
