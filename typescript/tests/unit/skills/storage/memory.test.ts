/**
 * Unit tests for RobutlerMemorySkill (memory tool, file-system metaphor)
 *
 * The memory tool is registered via registerTool() with a handler,
 * so we invoke it through the skill's tools array.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { RobutlerMemorySkill } from '../../../../src/skills/storage/skill.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockContext(): any {
  return {
    auth: { authenticated: true, agentId: 'test-agent', userId: 'owner-123' },
    metadata: {},
    session: {},
    get: vi.fn(),
    set: vi.fn(),
    delete: vi.fn(),
    hasScope: vi.fn().mockReturnValue(true),
    hasScopes: vi.fn().mockReturnValue(true),
  };
}

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

/**
 * Invoke the 'memory' tool registered on the skill.
 */
function callMemory(skill: RobutlerMemorySkill, params: Record<string, unknown>, ctx: any): Promise<unknown> {
  const memoryTool = (skill as any).tools.find((t: any) => t.name === 'memory');
  if (!memoryTool?.handler) throw new Error('memory tool not found on skill');
  return memoryTool.handler(params, ctx);
}

// ---------------------------------------------------------------------------
// Global fetch mock
// ---------------------------------------------------------------------------

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('RobutlerMemorySkill (file-system interface)', () => {
  const skill = new RobutlerMemorySkill({
    portalUrl: 'http://localhost:3000',
    apiKey: 'test-key',
    agentId: 'agent-1',
  });

  const ctx = createMockContext();

  describe('view command', () => {
    it('view directory lists entries', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { keys: ['topic-a', 'topic-b'] }),
      );

      const result = await callMemory(skill, { command: 'view', path: '/memories/' }, ctx);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory?'),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ keys: ['topic-a', 'topic-b'] });
    });

    it('view file reads entry', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: 'stored content' }),
      );

      const result = await callMemory(skill, { command: 'view', path: '/memories/topic.md' }, ctx);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/topic'),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ value: 'stored content' });
    });

    it('view returns null for 404', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(404, {}),
      );

      const result = await callMemory(skill, { command: 'view', path: '/memories/missing.md' }, ctx);
      expect(result).toBeNull();
    });
  });

  describe('create command', () => {
    it('create calls PUT with content', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await callMemory(
        skill,
        { command: 'create', path: '/memories/notes.md', content: 'hello world' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/notes'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        }),
      );
      const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(call[1].body);
      expect(body.value).toBe('hello world');
      expect(body.agentId).toBe('agent-1');
      expect(result).toBe('Created /memories/notes.md');
    });

    it('create requires path and content', async () => {
      const noPath = await callMemory(skill, { command: 'create', content: 'x' }, ctx);
      expect(noPath).toEqual({ error: 'path is required for create' });

      const noContent = await callMemory(skill, { command: 'create', path: '/memories/x.md' }, ctx);
      expect(noContent).toEqual({ error: 'content is required for create' });

      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('edit command', () => {
    it('edit performs str_replace on stored value', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockResponse(200, { value: 'hello world' }))
        .mockResolvedValueOnce(mockResponse(200, {}));

      const result = await callMemory(
        skill,
        { command: 'edit', path: '/memories/notes.md', old_str: 'hello', new_str: 'goodbye' },
        ctx,
      );

      expect(result).toBe('Edited /memories/notes.md');
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });

    it('edit requires path, old_str, new_str', async () => {
      const result = await callMemory(skill, { command: 'edit', path: '/memories/x.md' }, ctx);
      expect(result).toEqual({ error: 'old_str and new_str are required for edit' });
    });
  });

  describe('delete command', () => {
    it('delete calls DELETE', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await callMemory(
        skill,
        { command: 'delete', path: '/memories/old.md' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/old'),
        expect.objectContaining({ method: 'DELETE' }),
      );
      expect(result).toBe('Deleted /memories/old.md');
    });

    it('delete requires path', async () => {
      const result = await callMemory(skill, { command: 'delete' }, ctx);
      expect(result).toEqual({ error: 'path is required for delete' });
    });
  });

  describe('rename command', () => {
    it('rename performs get + set + delete', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockResponse(200, { value: 'content' }))
        .mockResolvedValueOnce(mockResponse(200, {}))
        .mockResolvedValueOnce(mockResponse(200, {}));

      const result = await callMemory(
        skill,
        { command: 'rename', path: '/memories/old.md', new_str: '/memories/new.md' },
        ctx,
      );

      expect(result).toContain('Renamed');
      expect(globalThis.fetch).toHaveBeenCalledTimes(3);
    });

    it('rename requires path and new_str', async () => {
      const result = await callMemory(skill, { command: 'rename', path: '/memories/x.md' }, ctx);
      expect(result).toEqual({ error: 'path and new_str (new path) are required for rename' });
    });
  });

  describe('search command', () => {
    it('search calls GET with query params', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { results: [{ key: 'notes', value: 'found' }] }),
      );

      const result = await callMemory(skill, { command: 'search', query: 'hello' }, ctx);

      const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toContain('action=search');
      expect(url).toContain('q=hello');
      expect(result).toEqual({ results: [{ key: 'notes', value: 'found' }] });
    });

    it('search requires query', async () => {
      const result = await callMemory(skill, { command: 'search' }, ctx);
      expect(result).toEqual({ error: 'query is required for search' });
    });
  });

  describe('stores command', () => {
    it('stores lists accessible stores', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { stores: ['store-1', 'store-2'] }),
      );

      const result = await callMemory(skill, { command: 'stores' }, ctx);
      expect(result).toEqual({ stores: ['store-1', 'store-2'] });
    });
  });

  describe('unknown command', () => {
    it('returns error', async () => {
      const result = await callMemory(skill, { command: 'invalid' }, ctx);
      expect(result).toEqual({ error: expect.stringContaining('Unknown command: invalid') });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });
});
