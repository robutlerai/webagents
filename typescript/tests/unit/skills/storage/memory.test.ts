/**
 * Unit tests for RobutlerMemorySkill (memory tool)
 *
 * The memory tool is registered via registerTool() with a handler,
 * so we invoke it through the skill's tools array.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { RobutlerKVSkill } from '../../../../src/skills/storage/skill.js';

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
function callMemory(skill: RobutlerKVSkill, params: Record<string, unknown>, ctx: any): Promise<unknown> {
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

describe('RobutlerKVSkill', () => {
  const skill = new RobutlerKVSkill({
    portalUrl: 'http://localhost:3000',
    apiKey: 'test-key',
    agentId: 'agent-1',
  });

  const ctx = createMockContext();

  describe('memory - get action', () => {
    it('get action calls GET /api/storage/memory/{key}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: 'stored-value' }),
      );

      const result = await callMemory(skill, { action: 'get', key: 'my-key' }, ctx);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/my-key'),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ value: 'stored-value' });
    });

    it('get returns null for 404', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(404, {}),
      );

      const result = await callMemory(skill, { action: 'get', key: 'missing-key' }, ctx);
      expect(result).toBeNull();
    });

    it('requires key for get', async () => {
      const result = await callMemory(skill, { action: 'get' }, ctx);
      expect(result).toEqual({ error: 'key is required for get' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('memory - set action', () => {
    it('set action calls PUT /api/storage/memory/{key}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await callMemory(
        skill,
        { action: 'set', key: 'pref', value: { theme: 'dark' }, ttl: 3600 },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/pref'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-key',
          }),
        }),
      );

      const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(call[1].body);
      expect(body.value).toEqual({ theme: 'dark' });
      expect(body.ttl).toBe(3600);
      expect(body.agentId).toBe('agent-1');
      expect(result).toBe('OK');
    });

    it('validates required params (key, value) for set', async () => {
      const noKey = await callMemory(skill, { action: 'set', value: 'x' }, ctx);
      expect(noKey).toEqual({ error: 'key is required for set' });

      const noValue = await callMemory(skill, { action: 'set', key: 'k' }, ctx);
      expect(noValue).toEqual({ error: 'value is required for set' });

      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('memory - delete action', () => {
    it('delete action calls DELETE /api/storage/memory/{key}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await callMemory(
        skill,
        { action: 'delete', key: 'old-key' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory/old-key'),
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toBe('OK');
    });

    it('requires key for delete', async () => {
      const result = await callMemory(skill, { action: 'delete' }, ctx);
      expect(result).toEqual({ error: 'key is required for delete' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('memory - list action', () => {
    it('list action calls GET /api/storage/memory with query params', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { keys: ['pref:a', 'pref:b'] }),
      );

      const result = await callMemory(
        skill,
        { action: 'list', prefix: 'pref:' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/storage/memory?'),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toContain('agentId=agent-1');
      expect(url).toContain('prefix=pref');
      expect(result).toEqual({ keys: ['pref:a', 'pref:b'] });
    });

    it('list without prefix', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { keys: ['a', 'b', 'c'] }),
      );

      await callMemory(skill, { action: 'list' }, ctx);

      const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toContain('/api/storage/memory?');
      expect(url).toContain('agentId=agent-1');
    });
  });

  describe('memory - unknown action', () => {
    it('unknown action returns error', async () => {
      const result = await callMemory(skill, { action: 'invalid' }, ctx);

      expect(result).toEqual({
        error: expect.stringContaining('Unknown action: invalid'),
      });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });
});
