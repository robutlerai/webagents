/**
 * Unit tests for RobutlerKVSkill (memory tool)
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
    it('get action calls GET /api/storage/kv/{key}?agentId=agent-1', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { value: 'stored-value' }),
      );

      const result = await skill.memory(
        { action: 'get', key: 'my-key' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/kv/my-key?agentId=agent-1',
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

      const result = await skill.memory(
        { action: 'get', key: 'missing-key' },
        ctx,
      );

      expect(result).toBeNull();
    });

    it('requires key for get', async () => {
      const result = await skill.memory({ action: 'get' }, ctx);
      expect(result).toEqual({ error: 'key is required for get' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('memory - set action', () => {
    it('set action calls PUT /api/storage/kv/{key} with {value, ttl, agentId}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await skill.memory(
        { action: 'set', key: 'pref', value: { theme: 'dark' }, ttl: 3600 },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/kv/pref',
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-key',
          }),
          body: JSON.stringify({
            value: { theme: 'dark' },
            ttl: 3600,
            agentId: 'agent-1',
          }),
        }),
      );
      expect(result).toBe('OK');
    });

    it('validates required params (key, value) for set', async () => {
      const noKey = await skill.memory({ action: 'set', value: 'x' }, ctx);
      expect(noKey).toEqual({ error: 'key is required for set' });

      const noValue = await skill.memory({ action: 'set', key: 'k' }, ctx);
      expect(noValue).toEqual({ error: 'value is required for set' });

      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('memory - delete action', () => {
    it('delete action calls DELETE /api/storage/kv/{key}?agentId=agent-1', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await skill.memory(
        { action: 'delete', key: 'old-key' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/kv/old-key?agentId=agent-1',
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toBe('OK');
    });

    it('requires key for delete', async () => {
      const result = await skill.memory({ action: 'delete' }, ctx);
      expect(result).toEqual({ error: 'key is required for delete' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('memory - list action', () => {
    it('list action calls GET /api/storage/kv?agentId=agent-1&prefix=...', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { keys: ['pref:a', 'pref:b'] }),
      );

      const result = await skill.memory(
        { action: 'list', prefix: 'pref:' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/kv?agentId=agent-1&prefix=pref%3A',
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ keys: ['pref:a', 'pref:b'] });
    });

    it('list without prefix', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { keys: ['a', 'b', 'c'] }),
      );

      await skill.memory({ action: 'list' }, ctx);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/kv?agentId=agent-1',
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
    });
  });

  describe('memory - unknown action', () => {
    it('unknown action returns error', async () => {
      const result = await skill.memory(
        { action: 'invalid' },
        ctx,
      );

      expect(result).toEqual({
        error: 'Unknown action: invalid. Use get, set, delete, or list.',
      });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });
});
