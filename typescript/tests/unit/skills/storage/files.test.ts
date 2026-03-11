/**
 * Unit tests for RobutlerFilesSkill (files tool)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { RobutlerFilesSkill } from '../../../../src/skills/storage/skill.js';

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

describe('RobutlerFilesSkill', () => {
  const skill = new RobutlerFilesSkill({
    portalUrl: 'http://localhost:3000',
    apiKey: 'test-key',
    agentId: 'agent-1',
  });

  const ctx = createMockContext();

  describe('files - upload action', () => {
    it('upload action calls POST /api/storage/files with {path, content, mimeType, visibility, agentId}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { path: 'reports/q1.pdf', size: 1024 }),
      );

      const result = await skill.files(
        {
          action: 'upload',
          path: 'reports/q1.pdf',
          content: 'base64content',
          mime_type: 'application/pdf',
        },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/files',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-key',
          }),
          body: JSON.stringify({
            path: 'reports/q1.pdf',
            content: 'base64content',
            mimeType: 'application/pdf',
            visibility: 'private',
            agentId: 'agent-1',
          }),
        }),
      );
      expect(result).toEqual({ path: 'reports/q1.pdf', size: 1024 });
    });

    it('validates required params (path, content, mime_type) for upload', async () => {
      const noPath = await skill.files(
        { action: 'upload', content: 'x', mime_type: 'text/plain' },
        ctx,
      );
      expect(noPath).toEqual({ error: 'path is required for upload' });

      const noContent = await skill.files(
        { action: 'upload', path: 'p', mime_type: 'text/plain' },
        ctx,
      );
      expect(noContent).toEqual({ error: 'content is required for upload' });

      const noMime = await skill.files(
        { action: 'upload', path: 'p', content: 'x' },
        ctx,
      );
      expect(noMime).toEqual({ error: 'mime_type is required for upload' });

      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('files - download action', () => {
    it('download action calls GET /api/storage/files/{path}?agentId=agent-1', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { content: 'base64data', mimeType: 'image/png' }),
      );

      const result = await skill.files(
        { action: 'download', path: 'images/photo.png' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/files/images%2Fphoto.png?agentId=agent-1',
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({ content: 'base64data', mimeType: 'image/png' });
    });

    it('requires path for download', async () => {
      const result = await skill.files({ action: 'download' }, ctx);
      expect(result).toEqual({ error: 'path is required for download' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('files - list action', () => {
    it('list action calls GET /api/storage/files?agentId=agent-1&prefix=...', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { files: [{ path: 'reports/a.pdf' }, { path: 'reports/b.pdf' }] }),
      );

      const result = await skill.files(
        { action: 'list', prefix: 'reports/' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/files?agentId=agent-1&prefix=reports%2F',
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({
        files: [{ path: 'reports/a.pdf' }, { path: 'reports/b.pdf' }],
      });
    });
  });

  describe('files - delete action', () => {
    it('delete action calls DELETE /api/storage/files/{path}?agentId=agent-1', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {}),
      );

      const result = await skill.files(
        { action: 'delete', path: 'temp/old.txt' },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/files/temp%2Fold.txt?agentId=agent-1',
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toBe('OK');
    });

    it('requires path for delete', async () => {
      const result = await skill.files({ action: 'delete' }, ctx);
      expect(result).toEqual({ error: 'path is required for delete' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('files - get_url action', () => {
    it('get_url action calls POST /api/storage/files/{path}/url with {expiresIn, agentId}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { url: 'https://cdn.example.com/signed-url', expiresAt: '2024-02-01T00:00:00Z' }),
      );

      const result = await skill.files(
        { action: 'get_url', path: 'shared/doc.pdf', expires_in: 7200 },
        ctx,
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/storage/files/shared%2Fdoc.pdf/url',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-key',
          }),
          body: JSON.stringify({
            expiresIn: 7200,
            agentId: 'agent-1',
          }),
        }),
      );
      expect(result).toEqual({
        url: 'https://cdn.example.com/signed-url',
        expiresAt: '2024-02-01T00:00:00Z',
      });
    });

    it('uses default expires_in of 3600 when not provided', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { url: 'https://example.com/url' }),
      );

      await skill.files(
        { action: 'get_url', path: 'file.txt' },
        ctx,
      );

      const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(call[1].body);
      expect(body.expiresIn).toBe(3600);
    });

    it('requires path for get_url', async () => {
      const result = await skill.files({ action: 'get_url' }, ctx);
      expect(result).toEqual({ error: 'path is required for get_url' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });

  describe('files - unknown action', () => {
    it('unknown action returns error', async () => {
      const result = await skill.files(
        { action: 'invalid' },
        ctx,
      );

      expect(result).toEqual({
        error: 'Unknown action: invalid. Use upload, download, list, delete, or get_url.',
      });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  });
});
