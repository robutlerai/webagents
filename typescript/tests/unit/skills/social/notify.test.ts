/**
 * Unit tests for NotificationsSkill
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NotificationsSkill } from '../../../../src/skills/social/skill.js';

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
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
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

describe('NotificationsSkill', () => {
  const skill = new NotificationsSkill({
    portalUrl: 'http://localhost:3000',
    apiKey: 'test-key',
    agentId: 'agent-1',
    ownerId: 'owner-123',
  });

  describe('notify - send action', () => {
    it('sends action calls POST /api/notifications/send with correct payload', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { notificationId: 'notif-123', sent: true }),
      );

      const result = await skill.notify(
        {
          action: 'send',
          title: 'Task complete',
          message: 'Image generated successfully',
        },
        createMockContext(),
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/notifications/send',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-key',
          }),
          body: JSON.stringify({
            title: 'Task complete',
            body: 'Image generated successfully',
            type: 'agent_update',
            priority: 'normal',
            userIds: ['owner-123'],
            requireInteraction: false,
            data: { agentName: 'agent-1', priority: 'normal' },
          }),
        }),
      );
      expect(result).toEqual({
        success: true,
        notification_id: 'notif-123',
        sent: true,
      });
    });

    it('requires title for send', async () => {
      const result = await skill.notify(
        { action: 'send', message: 'Body only' },
        createMockContext(),
      );
      expect(result).toEqual({ error: 'title is required' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('requires message for send', async () => {
      const result = await skill.notify(
        { action: 'send', title: 'Title only' },
        createMockContext(),
      );
      expect(result).toEqual({ error: 'message is required' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('passes CTA actions through in data.actions', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { notificationId: 'notif-456', sent: true }),
      );

      await skill.notify(
        {
          action: 'send',
          title: 'Deploy to prod?',
          message: 'v2.1.0 ready',
          actions: [
            { label: 'Approve', action: 'approve' },
            { label: 'Deny', action: 'deny' },
          ],
        },
        createMockContext(),
      );

      const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(call[1].body);
      expect(body.data.actions).toEqual([
        { label: 'Approve', action: 'approve' },
        { label: 'Deny', action: 'deny' },
      ]);
      expect(body.requireInteraction).toBe(true);
    });

    it('returns notification_id on success', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, { id: 'alt-id' }),
      );

      const result = await skill.notify(
        { title: 'Hi', message: 'Hello' },
        createMockContext(),
      );

      expect(result).toMatchObject({
        success: true,
        notification_id: 'alt-id',
      });
    });

    it('handles error response', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(500, { error: 'Internal server error' }),
      );

      const result = await skill.notify(
        { title: 'Hi', message: 'Hello' },
        createMockContext(),
      );

      expect(result).toMatchObject({
        error: expect.stringContaining('Notification failed'),
      });
    });
  });

  describe('notify - status action', () => {
    it('status action calls GET /api/notifications/{id}', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(200, {
          actionTaken: 'approve',
          updatedAt: '2024-01-15T12:00:00Z',
          isRead: true,
        }),
      );

      const result = await skill.notify(
        { action: 'status', notification_id: 'notif-789' },
        createMockContext(),
      );

      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:3000/api/notifications/notif-789',
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer test-key' }),
        }),
      );
      expect(result).toEqual({
        notification_id: 'notif-789',
        actionTaken: 'approve',
        actionAt: '2024-01-15T12:00:00Z',
        isRead: true,
      });
    });

    it('requires notification_id for status', async () => {
      const result = await skill.notify(
        { action: 'status' },
        createMockContext(),
      );
      expect(result).toEqual({ error: 'notification_id is required for status check' });
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('handles 404 for status', async () => {
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockResponse(404, {}),
      );

      const result = await skill.notify(
        { action: 'status', notification_id: 'missing' },
        createMockContext(),
      );

      expect(result).toEqual({ error: 'Notification not found' });
    });
  });
});
