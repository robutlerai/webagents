/**
 * AgentRegistry Unit Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AgentRegistry } from '../../../src/daemon/registry.js';
import type { IAgent } from '../../../src/core/types.js';
import type { Capabilities } from '../../../src/uamp/types.js';

// Mock fetch for health checks
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('AgentRegistry', () => {
  let registry: AgentRegistry;

  beforeEach(() => {
    registry = new AgentRegistry();
    mockFetch.mockReset();
  });

  afterEach(() => {
    registry.stopHealthChecks();
  });

  function createMockAgent(name: string, caps?: Partial<Capabilities>): IAgent {
    return {
      name,
      getCapabilities: () => ({
        id: name,
        provider: 'test',
        modalities: ['text'] as const,
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        ...caps,
      }),
      run: vi.fn(),
      runStreaming: vi.fn(),
      processUAMP: vi.fn(),
      executeTool: vi.fn(),
      getToolDefinitions: () => [],
      initialize: vi.fn(),
      cleanup: vi.fn(),
      addSkill: vi.fn(),
      removeSkill: vi.fn(),
      getHttpHandler: vi.fn(),
      getWebSocketHandler: vi.fn(),
    } as unknown as IAgent;
  }

  describe('registerLocal', () => {
    it('registers a local agent', () => {
      const agent = createMockAgent('test-agent');
      
      registry.registerLocal(agent);
      
      const entry = registry.get('test-agent');
      expect(entry).toBeDefined();
      expect(entry?.name).toBe('test-agent');
      expect(entry?.source).toBe('manual');
      expect(entry?.agent).toBe(agent);
    });

    it('marks agent as healthy', () => {
      const agent = createMockAgent('healthy-agent');
      
      registry.registerLocal(agent);
      
      expect(registry.get('healthy-agent')?.healthy).toBe(true);
    });

    it('sets registration timestamp', () => {
      const before = Date.now();
      const agent = createMockAgent('timed-agent');
      
      registry.registerLocal(agent);
      
      const after = Date.now();
      const entry = registry.get('timed-agent');
      expect(entry?.registeredAt).toBeGreaterThanOrEqual(before);
      expect(entry?.registeredAt).toBeLessThanOrEqual(after);
    });

    it('stores agent capabilities', () => {
      const agent = createMockAgent('caps-agent', {
        provides: ['search', 'weather'],
      });
      
      registry.registerLocal(agent);
      
      const entry = registry.get('caps-agent');
      expect(entry?.capabilities.provides).toContain('search');
    });
  });

  describe('registerRemote', () => {
    it('registers a remote agent by URL', () => {
      const caps: Capabilities = {
        id: 'remote-agent',
        provider: 'external',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      
      registry.registerRemote('remote-agent', 'https://agent.example.com', caps);
      
      const entry = registry.get('remote-agent');
      expect(entry).toBeDefined();
      expect(entry?.url).toBe('https://agent.example.com');
      expect(entry?.source).toBe('api');
    });

    it('supports websocket source', () => {
      const caps: Capabilities = {
        id: 'ws-agent',
        provider: 'portal',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      
      registry.registerRemote('ws-agent', 'wss://portal.example.com/ws', caps, 'websocket');
      
      const entry = registry.get('ws-agent');
      expect(entry?.source).toBe('websocket');
    });

    it('does not have local agent instance', () => {
      const caps: Capabilities = {
        id: 'no-instance',
        provider: 'external',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      
      registry.registerRemote('no-instance', 'https://example.com', caps);
      
      expect(registry.get('no-instance')?.agent).toBeUndefined();
    });
  });

  describe('unregister', () => {
    it('removes registered agent', () => {
      const agent = createMockAgent('to-remove');
      registry.registerLocal(agent);
      
      const removed = registry.unregister('to-remove');
      
      expect(removed).toBe(true);
      expect(registry.get('to-remove')).toBeUndefined();
    });

    it('returns false for unknown agent', () => {
      const removed = registry.unregister('nonexistent');
      expect(removed).toBe(false);
    });
  });

  describe('getAll', () => {
    it('returns empty array when no agents', () => {
      expect(registry.getAll()).toEqual([]);
    });

    it('returns all registered agents', () => {
      registry.registerLocal(createMockAgent('agent1'));
      registry.registerLocal(createMockAgent('agent2'));
      
      const all = registry.getAll();
      expect(all).toHaveLength(2);
      expect(all.map(a => a.name)).toContain('agent1');
      expect(all.map(a => a.name)).toContain('agent2');
    });
  });

  describe('findByCapability', () => {
    beforeEach(() => {
      registry.registerLocal(createMockAgent('search-agent', { provides: ['search'] }));
      registry.registerLocal(createMockAgent('weather-agent', { provides: ['weather'] }));
      registry.registerLocal(createMockAgent('multi-agent', { provides: ['search', 'weather'] }));
    });

    it('finds agents with matching capability', () => {
      const searchAgents = registry.findByCapability('search');
      
      expect(searchAgents).toHaveLength(2);
      expect(searchAgents.map(a => a.name)).toContain('search-agent');
      expect(searchAgents.map(a => a.name)).toContain('multi-agent');
    });

    it('returns empty array for unknown capability', () => {
      const results = registry.findByCapability('unknown');
      expect(results).toEqual([]);
    });
  });

  describe('updateActivity', () => {
    it('updates lastActivity timestamp', async () => {
      registry.registerLocal(createMockAgent('activity-agent'));
      const initial = registry.get('activity-agent')?.lastActivity;
      
      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 10));
      
      registry.updateActivity('activity-agent');
      
      const updated = registry.get('activity-agent')?.lastActivity;
      expect(updated).toBeGreaterThan(initial!);
    });

    it('ignores unknown agent', () => {
      expect(() => registry.updateActivity('unknown')).not.toThrow();
    });
  });

  describe('health status', () => {
    it('can mark agent unhealthy', () => {
      registry.registerLocal(createMockAgent('health-test'));
      expect(registry.get('health-test')?.healthy).toBe(true);
      
      registry.markUnhealthy('health-test');
      
      expect(registry.get('health-test')?.healthy).toBe(false);
    });

    it('can mark agent healthy again', () => {
      registry.registerLocal(createMockAgent('health-test'));
      registry.markUnhealthy('health-test');
      
      registry.markHealthy('health-test');
      
      expect(registry.get('health-test')?.healthy).toBe(true);
    });
  });

  describe('health checks', () => {
    it('checks remote agents health', async () => {
      mockFetch.mockResolvedValueOnce({ ok: true });
      
      const caps: Capabilities = {
        id: 'remote',
        provider: 'test',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      registry.registerRemote('remote', 'https://example.com', caps);
      
      // Start health checks with short interval
      registry.startHealthChecks(100);
      
      // Wait for health check
      await new Promise(resolve => setTimeout(resolve, 150));
      
      expect(mockFetch).toHaveBeenCalledWith(
        'https://example.com/health',
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('marks agent unhealthy on fetch failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));
      
      const caps: Capabilities = {
        id: 'failing',
        provider: 'test',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      registry.registerRemote('failing', 'https://failing.com', caps);
      
      registry.startHealthChecks(100);
      await new Promise(resolve => setTimeout(resolve, 150));
      
      expect(registry.get('failing')?.healthy).toBe(false);
    });

    it('marks agent unhealthy on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
      
      const caps: Capabilities = {
        id: 'error',
        provider: 'test',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      registry.registerRemote('error', 'https://error.com', caps);
      
      registry.startHealthChecks(100);
      await new Promise(resolve => setTimeout(resolve, 150));
      
      expect(registry.get('error')?.healthy).toBe(false);
    });

    it('stops health checks', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      
      const caps: Capabilities = {
        id: 'stop-test',
        provider: 'test',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      registry.registerRemote('stop-test', 'https://example.com', caps);
      
      registry.startHealthChecks(100);
      registry.stopHealthChecks();
      
      // Wait and verify no more calls
      const callCount = mockFetch.mock.calls.length;
      await new Promise(resolve => setTimeout(resolve, 200));
      
      expect(mockFetch.mock.calls.length).toBe(callCount);
    });

    it('does not check local agents', async () => {
      registry.registerLocal(createMockAgent('local'));
      
      registry.startHealthChecks(100);
      await new Promise(resolve => setTimeout(resolve, 150));
      
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('getStats', () => {
    it('returns stats for empty registry', () => {
      const stats = registry.getStats();
      
      expect(stats.total).toBe(0);
      expect(stats.healthy).toBe(0);
      expect(stats.local).toBe(0);
      expect(stats.remote).toBe(0);
    });

    it('counts agents correctly', () => {
      registry.registerLocal(createMockAgent('local1'));
      registry.registerLocal(createMockAgent('local2'));
      
      const caps: Capabilities = {
        id: 'remote1',
        provider: 'test',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      registry.registerRemote('remote1', 'https://example.com', caps);
      
      const stats = registry.getStats();
      
      expect(stats.total).toBe(3);
      expect(stats.local).toBe(2);
      expect(stats.remote).toBe(1);
      expect(stats.healthy).toBe(3);
    });

    it('counts unhealthy agents', () => {
      registry.registerLocal(createMockAgent('healthy'));
      registry.registerLocal(createMockAgent('unhealthy'));
      registry.markUnhealthy('unhealthy');
      
      const stats = registry.getStats();
      
      expect(stats.healthy).toBe(1);
      expect(stats.total).toBe(2);
    });
  });
});
