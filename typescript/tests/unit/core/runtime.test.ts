/**
 * AgentRuntime Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DefaultAgentRuntime } from '../../../src/core/runtime.js';
import type {
  Extension,
  AgentSource,
  SkillFactory,
  AgentRuntime,
  AgentInfo,
  RuntimeHooks,
} from '../../../src/core/runtime.js';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { handoff } from '../../../src/core/decorators.js';
import type { AgentConfig, ISkill, Context } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import { createResponseDoneEvent, generateEventId } from '../../../src/uamp/events.js';

// ============================================================================
// Helpers
// ============================================================================

class EchoLLM extends Skill {
  @handoff({ name: 'echo' })
  async *processUAMP(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'echo response' }]);
  }
}

function createMockAgent(name: string, skills: ISkill[] = []): BaseAgent {
  return new BaseAgent({ name, skills: [new EchoLLM(), ...skills] });
}

function createMockSource(agents: Map<string, BaseAgent>): AgentSource {
  return {
    type: 'mock',
    getAgent: async (name) => agents.get(name) ?? null,
    listAgents: async () =>
      Array.from(agents.entries()).map(([name]) => ({
        name,
        source: 'mock',
        loaded: true,
      })),
    searchAgents: async (query) =>
      Array.from(agents.entries())
        .filter(([name]) => name.includes(query))
        .map(([name]) => ({ name, source: 'mock', loaded: true })),
    invalidate: vi.fn(),
    invalidateAll: vi.fn(),
  };
}

function createMockExtension(
  name: string,
  sources: AgentSource[] = [],
  factories: SkillFactory[] = [],
  hooks?: RuntimeHooks
): Extension {
  return {
    name,
    initialize: vi.fn(),
    cleanup: vi.fn(),
    getAgentSources: () => sources,
    getSkillFactories: () => factories,
    getHooks: () => hooks ?? {},
  };
}

// ============================================================================
// Tests
// ============================================================================

describe('DefaultAgentRuntime', () => {
  let runtime: DefaultAgentRuntime;

  beforeEach(() => {
    runtime = new DefaultAgentRuntime();
  });

  describe('initialization', () => {
    it('initializes extensions in order', async () => {
      const order: string[] = [];
      const ext1 = createMockExtension('ext1');
      (ext1.initialize as ReturnType<typeof vi.fn>).mockImplementation(async () => { order.push('ext1'); });
      const ext2 = createMockExtension('ext2');
      (ext2.initialize as ReturnType<typeof vi.fn>).mockImplementation(async () => { order.push('ext2'); });

      runtime.registerExtension(ext1);
      runtime.registerExtension(ext2);
      await runtime.initialize();

      expect(order).toEqual(['ext1', 'ext2']);
    });

    it('only initializes once', async () => {
      const ext = createMockExtension('test');
      runtime.registerExtension(ext);

      await runtime.initialize();
      await runtime.initialize();

      expect(ext.initialize).toHaveBeenCalledTimes(1);
    });
  });

  describe('resolveAgent', () => {
    it('resolves from registered sources', async () => {
      const agent = createMockAgent('test-agent');
      const source = createMockSource(new Map([['test-agent', agent]]));
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      const resolved = await runtime.resolveAgent('test-agent');
      expect(resolved).toBe(agent);
    });

    it('returns null for unknown agents', async () => {
      const source = createMockSource(new Map());
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      const resolved = await runtime.resolveAgent('nonexistent');
      expect(resolved).toBeNull();
    });

    it('caches resolved agents', async () => {
      const agent = createMockAgent('cached');
      const getAgent = vi.fn(async () => agent);
      const source: AgentSource = {
        type: 'mock',
        getAgent,
        listAgents: async () => [],
      };
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      await runtime.resolveAgent('cached');
      await runtime.resolveAgent('cached');

      expect(getAgent).toHaveBeenCalledTimes(1);
    });

    it('searches multiple sources in order', async () => {
      const agent1 = createMockAgent('agent-a');
      const agent2 = createMockAgent('agent-b');
      const source1 = createMockSource(new Map([['agent-a', agent1]]));
      const source2 = createMockSource(new Map([['agent-b', agent2]]));
      const ext = createMockExtension('test', [source1, source2]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      expect(await runtime.resolveAgent('agent-a')).toBe(agent1);
      expect(await runtime.resolveAgent('agent-b')).toBe(agent2);
    });
  });

  describe('execute', () => {
    it('executes agent and returns response', async () => {
      const agent = createMockAgent('test');
      const source = createMockSource(new Map([['test', agent]]));
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      const response = await runtime.execute('test', [{ role: 'user', content: 'hi' }]);
      expect(response.content).toBeDefined();
    });

    it('throws for unknown agent', async () => {
      const source = createMockSource(new Map());
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      await expect(runtime.execute('unknown', [{ role: 'user', content: 'hi' }]))
        .rejects.toThrow('Agent not found: unknown');
    });

    it('calls runtime hooks', async () => {
      const agent = createMockAgent('hooked');
      const source = createMockSource(new Map([['hooked', agent]]));
      const onBeforeExecute = vi.fn();
      const onAfterExecute = vi.fn();
      const ext = createMockExtension('test', [source], [], {
        onBeforeExecute,
        onAfterExecute,
      });

      runtime.registerExtension(ext);
      await runtime.initialize();

      await runtime.execute('hooked', [{ role: 'user', content: 'hi' }]);

      expect(onBeforeExecute).toHaveBeenCalledWith('hooked', expect.any(Array), expect.any(Object));
      expect(onAfterExecute).toHaveBeenCalledWith('hooked', expect.objectContaining({ content: expect.any(String) }));
    });
  });

  describe('listAgents', () => {
    it('aggregates agents from all sources', async () => {
      const a1 = createMockAgent('alpha');
      const a2 = createMockAgent('beta');
      const source1 = createMockSource(new Map([['alpha', a1]]));
      const source2 = createMockSource(new Map([['beta', a2]]));
      const ext1 = createMockExtension('ext1', [source1]);
      const ext2 = createMockExtension('ext2', [source2]);

      runtime.registerExtension(ext1);
      runtime.registerExtension(ext2);
      await runtime.initialize();

      const agents = await runtime.listAgents();
      const names = agents.map(a => a.name);
      expect(names).toContain('alpha');
      expect(names).toContain('beta');
    });
  });

  describe('searchAgents', () => {
    it('searches across sources', async () => {
      const a1 = createMockAgent('search-one');
      const a2 = createMockAgent('search-two');
      const a3 = createMockAgent('other');
      const source = createMockSource(new Map([
        ['search-one', a1],
        ['search-two', a2],
        ['other', a3],
      ]));
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      const results = await runtime.searchAgents('search');
      expect(results).toHaveLength(2);
    });
  });

  describe('invalidation', () => {
    it('invalidates specific agent cache', async () => {
      const agent = createMockAgent('inv');
      const getAgent = vi.fn(async () => agent);
      const invalidate = vi.fn();
      const source: AgentSource = {
        type: 'mock',
        getAgent,
        listAgents: async () => [],
        invalidate,
      };
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      await runtime.resolveAgent('inv');
      expect(getAgent).toHaveBeenCalledTimes(1);

      runtime.invalidateAgent('inv');
      expect(invalidate).toHaveBeenCalledWith('inv');

      await runtime.resolveAgent('inv');
      expect(getAgent).toHaveBeenCalledTimes(2);
    });

    it('invalidates all cached agents', async () => {
      const invalidateAll = vi.fn();
      const source: AgentSource = {
        type: 'mock',
        getAgent: async () => null,
        listAgents: async () => [],
        invalidateAll,
      };
      const ext = createMockExtension('test', [source]);

      runtime.registerExtension(ext);
      await runtime.initialize();

      runtime.invalidateAll();
      expect(invalidateAll).toHaveBeenCalled();
    });
  });

  describe('cleanup', () => {
    it('cleans up all extensions', async () => {
      const ext1 = createMockExtension('ext1');
      const ext2 = createMockExtension('ext2');

      runtime.registerExtension(ext1);
      runtime.registerExtension(ext2);
      await runtime.initialize();
      await runtime.cleanup();

      expect(ext1.cleanup).toHaveBeenCalled();
      expect(ext2.cleanup).toHaveBeenCalled();
    });
  });

  describe('middleware', () => {
    it('runs middleware chain in order', async () => {
      const order: number[] = [];
      const agent = createMockAgent('mw');
      const source = createMockSource(new Map([['mw', agent]]));

      const ext: Extension = {
        name: 'mw-test',
        initialize: async () => {},
        getAgentSources: () => [source],
        getSkillFactories: () => [],
        getMiddleware: () => [
          async (_ctx, next) => { order.push(1); await next(); order.push(3); },
          async (_ctx, next) => { order.push(2); await next(); },
        ],
      };

      runtime.registerExtension(ext);
      await runtime.initialize();
      await runtime.execute('mw', [{ role: 'user', content: 'hi' }]);

      expect(order).toEqual([1, 2, 3]);
    });
  });
});
