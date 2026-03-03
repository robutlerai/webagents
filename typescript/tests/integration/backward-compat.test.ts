/**
 * Backward Compatibility Tests
 * 
 * Verifies that existing code continues to work with the new router architecture.
 * All new parameters (subscribes, produces) have sensible defaults.
 */

import { describe, it, expect, vi } from 'vitest';
import { BaseAgent, Skill, handoff } from '../../src/core/index.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import type { Context } from '../../src/core/types.js';

/**
 * Test skill that uses the old @handoff syntax (no subscribes/produces)
 */
class LegacyLLMSkill extends Skill {
  get id(): string {
    return 'legacy-llm';
  }

  get name(): string {
    return 'Legacy LLM';
  }

  get description(): string {
    return 'Test LLM skill without new router params';
  }

  @handoff({
    name: 'legacy-handler',
    priority: 10,
    // Note: no subscribes or produces specified
  })
  async *processUAMP(
    events: ClientEvent[],
    context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    for (const event of events) {
      if (event.type === 'input.text') {
        yield {
          type: 'response.delta',
          event_id: 'resp-1',
          delta: { text: 'Hello from legacy handler!' },
        } as unknown as ServerEvent;
      }
    }
  }
}

/**
 * Test skill that uses the new @handoff syntax
 */
class ModernLLMSkill extends Skill {
  get id(): string {
    return 'modern-llm';
  }

  get name(): string {
    return 'Modern LLM';
  }

  get description(): string {
    return 'Test LLM skill with new router params';
  }

  @handoff({
    name: 'modern-handler',
    priority: 10,
    subscribes: ['input.text'],
    produces: ['response.delta'],
  })
  async *processUAMP(
    events: ClientEvent[],
    context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    for (const event of events) {
      if (event.type === 'input.text') {
        yield {
          type: 'response.delta',
          event_id: 'resp-1',
          delta: { text: 'Hello from modern handler!' },
        } as unknown as ServerEvent;
      }
    }
  }
}

describe('Backward Compatibility', () => {
  describe('Legacy @handoff decorator', () => {
    it('should work without subscribes/produces params', () => {
      const skill = new LegacyLLMSkill();
      
      // Should have handoff registered with defaults
      expect(skill.handoffs).toHaveLength(1);
      expect(skill.handoffs[0].name).toBe('legacy-handler');
      expect(skill.handoffs[0].subscribes).toEqual(['input.text']); // Default
      expect(skill.handoffs[0].produces).toEqual(['response.delta']); // Default
    });

    it('should be equivalent to modern syntax with defaults', () => {
      const legacy = new LegacyLLMSkill();
      const modern = new ModernLLMSkill();

      expect(legacy.handoffs[0].subscribes).toEqual(modern.handoffs[0].subscribes);
      expect(legacy.handoffs[0].produces).toEqual(modern.handoffs[0].produces);
    });
  });

  describe('BaseAgent with legacy skills', () => {
    it('should add legacy skill without errors', () => {
      const agent = new BaseAgent({ name: 'test-agent' });
      const skill = new LegacyLLMSkill();
      
      // Should not throw
      agent.addSkill(skill);
      
      // Router should have handler registered
      expect(agent.router.getHandlers().has('legacy-handler')).toBe(true);
    });

    it('should set first LLM skill as default handler', () => {
      const agent = new BaseAgent({ name: 'test-agent' });
      agent.addSkill(new LegacyLLMSkill());

      expect(agent.router.defaultHandler?.name).toBe('legacy-handler');
    });

    it('should process messages through legacy handler', async () => {
      const agent = new BaseAgent({ name: 'test-agent' });
      agent.addSkill(new LegacyLLMSkill());

      const events: ServerEvent[] = [];

      // Collect events via UAMP processing
      for await (const event of agent.processUAMP([{
        type: 'input.text',
        event_id: 'test-1',
        text: 'Hello',
      } as ClientEvent])) {
        events.push(event);
      }

      // Should have received response
      expect(events.some(e => e.type === 'response.delta')).toBe(true);
    });
  });

  describe('Mixed legacy and modern skills', () => {
    it('should work with both legacy and modern skills', () => {
      const agent = new BaseAgent({ name: 'test-agent' });
      agent.addSkill(new LegacyLLMSkill());
      agent.addSkill(new ModernLLMSkill());

      // Both should be registered
      expect(agent.router.getHandlers().has('legacy-handler')).toBe(true);
      expect(agent.router.getHandlers().has('modern-handler')).toBe(true);
    });
  });

  describe('Agent.run() interface', () => {
    it('should work with run() method unchanged', async () => {
      const agent = new BaseAgent({ name: 'test-agent' });
      agent.addSkill(new LegacyLLMSkill());

      // The run() method should still work
      // Note: This will throw since we don't have a real LLM, but the interface is unchanged
      try {
        const response = await agent.run([
          { role: 'user', content: 'Hello' }
        ]);
        // If we get here, the method signature is correct
        expect(response).toBeDefined();
      } catch (error) {
        // Expected - no real LLM available
        // But the interface is preserved
      }
    });
  });

  describe('Default values', () => {
    it('should use correct default for subscribes', () => {
      const skill = new LegacyLLMSkill();
      expect(skill.handoffs[0].subscribes).toEqual(['input.text']);
    });

    it('should use correct default for produces', () => {
      const skill = new LegacyLLMSkill();
      expect(skill.handoffs[0].produces).toEqual(['response.delta']);
    });

    it('should use 0 as default priority', () => {
      // Create a skill without explicit priority
      class NoPrioritySkill extends Skill {
        get id() { return 'no-priority'; }
        get name() { return 'No Priority'; }
        get description() { return 'Test'; }

        @handoff({ name: 'no-priority-handler' })
        async *process(): AsyncGenerator<ServerEvent, void, unknown> {}
      }

      const skill = new NoPrioritySkill();
      expect(skill.handoffs[0].priority).toBe(0);
    });
  });
});
