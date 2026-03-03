/**
 * Skill Base Class Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Skill } from '../../../src/core/skill.js';
import { tool, hook, handoff, http, websocket } from '../../../src/core/decorators.js';
import type { Context, HookData, HookResult } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';

describe('Skill Base Class', () => {
  describe('constructor', () => {
    it('uses constructor name as default skill name', () => {
      class MyCustomSkill extends Skill {}
      const skill = new MyCustomSkill();
      expect(skill.name).toBe('MyCustomSkill');
    });

    it('uses config name when provided', () => {
      class TestSkill extends Skill {}
      const skill = new TestSkill({ name: 'custom-name' });
      expect(skill.name).toBe('custom-name');
    });

    it('is enabled by default', () => {
      class TestSkill extends Skill {}
      const skill = new TestSkill();
      expect(skill.enabled).toBe(true);
    });

    it('respects enabled config', () => {
      class TestSkill extends Skill {}
      const skill = new TestSkill({ enabled: false });
      expect(skill.enabled).toBe(false);
    });
  });

  describe('tools collection', () => {
    it('collects tools from decorated methods', () => {
      class ToolSkill extends Skill {
        @tool({ description: 'Search the web' })
        async search(_params: { query: string }, _ctx: Context) {
          return { results: [] };
        }

        @tool({ name: 'read_file', description: 'Read a file' })
        async readFile(_params: { path: string }, _ctx: Context) {
          return 'content';
        }
      }

      const skill = new ToolSkill();
      expect(skill.tools).toHaveLength(2);
      expect(skill.tools.map(t => t.name)).toContain('search');
      expect(skill.tools.map(t => t.name)).toContain('read_file');
    });

    it('includes parameters and provides', () => {
      class ParamSkill extends Skill {
        @tool({
          provides: 'weather',
          parameters: {
            type: 'object',
            properties: {
              city: { type: 'string' },
            },
            required: ['city'],
          },
        })
        async getWeather(_params: { city: string }, _ctx: Context) {
          return { temp: 72 };
        }
      }

      const skill = new ParamSkill();
      const weatherTool = skill.tools[0];
      expect(weatherTool.provides).toBe('weather');
      expect(weatherTool.parameters?.type).toBe('object');
    });

    it('filters disabled tools', () => {
      class DisabledToolSkill extends Skill {
        @tool({ enabled: false })
        async disabledTool(_p: Record<string, unknown>, _c: Context) {}

        @tool({})
        async enabledTool(_p: Record<string, unknown>, _c: Context) {}
      }

      const skill = new DisabledToolSkill();
      expect(skill.tools).toHaveLength(1);
      expect(skill.tools[0].name).toBe('enabledTool');
    });
  });

  describe('hooks collection', () => {
    it('collects hooks from decorated methods', () => {
      class HookSkill extends Skill {
        @hook({ lifecycle: 'before_run' })
        async beforeRun(_data: HookData, _ctx: Context): Promise<HookResult | void> {}

        @hook({ lifecycle: 'after_run' })
        async afterRun(_data: HookData, _ctx: Context): Promise<HookResult | void> {}
      }

      const skill = new HookSkill();
      expect(skill.hooks).toHaveLength(2);
      expect(skill.hooks.map(h => h.lifecycle)).toContain('before_run');
      expect(skill.hooks.map(h => h.lifecycle)).toContain('after_run');
    });

    it('uses default priority of 50', () => {
      class PrioritySkill extends Skill {
        @hook({ lifecycle: 'before_run' })
        async myHook(_data: HookData, _ctx: Context): Promise<HookResult | void> {}
      }

      const skill = new PrioritySkill();
      expect(skill.hooks[0].priority).toBe(50);
    });

    it('respects custom priority', () => {
      class PrioritySkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 10 })
        async highPriority(_data: HookData, _ctx: Context): Promise<HookResult | void> {}
      }

      const skill = new PrioritySkill();
      expect(skill.hooks[0].priority).toBe(10);
    });
  });

  describe('handoffs collection', () => {
    it('collects handoffs from decorated methods', () => {
      class LLMSkill extends Skill {
        @handoff({ name: 'test-llm', description: 'Test LLM handoff' })
        async *processUAMP(_events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
          // Generator implementation
        }
      }

      const skill = new LLMSkill();
      expect(skill.handoffs).toHaveLength(1);
      expect(skill.handoffs[0].name).toBe('test-llm');
      expect(skill.handoffs[0].description).toBe('Test LLM handoff');
    });

    it('uses default priority of 0 for handoffs', () => {
      class LLMSkill extends Skill {
        @handoff({ name: 'default-priority' })
        async *processUAMP(_events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {}
      }

      const skill = new LLMSkill();
      expect(skill.handoffs[0].priority).toBe(0);
    });
  });

  describe('HTTP endpoints collection', () => {
    it('collects HTTP endpoints from decorated methods', () => {
      class APISkill extends Skill {
        @http({ path: '/api/data', method: 'GET' })
        async getData(_req: Request, _ctx: Context): Promise<Response> {
          return new Response('data');
        }

        @http({ path: '/api/submit', method: 'POST' })
        async postData(_req: Request, _ctx: Context): Promise<Response> {
          return new Response('ok');
        }
      }

      const skill = new APISkill();
      expect(skill.httpEndpoints).toHaveLength(2);
      expect(skill.httpEndpoints.map(e => e.path)).toContain('/api/data');
      expect(skill.httpEndpoints.map(e => e.method)).toContain('POST');
    });

    it('defaults to GET method', () => {
      class GetSkill extends Skill {
        @http({ path: '/status' })
        async status(_req: Request, _ctx: Context): Promise<Response> {
          return new Response('ok');
        }
      }

      const skill = new GetSkill();
      expect(skill.httpEndpoints[0].method).toBe('GET');
    });
  });

  describe('WebSocket endpoints collection', () => {
    it('collects WebSocket endpoints from decorated methods', () => {
      class WSSkill extends Skill {
        @websocket({ path: '/ws/stream' })
        handleStream(_ws: WebSocket, _ctx: Context): void {}
      }

      const skill = new WSSkill();
      expect(skill.wsEndpoints).toHaveLength(1);
      expect(skill.wsEndpoints[0].path).toBe('/ws/stream');
    });

    it('includes protocols when provided', () => {
      class ProtocolSkill extends Skill {
        @websocket({ path: '/ws/chat', protocols: ['uamp', 'json'] })
        handleChat(_ws: WebSocket, _ctx: Context): void {}
      }

      const skill = new ProtocolSkill();
      expect(skill.wsEndpoints[0].protocols).toEqual(['uamp', 'json']);
    });
  });

  describe('manual registration', () => {
    it('supports manual tool registration', () => {
      class ManualSkill extends Skill {
        constructor() {
          super();
          this.registerTool({
            name: 'manual_tool',
            description: 'Manually registered',
            enabled: true,
            handler: async () => 'result',
          });
        }

        // Need protected access
        protected registerTool = super['registerTool'].bind(this);
      }

      const skill = new ManualSkill();
      expect(skill.tools).toHaveLength(1);
      expect(skill.tools[0].name).toBe('manual_tool');
    });
  });

  describe('enable/disable tools', () => {
    let skill: InstanceType<typeof TestSkill>;
    
    class TestSkill extends Skill {
      @tool({ name: 'toggleable' })
      async toggleable(_p: Record<string, unknown>, _c: Context) {}
    }

    beforeEach(() => {
      skill = new TestSkill();
    });

    it('can disable a tool by name', () => {
      expect(skill.tools).toHaveLength(1);
      
      skill.setToolEnabled('toggleable', false);
      expect(skill.tools).toHaveLength(0);
    });

    it('can re-enable a disabled tool', () => {
      skill.setToolEnabled('toggleable', false);
      expect(skill.tools).toHaveLength(0);
      
      skill.setToolEnabled('toggleable', true);
      expect(skill.tools).toHaveLength(1);
    });

    it('ignores unknown tool names', () => {
      skill.setToolEnabled('nonexistent', false);
      expect(skill.tools).toHaveLength(1);
    });
  });

  describe('getTool / getHandoff', () => {
    it('returns tool by name', () => {
      class LookupSkill extends Skill {
        @tool({ name: 'find_me' })
        async findMe(_p: Record<string, unknown>, _c: Context) {}
      }

      const skill = new LookupSkill();
      const foundTool = skill.getTool('find_me');
      expect(foundTool).toBeDefined();
      expect(foundTool?.name).toBe('find_me');
    });

    it('returns undefined for unknown tool', () => {
      class EmptySkill extends Skill {}
      const skill = new EmptySkill();
      expect(skill.getTool('unknown')).toBeUndefined();
    });

    it('returns handoff by name', () => {
      class HandoffSkill extends Skill {
        @handoff({ name: 'my-llm' })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {}
      }

      const skill = new HandoffSkill();
      const foundHandoff = skill.getHandoff('my-llm');
      expect(foundHandoff).toBeDefined();
      expect(foundHandoff?.name).toBe('my-llm');
    });
  });

  describe('lifecycle methods', () => {
    it('has default initialize that does nothing', async () => {
      class LifecycleSkill extends Skill {}
      const skill = new LifecycleSkill();
      await expect(skill.initialize()).resolves.toBeUndefined();
    });

    it('has default cleanup that does nothing', async () => {
      class LifecycleSkill extends Skill {}
      const skill = new LifecycleSkill();
      await expect(skill.cleanup()).resolves.toBeUndefined();
    });

    it('allows overriding initialize', async () => {
      const initFn = vi.fn();
      
      class CustomInitSkill extends Skill {
        async initialize() {
          initFn();
        }
      }

      const skill = new CustomInitSkill();
      await skill.initialize();
      expect(initFn).toHaveBeenCalledOnce();
    });

    it('allows overriding cleanup', async () => {
      const cleanupFn = vi.fn();
      
      class CustomCleanupSkill extends Skill {
        async cleanup() {
          cleanupFn();
        }
      }

      const skill = new CustomCleanupSkill();
      await skill.cleanup();
      expect(cleanupFn).toHaveBeenCalledOnce();
    });
  });

  describe('tool execution', () => {
    it('binds handler to skill instance', async () => {
      class StatefulSkill extends Skill {
        private value = 'secret';

        @tool({})
        async getValue(_p: Record<string, unknown>, _c: Context) {
          return this.value;
        }
      }

      const skill = new StatefulSkill();
      const registeredTool = skill.tools[0];
      const result = await registeredTool.handler({}, {} as Context);
      expect(result).toBe('secret');
    });
  });
});
