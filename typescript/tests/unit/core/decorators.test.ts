/**
 * Decorators Unit Tests
 */

import { describe, it, expect } from 'vitest';
import {
  tool,
  hook,
  handoff,
  http,
  websocket,
  getTools,
  getHooks,
  getHandoffs,
  getHttpEndpoints,
  getWebSocketEndpoints,
} from '../../../src/core/decorators.js';
import { Skill } from '../../../src/core/skill.js';
import type { Context } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';

describe('Decorators', () => {
  describe('@tool', () => {
    it('registers function with metadata', () => {
      class TestSkill extends Skill {
        @tool({ provides: 'test' })
        async testTool(_params: Record<string, unknown>, _ctx: Context) {
          return 'result';
        }
      }
      
      const skill = new TestSkill();
      expect(skill.tools).toHaveLength(1);
      expect(skill.tools[0].name).toBe('testTool');
      expect(skill.tools[0].provides).toBe('test');
    });
    
    it('uses custom name when provided', () => {
      class TestSkill extends Skill {
        @tool({ name: 'custom_name' })
        async myTool(_params: Record<string, unknown>, _ctx: Context) {
          return 'result';
        }
      }
      
      const skill = new TestSkill();
      expect(skill.tools[0].name).toBe('custom_name');
    });
    
    it('includes description and parameters', () => {
      class TestSkill extends Skill {
        @tool({
          description: 'A test tool',
          parameters: {
            type: 'object',
            properties: {
              query: { type: 'string' },
            },
          },
        })
        async search(_params: { query: string }, _ctx: Context) {
          return [];
        }
      }
      
      const skill = new TestSkill();
      expect(skill.tools[0].description).toBe('A test tool');
      expect(skill.tools[0].parameters?.type).toBe('object');
    });
    
    it('handles multiple tools', () => {
      class TestSkill extends Skill {
        @tool({ provides: 'read' })
        async readFile(_p: Record<string, unknown>, _c: Context) { return ''; }
        
        @tool({ provides: 'write' })
        async writeFile(_p: Record<string, unknown>, _c: Context) { return true; }
      }
      
      const skill = new TestSkill();
      expect(skill.tools).toHaveLength(2);
    });
  });
  
  describe('@hook', () => {
    it('registers hook with lifecycle', () => {
      class TestSkill extends Skill {
        @hook({ lifecycle: 'before_run' })
        async beforeRun() {}
      }
      
      const skill = new TestSkill();
      expect(skill.hooks).toHaveLength(1);
      expect(skill.hooks[0].lifecycle).toBe('before_run');
    });
    
    it('respects priority', () => {
      class TestSkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 100 })
        async lowPriority() {}
        
        @hook({ lifecycle: 'before_run', priority: 10 })
        async highPriority() {}
      }
      
      const skill = new TestSkill();
      // Both hooks are registered (sorting happens in BaseAgent)
      expect(skill.hooks).toHaveLength(2);
      // Check that both priorities are present
      const priorities = skill.hooks.map(h => h.priority).sort((a, b) => a - b);
      expect(priorities).toEqual([10, 100]);
    });
  });
  
  describe('@handoff', () => {
    it('registers handoff with name', () => {
      class TestSkill extends Skill {
        @handoff({ name: 'test-llm', priority: 5 })
        async *processUAMP(_events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
          // Generator
        }
      }
      
      const skill = new TestSkill();
      expect(skill.handoffs).toHaveLength(1);
      expect(skill.handoffs[0].name).toBe('test-llm');
      expect(skill.handoffs[0].priority).toBe(5);
    });
  });
  
  describe('@http', () => {
    it('registers HTTP endpoint', () => {
      class TestSkill extends Skill {
        @http({ path: '/api/test', method: 'POST' })
        async handleTest(_req: Request, _ctx: Context) {
          return new Response('ok');
        }
      }
      
      const skill = new TestSkill();
      expect(skill.httpEndpoints).toHaveLength(1);
      expect(skill.httpEndpoints[0].path).toBe('/api/test');
      expect(skill.httpEndpoints[0].method).toBe('POST');
    });
    
    it('defaults to GET method', () => {
      class TestSkill extends Skill {
        @http({ path: '/health' })
        async health(_req: Request, _ctx: Context) {
          return new Response('ok');
        }
      }
      
      const skill = new TestSkill();
      expect(skill.httpEndpoints[0].method).toBe('GET');
    });
  });
  
  describe('@websocket', () => {
    it('registers WebSocket endpoint', () => {
      class TestSkill extends Skill {
        @websocket({ path: '/ws/stream' })
        handleWs(_ws: WebSocket, _ctx: Context) {}
      }
      
      const skill = new TestSkill();
      expect(skill.wsEndpoints).toHaveLength(1);
      expect(skill.wsEndpoints[0].path).toBe('/ws/stream');
    });
  });
  
  describe('enabled flag', () => {
    it('respects enabled: false for tools', () => {
      class TestSkill extends Skill {
        @tool({ enabled: false })
        async disabledTool(_p: Record<string, unknown>, _c: Context) {}
      }
      
      const skill = new TestSkill();
      // tools getter filters by enabled
      expect(skill.tools).toHaveLength(0);
    });
    
    it('can enable/disable tools', () => {
      class TestSkill extends Skill {
        @tool({ name: 'myTool' })
        async myTool(_p: Record<string, unknown>, _c: Context) {}
      }
      
      const skill = new TestSkill();
      expect(skill.tools).toHaveLength(1);
      
      skill.setToolEnabled('myTool', false);
      expect(skill.tools).toHaveLength(0);
      
      skill.setToolEnabled('myTool', true);
      expect(skill.tools).toHaveLength(1);
    });
  });
});
