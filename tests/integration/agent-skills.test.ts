/**
 * Agent + Skills Integration Tests
 * 
 * Tests the interaction between BaseAgent and various skills.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { tool, hook, handoff, http } from '../../src/core/decorators.js';
import type { Context, HookData, HookResult } from '../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../src/uamp/events.js';

describe('Agent + Skills Integration', () => {
  describe('multi-skill agent', () => {
    class SearchSkill extends Skill {
      @tool({ provides: 'search', description: 'Search the web' })
      async search(params: { query: string }, _ctx: Context) {
        return {
          results: [`Result for: ${params.query}`],
        };
      }
    }

    class WeatherSkill extends Skill {
      @tool({ provides: 'weather', description: 'Get weather' })
      async getWeather(params: { city: string }, _ctx: Context) {
        return {
          city: params.city,
          temp: 72,
          condition: 'sunny',
        };
      }
    }

    class MockLLMSkill extends Skill {
      @handoff({ name: 'mock-llm', priority: 10 })
      async *processUAMP(events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
        // Simple mock that echoes the input
        for (const e of events) {
          if (e.type === 'input.text') {
            const text = (e as { text: string }).text;
            yield createResponseDeltaEvent('r1', { type: 'text', text: `Response: ${text}` });
          }
        }
        yield createResponseDoneEvent('r1', [{ type: 'text', text: 'done' }]);
      }
    }

    let agent: BaseAgent;

    beforeEach(() => {
      agent = new BaseAgent({
        name: 'multi-skill-agent',
        skills: [new SearchSkill(), new WeatherSkill(), new MockLLMSkill()],
      });
    });

    it('aggregates tools from multiple skills', () => {
      const caps = agent.getCapabilities();
      expect(caps.provides).toContain('search');
      expect(caps.provides).toContain('weather');
    });

    it('executes tools from any skill', async () => {
      const searchResult = await agent.executeTool('search', { query: 'cats' });
      expect(searchResult).toEqual({ results: ['Result for: cats'] });

      const weatherResult = await agent.executeTool('getWeather', { city: 'NYC' });
      expect(weatherResult).toEqual({ city: 'NYC', temp: 72, condition: 'sunny' });
    });

    it('processes messages through LLM skill', async () => {
      const response = await agent.run([
        { role: 'user', content: 'Hello world' },
      ]);
      
      expect(response.content).toContain('Response: Hello world');
    });

    it('can remove skills dynamically', async () => {
      agent.removeSkill('SearchSkill');
      
      const caps = agent.getCapabilities();
      expect(caps.provides).not.toContain('search');
      
      await expect(agent.executeTool('search', { query: 'test' }))
        .rejects.toThrow('Tool not found');
    });
  });

  describe('hooks integration', () => {
    it('runs hooks in correct order across skills', async () => {
      const order: string[] = [];

      class FirstHookSkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 1 })
        async first(_d: HookData, _c: Context): Promise<HookResult | void> {
          order.push('first');
        }
      }

      class SecondHookSkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 2 })
        async second(_d: HookData, _c: Context): Promise<HookResult | void> {
          order.push('second');
        }
      }

      class ThirdHookSkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 3 })
        async third(_d: HookData, _c: Context): Promise<HookResult | void> {
          order.push('third');
        }
      }

      class MockLLM extends Skill {
        @handoff({ name: 'llm' })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          yield createResponseDoneEvent('r', [{ type: 'text', text: 'ok' }]);
        }
      }

      const agent = new BaseAgent({
        skills: [
          new SecondHookSkill(),
          new ThirdHookSkill(),
          new FirstHookSkill(),
          new MockLLM(),
        ],
      });

      await agent.run([{ role: 'user', content: 'test' }]);

      expect(order).toEqual(['first', 'second', 'third']);
    });

    it('can abort processing via hook', async () => {
      class AuthSkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 1 })
        async checkAuth(_d: HookData, ctx: Context): Promise<HookResult> {
          if (!ctx.auth.authenticated) {
            return { abort: true, abort_reason: 'Unauthorized' };
          }
          return {};
        }
      }

      class MockLLM extends Skill {
        @handoff({ name: 'llm' })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          yield createResponseDoneEvent('r', [{ type: 'text', text: 'should not reach' }]);
        }
      }

      const agent = new BaseAgent({
        skills: [new AuthSkill(), new MockLLM()],
      });

      await expect(agent.run([{ role: 'user', content: 'hi' }]))
        .rejects.toThrow('Unauthorized');
    });

    it('can modify tool params via before_tool hook', async () => {
      class AuditSkill extends Skill {
        auditLog: string[] = [];

        @hook({ lifecycle: 'before_tool' })
        async audit(data: HookData, _c: Context): Promise<HookResult | void> {
          this.auditLog.push(`Called: ${data.tool_name}`);
        }

        @tool({})
        async myTool(_p: Record<string, unknown>, _c: Context) {
          return 'result';
        }
      }

      const auditSkill = new AuditSkill();
      const agent = new BaseAgent({ skills: [auditSkill] });

      await agent.executeTool('myTool', {});

      expect(auditSkill.auditLog).toContain('Called: myTool');
    });
  });

  describe('handoff priority', () => {
    it('selects highest priority handoff', async () => {
      const called: string[] = [];

      class LowPriorityLLM extends Skill {
        @handoff({ name: 'low', priority: 1 })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          called.push('low');
          yield createResponseDoneEvent('r', [{ type: 'text', text: 'low' }]);
        }
      }

      class HighPriorityLLM extends Skill {
        @handoff({ name: 'high', priority: 100 })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          called.push('high');
          yield createResponseDoneEvent('r', [{ type: 'text', text: 'high' }]);
        }
      }

      const agent = new BaseAgent({
        skills: [new LowPriorityLLM(), new HighPriorityLLM()],
      });

      await agent.run([{ role: 'user', content: 'test' }]);

      expect(called).toEqual(['high']);
    });

    it('falls back to lower priority when higher is disabled', async () => {
      const called: string[] = [];

      class DisabledLLM extends Skill {
        @handoff({ name: 'disabled', priority: 100, enabled: false })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          called.push('disabled');
          yield createResponseDoneEvent('r', [{ type: 'text', text: 'disabled' }]);
        }
      }

      class EnabledLLM extends Skill {
        @handoff({ name: 'enabled', priority: 1 })
        async *process(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          called.push('enabled');
          yield createResponseDoneEvent('r', [{ type: 'text', text: 'enabled' }]);
        }
      }

      const agent = new BaseAgent({
        skills: [new DisabledLLM(), new EnabledLLM()],
      });

      await agent.run([{ role: 'user', content: 'test' }]);

      expect(called).toEqual(['enabled']);
    });
  });

  describe('HTTP endpoints integration', () => {
    class APISkill extends Skill {
      @http({ path: '/api/data', method: 'GET' })
      async getData(_req: Request, _ctx: Context): Promise<Response> {
        return new Response(JSON.stringify({ data: 'value' }), {
          headers: { 'Content-Type': 'application/json' },
        });
      }

      @http({ path: '/api/submit', method: 'POST' })
      async submit(req: Request, _ctx: Context): Promise<Response> {
        const body = await req.json();
        return new Response(JSON.stringify({ received: body }));
      }
    }

    let agent: BaseAgent;

    beforeEach(() => {
      agent = new BaseAgent({
        skills: [new APISkill()],
      });
    });

    it('provides HTTP handlers', () => {
      const getHandler = agent.getHttpHandler('/api/data', 'GET');
      expect(getHandler).toBeDefined();

      const postHandler = agent.getHttpHandler('/api/submit', 'POST');
      expect(postHandler).toBeDefined();
    });

    it('executes HTTP handlers correctly', async () => {
      const handler = agent.getHttpHandler('/api/data', 'GET');
      const response = await handler!.handler(
        new Request('http://localhost/api/data'),
        {} as Context
      );

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.data).toBe('value');
    });
  });

  describe('tool execution with context', () => {
    class ScopedSkill extends Skill {
      @tool({ scopes: ['admin'] })
      async adminTool(_p: Record<string, unknown>, _c: Context) {
        return 'admin secret';
      }

      @tool({ scopes: ['user'] })
      async userTool(_p: Record<string, unknown>, _c: Context) {
        return 'user data';
      }

      @tool({})
      async publicTool(_p: Record<string, unknown>, _c: Context) {
        return 'public';
      }
    }

    it('allows public tools without auth', async () => {
      const agent = new BaseAgent({ skills: [new ScopedSkill()] });
      const result = await agent.executeTool('publicTool', {});
      expect(result).toBe('public');
    });

    it('denies scoped tools without auth', async () => {
      const agent = new BaseAgent({ skills: [new ScopedSkill()] });
      
      await expect(agent.executeTool('adminTool', {}))
        .rejects.toThrow('Insufficient permissions');
    });
  });

  describe('streaming integration', () => {
    class StreamingLLM extends Skill {
      @handoff({ name: 'streaming-llm' })
      async *processUAMP(_events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
        const words = ['Hello', ' ', 'streaming', ' ', 'world', '!'];
        
        for (const word of words) {
          yield createResponseDeltaEvent('r1', { type: 'text', text: word });
        }
        
        yield createResponseDoneEvent('r1', [
          { type: 'text', text: 'Hello streaming world!' },
        ], 'completed', {
          input_tokens: 5,
          output_tokens: 10,
          total_tokens: 15,
        });
      }
    }

    it('streams chunks correctly', async () => {
      const agent = new BaseAgent({ skills: [new StreamingLLM()] });
      
      const chunks: string[] = [];
      for await (const chunk of agent.runStreaming([{ role: 'user', content: 'test' }])) {
        if (chunk.type === 'delta') {
          chunks.push(chunk.delta);
        }
      }

      expect(chunks.join('')).toBe('Hello streaming world!');
    });

    it('provides usage stats at end', async () => {
      const agent = new BaseAgent({ skills: [new StreamingLLM()] });
      
      let finalChunk: unknown;
      for await (const chunk of agent.runStreaming([{ role: 'user', content: 'test' }])) {
        if (chunk.type === 'done') {
          finalChunk = chunk;
        }
      }

      expect(finalChunk).toBeDefined();
      expect((finalChunk as { response: { usage: { total_tokens: number } } })
        .response.usage.total_tokens).toBe(15);
    });
  });

  describe('error handling', () => {
    class FailingSkill extends Skill {
      @tool({})
      async failingTool(_p: Record<string, unknown>, _c: Context) {
        throw new Error('Tool execution failed');
      }
    }

    class ErrorRecoverySkill extends Skill {
      errorsCaught: Error[] = [];

      @hook({ lifecycle: 'on_error' })
      async catchError(data: HookData, _c: Context): Promise<HookResult | void> {
        if (data.error) {
          this.errorsCaught.push(data.error);
        }
      }
    }

    it('propagates tool errors', async () => {
      const agent = new BaseAgent({ skills: [new FailingSkill()] });
      
      await expect(agent.executeTool('failingTool', {}))
        .rejects.toThrow('Tool execution failed');
    });

    it('runs on_error hooks on tool failure', async () => {
      const errorSkill = new ErrorRecoverySkill();
      const agent = new BaseAgent({
        skills: [new FailingSkill(), errorSkill],
      });

      await expect(agent.executeTool('failingTool', {})).rejects.toThrow();

      expect(errorSkill.errorsCaught).toHaveLength(1);
      expect(errorSkill.errorsCaught[0].message).toBe('Tool execution failed');
    });
  });

  describe('lifecycle management', () => {
    it('initializes all skills', async () => {
      const initOrder: string[] = [];

      class InitSkill1 extends Skill {
        async initialize() {
          initOrder.push('skill1');
        }
      }

      class InitSkill2 extends Skill {
        async initialize() {
          initOrder.push('skill2');
        }
      }

      const agent = new BaseAgent({
        skills: [new InitSkill1(), new InitSkill2()],
      });

      await agent.initialize();

      expect(initOrder).toEqual(['skill1', 'skill2']);
    });

    it('cleans up all skills', async () => {
      const cleanupOrder: string[] = [];

      class CleanupSkill1 extends Skill {
        async cleanup() {
          cleanupOrder.push('skill1');
        }
      }

      class CleanupSkill2 extends Skill {
        async cleanup() {
          cleanupOrder.push('skill2');
        }
      }

      const agent = new BaseAgent({
        skills: [new CleanupSkill1(), new CleanupSkill2()],
      });

      await agent.cleanup();

      expect(cleanupOrder).toEqual(['skill1', 'skill2']);
    });
  });
});
