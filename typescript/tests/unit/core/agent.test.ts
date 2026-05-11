/**
 * BaseAgent Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { tool, hook, handoff, http, websocket } from '../../../src/core/decorators.js';
import type { Context, HookData, HookResult } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../../src/uamp/events.js';

describe('BaseAgent', () => {
  describe('constructor', () => {
    it('creates agent with default name', () => {
      const agent = new BaseAgent();
      expect(agent.name).toBe('agent');
    });

    it('uses provided config', () => {
      const agent = new BaseAgent({
        name: 'test-agent',
        description: 'A test agent',
        instructions: 'Be helpful',
        model: 'gpt-4',
      });
      expect(agent.name).toBe('test-agent');
      expect(agent.description).toBe('A test agent');
    });

    it('initializes default capabilities', () => {
      const agent = new BaseAgent({ name: 'test' });
      const caps = agent.getCapabilities();
      
      expect(caps.id).toBe('test');
      expect(caps.provider).toBe('webagents');
      expect(caps.modalities).toContain('text');
      expect(caps.supports_streaming).toBe(true);
    });

    it('accepts custom capabilities', () => {
      const agent = new BaseAgent({
        capabilities: {
          id: 'custom',
          provider: 'custom-provider',
          modalities: ['text', 'audio'],
          supports_streaming: false,
          supports_thinking: true,
          supports_caching: true,
        },
      });
      
      const caps = agent.getCapabilities();
      expect(caps.modalities).toEqual(['text', 'audio']);
      expect(caps.supports_streaming).toBe(false);
      expect(caps.supports_thinking).toBe(true);
    });
  });

  describe('skill management', () => {
    class TestTool extends Skill {
      @tool({ provides: 'test', description: 'Test tool' })
      async testTool(_p: Record<string, unknown>, _c: Context) {
        return 'result';
      }
    }

    it('adds skills from config', () => {
      const agent = new BaseAgent({
        skills: [new TestTool()],
      });
      
      const caps = agent.getCapabilities();
      expect(caps.tools?.built_in_tools).toContain('testTool');
    });

    it('adds skills via addSkill', () => {
      const agent = new BaseAgent();
      agent.addSkill(new TestTool());
      
      const caps = agent.getCapabilities();
      expect(caps.provides).toContain('test');
    });

    it('removes skills via removeSkill', () => {
      const skill = new TestTool();
      const agent = new BaseAgent({ skills: [skill] });
      
      expect(agent.getCapabilities().provides).toContain('test');
      
      agent.removeSkill('TestTool');
      
      expect(agent.getCapabilities().provides || []).not.toContain('test');
    });

    it('warns on duplicate tool names', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      
      class DuplicateSkill extends Skill {
        @tool({ name: 'testTool' })
        async testTool(_p: Record<string, unknown>, _c: Context) {}
      }

      const agent = new BaseAgent({
        skills: [new TestTool(), new DuplicateSkill()],
      });
      
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('testTool')
      );
      warnSpy.mockRestore();
    });
  });

  describe('tool execution', () => {
    class MathSkill extends Skill {
      @tool({
        parameters: {
          type: 'object',
          properties: { a: { type: 'number' }, b: { type: 'number' } },
        },
      })
      async add(params: { a: number; b: number }, _c: Context) {
        return params.a + params.b;
      }

      @tool({ scopes: ['admin'] })
      async adminOnly(_p: Record<string, unknown>, _c: Context) {
        return 'admin secret';
      }
    }

    let agent: BaseAgent;

    beforeEach(() => {
      agent = new BaseAgent({
        skills: [new MathSkill()],
      });
    });

    it('executes tools by name', async () => {
      const result = await agent.executeTool('add', { a: 2, b: 3 });
      expect(result).toBe(5);
    });

    it('throws on unknown tool', async () => {
      await expect(agent.executeTool('unknown', {}))
        .rejects.toThrow('Tool not found: unknown');
    });

    it('respects tool scopes', async () => {
      // No scopes = should fail
      await expect(agent.executeTool('adminOnly', {}))
        .rejects.toThrow('Insufficient permissions');
    });
  });

  describe('hooks', () => {
    it('runs hooks in priority order', async () => {
      const order: number[] = [];
      
      class Hook1Skill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 20 })
        async second(_d: HookData, _c: Context): Promise<HookResult | void> {
          order.push(2);
        }
      }

      class Hook2Skill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 10 })
        async first(_d: HookData, _c: Context): Promise<HookResult | void> {
          order.push(1);
        }
      }

      const agent = new BaseAgent({
        skills: [new Hook1Skill(), new Hook2Skill()],
      });

      // Trigger hooks via run (which calls before_run)
      // Need a handoff to avoid error
      class MockLLM extends Skill {
        @handoff({ name: 'mock-llm' })
        async *processUAMP(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
        }
      }
      agent.addSkill(new MockLLM());

      await agent.run([{ role: 'user', content: 'hi' }]);
      
      expect(order).toEqual([1, 2]);
    });

    it('can abort with hook result', async () => {
      class AbortSkill extends Skill {
        @hook({ lifecycle: 'before_run', priority: 1 })
        async abort(_d: HookData, _c: Context): Promise<HookResult> {
          return { abort: true, abort_reason: 'Blocked by policy' };
        }
      }

      const agent = new BaseAgent({ skills: [new AbortSkill()] });
      
      await expect(agent.run([{ role: 'user', content: 'hi' }]))
        .rejects.toThrow('Blocked by policy');
    });

    it('can modify tool params via before_tool hook', async () => {
      class ModifySkill extends Skill {
        @hook({ lifecycle: 'before_tool' })
        async modify(data: HookData, _c: Context): Promise<HookResult> {
          return {
            tool_params: {
              ...(data.tool_params || {}),
              injected: true,
            },
          };
        }

        @tool({})
        async checkParams(params: Record<string, unknown>, _c: Context) {
          return params;
        }
      }

      const agent = new BaseAgent({ skills: [new ModifySkill()] });
      const result = await agent.executeTool('checkParams', { original: true });
      
      expect(result).toEqual({ original: true, injected: true });
    });
  });

  describe('handoffs', () => {
    class SimpleLLM extends Skill {
      @handoff({ name: 'simple-llm', priority: 10 })
      async *processUAMP(events: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
        // Find user input
        const userInput = events.find(e => e.type === 'input.text');
        const text = (userInput as { text: string })?.text || 'no input';
        
        yield createResponseDeltaEvent('r1', {
          type: 'text',
          text: `Echo: ${text}`,
        });
        yield createResponseDoneEvent('r1', [
          { type: 'text', text: `Echo: ${text}` },
        ]);
      }
    }

    it('selects best handoff by priority', async () => {
      class LowPriorityLLM extends Skill {
        @handoff({ name: 'low-priority', priority: 1 })
        async *processUAMP(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          yield createResponseDoneEvent('r2', [{ type: 'text', text: 'low' }]);
        }
      }

      const agent = new BaseAgent({
        skills: [new LowPriorityLLM(), new SimpleLLM()],
      });

      const response = await agent.run([{ role: 'user', content: 'test' }]);
      expect(response.content).toContain('Echo:');
    });

    it('returns error when no handoff available', async () => {
      const agent = new BaseAgent();
      
      const events: ServerEvent[] = [];
      for await (const event of agent.processUAMP([
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('hello'),
        createResponseCreateEvent(),
      ])) {
        events.push(event);
      }
      
      expect(events).toHaveLength(1);
      expect(events[0].type).toBe('response.error');
    });
  });

  describe('processUAMP() system-prompt injection', () => {
    // A skill that captures the conversation the agent assembled, so tests can
    // assert the system-message contract without spinning up a real LLM.
    class CaptureLLM extends Skill {
      public captured: Array<Record<string, unknown>> = [];
      @handoff({ name: 'capture-llm' })
      async *processUAMP(_e: ClientEvent[], ctx: Context): AsyncGenerator<ServerEvent> {
        this.captured = (ctx.get('_agentic_messages') as Array<Record<string, unknown>>) ?? [];
        yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
      }
    }

    it('injects agent.instructions as a leading system message by default', async () => {
      const capture = new CaptureLLM();
      const agent = new BaseAgent({
        instructions: 'I am the platform agent.',
        skills: [capture],
      });

      for await (const _ of agent.processUAMP([
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('hello'),
        createResponseCreateEvent(),
      ])) {
        // drain
      }

      expect(capture.captured[0]).toMatchObject({
        role: 'system',
        content: 'I am the platform agent.',
      });
    });

    it('appends session.instructions after agent.instructions', async () => {
      const capture = new CaptureLLM();
      const agent = new BaseAgent({
        instructions: 'I am the platform agent.',
        skills: [capture],
      });

      for await (const _ of agent.processUAMP([
        createSessionCreateEvent({
          modalities: ['text'],
          instructions: 'Caller hint: be brief.',
        }),
        createInputTextEvent('hello'),
        createResponseCreateEvent(),
      ])) {
        // drain
      }

      expect(capture.captured[0]).toMatchObject({
        role: 'system',
        content: 'I am the platform agent.\n\nCaller hint: be brief.',
      });
    });

    it('does not inject a system message when instructions are empty', async () => {
      const capture = new CaptureLLM();
      const agent = new BaseAgent({ skills: [capture] });

      for await (const _ of agent.processUAMP([
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('hello'),
        createResponseCreateEvent(),
      ])) {
        // drain
      }

      expect(capture.captured[0]?.role).not.toBe('system');
    });
  });

  describe('run()', () => {
    class EchoLLM extends Skill {
      @handoff({ name: 'echo-llm' })
      async *processUAMP(events: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
        const texts: string[] = [];
        for (const e of events) {
          if (e.type === 'input.text') {
            texts.push((e as { text: string }).text);
          }
        }
        
        const response = texts.join(' | ');
        yield createResponseDeltaEvent('r1', { type: 'text', text: response });
        yield createResponseDoneEvent('r1', [{ type: 'text', text: response }], 'completed', {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15,
        });
      }
    }

    let agent: BaseAgent;

    beforeEach(() => {
      agent = new BaseAgent({ skills: [new EchoLLM()] });
    });

    it('processes messages and returns response', async () => {
      const response = await agent.run([
        { role: 'user', content: 'Hello' },
      ]);
      
      expect(response.content).toBe('Hello');
      expect(response.content_items).toHaveLength(1);
    });

    it('includes usage stats', async () => {
      const response = await agent.run([
        { role: 'user', content: 'test' },
      ]);
      
      expect(response.usage?.total_tokens).toBe(15);
    });

    it('handles system messages', async () => {
      const response = await agent.run([
        { role: 'system', content: 'Be brief' },
        { role: 'user', content: 'Hi' },
      ]);
      
      expect(response.content).toBe('Be brief | Hi');
    });
  });

  describe('runStreaming()', () => {
    class StreamingLLM extends Skill {
      @handoff({ name: 'streaming-llm' })
      async *processUAMP(_events: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
        yield createResponseDeltaEvent('r1', { type: 'text', text: 'Hello' });
        yield createResponseDeltaEvent('r1', { type: 'text', text: ' World' });
        yield createResponseDoneEvent('r1', [{ type: 'text', text: 'Hello World' }]);
      }
    }

    it('yields delta chunks', async () => {
      const agent = new BaseAgent({ skills: [new StreamingLLM()] });
      
      const chunks: string[] = [];
      for await (const chunk of agent.runStreaming([{ role: 'user', content: 'test' }])) {
        if (chunk.type === 'delta') {
          chunks.push(chunk.delta);
        }
      }
      
      expect(chunks).toEqual(['Hello', ' World']);
    });

    it('yields done chunk at end', async () => {
      const agent = new BaseAgent({ skills: [new StreamingLLM()] });
      
      let doneChunk: unknown;
      for await (const chunk of agent.runStreaming([{ role: 'user', content: 'test' }])) {
        if (chunk.type === 'done') {
          doneChunk = chunk;
        }
      }
      
      expect(doneChunk).toBeDefined();
      expect((doneChunk as { response: { content: string } }).response.content).toBe('Hello World');
    });
  });

  describe('HTTP/WebSocket handlers', () => {
    class APISkill extends Skill {
      @http({ path: '/api/test', method: 'GET' })
      async handleGet(_req: Request, _ctx: Context): Promise<Response> {
        return new Response('ok');
      }

      @websocket({ path: '/ws/stream' })
      handleWs(_ws: WebSocket, _ctx: Context): void {}
    }

    it('provides access to HTTP handlers', () => {
      const agent = new BaseAgent({ skills: [new APISkill()] });
      
      const handler = agent.getHttpHandler('/api/test', 'GET');
      expect(handler).toBeDefined();
      expect(handler?.path).toBe('/api/test');
    });

    it('returns undefined for unknown HTTP handler', () => {
      const agent = new BaseAgent({ skills: [new APISkill()] });
      
      const handler = agent.getHttpHandler('/unknown', 'GET');
      expect(handler).toBeUndefined();
    });

    it('provides access to WebSocket handlers', () => {
      const agent = new BaseAgent({ skills: [new APISkill()] });
      
      const handler = agent.getWebSocketHandler('/ws/stream');
      expect(handler).toBeDefined();
    });
  });

  describe('addSkill setAgent', () => {
    it('calls setAgent on transport skills that define it', () => {
      const setAgentSpy = vi.fn();

      class TransportSkill extends Skill {
        setAgent(agent: unknown) {
          setAgentSpy(agent);
        }
      }

      const agent = new BaseAgent();
      agent.addSkill(new TransportSkill());

      expect(setAgentSpy).toHaveBeenCalledOnce();
      expect(setAgentSpy).toHaveBeenCalledWith(agent);
    });

    it('does not fail if skill has no setAgent', () => {
      class PlainSkill extends Skill {}

      const agent = new BaseAgent();
      expect(() => agent.addSkill(new PlainSkill())).not.toThrow();
    });

    it('populates wsRegistry when skill has @websocket decorator', () => {
      class WsSkill extends Skill {
        @websocket({ path: '/uamp' })
        handleConnection(_ws: WebSocket, _ctx: Context): void {}
      }

      const agent = new BaseAgent();
      agent.addSkill(new WsSkill());

      const handler = agent.getWebSocketHandler('/uamp');
      expect(handler).toBeDefined();
      expect(handler!.path).toBe('/uamp');
    });

    it('populates httpRegistry when skill has @http decorator', () => {
      class HttpSkill extends Skill {
        @http({ path: '/v1/chat/completions', method: 'POST' })
        async handleCompletions(_req: Request, _ctx: Context): Promise<Response> {
          return new Response('ok');
        }
      }

      const agent = new BaseAgent();
      agent.addSkill(new HttpSkill());

      const handler = agent.getHttpHandler('/v1/chat/completions', 'POST');
      expect(handler).toBeDefined();
      expect(handler!.path).toBe('/v1/chat/completions');
    });
  });

  describe('getToolDefinitions()', () => {
    it('returns tools in ToolDefinition format', () => {
      class DefSkill extends Skill {
        @tool({ description: 'Search for things', parameters: { type: 'object' } })
        async search(_p: Record<string, unknown>, _c: Context) {}
      }

      const agent = new BaseAgent({ skills: [new DefSkill()] });
      const defs = agent.getToolDefinitions();
      
      expect(defs).toHaveLength(1);
      expect(defs[0].type).toBe('function');
      expect(defs[0].function.name).toBe('search');
      expect(defs[0].function.description).toBe('Search for things');
    });
  });

  describe('requiresBridge gate', () => {
    class BridgeSkill extends Skill {
      @tool({ requiresBridge: 'discord', description: 'Discord-only DM' })
      async discordOnly(_p: Record<string, unknown>, _c: Context) {
        return 'sent';
      }
      @tool({ description: 'Always available' })
      async always(_p: Record<string, unknown>, _c: Context) {
        return 'ok';
      }
    }

    it('hides bridge-gated tools from getToolDefinitions when bridge mismatches', () => {
      const agent = new BaseAgent({ skills: [new BridgeSkill()] });
      // No bridge in context at all → discordOnly is hidden, always is visible.
      const names = agent.getToolDefinitions().map((d) => d.function.name);
      expect(names).toContain('always');
      expect(names).not.toContain('discordOnly');
    });

    it('shows bridge-gated tools when bridge.source matches', () => {
      const agent = new BaseAgent({ skills: [new BridgeSkill()] });
      // Inject bridge metadata onto the agent's Context the same way the
      // gateway does on inbound Discord messages.
      (agent as unknown as { context: Context }).context.metadata = {
        bridge: { source: 'discord', user_id: 'u123' },
      };
      const names = agent.getToolDefinitions().map((d) => d.function.name);
      expect(names).toContain('discordOnly');
      expect(names).toContain('always');
    });

    it('rejects executeTool with tool_unavailable_in_this_context when bridge mismatches', async () => {
      const agent = new BaseAgent({ skills: [new BridgeSkill()] });
      await expect(agent.executeTool('discordOnly', {})).rejects.toThrow(
        /tool_unavailable_in_this_context/,
      );
    });

    it('allows executeTool when bridge matches', async () => {
      const agent = new BaseAgent({ skills: [new BridgeSkill()] });
      (agent as unknown as { context: Context }).context.metadata = {
        bridge: { source: 'discord' },
      };
      await expect(agent.executeTool('discordOnly', {})).resolves.toBe('sent');
    });
  });

  describe('requiresConfirmation flow', () => {
    class SensitiveSkill extends Skill {
      ran = 0;
      @tool({ requiresConfirmation: true, description: 'Sensitive' })
      async sensitive(_p: Record<string, unknown>, _c: Context) {
        this.ran++;
        return 'done';
      }
      @tool({
        requiresConfirmation: (args: unknown) => (args as { needsApproval?: boolean }).needsApproval === true,
        description: 'Sometimes sensitive',
      })
      async sometimes(_p: Record<string, unknown>, _c: Context) {
        return 'ok';
      }
    }

    it('auto-approves when context.requestToolApproval is undefined (no NotificationSkill)', async () => {
      const skill = new SensitiveSkill();
      const agent = new BaseAgent({ skills: [skill] });
      // BaseAgent auto-injects LocalNotificationSkill; force the field
      // back to undefined to simulate a host that explicitly skips it.
      (agent as unknown as { context: Context }).context.requestToolApproval = undefined;
      await expect(agent.executeTool('sensitive', {})).resolves.toBe('done');
      expect(skill.ran).toBe(1);
    });

    it('returns { error: rejected_by_owner } when requestToolApproval rejects', async () => {
      const skill = new SensitiveSkill();
      const agent = new BaseAgent({ skills: [skill] });
      (agent as unknown as { context: Context }).context.requestToolApproval = vi
        .fn()
        .mockResolvedValue('rejected');
      const out = (await agent.executeTool('sensitive', { x: 1 })) as { error: string; tool_name: string };
      expect(out).toEqual({ error: 'rejected_by_owner', tool_name: 'sensitive' });
      expect(skill.ran).toBe(0);
    });

    it('treats a thrown approval surface as rejected (fail-safe)', async () => {
      const skill = new SensitiveSkill();
      const agent = new BaseAgent({ skills: [skill] });
      (agent as unknown as { context: Context }).context.requestToolApproval = vi
        .fn()
        .mockRejectedValue(new Error('boom'));
      const out = (await agent.executeTool('sensitive', {})) as { error: string };
      expect(out.error).toBe('rejected_by_owner');
      expect(skill.ran).toBe(0);
    });

    it('callback-style requiresConfirmation only prompts when callback returns true', async () => {
      const agent = new BaseAgent({ skills: [new SensitiveSkill()] });
      const approver = vi.fn().mockResolvedValue('approved');
      (agent as unknown as { context: Context }).context.requestToolApproval = approver;

      await agent.executeTool('sometimes', { needsApproval: false });
      expect(approver).not.toHaveBeenCalled();

      await agent.executeTool('sometimes', { needsApproval: true });
      expect(approver).toHaveBeenCalledTimes(1);
    });
  });

  describe('LocalNotificationSkill auto-injection', () => {
    it('installs context.requestToolApproval after run when no NotificationSkill is provided', async () => {
      class EchoLLM extends Skill {
        @handoff({ name: 'echo' })
        async *processUAMP(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
          yield createResponseDoneEvent('r1', [{ type: 'text', text: 'hi' }]);
        }
      }
      const agent = new BaseAgent({ skills: [new EchoLLM()] });
      // The before_run hook on LocalNotificationSkill runs during run(),
      // so kick the loop once and then assert the context callable is wired.
      await agent.run([{ role: 'user', content: 'hi' }]);
      const ctx = (agent as unknown as { context: Context }).context;
      expect(typeof ctx.requestToolApproval).toBe('function');
      expect(typeof ctx.notify).toBe('function');
      // Default is to auto-approve.
      const decision = await ctx.requestToolApproval!({
        toolName: 'foo',
        args: { a: 1 },
      });
      expect(decision).toBe('approved');
    });
  });

  describe('initialize / cleanup', () => {
    it('initializes all skills', async () => {
      const initFn = vi.fn();
      
      class InitSkill extends Skill {
        async initialize() {
          initFn();
        }
      }

      const agent = new BaseAgent({
        skills: [new InitSkill(), new InitSkill()],
      });
      
      await agent.initialize();
      expect(initFn).toHaveBeenCalledTimes(2);
    });

    it('cleans up all skills', async () => {
      const cleanupFn = vi.fn();
      
      class CleanupSkill extends Skill {
        async cleanup() {
          cleanupFn();
        }
      }

      const agent = new BaseAgent({
        skills: [new CleanupSkill()],
      });
      
      await agent.cleanup();
      expect(cleanupFn).toHaveBeenCalledOnce();
    });
  });
});
