/**
 * WebLLM Skill Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WebLLMSkill } from '../../../../src/skills/llm/webllm/skill.js';
import type { ClientEvent } from '../../../../src/uamp/events.js';

// Mock WebLLM engine
const mockCreate = vi.fn();
const mockUnload = vi.fn();
const mockResetChat = vi.fn();
const mockGetMessage = vi.fn();

const mockEngine = {
  chat: {
    completions: {
      create: mockCreate,
    },
  },
  getMessage: mockGetMessage,
  resetChat: mockResetChat,
  unload: mockUnload,
};

const mockCreateMLCEngine = vi.fn().mockResolvedValue(mockEngine);

// Mock the dynamic import for @mlc-ai/web-llm
vi.mock('@mlc-ai/web-llm', () => ({
  CreateMLCEngine: mockCreateMLCEngine,
  default: { CreateMLCEngine: mockCreateMLCEngine },
}));

describe('WebLLMSkill', () => {
  let skill: WebLLMSkill;

  beforeEach(() => {
    vi.clearAllMocks();
    skill = new WebLLMSkill({
      model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC',
    });
  });

  afterEach(async () => {
    await skill.cleanup();
  });

  describe('constructor', () => {
    it('creates skill with model config', () => {
      expect(skill.name).toBe('webllm');
    });

    it('uses custom name if provided', () => {
      const customSkill = new WebLLMSkill({
        model: 'test-model',
        name: 'custom-webllm',
      });
      expect(customSkill.name).toBe('custom-webllm');
    });

    it('accepts optional config parameters', () => {
      const configuredSkill = new WebLLMSkill({
        model: 'test-model',
        temperature: 0.5,
        max_tokens: 1024,
        top_p: 0.9,
      });
      expect(configuredSkill).toBeDefined();
    });
  });

  describe('getCapabilities', () => {
    it('returns capabilities with model id', () => {
      const caps = skill.getCapabilities();
      expect(caps.id).toBe('Llama-3.1-8B-Instruct-q4f32_1-MLC');
    });

    it('identifies as webllm provider', () => {
      const caps = skill.getCapabilities();
      expect(caps.provider).toBe('webllm');
    });

    it('supports text modality', () => {
      const caps = skill.getCapabilities();
      expect(caps.modalities).toContain('text');
    });

    it('supports streaming', () => {
      const caps = skill.getCapabilities();
      expect(caps.supports_streaming).toBe(true);
    });

    it('does not support thinking', () => {
      const caps = skill.getCapabilities();
      expect(caps.supports_thinking).toBe(false);
    });

    it('supports caching', () => {
      const caps = skill.getCapabilities();
      expect(caps.supports_caching).toBe(true);
    });

    it('includes browser runtime extension', () => {
      const caps = skill.getCapabilities();
      expect(caps.extensions?.runtime).toBe('browser');
    });

    it('includes webgpu engine extension', () => {
      const caps = skill.getCapabilities();
      expect(caps.extensions?.engine).toBe('webgpu');
    });
  });

  describe('initialize', () => {
    it('initializes without error', async () => {
      await expect(skill.initialize()).resolves.not.toThrow();
    });
  });

  describe('cleanup', () => {
    it('handles cleanup when not initialized', async () => {
      await expect(skill.cleanup()).resolves.not.toThrow();
    });
  });

  describe('handoff registration', () => {
    it('registers processUAMP as handoff', () => {
      expect(skill.handoffs.length).toBeGreaterThan(0);
    });

    it('handoff has name webllm', () => {
      const handoff = skill.handoffs.find(h => h.name === 'webllm');
      expect(handoff).toBeDefined();
    });

    it('handoff has priority 5', () => {
      const handoff = skill.handoffs.find(h => h.name === 'webllm');
      expect(handoff?.priority).toBe(5);
    });
  });
});

// Test with manual engine injection to avoid dynamic import issues
describe('WebLLMSkill with mocked engine', () => {
  let skill: WebLLMSkill;
  let mockEngine: any;
  let mockCreate: ReturnType<typeof vi.fn>;
  let mockUnload: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockCreate = vi.fn();
    mockUnload = vi.fn().mockResolvedValue(undefined);
    
    mockEngine = {
      chat: {
        completions: {
          create: mockCreate,
        },
      },
      unload: mockUnload,
    };

    skill = new WebLLMSkill({
      model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC',
    });

    // Inject the mock engine and createEngine function directly
    (skill as any).engine = null;
    (skill as any).createEngine = vi.fn().mockResolvedValue(mockEngine);
  });

  afterEach(async () => {
    vi.clearAllMocks();
  });

  describe('processUAMP', () => {
    it('yields response.created event first', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Hello']));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      expect(responses[0].type).toBe('response.created');
      expect(responses[0].event_id).toBeDefined();
      expect(responses[0].response_id).toBeDefined();
    });

    it('extracts system message from session.create', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events: ClientEvent[] = [
        {
          type: 'session.create',
          event_id: 'e1',
          uamp_version: '1.0',
          session: {
            modalities: ['text'],
            instructions: 'You are a helpful assistant',
          },
        },
        {
          type: 'input.text',
          event_id: 'e2',
          text: 'Hello',
          role: 'user',
        },
        {
          type: 'response.create',
          event_id: 'e3',
        },
      ];
      
      const responses: any[] = [];
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            { role: 'system', content: 'You are a helpful assistant' },
            { role: 'user', content: 'Hello' },
          ]),
        })
      );
    });

    it('extracts user message from input.text', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Hi there!']));
      
      const events = createTestEvents('Hello world');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            { role: 'user', content: 'Hello world' },
          ]),
        })
      );
    });

    it('yields delta events for streaming content', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Hello', ' ', 'World']));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const deltas = responses.filter(e => e.type === 'response.delta');
      expect(deltas.length).toBe(3);
      expect(deltas[0].delta.text).toBe('Hello');
      expect(deltas[1].delta.text).toBe(' ');
      expect(deltas[2].delta.text).toBe('World');
    });

    it('yields response.done event at end', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Hello']));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect(doneEvent.response.status).toBe('completed');
    });

    it('includes full content in response.done output', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Hello', ' ', 'World']));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent.response.output[0].text).toBe('Hello World');
    });

    it('includes usage stats when available', async () => {
      mockCreate.mockResolvedValueOnce(createMockStreamWithUsage(['Hello'], {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      }));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent.response.usage).toEqual({
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
      });
    });

    it('yields error when no input messages', async () => {
      const events: ClientEvent[] = [
        { type: 'response.create', event_id: 'e1' },
      ];
      
      const responses: any[] = [];
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const errorEvent = responses.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent.error.code).toBe('no_input');
    });

    it('yields error on engine failure', async () => {
      mockCreate.mockRejectedValueOnce(new Error('WebGPU not supported'));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const errorEvent = responses.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent.error.code).toBe('webllm_error');
      expect(errorEvent.error.message).toBe('WebGPU not supported');
    });

    it('uses configured temperature', async () => {
      const configuredSkill = new WebLLMSkill({
        model: 'test-model',
        temperature: 0.3,
      });
      (configuredSkill as any).createEngine = vi.fn().mockResolvedValue(mockEngine);
      
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of configuredSkill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          temperature: 0.3,
        })
      );
    });

    it('uses configured max_tokens', async () => {
      const configuredSkill = new WebLLMSkill({
        model: 'test-model',
        max_tokens: 512,
      });
      (configuredSkill as any).createEngine = vi.fn().mockResolvedValue(mockEngine);
      
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of configuredSkill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          max_tokens: 512,
        })
      );
    });

    it('uses configured top_p', async () => {
      const configuredSkill = new WebLLMSkill({
        model: 'test-model',
        top_p: 0.8,
      });
      (configuredSkill as any).createEngine = vi.fn().mockResolvedValue(mockEngine);
      
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of configuredSkill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          top_p: 0.8,
        })
      );
    });

    it('uses default temperature when not configured', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          temperature: 0.7,
        })
      );
    });

    it('uses default max_tokens when not configured', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          max_tokens: 2048,
        })
      );
    });

    it('uses default top_p when not configured', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          top_p: 0.95,
        })
      );
    });

    it('enables streaming by default', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          stream: true,
          stream_options: { include_usage: true },
        })
      );
    });

    it('generates unique event IDs', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['A', 'B']));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const eventIds = responses.map(e => e.event_id);
      const uniqueIds = new Set(eventIds);
      expect(uniqueIds.size).toBe(eventIds.length);
    });

    it('maintains consistent response_id across events', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['A', 'B', 'C']));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const responseIds = responses.filter(e => e.response_id).map(e => e.response_id);
      const uniqueResponseIds = new Set(responseIds);
      expect(uniqueResponseIds.size).toBe(1);
    });

    it('handles assistant role in input', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream(['Response']));
      
      const events: ClientEvent[] = [
        { type: 'input.text', event_id: 'e1', text: 'Hi', role: 'user' },
        { type: 'input.text', event_id: 'e2', text: 'Hello', role: 'assistant' },
        { type: 'input.text', event_id: 'e3', text: 'How are you?', role: 'user' },
        { type: 'response.create', event_id: 'e4' },
      ];
      
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            { role: 'user', content: 'Hi' },
            { role: 'assistant', content: 'Hello' },
            { role: 'user', content: 'How are you?' },
          ],
        })
      );
    });

    it('skips empty delta content', async () => {
      // Create stream with some empty deltas
      const mockStream = {
        async *[Symbol.asyncIterator]() {
          yield { id: 'c1', choices: [{ index: 0, delta: { content: 'Hello' }, finish_reason: null }] };
          yield { id: 'c2', choices: [{ index: 0, delta: { content: '' }, finish_reason: null }] };
          yield { id: 'c3', choices: [{ index: 0, delta: {}, finish_reason: null }] };
          yield { id: 'c4', choices: [{ index: 0, delta: { content: ' World' }, finish_reason: 'stop' }] };
        }
      };
      mockCreate.mockResolvedValueOnce(mockStream);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const deltas = responses.filter(e => e.type === 'response.delta');
      expect(deltas.length).toBe(2);
      expect(deltas[0].delta.text).toBe('Hello');
      expect(deltas[1].delta.text).toBe(' World');
    });

    it('handles empty response gracefully', async () => {
      mockCreate.mockResolvedValueOnce(createMockStream([]));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect(doneEvent.response.output).toEqual([]);
    });
  });

  describe('cleanup with engine', () => {
    it('unloads engine on cleanup', async () => {
      // Force engine to be set
      (skill as any).engine = mockEngine;
      
      await skill.cleanup();
      
      expect(mockUnload).toHaveBeenCalled();
    });

    it('resets engine to null after cleanup', async () => {
      (skill as any).engine = mockEngine;
      
      await skill.cleanup();
      
      expect((skill as any).engine).toBeNull();
    });
  });

  describe('ensureEngine', () => {
    it('throws when createEngine not available', async () => {
      const uninitSkill = new WebLLMSkill({ model: 'test' });
      // Don't inject createEngine
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of uninitSkill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const errorEvent = responses.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent.error.message).toContain('WebLLM not available');
    });

    it('reuses existing engine', async () => {
      (skill as any).engine = mockEngine;
      mockCreate.mockResolvedValueOnce(createMockStream(['A']));
      mockCreate.mockResolvedValueOnce(createMockStream(['B']));
      
      const events = createTestEvents('Hi');
      
      // First call
      for await (const _ of skill.processUAMP(events, {} as any)) {}
      
      // Second call
      for await (const _ of skill.processUAMP(events, {} as any)) {}
      
      // createEngine should not be called since engine already exists
      expect((skill as any).createEngine).not.toHaveBeenCalled();
    });
  });
});

// Helper functions

function createTestEvents(userMessage: string): ClientEvent[] {
  return [
    {
      type: 'session.create',
      event_id: 'e1',
      uamp_version: '1.0',
      session: {
        modalities: ['text'],
      },
    },
    {
      type: 'input.text',
      event_id: 'e2',
      text: userMessage,
      role: 'user',
    },
    {
      type: 'response.create',
      event_id: 'e3',
    },
  ];
}

function createMockStream(chunks: string[]) {
  return {
    async *[Symbol.asyncIterator]() {
      for (let i = 0; i < chunks.length; i++) {
        yield {
          id: `chunk-${i}`,
          choices: [{
            index: 0,
            delta: { content: chunks[i] },
            finish_reason: i === chunks.length - 1 ? 'stop' : null,
          }],
        };
      }
    }
  };
}

function createMockStreamWithUsage(chunks: string[], usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }) {
  return {
    async *[Symbol.asyncIterator]() {
      for (let i = 0; i < chunks.length; i++) {
        const isLast = i === chunks.length - 1;
        yield {
          id: `chunk-${i}`,
          choices: [{
            index: 0,
            delta: { content: chunks[i] },
            finish_reason: isLast ? 'stop' : null,
          }],
          ...(isLast ? { usage } : {}),
        };
      }
    }
  };
}
