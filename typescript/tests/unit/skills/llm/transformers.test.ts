/**
 * Transformers.js Skill Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TransformersSkill } from '../../../../src/skills/llm/transformers/skill.js';
import type { ClientEvent } from '../../../../src/uamp/events.js';

describe('TransformersSkill', () => {
  let skill: TransformersSkill;

  beforeEach(() => {
    vi.clearAllMocks();
    skill = new TransformersSkill({
      model: 'Xenova/Phi-3-mini-4k-instruct',
    });
  });

  afterEach(async () => {
    await skill.cleanup();
  });

  describe('constructor', () => {
    it('creates skill with model config', () => {
      expect(skill.name).toBe('transformers');
    });

    it('uses custom name if provided', () => {
      const customSkill = new TransformersSkill({
        model: 'test-model',
        name: 'custom-transformers',
      });
      expect(customSkill.name).toBe('custom-transformers');
    });

    it('accepts optional config parameters', () => {
      const configuredSkill = new TransformersSkill({
        model: 'test-model',
        device: 'wasm',
        temperature: 0.5,
        max_tokens: 1024,
        top_p: 0.9,
        dtype: 'fp16',
      });
      expect(configuredSkill).toBeDefined();
    });
  });

  describe('getCapabilities', () => {
    it('returns capabilities with model id', () => {
      const caps = skill.getCapabilities();
      expect(caps.id).toBe('Xenova/Phi-3-mini-4k-instruct');
    });

    it('identifies as transformers.js provider', () => {
      const caps = skill.getCapabilities();
      expect(caps.provider).toBe('transformers.js');
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

    it('includes webgpu+wasm engine extension', () => {
      const caps = skill.getCapabilities();
      expect(caps.extensions?.engine).toBe('webgpu+wasm');
    });

    it('uses default webgpu device', () => {
      const caps = skill.getCapabilities();
      expect(caps.extensions?.device).toBe('webgpu');
    });

    it('uses configured device', () => {
      const wasmSkill = new TransformersSkill({
        model: 'test',
        device: 'wasm',
      });
      const caps = wasmSkill.getCapabilities();
      expect(caps.extensions?.device).toBe('wasm');
    });
  });

  describe('initialize', () => {
    it('initializes without error', async () => {
      await expect(skill.initialize()).resolves.not.toThrow();
    });
  });

  describe('cleanup', () => {
    it('handles cleanup', async () => {
      await expect(skill.cleanup()).resolves.not.toThrow();
    });
  });

  describe('handoff registration', () => {
    it('registers processUAMP as handoff', () => {
      expect(skill.handoffs.length).toBeGreaterThan(0);
    });

    it('handoff has name transformers', () => {
      const handoff = skill.handoffs.find(h => h.name === 'transformers');
      expect(handoff).toBeDefined();
    });

    it('handoff has priority 4', () => {
      const handoff = skill.handoffs.find(h => h.name === 'transformers');
      expect(handoff?.priority).toBe(4);
    });
  });
});

// Test with manual pipeline injection to avoid dynamic import issues
describe('TransformersSkill with mocked pipeline', () => {
  let skill: TransformersSkill;
  let mockPipeline: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockPipeline = vi.fn();

    skill = new TransformersSkill({
      model: 'Xenova/Phi-3-mini-4k-instruct',
    });

    // Inject the mock pipeline and pipelineFn directly
    (skill as any).pipeline = null;
    (skill as any).pipelineFn = vi.fn().mockResolvedValue(mockPipeline);
  });

  afterEach(async () => {
    vi.clearAllMocks();
  });

  describe('processUAMP', () => {
    it('yields response.created event first', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt\n\nAssistant: Hello' }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      expect(responses[0].type).toBe('response.created');
      expect(responses[0].event_id).toBeDefined();
      expect(responses[0].response_id).toBeDefined();
    });

    it('formats system message in prompt', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'System: Be helpful\n\nUser: Hello\n\nAssistant: Hi there!' }]);
      
      const events: ClientEvent[] = [
        {
          type: 'session.create',
          event_id: 'e1',
          uamp_version: '1.0',
          session: {
            modalities: ['text'],
            instructions: 'Be helpful',
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
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.stringContaining('System: Be helpful'),
        expect.any(Object)
      );
    });

    it('formats user message in prompt', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'User: Hello world\n\nAssistant: Hi!' }]);
      
      const events = createTestEvents('Hello world');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.stringContaining('User: Hello world'),
        expect.any(Object)
      );
    });

    it('appends Assistant: to prompt', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'User: Hi\n\nAssistant: Hello!' }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.stringMatching(/Assistant:$/),
        expect.any(Object)
      );
    });

    it('yields delta event with response text', async () => {
      const fullPrompt = 'User: Hi\n\nAssistant:';
      mockPipeline.mockResolvedValueOnce([{ generated_text: `${fullPrompt} Hello there!` }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const deltas = responses.filter(e => e.type === 'response.delta');
      expect(deltas.length).toBe(1);
      expect(deltas[0].delta.text).toBe('Hello there!');
    });

    it('yields response.done event at end', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect(doneEvent.response.status).toBe('completed');
    });

    it('includes response text in output', async () => {
      const fullPrompt = 'User: Hi\n\nAssistant:';
      mockPipeline.mockResolvedValueOnce([{ generated_text: `${fullPrompt} Hello World` }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent.response.output[0].text).toBe('Hello World');
    });

    it('includes usage stats (zeroed)', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent.response.usage).toEqual({
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
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

    it('yields error on pipeline failure', async () => {
      mockPipeline.mockRejectedValueOnce(new Error('Model loading failed'));
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const errorEvent = responses.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent.error.code).toBe('transformers_error');
      expect(errorEvent.error.message).toBe('Model loading failed');
    });

    it('uses configured temperature', async () => {
      const configuredSkill = new TransformersSkill({
        model: 'test-model',
        temperature: 0.3,
      });
      (configuredSkill as any).pipelineFn = vi.fn().mockResolvedValue(mockPipeline);
      
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of configuredSkill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          temperature: 0.3,
        })
      );
    });

    it('uses configured max_tokens', async () => {
      const configuredSkill = new TransformersSkill({
        model: 'test-model',
        max_tokens: 256,
      });
      (configuredSkill as any).pipelineFn = vi.fn().mockResolvedValue(mockPipeline);
      
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of configuredSkill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          max_new_tokens: 256,
        })
      );
    });

    it('uses configured top_p', async () => {
      const configuredSkill = new TransformersSkill({
        model: 'test-model',
        top_p: 0.8,
      });
      (configuredSkill as any).pipelineFn = vi.fn().mockResolvedValue(mockPipeline);
      
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of configuredSkill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          top_p: 0.8,
        })
      );
    });

    it('uses default temperature when not configured', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          temperature: 0.7,
        })
      );
    });

    it('uses default max_new_tokens when not configured', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          max_new_tokens: 512,
        })
      );
    });

    it('uses default top_p when not configured', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          top_p: 0.95,
        })
      );
    });

    it('enables sampling by default', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of skill.processUAMP(events, {} as any)) {
        // consume
      }
      
      expect(mockPipeline).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          do_sample: true,
        })
      );
    });

    it('generates unique event IDs', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: A B' }]);
      
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
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const responseIds = responses.filter(e => e.response_id).map(e => e.response_id);
      const uniqueResponseIds = new Set(responseIds);
      expect(uniqueResponseIds.size).toBe(1);
    });

    it('handles empty response gracefully', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: '' }]);
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect(doneEvent.response.output).toEqual([]);
    });

    it('strips prompt from response', async () => {
      const prompt = 'System: Be helpful\n\nUser: Hi\n\nAssistant:';
      mockPipeline.mockResolvedValueOnce([{ generated_text: `${prompt} This is the response` }]);
      
      const events: ClientEvent[] = [
        {
          type: 'session.create',
          event_id: 'e1',
          uamp_version: '1.0',
          session: { modalities: ['text'], instructions: 'Be helpful' },
        },
        { type: 'input.text', event_id: 'e2', text: 'Hi', role: 'user' },
        { type: 'response.create', event_id: 'e3' },
      ];
      
      const responses: any[] = [];
      for await (const event of skill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent.response.output[0].text).toBe('This is the response');
    });
  });

  describe('ensurePipeline', () => {
    it('throws when pipelineFn not available', async () => {
      const uninitSkill = new TransformersSkill({ model: 'test' });
      // Don't inject pipelineFn
      
      const events = createTestEvents('Hi');
      const responses: any[] = [];
      
      for await (const event of uninitSkill.processUAMP(events, {} as any)) {
        responses.push(event);
      }
      
      const errorEvent = responses.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent.error.message).toContain('Transformers.js not available');
    });

    it('reuses existing pipeline', async () => {
      const mockExistingPipeline = vi.fn().mockResolvedValue([{ generated_text: 'prompt Assistant: A' }]);
      (skill as any).pipeline = mockExistingPipeline;
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: B' }]);
      
      const events = createTestEvents('Hi');
      
      // First call - should use existing pipeline
      for await (const _ of skill.processUAMP(events, {} as any)) {}
      
      // Second call
      for await (const _ of skill.processUAMP(events, {} as any)) {}
      
      // pipelineFn should not be called since pipeline already exists
      expect((skill as any).pipelineFn).not.toHaveBeenCalled();
    });

    it('creates pipeline with configured device', async () => {
      const wasmSkill = new TransformersSkill({
        model: 'test-model',
        device: 'wasm',
      });
      const mockPipelineFn = vi.fn().mockResolvedValue(mockPipeline);
      (wasmSkill as any).pipelineFn = mockPipelineFn;
      
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of wasmSkill.processUAMP(events, {} as any)) {}
      
      expect(mockPipelineFn).toHaveBeenCalledWith(
        'text-generation',
        'test-model',
        expect.objectContaining({
          device: 'wasm',
        })
      );
    });

    it('creates pipeline with configured dtype', async () => {
      const dtypeSkill = new TransformersSkill({
        model: 'test-model',
        dtype: 'fp16',
      });
      const mockPipelineFn = vi.fn().mockResolvedValue(mockPipeline);
      (dtypeSkill as any).pipelineFn = mockPipelineFn;
      
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events = createTestEvents('Hi');
      for await (const _ of dtypeSkill.processUAMP(events, {} as any)) {}
      
      expect(mockPipelineFn).toHaveBeenCalledWith(
        'text-generation',
        'test-model',
        expect.objectContaining({
          dtype: 'fp16',
        })
      );
    });
  });

  describe('prompt formatting', () => {
    it('handles multiple user messages', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events: ClientEvent[] = [
        { type: 'input.text', event_id: 'e1', text: 'First', role: 'user' },
        { type: 'input.text', event_id: 'e2', text: 'Second', role: 'user' },
        { type: 'response.create', event_id: 'e3' },
      ];
      
      for await (const _ of skill.processUAMP(events, {} as any)) {}
      
      const calledPrompt = mockPipeline.mock.calls[0][0];
      expect(calledPrompt).toContain('User: First');
      expect(calledPrompt).toContain('User: Second');
    });

    it('formats system role as System', async () => {
      mockPipeline.mockResolvedValueOnce([{ generated_text: 'prompt Assistant: Response' }]);
      
      const events: ClientEvent[] = [
        { type: 'input.text', event_id: 'e1', text: 'System message', role: 'system' },
        { type: 'input.text', event_id: 'e2', text: 'User message', role: 'user' },
        { type: 'response.create', event_id: 'e3' },
      ];
      
      for await (const _ of skill.processUAMP(events, {} as any)) {}
      
      const calledPrompt = mockPipeline.mock.calls[0][0];
      expect(calledPrompt).toContain('System: System message');
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
