/**
 * Portal Transport + LLM Integration Tests
 * 
 * Tests the portal transport skill with WebLLM/Transformers LLM skills.
 * Uses mocked engines for fast CI, with option for real models in E2E.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { handoff } from '../../src/core/decorators.js';
import { WebLLMSkill } from '../../src/skills/llm/webllm/skill.js';
import { TransformersSkill } from '../../src/skills/llm/transformers/skill.js';
import { PortalTransportSkill } from '../../src/skills/transport/portal/skill.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import { generateEventId } from '../../src/uamp/events.js';
import type { Context } from '../../src/core/types.js';

// Mock WebSocket for testing
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.OPEN;
  onopen: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;

  private messageHandlers: ((ev: MessageEvent) => void)[] = [];

  constructor(public url: string) {
    // Simulate connection
    setTimeout(() => {
      if (this.onopen) this.onopen(new Event('open'));
    }, 0);
  }

  send(data: string): void {
    // Store for assertions
    (this as any).lastSent = data;
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose(new CloseEvent('close'));
  }

  addEventListener(type: string, handler: (ev: any) => void): void {
    if (type === 'message') {
      this.messageHandlers.push(handler);
    }
  }

  removeEventListener(type: string, handler: (ev: any) => void): void {
    if (type === 'message') {
      const idx = this.messageHandlers.indexOf(handler);
      if (idx >= 0) this.messageHandlers.splice(idx, 1);
    }
  }

  // Test helper: simulate incoming message
  simulateMessage(data: string): void {
    const event = new MessageEvent('message', { data });
    if (this.onmessage) this.onmessage(event);
    for (const handler of this.messageHandlers) {
      handler(event);
    }
  }
}

// Mock LLM Skill that returns predictable responses
class MockLLMSkill extends Skill {
  private responseText: string;
  private responseDelay: number;

  constructor(responseText = 'Hello! I am a mock LLM.', delay = 0) {
    super({ name: 'mock-llm' });
    this.responseText = responseText;
    this.responseDelay = delay;
  }

  @handoff({ name: 'mock-llm', priority: 10 })
  async *processUAMP(
    events: ClientEvent[],
    _context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield {
      type: 'response.created',
      event_id: generateEventId(),
      response_id: responseId,
    };

    if (this.responseDelay > 0) {
      await new Promise(resolve => setTimeout(resolve, this.responseDelay));
    }

    // Stream response in chunks
    const words = this.responseText.split(' ');
    for (let i = 0; i < words.length; i++) {
      yield {
        type: 'response.delta',
        event_id: generateEventId(),
        response_id: responseId,
        delta: {
          type: 'text',
          text: (i > 0 ? ' ' : '') + words[i],
        },
      };
    }

    yield {
      type: 'response.done',
      event_id: generateEventId(),
      response_id: responseId,
      response: {
        id: responseId,
        status: 'completed',
        output: [{ type: 'text', text: this.responseText }],
        usage: {
          input_tokens: 10,
          output_tokens: words.length,
          total_tokens: 10 + words.length,
        },
      },
    };
  }
}

describe('Portal Transport + LLM Integration', () => {
  let agent: BaseAgent;
  let portalSkill: PortalTransportSkill;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(async () => {
    if (agent) {
      await agent.cleanup();
    }
  });

  describe('with MockLLMSkill', () => {
    beforeEach(() => {
      portalSkill = new PortalTransportSkill();
      agent = new BaseAgent({
        name: 'test-agent',
        skills: [new MockLLMSkill('Hello from mock LLM!'), portalSkill],
      });
      portalSkill.setAgent(agent);
    });

    it('agent processes UAMP events through portal WebSocket', async () => {
      const mockWs = new MockWebSocket('ws://test');
      const responses: ServerEvent[] = [];

      // Simulate handleConnection being called
      portalSkill.handleConnection(mockWs as any, {} as any);

      // Set up response capture
      const originalSend = mockWs.send.bind(mockWs);
      mockWs.send = (data: string) => {
        originalSend(data);
        try {
          responses.push(JSON.parse(data));
        } catch {
          // Ignore non-JSON
        }
      };

      // Simulate incoming UAMP message
      const uampMessage = JSON.stringify({
        type: 'uamp',
        events: [
          {
            type: 'session.create',
            event_id: 'e1',
            uamp_version: '1.0',
            session: { modalities: ['text'] },
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
        ],
      });

      mockWs.simulateMessage(uampMessage);

      // Wait for async processing
      await new Promise(resolve => setTimeout(resolve, 50));

      // Verify responses
      expect(responses.length).toBeGreaterThan(0);
      const createdEvent = responses.find(e => e.type === 'response.created');
      expect(createdEvent).toBeDefined();

      const deltas = responses.filter(e => e.type === 'response.delta');
      expect(deltas.length).toBeGreaterThan(0);

      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect((doneEvent as any).response.output[0].text).toBe('Hello from mock LLM!');
    });

    it('handles discover message', async () => {
      const mockWs = new MockWebSocket('ws://test');
      let response: any = null;

      portalSkill.handleConnection(mockWs as any, {} as any);

      mockWs.send = (data: string) => {
        response = JSON.parse(data);
      };

      mockWs.simulateMessage(JSON.stringify({ type: 'discover' }));

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(response).toBeDefined();
      expect(response.type).toBe('agents');
      expect(response.agents[0].name).toBe('test-agent');
      expect(response.agents[0].capabilities).toBeDefined();
    });

    it('broadcasts events to connected clients', async () => {
      const ws1 = new MockWebSocket('ws://test');
      const ws2 = new MockWebSocket('ws://test');
      const received1: string[] = [];
      const received2: string[] = [];

      portalSkill.handleConnection(ws1 as any, {} as any);
      portalSkill.handleConnection(ws2 as any, {} as any);

      ws1.send = (data: string) => received1.push(data);
      ws2.send = (data: string) => received2.push(data);

      const event: ServerEvent = {
        type: 'response.delta',
        event_id: 'test-id',
        response_id: 'resp-id',
        delta: { type: 'text', text: 'Broadcast test' },
      };

      portalSkill.broadcast(event);

      expect(received1.length).toBe(1);
      expect(received2.length).toBe(1);
      expect(JSON.parse(received1[0]).delta.text).toBe('Broadcast test');
    });

    it('handles WebSocket errors gracefully', async () => {
      const mockWs = new MockWebSocket('ws://test');
      let errorHandled = false;

      portalSkill.handleConnection(mockWs as any, {} as any);

      // Capture error response
      mockWs.send = () => {
        errorHandled = true;
      };

      // Simulate malformed message
      mockWs.simulateMessage('not valid json');

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(errorHandled).toBe(true);
    });
  });

  describe('with mocked WebLLMSkill', () => {
    let webllmSkill: WebLLMSkill;
    let mockCreate: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      mockCreate = vi.fn();
      
      webllmSkill = new WebLLMSkill({
        model: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', // Small model
      });

      // Inject mock engine
      const mockEngine = {
        chat: { completions: { create: mockCreate } },
        unload: vi.fn(),
      };
      (webllmSkill as any).engine = null;
      (webllmSkill as any).createEngine = vi.fn().mockResolvedValue(mockEngine);

      portalSkill = new PortalTransportSkill();
      agent = new BaseAgent({
        name: 'webllm-agent',
        skills: [webllmSkill, portalSkill],
      });
      portalSkill.setAgent(agent);
    });

    it('processes UAMP through portal with WebLLM', async () => {
      // Mock streaming response
      mockCreate.mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield { id: 'c1', choices: [{ index: 0, delta: { content: 'Hello' }, finish_reason: null }] };
          yield { id: 'c2', choices: [{ index: 0, delta: { content: ' from' }, finish_reason: null }] };
          yield { id: 'c3', choices: [{ index: 0, delta: { content: ' WebLLM!' }, finish_reason: 'stop' }] };
        }
      });

      const mockWs = new MockWebSocket('ws://test');
      const responses: ServerEvent[] = [];

      portalSkill.handleConnection(mockWs as any, {} as any);

      mockWs.send = (data: string) => {
        try {
          responses.push(JSON.parse(data));
        } catch {
          // Ignore
        }
      };

      const uampMessage = JSON.stringify({
        type: 'uamp',
        events: [
          { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } },
          { type: 'input.text', event_id: 'e2', text: 'Hi', role: 'user' },
          { type: 'response.create', event_id: 'e3' },
        ],
      });

      mockWs.simulateMessage(uampMessage);

      await new Promise(resolve => setTimeout(resolve, 100));

      // Verify WebLLM was called
      expect(mockCreate).toHaveBeenCalled();

      // Verify streaming response
      const deltas = responses.filter(e => e.type === 'response.delta');
      expect(deltas.length).toBe(3);
      expect((deltas[0] as any).delta.text).toBe('Hello');
      expect((deltas[1] as any).delta.text).toBe(' from');
      expect((deltas[2] as any).delta.text).toBe(' WebLLM!');

      // Verify done event
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect((doneEvent as any).response.output[0].text).toBe('Hello from WebLLM!');
    });

    it('handles WebLLM errors through portal', async () => {
      mockCreate.mockRejectedValueOnce(new Error('WebGPU not available'));

      const mockWs = new MockWebSocket('ws://test');
      const responses: ServerEvent[] = [];

      portalSkill.handleConnection(mockWs as any, {} as any);

      mockWs.send = (data: string) => {
        try {
          responses.push(JSON.parse(data));
        } catch {
          // Ignore
        }
      };

      const uampMessage = JSON.stringify({
        type: 'uamp',
        events: [
          { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } },
          { type: 'input.text', event_id: 'e2', text: 'Hi', role: 'user' },
          { type: 'response.create', event_id: 'e3' },
        ],
      });

      mockWs.simulateMessage(uampMessage);

      await new Promise(resolve => setTimeout(resolve, 100));

      const errorEvent = responses.find(e => e.type === 'response.error');
      expect(errorEvent).toBeDefined();
      expect((errorEvent as any).error.message).toBe('WebGPU not available');
    });
  });

  describe('with mocked TransformersSkill', () => {
    let transformersSkill: TransformersSkill;
    let mockPipeline: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      mockPipeline = vi.fn();

      transformersSkill = new TransformersSkill({
        model: 'Xenova/Phi-3-mini-4k-instruct',
        device: 'wasm', // Use WASM for broader compatibility
      });

      // Inject mock pipeline
      (transformersSkill as any).pipeline = null;
      (transformersSkill as any).pipelineFn = vi.fn().mockResolvedValue(mockPipeline);

      portalSkill = new PortalTransportSkill();
      agent = new BaseAgent({
        name: 'transformers-agent',
        skills: [transformersSkill, portalSkill],
      });
      portalSkill.setAgent(agent);
    });

    it('processes UAMP through portal with Transformers.js', async () => {
      const prompt = 'User: Hello\n\nAssistant:';
      mockPipeline.mockResolvedValueOnce([{
        generated_text: `${prompt} Hi there! I'm running in the browser.`
      }]);

      const mockWs = new MockWebSocket('ws://test');
      const responses: ServerEvent[] = [];

      portalSkill.handleConnection(mockWs as any, {} as any);

      mockWs.send = (data: string) => {
        try {
          responses.push(JSON.parse(data));
        } catch {
          // Ignore
        }
      };

      const uampMessage = JSON.stringify({
        type: 'uamp',
        events: [
          { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } },
          { type: 'input.text', event_id: 'e2', text: 'Hello', role: 'user' },
          { type: 'response.create', event_id: 'e3' },
        ],
      });

      mockWs.simulateMessage(uampMessage);

      await new Promise(resolve => setTimeout(resolve, 100));

      // Verify pipeline was called
      expect(mockPipeline).toHaveBeenCalled();

      // Verify response
      const doneEvent = responses.find(e => e.type === 'response.done');
      expect(doneEvent).toBeDefined();
      expect((doneEvent as any).response.output[0].text).toBe("Hi there! I'm running in the browser.");
    });
  });

  describe('agent.run() with portal transport', () => {
    it('allows direct agent.run() calls alongside portal', async () => {
      const mockLLM = new MockLLMSkill('Direct call response');
      portalSkill = new PortalTransportSkill();
      
      agent = new BaseAgent({
        name: 'dual-mode-agent',
        skills: [mockLLM, portalSkill],
      });
      portalSkill.setAgent(agent);

      // Direct call
      const response = await agent.run([
        { role: 'user', content: 'Hello' },
      ]);

      expect(response.content).toBe('Direct call response');

      // Portal call simultaneously
      const mockWs = new MockWebSocket('ws://test');
      const portalResponses: ServerEvent[] = [];

      portalSkill.handleConnection(mockWs as any, {} as any);

      mockWs.send = (data: string) => {
        try {
          portalResponses.push(JSON.parse(data));
        } catch {
          // Ignore
        }
      };

      mockWs.simulateMessage(JSON.stringify({
        type: 'uamp',
        events: [
          { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } },
          { type: 'input.text', event_id: 'e2', text: 'Portal call', role: 'user' },
          { type: 'response.create', event_id: 'e3' },
        ],
      }));

      await new Promise(resolve => setTimeout(resolve, 50));

      const portalDone = portalResponses.find(e => e.type === 'response.done');
      expect(portalDone).toBeDefined();
    });
  });

  describe('streaming through portal', () => {
    it('streams chunks in real-time', async () => {
      const slowLLM = new MockLLMSkill('Word by word streaming test', 10);
      portalSkill = new PortalTransportSkill();

      agent = new BaseAgent({
        name: 'streaming-agent',
        skills: [slowLLM, portalSkill],
      });
      portalSkill.setAgent(agent);

      const mockWs = new MockWebSocket('ws://test');
      const responses: ServerEvent[] = [];
      const timestamps: number[] = [];

      portalSkill.handleConnection(mockWs as any, {} as any);

      mockWs.send = (data: string) => {
        timestamps.push(Date.now());
        try {
          responses.push(JSON.parse(data));
        } catch {
          // Ignore
        }
      };

      mockWs.simulateMessage(JSON.stringify({
        type: 'uamp',
        events: [
          { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } },
          { type: 'input.text', event_id: 'e2', text: 'Stream test', role: 'user' },
          { type: 'response.create', event_id: 'e3' },
        ],
      }));

      // Wait for all chunks
      await new Promise(resolve => setTimeout(resolve, 200));

      const deltas = responses.filter(e => e.type === 'response.delta');
      expect(deltas.length).toBe(5); // "Word by word streaming test" = 5 words

      // Verify chunks arrived progressively
      expect(responses[0].type).toBe('response.created');
      expect((deltas[0] as any).delta.text).toBe('Word');
      expect((deltas[1] as any).delta.text).toBe(' by');
    });
  });
});

// E2E test placeholder - requires real models
describe.skip('Portal + LLM E2E (requires models)', () => {
  it('runs with real WebLLM model', async () => {
    // This test would download and run a real model
    // Only run manually or in E2E environment with RUN_E2E=true

    const webllmSkill = new WebLLMSkill({
      model: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', // ~600MB
    });

    await webllmSkill.initialize();

    const portalSkill = new PortalTransportSkill();
    const agent = new BaseAgent({
      name: 'real-webllm-agent',
      skills: [webllmSkill, portalSkill],
    });
    portalSkill.setAgent(agent);

    const response = await agent.run([
      { role: 'user', content: 'Say hello in exactly 5 words.' },
    ]);

    expect(response.content.length).toBeGreaterThan(0);
    console.log('WebLLM response:', response.content);

    await agent.cleanup();
  });

  it('runs with real Transformers.js model', async () => {
    const transformersSkill = new TransformersSkill({
      model: 'Xenova/distilgpt2', // Tiny model ~250MB
      device: 'wasm',
      max_tokens: 50,
    });

    await transformersSkill.initialize();

    const portalSkill = new PortalTransportSkill();
    const agent = new BaseAgent({
      name: 'real-transformers-agent',
      skills: [transformersSkill, portalSkill],
    });
    portalSkill.setAgent(agent);

    const response = await agent.run([
      { role: 'user', content: 'Hello' },
    ]);

    expect(response.content.length).toBeGreaterThan(0);
    console.log('Transformers.js response:', response.content);

    await agent.cleanup();
  });
});
