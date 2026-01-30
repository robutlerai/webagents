/**
 * UAMP Protocol Compliance Tests
 * 
 * These tests verify that the implementation conforms to the UAMP specification.
 * They can be run against any UAMP-compliant SDK.
 */

import { describe, it, expect } from 'vitest';
import {
  // Types
  generateEventId,
  createBaseEvent,
  parseEvent,
  serializeEvent,
  isClientEvent,
  isServerEvent,
  
  // Event factories
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createToolResultEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
  createProgressEvent,
} from '../../src/uamp/events.js';
import type {
  SessionCreateEvent,
  InputTextEvent,
  ResponseDeltaEvent,
  ResponseDoneEvent,
  ToolResultEvent,
  UAMPEvent,
  ClientEvent,
  ServerEvent,
} from '../../src/uamp/events.js';
import type { Capabilities, Modality } from '../../src/uamp/types.js';

describe('UAMP Protocol Compliance', () => {
  describe('Event IDs', () => {
    it('MUST generate valid UUIDs for event_id', () => {
      const id = generateEventId();
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      expect(id).toMatch(uuidRegex);
    });

    it('MUST generate unique IDs', () => {
      const ids = new Set();
      for (let i = 0; i < 1000; i++) {
        ids.add(generateEventId());
      }
      expect(ids.size).toBe(1000);
    });
  });

  describe('Base Event Structure', () => {
    it('MUST include type field', () => {
      const event = createBaseEvent('test.event');
      expect(event.type).toBe('test.event');
    });

    it('MUST include event_id field', () => {
      const event = createBaseEvent('test.event');
      expect(event.event_id).toBeDefined();
      expect(typeof event.event_id).toBe('string');
    });

    it('SHOULD include timestamp field', () => {
      const event = createBaseEvent('test.event');
      expect(event.timestamp).toBeDefined();
      expect(typeof event.timestamp).toBe('number');
    });

    it('timestamp SHOULD be Unix milliseconds', () => {
      const before = Date.now();
      const event = createBaseEvent('test.event');
      const after = Date.now();
      
      expect(event.timestamp).toBeGreaterThanOrEqual(before);
      expect(event.timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe('Session Events', () => {
    describe('session.create', () => {
      it('MUST include uamp_version field', () => {
        const event = createSessionCreateEvent({ modalities: ['text'] });
        expect(event.uamp_version).toBeDefined();
        expect(event.uamp_version).toBe('1.0');
      });

      it('MUST include session.modalities array', () => {
        const event = createSessionCreateEvent({ modalities: ['text', 'audio'] });
        expect(Array.isArray(event.session.modalities)).toBe(true);
        expect(event.session.modalities).toContain('text');
        expect(event.session.modalities).toContain('audio');
      });

      it('MAY include client_capabilities', () => {
        const caps: Capabilities = {
          id: 'test-client',
          provider: 'test',
          modalities: ['text'] as Modality[],
          supports_streaming: true,
          supports_thinking: false,
          supports_caching: false,
        };
        
        const event = createSessionCreateEvent({ modalities: ['text'] }, caps);
        expect(event.client_capabilities).toBeDefined();
        expect(event.client_capabilities?.id).toBe('test-client');
      });

      it('MAY include session.tools array', () => {
        const event = createSessionCreateEvent({
          modalities: ['text'],
          tools: [
            {
              type: 'function',
              function: {
                name: 'search',
                description: 'Search the web',
              },
            },
          ],
        });
        
        expect(event.session.tools).toHaveLength(1);
        expect(event.session.tools![0].function.name).toBe('search');
      });

      it('MAY include session.instructions', () => {
        const event = createSessionCreateEvent({
          modalities: ['text'],
          instructions: 'Be helpful',
        });
        
        expect(event.session.instructions).toBe('Be helpful');
      });
    });
  });

  describe('Input Events', () => {
    describe('input.text', () => {
      it('MUST include text field', () => {
        const event = createInputTextEvent('Hello');
        expect(event.text).toBe('Hello');
      });

      it('SHOULD default role to user', () => {
        const event = createInputTextEvent('Hello');
        expect(event.role).toBe('user');
      });

      it('MAY specify system role', () => {
        const event = createInputTextEvent('Be concise', 'system');
        expect(event.role).toBe('system');
      });
    });
  });

  describe('Response Events', () => {
    describe('response.create', () => {
      it('MUST have type response.create', () => {
        const event = createResponseCreateEvent();
        expect(event.type).toBe('response.create');
      });

      it('MAY include response configuration', () => {
        const event = createResponseCreateEvent({
          modalities: ['text', 'audio'],
          instructions: 'Override instructions',
        });
        
        expect(event.response?.modalities).toEqual(['text', 'audio']);
      });
    });

    describe('response.delta', () => {
      it('MUST include response_id', () => {
        const event = createResponseDeltaEvent('r123', { type: 'text', text: 'Hi' });
        expect(event.response_id).toBe('r123');
      });

      it('MUST include delta object', () => {
        const event = createResponseDeltaEvent('r123', { type: 'text', text: 'Hi' });
        expect(event.delta).toBeDefined();
        expect(event.delta.type).toBe('text');
      });

      it('text delta MUST include text field', () => {
        const event = createResponseDeltaEvent('r123', { type: 'text', text: 'Hello' });
        expect(event.delta.text).toBe('Hello');
      });

      it('tool_call delta MUST include tool_call object', () => {
        const event = createResponseDeltaEvent('r123', {
          type: 'tool_call',
          tool_call: {
            id: 'tc123',
            name: 'search',
            arguments: '{}',
          },
        });
        
        expect(event.delta.tool_call).toBeDefined();
        expect(event.delta.tool_call!.id).toBe('tc123');
        expect(event.delta.tool_call!.name).toBe('search');
      });
    });

    describe('response.done', () => {
      it('MUST include response_id', () => {
        const event = createResponseDoneEvent('r123', []);
        expect(event.response_id).toBe('r123');
      });

      it('MUST include response object', () => {
        const event = createResponseDoneEvent('r123', [{ type: 'text', text: 'Done' }]);
        expect(event.response).toBeDefined();
      });

      it('response MUST include status', () => {
        const event = createResponseDoneEvent('r123', [], 'completed');
        expect(event.response.status).toBe('completed');
      });

      it('response MUST include output array', () => {
        const event = createResponseDoneEvent('r123', [
          { type: 'text', text: 'Hello' },
          { type: 'text', text: 'World' },
        ]);
        
        expect(Array.isArray(event.response.output)).toBe(true);
        expect(event.response.output).toHaveLength(2);
      });

      it('response MAY include usage stats', () => {
        const event = createResponseDoneEvent('r123', [], 'completed', {
          input_tokens: 10,
          output_tokens: 20,
          total_tokens: 30,
        });
        
        expect(event.response.usage).toBeDefined();
        expect(event.response.usage!.total_tokens).toBe(30);
      });
    });

    describe('response.error', () => {
      it('MUST include error object', () => {
        const event = createResponseErrorEvent('test_error', 'Test message');
        expect(event.error).toBeDefined();
      });

      it('error MUST include code', () => {
        const event = createResponseErrorEvent('test_error', 'Test message');
        expect(event.error.code).toBe('test_error');
      });

      it('error MUST include message', () => {
        const event = createResponseErrorEvent('test_error', 'Test message');
        expect(event.error.message).toBe('Test message');
      });

      it('MAY include response_id', () => {
        const event = createResponseErrorEvent('test_error', 'Test message', 'r123');
        expect(event.response_id).toBe('r123');
      });
    });
  });

  describe('Tool Events', () => {
    describe('tool.result', () => {
      it('MUST include call_id', () => {
        const event = createToolResultEvent('call123', { result: 'data' });
        expect(event.call_id).toBe('call123');
      });

      it('MUST include result as string', () => {
        const event = createToolResultEvent('call123', { result: 'data' });
        expect(typeof event.result).toBe('string');
      });

      it('MUST stringify non-string results', () => {
        const event = createToolResultEvent('call123', { result: 'data' });
        expect(event.result).toBe('{"result":"data"}');
      });

      it('SHOULD preserve string results', () => {
        const event = createToolResultEvent('call123', 'plain string');
        expect(event.result).toBe('plain string');
      });

      it('MAY include is_error flag', () => {
        const event = createToolResultEvent('call123', { error: 'failed' }, true);
        expect(event.is_error).toBe(true);
      });
    });
  });

  describe('Progress Events', () => {
    it('MUST include target field', () => {
      const event = createProgressEvent('tool', 'Processing');
      expect(event.target).toBe('tool');
    });

    it('MAY include message', () => {
      const event = createProgressEvent('tool', 'Processing...');
      expect(event.message).toBe('Processing...');
    });

    it('MAY include percent (0-100)', () => {
      const event = createProgressEvent('tool', 'Processing', 50);
      expect(event.percent).toBe(50);
    });

    it('MAY include target_id', () => {
      const event = createProgressEvent('tool', 'Processing', 50, 'tc123');
      expect(event.target_id).toBe('tc123');
    });
  });

  describe('Event Serialization', () => {
    it('MUST serialize to valid JSON', () => {
      const event = createInputTextEvent('Hello');
      const json = serializeEvent(event);
      
      expect(() => JSON.parse(json)).not.toThrow();
    });

    it('MUST preserve all fields during round-trip', () => {
      const original = createSessionCreateEvent({
        modalities: ['text', 'audio'],
        instructions: 'Be helpful',
        tools: [{ type: 'function', function: { name: 'search' } }],
      });
      
      const json = serializeEvent(original);
      const parsed = parseEvent(json) as SessionCreateEvent;
      
      expect(parsed.type).toBe(original.type);
      expect(parsed.uamp_version).toBe(original.uamp_version);
      expect(parsed.session.modalities).toEqual(original.session.modalities);
      expect(parsed.session.instructions).toBe(original.session.instructions);
    });

    it('MUST throw on invalid JSON', () => {
      expect(() => parseEvent('not json')).toThrow();
    });

    it('MUST throw on missing required fields', () => {
      expect(() => parseEvent('{}')).toThrow();
      expect(() => parseEvent('{"type": "test"}')).toThrow();
    });
  });

  describe('Event Classification', () => {
    const clientEventTypes = [
      'session.create',
      'session.update',
      'input.text',
      'response.create',
      'tool.result',
      'ping',
    ];

    const serverEventTypes = [
      'session.created',
      'response.delta',
      'response.done',
      'response.error',
      'tool.call',
      'progress',
      'thinking',
      'pong',
    ];

    for (const type of clientEventTypes) {
      it(`${type} MUST be classified as client event`, () => {
        const event = { type, event_id: 'test' } as UAMPEvent;
        expect(isClientEvent(event)).toBe(true);
        expect(isServerEvent(event)).toBe(false);
      });
    }

    for (const type of serverEventTypes) {
      it(`${type} MUST be classified as server event`, () => {
        const event = { type, event_id: 'test' } as UAMPEvent;
        expect(isServerEvent(event)).toBe(true);
        expect(isClientEvent(event)).toBe(false);
      });
    }
  });

  describe('Version Compatibility', () => {
    it('UAMP version 1.0 MUST be supported', () => {
      const event = createSessionCreateEvent({ modalities: ['text'] });
      expect(event.uamp_version).toBe('1.0');
    });
  });

  describe('Modality Support', () => {
    const validModalities: Modality[] = ['text', 'audio', 'image', 'video', 'file'];

    for (const modality of validModalities) {
      it(`${modality} MUST be a valid modality`, () => {
        const event = createSessionCreateEvent({ modalities: [modality] });
        expect(event.session.modalities).toContain(modality);
      });
    }

    it('multiple modalities MUST be supported', () => {
      const event = createSessionCreateEvent({
        modalities: ['text', 'audio', 'image'],
      });
      expect(event.session.modalities).toHaveLength(3);
    });
  });

  describe('Content Item Types', () => {
    it('text content MUST have type and text', () => {
      const event = createResponseDoneEvent('r1', [
        { type: 'text', text: 'Hello' },
      ]);
      
      expect(event.response.output[0].type).toBe('text');
      expect(event.response.output[0].text).toBe('Hello');
    });

    it('tool_call content MUST have id, name, arguments', () => {
      const event = createResponseDoneEvent('r1', [
        {
          type: 'tool_call',
          id: 'tc1',
          name: 'search',
          arguments: '{"q":"test"}',
        },
      ]);
      
      const toolCall = event.response.output[0];
      expect(toolCall.type).toBe('tool_call');
      expect(toolCall.id).toBe('tc1');
      expect(toolCall.name).toBe('search');
      expect(toolCall.arguments).toBe('{"q":"test"}');
    });
  });

  describe('Usage Stats', () => {
    it('MUST support input_tokens', () => {
      const event = createResponseDoneEvent('r1', [], 'completed', {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
      });
      
      expect(event.response.usage!.input_tokens).toBe(100);
    });

    it('MUST support output_tokens', () => {
      const event = createResponseDoneEvent('r1', [], 'completed', {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
      });
      
      expect(event.response.usage!.output_tokens).toBe(50);
    });

    it('MUST support total_tokens', () => {
      const event = createResponseDoneEvent('r1', [], 'completed', {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
      });
      
      expect(event.response.usage!.total_tokens).toBe(150);
    });
  });
});
