/**
 * UAMP Events Unit Tests
 */

import { describe, it, expect } from 'vitest';
import {
  generateEventId,
  createBaseEvent,
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  createToolResultEvent,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
  createProgressEvent,
  parseEvent,
  serializeEvent,
  isClientEvent,
  isServerEvent,
} from '../../../src/uamp/events.js';
import type {
  SessionCreateEvent,
  InputTextEvent,
  ResponseDeltaEvent,
} from '../../../src/uamp/events.js';

describe('UAMP Events', () => {
  describe('generateEventId', () => {
    it('generates unique UUIDs', () => {
      const id1 = generateEventId();
      const id2 = generateEventId();
      
      expect(id1).toBeDefined();
      expect(id2).toBeDefined();
      expect(id1).not.toBe(id2);
    });
    
    it('generates valid UUID format', () => {
      const id = generateEventId();
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      expect(id).toMatch(uuidRegex);
    });
  });
  
  describe('createBaseEvent', () => {
    it('creates event with type, event_id, and timestamp', () => {
      const event = createBaseEvent('test.event');
      
      expect(event.type).toBe('test.event');
      expect(event.event_id).toBeDefined();
      expect(event.timestamp).toBeDefined();
      expect(typeof event.timestamp).toBe('number');
    });
  });
  
  describe('createSessionCreateEvent', () => {
    it('creates session.create event with required fields', () => {
      const event = createSessionCreateEvent({
        modalities: ['text'],
      });
      
      expect(event.type).toBe('session.create');
      expect(event.uamp_version).toBe('1.0');
      expect(event.session.modalities).toEqual(['text']);
      expect(event.event_id).toBeDefined();
    });
    
    it('includes client capabilities when provided', () => {
      const event = createSessionCreateEvent(
        { modalities: ['text', 'image'] },
        {
          id: 'test-client',
          provider: 'webagents',
          modalities: ['text', 'image'],
          supports_streaming: true,
          supports_thinking: false,
          supports_caching: false,
        }
      );
      
      expect(event.client_capabilities).toBeDefined();
      expect(event.client_capabilities?.id).toBe('test-client');
    });
  });
  
  describe('createInputTextEvent', () => {
    it('creates input.text event', () => {
      const event = createInputTextEvent('Hello, world!');
      
      expect(event.type).toBe('input.text');
      expect(event.text).toBe('Hello, world!');
      expect(event.role).toBe('user');
    });
    
    it('accepts custom role', () => {
      const event = createInputTextEvent('System instruction', 'system');
      
      expect(event.role).toBe('system');
    });
  });
  
  describe('createResponseCreateEvent', () => {
    it('creates response.create event', () => {
      const event = createResponseCreateEvent();
      
      expect(event.type).toBe('response.create');
      expect(event.event_id).toBeDefined();
    });
    
    it('includes response config when provided', () => {
      const event = createResponseCreateEvent({
        modalities: ['text', 'audio'],
        instructions: 'Be concise',
      });
      
      expect(event.response?.modalities).toEqual(['text', 'audio']);
      expect(event.response?.instructions).toBe('Be concise');
    });
  });
  
  describe('createToolResultEvent', () => {
    it('creates tool.result event', () => {
      const event = createToolResultEvent('call-123', { result: 'success' });
      
      expect(event.type).toBe('tool.result');
      expect(event.call_id).toBe('call-123');
      expect(event.result).toBe('{"result":"success"}');
      expect(event.is_error).toBe(false);
    });
    
    it('handles string result', () => {
      const event = createToolResultEvent('call-456', 'plain string result');
      
      expect(event.result).toBe('plain string result');
    });
    
    it('marks errors correctly', () => {
      const event = createToolResultEvent('call-789', { error: 'failed' }, true);
      
      expect(event.is_error).toBe(true);
    });
  });
  
  describe('createResponseDeltaEvent', () => {
    it('creates response.delta event with text', () => {
      const event = createResponseDeltaEvent('resp-123', {
        type: 'text',
        text: 'Hello',
      });
      
      expect(event.type).toBe('response.delta');
      expect(event.response_id).toBe('resp-123');
      expect(event.delta.text).toBe('Hello');
    });
    
    it('creates response.delta event with tool_call', () => {
      const event = createResponseDeltaEvent('resp-456', {
        type: 'tool_call',
        tool_call: {
          id: 'tc-1',
          name: 'search',
          arguments: '{"query":"test"}',
        },
      });
      
      expect(event.delta.type).toBe('tool_call');
      expect(event.delta.tool_call?.name).toBe('search');
    });
  });
  
  describe('createResponseDoneEvent', () => {
    it('creates response.done event', () => {
      const event = createResponseDoneEvent(
        'resp-123',
        [{ type: 'text', text: 'Complete response' }]
      );
      
      expect(event.type).toBe('response.done');
      expect(event.response_id).toBe('resp-123');
      expect(event.response.status).toBe('completed');
      expect(event.response.output).toHaveLength(1);
    });
    
    it('includes usage stats', () => {
      const event = createResponseDoneEvent(
        'resp-456',
        [{ type: 'text', text: 'Response' }],
        'completed',
        {
          input_tokens: 10,
          output_tokens: 20,
          total_tokens: 30,
        }
      );
      
      expect(event.response.usage?.input_tokens).toBe(10);
      expect(event.response.usage?.total_tokens).toBe(30);
    });
  });
  
  describe('createResponseErrorEvent', () => {
    it('creates response.error event', () => {
      const event = createResponseErrorEvent('test_error', 'Something went wrong');
      
      expect(event.type).toBe('response.error');
      expect(event.error.code).toBe('test_error');
      expect(event.error.message).toBe('Something went wrong');
    });
    
    it('includes response_id when provided', () => {
      const event = createResponseErrorEvent('api_error', 'API failed', 'resp-789');
      
      expect(event.response_id).toBe('resp-789');
    });
  });
  
  describe('createProgressEvent', () => {
    it('creates progress event', () => {
      const event = createProgressEvent('tool', 'Processing...', 50, 'tc-123');
      
      expect(event.type).toBe('progress');
      expect(event.target).toBe('tool');
      expect(event.message).toBe('Processing...');
      expect(event.percent).toBe(50);
      expect(event.target_id).toBe('tc-123');
    });
  });
  
  describe('parseEvent / serializeEvent', () => {
    it('serializes and parses events correctly', () => {
      const original = createInputTextEvent('Test message');
      const serialized = serializeEvent(original);
      const parsed = parseEvent(serialized);
      
      expect(parsed.type).toBe(original.type);
      expect((parsed as InputTextEvent).text).toBe('Test message');
    });
    
    it('throws on invalid JSON', () => {
      expect(() => parseEvent('invalid json')).toThrow();
    });
    
    it('throws on missing required fields', () => {
      expect(() => parseEvent('{"foo": "bar"}')).toThrow();
    });
  });
  
  describe('isClientEvent / isServerEvent', () => {
    it('correctly identifies client events', () => {
      const sessionCreate = createSessionCreateEvent({ modalities: ['text'] });
      const inputText = createInputTextEvent('Hello');
      const responseCreate = createResponseCreateEvent();
      
      expect(isClientEvent(sessionCreate)).toBe(true);
      expect(isClientEvent(inputText)).toBe(true);
      expect(isClientEvent(responseCreate)).toBe(true);
    });
    
    it('correctly identifies server events', () => {
      const responseDelta = createResponseDeltaEvent('r-1', { type: 'text', text: 'Hi' });
      const responseDone = createResponseDoneEvent('r-1', []);
      const responseError = createResponseErrorEvent('err', 'error');
      
      expect(isServerEvent(responseDelta)).toBe(true);
      expect(isServerEvent(responseDone)).toBe(true);
      expect(isServerEvent(responseError)).toBe(true);
    });
  });
});
