/**
 * CompletionsTransportSkill Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CompletionsTransportSkill } from '../../../src/skills/transport/completions/skill.js';
import type { IAgent } from '../../../src/core/types.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
} from '../../../src/uamp/events.js';
import type { ServerEvent } from '../../../src/uamp/events.js';

describe('CompletionsTransportSkill', () => {
  let skill: CompletionsTransportSkill;

  beforeEach(() => {
    skill = new CompletionsTransportSkill();
  });

  describe('constructor', () => {
    it('uses default name', () => {
      expect(skill.name).toBe('completions');
    });

    it('accepts custom name', () => {
      const custom = new CompletionsTransportSkill({ name: 'custom-completions' });
      expect(custom.name).toBe('custom-completions');
    });
  });

  describe('toUAMP', () => {
    it('converts basic request to UAMP events', () => {
      const request = {
        model: 'gpt-4',
        messages: [
          { role: 'user' as const, content: 'Hello' },
        ],
      };

      const events = skill.toUAMP(request);

      expect(events).toHaveLength(3);
      expect(events[0].type).toBe('session.create');
      expect(events[1].type).toBe('input.text');
      expect(events[2].type).toBe('response.create');
    });

    it('handles system messages', () => {
      const request = {
        model: 'gpt-4',
        messages: [
          { role: 'system' as const, content: 'You are helpful' },
          { role: 'user' as const, content: 'Hi' },
        ],
      };

      const events = skill.toUAMP(request);

      // session.create + system input + user input + response.create
      expect(events).toHaveLength(4);
      expect(events[1].type).toBe('input.text');
      expect((events[1] as { role: string }).role).toBe('system');
    });

    it('includes tools in session config', () => {
      const request = {
        model: 'gpt-4',
        messages: [{ role: 'user' as const, content: 'Search for cats' }],
        tools: [
          {
            type: 'function' as const,
            function: {
              name: 'search',
              description: 'Search the web',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      };

      const events = skill.toUAMP(request);
      const sessionEvent = events[0] as { session: { tools: unknown[] } };

      expect(sessionEvent.session.tools).toHaveLength(1);
      expect(sessionEvent.session.tools[0]).toHaveProperty('function');
    });

    it('preserves model-specific options in extensions', () => {
      const request = {
        model: 'gpt-4',
        messages: [{ role: 'user' as const, content: 'Hello' }],
        temperature: 0.7,
        max_tokens: 100,
      };

      const events = skill.toUAMP(request);
      const sessionEvent = events[0] as { session: { extensions: { openai: unknown } } };

      expect(sessionEvent.session.extensions.openai).toEqual({
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 100,
      });
    });

    it('handles empty content', () => {
      const request = {
        model: 'gpt-4',
        messages: [{ role: 'user' as const, content: null }],
      };

      const events = skill.toUAMP(request);
      // null content produces no input events - only session.create + response.create
      expect(events.length).toBe(2);
      expect(events[0].type).toBe('session.create');
      expect(events[1].type).toBe('response.create');
    });
  });

  describe('fromUAMP', () => {
    it('converts UAMP events to OpenAI response', () => {
      const events: ServerEvent[] = [
        createResponseDeltaEvent('r1', { type: 'text', text: 'Hello' }),
        createResponseDeltaEvent('r1', { type: 'text', text: ' World' }),
        createResponseDoneEvent('r1', [{ type: 'text', text: 'Hello World' }], 'completed', {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15,
        }),
      ];

      const response = skill.fromUAMP(events, 'gpt-4');

      expect(response.object).toBe('chat.completion');
      expect(response.model).toBe('gpt-4');
      expect(response.choices).toHaveLength(1);
      expect(response.choices[0].message.content).toBe('Hello World');
      expect(response.choices[0].finish_reason).toBe('stop');
    });

    it('includes usage stats', () => {
      const events: ServerEvent[] = [
        createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }], 'completed', {
          input_tokens: 100,
          output_tokens: 50,
          total_tokens: 150,
        }),
      ];

      const response = skill.fromUAMP(events, 'gpt-4');

      expect(response.usage?.prompt_tokens).toBe(100);
      expect(response.usage?.completion_tokens).toBe(50);
      expect(response.usage?.total_tokens).toBe(150);
    });

    it('handles tool calls', () => {
      const events: ServerEvent[] = [
        createResponseDeltaEvent('r1', {
          type: 'tool_call',
          tool_call: {
            id: 'call_123',
            name: 'search',
            arguments: '{"query"',
          },
        }),
        createResponseDeltaEvent('r1', {
          type: 'tool_call',
          tool_call: {
            id: 'call_123',
            name: 'search',
            arguments: ': "cats"}',
          },
        }),
        createResponseDoneEvent('r1', []),
      ];

      const response = skill.fromUAMP(events, 'gpt-4');

      expect(response.choices[0].message.tool_calls).toHaveLength(1);
      expect(response.choices[0].message.tool_calls![0].function.name).toBe('search');
      expect(response.choices[0].message.tool_calls![0].function.arguments).toBe('{"query": "cats"}');
      expect(response.choices[0].finish_reason).toBe('tool_calls');
    });

    it('generates unique response IDs', () => {
      const events: ServerEvent[] = [
        createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]),
      ];

      const response1 = skill.fromUAMP(events, 'gpt-4');
      const response2 = skill.fromUAMP(events, 'gpt-4');

      expect(response1.id).not.toBe(response2.id);
      expect(response1.id).toMatch(/^chatcmpl-/);
    });
  });

  describe('fromUAMPStreaming', () => {
    it('converts delta events to SSE chunks', () => {
      const event = createResponseDeltaEvent('r1', { type: 'text', text: 'Hello' });
      
      const chunk = skill.fromUAMPStreaming(event, 'gpt-4');

      expect(chunk).toMatch(/^data: /);
      expect(chunk).toMatch(/\n\n$/);
      
      const parsed = JSON.parse(chunk!.replace('data: ', '').trim());
      expect(parsed.object).toBe('chat.completion.chunk');
      expect(parsed.choices[0].delta.content).toBe('Hello');
    });

    it('converts done events to final SSE chunk with [DONE]', () => {
      const event = createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
      
      const chunk = skill.fromUAMPStreaming(event, 'gpt-4');

      expect(chunk).toContain('[DONE]');
      expect(chunk).toContain('finish_reason');
    });

    it('returns null for non-streaming events', () => {
      const event = createResponseErrorEvent('error', 'Something went wrong');
      
      const chunk = skill.fromUAMPStreaming(event, 'gpt-4');

      expect(chunk).toBeNull();
    });

    it('includes tool calls in streaming chunks', () => {
      const event = createResponseDeltaEvent('r1', {
        type: 'tool_call',
        tool_call: {
          id: 'call_123',
          name: 'search',
          arguments: '{}',
        },
      });

      const chunk = skill.fromUAMPStreaming(event, 'gpt-4');
      const parsed = JSON.parse(chunk!.replace('data: ', '').trim());

      expect(parsed.choices[0].delta.tool_calls).toBeDefined();
      expect(parsed.choices[0].delta.tool_calls[0].function.name).toBe('search');
    });
  });

  describe('HTTP endpoints', () => {
    it('registers /v1/chat/completions endpoint', () => {
      expect(skill.httpEndpoints).toHaveLength(2);
      
      const completionsEndpoint = skill.httpEndpoints.find(
        e => e.path === '/v1/chat/completions'
      );
      expect(completionsEndpoint).toBeDefined();
      expect(completionsEndpoint?.method).toBe('POST');
    });

    it('registers /v1/models endpoint', () => {
      const modelsEndpoint = skill.httpEndpoints.find(
        e => e.path === '/v1/models'
      );
      expect(modelsEndpoint).toBeDefined();
      expect(modelsEndpoint?.method).toBe('GET');
    });
  });

  describe('handleCompletions', () => {
    it('returns error when no agent configured', async () => {
      const request = new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [{ role: 'user', content: 'Hello' }],
        }),
      });

      const endpoint = skill.httpEndpoints.find(e => e.path === '/v1/chat/completions');
      const response = await endpoint!.handler(request, {} as never);

      expect(response.status).toBe(500);
      const body = await response.json();
      expect(body.error.message).toBe('No agent configured');
    });

    it('returns non-streaming response when agent is set', async () => {
      const mockAgent: Partial<IAgent> = {
        processUAMP: async function* () {
          yield createResponseDeltaEvent('r1', { type: 'text', text: 'Hello' });
          yield createResponseDoneEvent('r1', [{ type: 'text', text: 'Hello' }]);
        },
      };
      skill.setAgent(mockAgent as IAgent);

      const request = new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [{ role: 'user', content: 'Hi' }],
          stream: false,
        }),
      });

      const endpoint = skill.httpEndpoints.find(e => e.path === '/v1/chat/completions');
      const response = await endpoint!.handler(request, {} as never);

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices[0].message.content).toBe('Hello');
    });

    it('returns streaming response when requested', async () => {
      const mockAgent: Partial<IAgent> = {
        processUAMP: async function* () {
          yield createResponseDeltaEvent('r1', { type: 'text', text: 'Hello' });
          yield createResponseDoneEvent('r1', [{ type: 'text', text: 'Hello' }]);
        },
      };
      skill.setAgent(mockAgent as IAgent);

      const request = new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [{ role: 'user', content: 'Hi' }],
          stream: true,
        }),
      });

      const endpoint = skill.httpEndpoints.find(e => e.path === '/v1/chat/completions');
      const response = await endpoint!.handler(request, {} as never);

      expect(response.headers.get('Content-Type')).toBe('text/event-stream');
      
      // Read the stream
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let result = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value);
      }

      expect(result).toContain('data:');
      expect(result).toContain('[DONE]');
    });

    it('handles invalid JSON gracefully', async () => {
      skill.setAgent({} as IAgent);

      const request = new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        body: 'not json',
      });

      const endpoint = skill.httpEndpoints.find(e => e.path === '/v1/chat/completions');
      const response = await endpoint!.handler(request, {} as never);

      expect(response.status).toBe(400);
      const body = await response.json();
      expect(body.error.type).toBe('invalid_request_error');
    });
  });

  describe('handleModels', () => {
    it('returns list of models', async () => {
      const mockAgent: Partial<IAgent> = {
        getCapabilities: () => ({
          id: 'test-model',
          provider: 'test-provider',
          modalities: ['text'] as const,
          supports_streaming: true,
          supports_thinking: false,
          supports_caching: false,
        }),
      };
      skill.setAgent(mockAgent as IAgent);

      const request = new Request('http://localhost/v1/models');
      const endpoint = skill.httpEndpoints.find(e => e.path === '/v1/models');
      const response = await endpoint!.handler(request, {} as never);

      const body = await response.json();
      expect(body.object).toBe('list');
      expect(body.data).toHaveLength(1);
      expect(body.data[0].id).toBe('test-model');
      expect(body.data[0].owned_by).toBe('test-provider');
    });

    it('returns default values when no agent', async () => {
      const request = new Request('http://localhost/v1/models');
      const endpoint = skill.httpEndpoints.find(e => e.path === '/v1/models');
      const response = await endpoint!.handler(request, {} as never);

      const body = await response.json();
      expect(body.data[0].id).toBe('default');
      expect(body.data[0].owned_by).toBe('webagents');
    });
  });
});
