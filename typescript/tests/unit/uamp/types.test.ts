/**
 * UAMP Types Unit Tests
 */

import { describe, it, expect } from 'vitest';
import type {
  Modality,
  SessionConfig,
  ContentItem,
  Capabilities,
  UsageStats,
  Message,
  ToolDefinition,
  AudioFormat,
  ImageCapabilities,
  AudioCapabilities,
  ToolCapabilities,
} from '../../../src/uamp/types.js';

describe('UAMP Types', () => {
  describe('Modality', () => {
    it('supports text modality', () => {
      const modality: Modality = 'text';
      expect(modality).toBe('text');
    });

    it('supports audio modality', () => {
      const modality: Modality = 'audio';
      expect(modality).toBe('audio');
    });

    it('supports image modality', () => {
      const modality: Modality = 'image';
      expect(modality).toBe('image');
    });

    it('supports video modality', () => {
      const modality: Modality = 'video';
      expect(modality).toBe('video');
    });

    it('supports file modality', () => {
      const modality: Modality = 'file';
      expect(modality).toBe('file');
    });
  });

  describe('AudioFormat', () => {
    it('supports pcm16 format', () => {
      const format: AudioFormat = 'pcm16';
      expect(format).toBe('pcm16');
    });

    it('supports g711_ulaw format', () => {
      const format: AudioFormat = 'g711_ulaw';
      expect(format).toBe('g711_ulaw');
    });

    it('supports g711_alaw format', () => {
      const format: AudioFormat = 'g711_alaw';
      expect(format).toBe('g711_alaw');
    });
  });

  describe('SessionConfig', () => {
    it('requires modalities', () => {
      const config: SessionConfig = {
        modalities: ['text'],
      };
      expect(config.modalities).toEqual(['text']);
    });

    it('supports multiple modalities', () => {
      const config: SessionConfig = {
        modalities: ['text', 'audio', 'image'],
      };
      expect(config.modalities).toHaveLength(3);
    });

    it('supports instructions', () => {
      const config: SessionConfig = {
        modalities: ['text'],
        instructions: 'Be helpful and concise',
      };
      expect(config.instructions).toBe('Be helpful and concise');
    });

    it('supports voice configuration', () => {
      const config: SessionConfig = {
        modalities: ['text', 'audio'],
        voice: {
          voice_id: 'alloy',
          language: 'en-US',
        },
      };
      expect(config.voice?.voice_id).toBe('alloy');
    });

    it('supports turn detection', () => {
      const config: SessionConfig = {
        modalities: ['audio'],
        turn_detection: {
          type: 'server_vad',
          threshold: 0.5,
          prefix_padding_ms: 300,
          silence_duration_ms: 500,
        },
      };
      expect(config.turn_detection?.type).toBe('server_vad');
    });

    it('supports tools', () => {
      const config: SessionConfig = {
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
      };
      expect(config.tools?.[0].function.name).toBe('search');
    });

    it('supports extensions', () => {
      const config: SessionConfig = {
        modalities: ['text'],
        extensions: {
          custom_key: 'custom_value',
          nested: { foo: 'bar' },
        },
      };
      expect(config.extensions?.custom_key).toBe('custom_value');
    });
  });

  describe('ContentItem', () => {
    it('supports text content', () => {
      const item: ContentItem = {
        type: 'text',
        text: 'Hello, world!',
      };
      expect(item.type).toBe('text');
      expect(item.text).toBe('Hello, world!');
    });

    it('supports image content with URL', () => {
      const item: ContentItem = {
        type: 'image',
        url: 'https://example.com/image.png',
      };
      expect(item.type).toBe('image');
      expect(item.url).toBe('https://example.com/image.png');
    });

    it('supports image content with base64 data', () => {
      const item: ContentItem = {
        type: 'image',
        data: 'base64encodeddata...',
        mime_type: 'image/png',
      };
      expect(item.data).toBeDefined();
      expect(item.mime_type).toBe('image/png');
    });

    it('supports audio content', () => {
      const item: ContentItem = {
        type: 'audio',
        data: 'base64audiodata',
        mime_type: 'audio/mp3',
      };
      expect(item.type).toBe('audio');
    });

    it('supports file content', () => {
      const item: ContentItem = {
        type: 'file',
        data: 'filedata',
        mime_type: 'application/pdf',
        name: 'document.pdf',
      };
      expect(item.type).toBe('file');
      expect(item.name).toBe('document.pdf');
    });

    it('supports tool_call content', () => {
      const item: ContentItem = {
        type: 'tool_call',
        id: 'call_123',
        name: 'search',
        arguments: '{"query": "test"}',
      };
      expect(item.type).toBe('tool_call');
      expect(item.name).toBe('search');
    });

    it('supports tool_result content', () => {
      const item: ContentItem = {
        type: 'tool_result',
        id: 'call_123',
        result: '{"results": []}',
      };
      expect(item.type).toBe('tool_result');
    });

    it('supports thinking content', () => {
      const item: ContentItem = {
        type: 'thinking',
        text: 'Let me analyze this...',
      };
      expect(item.type).toBe('thinking');
    });
  });

  describe('UsageStats', () => {
    it('tracks token usage', () => {
      const usage: UsageStats = {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
      };
      expect(usage.total_tokens).toBe(150);
    });

    it('supports cached tokens', () => {
      const usage: UsageStats = {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        cached_tokens: 20,
      };
      expect(usage.cached_tokens).toBe(20);
    });

    it('supports audio tokens', () => {
      const usage: UsageStats = {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        audio_input_tokens: 10,
        audio_output_tokens: 5,
      };
      expect(usage.audio_input_tokens).toBe(10);
    });

    it('supports reasoning tokens', () => {
      const usage: UsageStats = {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        reasoning_tokens: 30,
      };
      expect(usage.reasoning_tokens).toBe(30);
    });

    it('supports cost tracking', () => {
      const usage: UsageStats = {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        cost: 0.005,
      };
      expect(usage.cost).toBe(0.005);
    });
  });

  describe('Message', () => {
    it('supports user message', () => {
      const msg: Message = {
        role: 'user',
        content: 'Hello!',
      };
      expect(msg.role).toBe('user');
    });

    it('supports assistant message', () => {
      const msg: Message = {
        role: 'assistant',
        content: 'Hi there!',
      };
      expect(msg.role).toBe('assistant');
    });

    it('supports system message', () => {
      const msg: Message = {
        role: 'system',
        content: 'You are a helpful assistant.',
      };
      expect(msg.role).toBe('system');
    });

    it('supports tool message', () => {
      const msg: Message = {
        role: 'tool',
        content: '{"result": "data"}',
        tool_call_id: 'call_123',
      };
      expect(msg.role).toBe('tool');
      expect(msg.tool_call_id).toBe('call_123');
    });

    it('supports content items array', () => {
      const msg: Message = {
        role: 'user',
        content_items: [
          { type: 'text', text: 'Look at this:' },
          { type: 'image', url: 'https://example.com/img.png' },
        ],
      };
      expect(msg.content_items).toHaveLength(2);
    });
  });

  describe('ToolDefinition', () => {
    it('defines function type', () => {
      const tool: ToolDefinition = {
        type: 'function',
        function: {
          name: 'search',
        },
      };
      expect(tool.type).toBe('function');
    });

    it('includes description', () => {
      const tool: ToolDefinition = {
        type: 'function',
        function: {
          name: 'search',
          description: 'Search the web for information',
        },
      };
      expect(tool.function.description).toBeDefined();
    });

    it('includes JSON schema parameters', () => {
      const tool: ToolDefinition = {
        type: 'function',
        function: {
          name: 'search',
          parameters: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The search query',
              },
              limit: {
                type: 'number',
                default: 10,
              },
            },
            required: ['query'],
          },
        },
      };
      expect(tool.function.parameters?.type).toBe('object');
      expect(tool.function.parameters?.required).toContain('query');
    });
  });

  describe('Capabilities', () => {
    it('requires core fields', () => {
      const caps: Capabilities = {
        id: 'test-agent',
        provider: 'webagents',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
      };
      expect(caps.id).toBe('test-agent');
    });

    it('supports image capabilities', () => {
      const imgCaps: ImageCapabilities = {
        supported_formats: ['png', 'jpeg', 'webp'],
        max_resolution: '4096x4096',
        max_file_size_mb: 10,
        supports_url: true,
        supports_base64: true,
      };
      
      const caps: Capabilities = {
        id: 'vision-agent',
        provider: 'webagents',
        modalities: ['text', 'image'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        image: imgCaps,
      };
      expect(caps.image?.supported_formats).toContain('png');
    });

    it('supports audio capabilities', () => {
      const audioCaps: AudioCapabilities = {
        supported_input_formats: ['pcm16', 'mp3'],
        supported_output_formats: ['pcm16'],
        sample_rates: [16000, 24000],
        supports_voice_activity: true,
      };
      
      const caps: Capabilities = {
        id: 'audio-agent',
        provider: 'webagents',
        modalities: ['text', 'audio'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        audio: audioCaps,
      };
      expect(caps.audio?.supports_voice_activity).toBe(true);
    });

    it('supports tool capabilities', () => {
      const toolCaps: ToolCapabilities = {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_streaming_tools: false,
        built_in_tools: ['web_search', 'code_interpreter'],
        max_tools: 128,
      };
      
      const caps: Capabilities = {
        id: 'tool-agent',
        provider: 'webagents',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        tools: toolCaps,
      };
      expect(caps.tools?.built_in_tools).toContain('web_search');
    });

    it('supports provides array', () => {
      const caps: Capabilities = {
        id: 'multi-agent',
        provider: 'webagents',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        provides: ['search', 'weather', 'calculator'],
      };
      expect(caps.provides).toHaveLength(3);
    });

    it('supports endpoints array', () => {
      const caps: Capabilities = {
        id: 'api-agent',
        provider: 'webagents',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        endpoints: ['/v1/chat/completions', '/v1/models', '/ws/realtime'],
      };
      expect(caps.endpoints).toContain('/v1/chat/completions');
    });

    it('supports extensions', () => {
      const caps: Capabilities = {
        id: 'extended-agent',
        provider: 'webagents',
        modalities: ['text'],
        supports_streaming: true,
        supports_thinking: false,
        supports_caching: false,
        extensions: {
          custom_feature: true,
          version: '2.0',
        },
      };
      expect(caps.extensions?.custom_feature).toBe(true);
    });
  });
});
