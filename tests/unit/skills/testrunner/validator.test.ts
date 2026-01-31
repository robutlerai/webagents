/**
 * StrictValidator Unit Tests
 * 
 * Tests the deterministic assertion validator.
 */

import { describe, it, expect } from 'vitest';
import { StrictValidator } from '../../../../src/skills/testrunner/validator.js';
import type { HttpResponse } from '../../../../src/skills/testrunner/types.js';

describe('StrictValidator', () => {
  const validator = new StrictValidator();

  describe('status validation', () => {
    it('should pass on exact status match', () => {
      const response: HttpResponse = { status: 200, headers: {}, body: {} };
      const result = validator.validate(response, { status: 200 });
      
      expect(result.passed).toBe(true);
    });

    it('should fail on status mismatch', () => {
      const response: HttpResponse = { status: 404, headers: {}, body: {} };
      const result = validator.validate(response, { status: 200 });
      
      expect(result.passed).toBe(false);
      expect(result.results[0].reason).toContain('404');
    });

    it('should pass when status is in allowed array', () => {
      const response: HttpResponse = { status: 201, headers: {}, body: {} };
      const result = validator.validate(response, { status: [200, 201, 204] });
      
      expect(result.passed).toBe(true);
    });

    it('should fail when status not in allowed array', () => {
      const response: HttpResponse = { status: 500, headers: {}, body: {} };
      const result = validator.validate(response, { status: [200, 201] });
      
      expect(result.passed).toBe(false);
    });
  });

  describe('header validation', () => {
    it('should validate header values', () => {
      const response: HttpResponse = {
        status: 200,
        headers: { 'content-type': 'application/json' },
        body: {},
      };
      const result = validator.validate(response, {
        headers: { 'content-type': 'application/json' },
      });
      
      expect(result.passed).toBe(true);
    });

    it('should match header names exactly or lowercase', () => {
      // Headers are checked: actual[header] || actual[header.toLowerCase()]
      const response: HttpResponse = {
        status: 200,
        headers: { 'content-type': 'application/json' },
        body: {},
      };
      const result = validator.validate(response, {
        headers: { 'content-type': 'application/json' },
      });
      
      expect(result.passed).toBe(true);
    });
  });

  describe('body validation', () => {
    it('should validate simple body values', () => {
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: { object: 'chat.completion', id: 'chatcmpl-123' },
      };
      const result = validator.validate(response, {
        body: { object: 'chat.completion' },
      });
      
      expect(result.passed).toBe(true);
    });

    it('should validate nested values with dot notation', () => {
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: {
          choices: [
            { message: { role: 'assistant', content: 'Hello!' } },
          ],
        },
      };
      const result = validator.validate(response, {
        body: { 'choices[0].message.role': 'assistant' },
      });
      
      expect(result.passed).toBe(true);
    });

    it('should validate array access', () => {
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: {
          choices: [
            { index: 0, finish_reason: 'stop' },
            { index: 1, finish_reason: 'length' },
          ],
        },
      };
      const result = validator.validate(response, {
        body: {
          'choices[0].finish_reason': 'stop',
          'choices[1].finish_reason': 'length',
        },
      });
      
      expect(result.passed).toBe(true);
    });
  });

  describe('type checking', () => {
    const response: HttpResponse = {
      status: 200,
      headers: {},
      body: {
        name: 'test',
        count: 42,
        enabled: true,
        items: [1, 2, 3],
        meta: { key: 'value' },
        nothing: null,
      },
    };

    it('should validate type(string)', () => {
      const result = validator.validate(response, {
        body: { name: 'type(string)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should validate type(number)', () => {
      const result = validator.validate(response, {
        body: { count: 'type(number)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should validate type(boolean)', () => {
      const result = validator.validate(response, {
        body: { enabled: 'type(boolean)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should validate type(array)', () => {
      const result = validator.validate(response, {
        body: { items: 'type(array)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should validate type(object)', () => {
      const result = validator.validate(response, {
        body: { meta: 'type(object)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should validate type(null)', () => {
      const result = validator.validate(response, {
        body: { nothing: 'type(null)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should fail on type mismatch', () => {
      const result = validator.validate(response, {
        body: { name: 'type(number)' },
      });
      expect(result.passed).toBe(false);
    });
  });

  describe('existence checks', () => {
    const response: HttpResponse = {
      status: 200,
      headers: {},
      body: { present: 'value', empty: '', zero: 0, nullValue: null },
    };

    it('should pass exists for present value', () => {
      const result = validator.validate(response, {
        body: { present: 'exists' },
      });
      expect(result.passed).toBe(true);
    });

    it('should pass exists for empty string', () => {
      const result = validator.validate(response, {
        body: { empty: 'exists' },
      });
      expect(result.passed).toBe(true);
    });

    it('should pass exists for zero', () => {
      const result = validator.validate(response, {
        body: { zero: 'exists' },
      });
      expect(result.passed).toBe(true);
    });

    it('should fail exists for null', () => {
      const result = validator.validate(response, {
        body: { nullValue: 'exists' },
      });
      expect(result.passed).toBe(false);
    });

    it('should fail exists for missing field', () => {
      const result = validator.validate(response, {
        body: { missing: 'exists' },
      });
      expect(result.passed).toBe(false);
    });

    it('should pass not_exists for missing field', () => {
      const result = validator.validate(response, {
        body: { missing: 'not_exists' },
      });
      expect(result.passed).toBe(true);
    });
  });

  describe('length checks', () => {
    const response: HttpResponse = {
      status: 200,
      headers: {},
      body: {
        items: [1, 2, 3],
        text: 'hello',
        choices: [{ index: 0 }],
      },
    };

    it('should validate array length', () => {
      const result = validator.validate(response, {
        body: { items: 'length(3)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should validate string length', () => {
      const result = validator.validate(response, {
        body: { text: 'length(5)' },
      });
      expect(result.passed).toBe(true);
    });

    it('should fail on incorrect length', () => {
      const result = validator.validate(response, {
        body: { items: 'length(5)' },
      });
      expect(result.passed).toBe(false);
    });

    it('should validate choices length (compliance spec pattern)', () => {
      const result = validator.validate(response, {
        body: { choices: 'length(1)' },
      });
      expect(result.passed).toBe(true);
    });
  });

  describe('contains checks', () => {
    const response: HttpResponse = {
      status: 200,
      headers: {},
      body: {
        message: 'Hello, Alice! How are you?',
        choices: [{ message: { content: 'The weather in NYC is sunny.' } }],
      },
    };

    it('should pass contains for matching substring', () => {
      const result = validator.validate(response, {
        body: { message: 'contains("Alice")' },
      });
      expect(result.passed).toBe(true);
    });

    it('should fail contains for non-matching substring', () => {
      const result = validator.validate(response, {
        body: { message: 'contains("Bob")' },
      });
      expect(result.passed).toBe(false);
    });

    it('should work with nested path', () => {
      const result = validator.validate(response, {
        body: { 'choices[0].message.content': 'contains("NYC")' },
      });
      expect(result.passed).toBe(true);
    });
  });

  describe('regex checks', () => {
    const response: HttpResponse = {
      status: 200,
      headers: {},
      body: {
        id: 'chatcmpl-abc123def456',
        timestamp: '2024-01-15T10:30:00Z',
      },
    };

    it('should pass regex match', () => {
      const result = validator.validate(response, {
        body: { id: '/^chatcmpl-[a-z0-9]+$/' },
      });
      expect(result.passed).toBe(true);
    });

    it('should fail regex non-match', () => {
      const result = validator.validate(response, {
        body: { id: '/^invalid-/' },
      });
      expect(result.passed).toBe(false);
    });

    it('should validate ISO timestamp format', () => {
      const result = validator.validate(response, {
        body: { timestamp: '/^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}/' },
      });
      expect(result.passed).toBe(true);
    });
  });

  describe('format validation', () => {
    it('should validate SSE format', () => {
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: {},
        format: 'sse',
      };
      const result = validator.validate(response, { format: 'sse' });
      
      expect(result.passed).toBe(true);
    });

    it('should fail on wrong format', () => {
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: {},
        format: 'json',
      };
      const result = validator.validate(response, { format: 'sse' });
      
      expect(result.passed).toBe(false);
    });
  });

  describe('chunk validation', () => {
    it('should validate SSE chunks', () => {
      const chunks = [
        { object: 'chat.completion.chunk', choices: [{ delta: { role: 'assistant' } }] },
        { object: 'chat.completion.chunk', choices: [{ delta: { content: 'Hello' } }] },
        { object: 'chat.completion.chunk', choices: [{ delta: { content: ' World' } }] },
        { object: 'chat.completion.chunk', choices: [{ delta: {}, finish_reason: 'stop' }] },
      ];
      
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: chunks,
        format: 'sse',
      };

      const result = validator.validate(response, {
        chunks: [
          { object: 'chat.completion.chunk' },
          { 'choices[0].delta': 'exists' },
        ],
      });
      
      expect(result.passed).toBe(true);
    });

    it('should validate final chunk', () => {
      const chunks = [
        { choices: [{ delta: { content: 'Hi' } }] },
        { choices: [{ delta: {}, finish_reason: 'stop' }] },
      ];
      
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: chunks,
        format: 'sse',
      };

      const result = validator.validate(response, {
        final_chunk: {
          'choices[0].finish_reason': 'stop',
        },
      });
      
      expect(result.passed).toBe(true);
    });
  });

  describe('compliance spec patterns', () => {
    it('should validate completions-basic response', () => {
      const response: HttpResponse = {
        status: 200,
        headers: { 'content-type': 'application/json' },
        body: {
          id: 'chatcmpl-123',
          object: 'chat.completion',
          created: 1705300000,
          model: 'echo-agent',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: 'Echo: Hello',
              },
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        },
      };

      const result = validator.validate(response, {
        status: 200,
        body: {
          object: 'chat.completion',
          choices: 'length(1)',
          'choices[0].message.role': 'assistant',
          'choices[0].message.content': 'type(string)',
          'choices[0].finish_reason': 'stop',
        },
      });

      expect(result.passed).toBe(true);
    });

    it('should validate error response', () => {
      const response: HttpResponse = {
        status: 400,
        headers: {},
        body: {
          error: {
            message: 'Messages array cannot be empty',
            type: 'invalid_request_error',
            code: 'invalid_messages',
          },
        },
      };

      const result = validator.validate(response, {
        status: 400,
        body: {
          error: 'exists',
          'error.message': 'type(string)',
        },
      });

      expect(result.passed).toBe(true);
    });
  });

  describe('multiple assertions', () => {
    it('should return all results even when some fail', () => {
      const response: HttpResponse = {
        status: 200,
        headers: {},
        body: { value: 'test' },
      };

      const result = validator.validate(response, {
        status: 201, // Will fail
        body: { value: 'test' }, // Will pass
      });

      expect(result.passed).toBe(false);
      expect(result.results).toHaveLength(2);
      expect(result.results.filter(r => r.passed)).toHaveLength(1);
      expect(result.results.filter(r => !r.passed)).toHaveLength(1);
    });
  });
});
