/**
 * Cross-Language Adapter Compatibility Tests (TypeScript side)
 *
 * Verifies that TS adapter message conversion matches the shared fixtures
 * that the Python adapters also consume. Both must produce identical output
 * for the same inputs.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { anthropicAdapter } from '../../src/adapters/anthropic';
import { openaiAdapter } from '../../src/adapters/openai';
import { googleAdapter } from '../../src/adapters/google';
import type { Message, ToolDefinition } from '../../src/adapters/types';

const FIXTURES_DIR = resolve(__dirname, '../../../test-fixtures/adapter-compat');

function loadFixture(name: string): any {
  return JSON.parse(readFileSync(resolve(FIXTURES_DIR, name), 'utf-8'));
}

// Helper: build request and parse the body
function buildBody(adapter: typeof anthropicAdapter | typeof openaiAdapter | typeof googleAdapter,
  messages: Message[], tools?: ToolDefinition[]): Record<string, unknown> {
  const req = adapter.buildRequest({
    messages,
    model: 'test-model',
    apiKey: 'test-key',
    tools: tools && tools.length > 0 ? tools : undefined,
    stream: false,
    maxTokens: 100,
  });
  return JSON.parse(req.body);
}

// -----------------------------------------------------------------------
// Anthropic
// -----------------------------------------------------------------------

describe('Anthropic adapter cross-lang compat', () => {
  const fixture = loadFixture('anthropic.json');

  for (const tc of fixture.tests) {
    it(tc.name, () => {
      const body = buildBody(anthropicAdapter, tc.input.messages as Message[], tc.input.tools);
      // Use ts_expected if available, otherwise expected
      const expected = tc.ts_expected || tc.expected;

      if (expected.messages) {
        expect(body.messages).toEqual(expected.messages);
      }

      if (expected.system !== undefined) {
        expect(body.system).toEqual(expected.system);
      } else if (!tc.ts_expected) {
        expect(body.system).toBeUndefined();
      }

      if (tc.expected_no_field) {
        const msgs = body.messages as any[];
        for (const m of msgs) {
          expect(m).not.toHaveProperty(tc.expected_no_field);
        }
      }
    });
  }

  if (fixture.tool_tests) {
    for (const tc of fixture.tool_tests) {
      it(`tools: ${tc.name}`, () => {
        const body = buildBody(
          anthropicAdapter,
          [{ role: 'user', content: 'test' }],
          tc.input as ToolDefinition[],
        );
        expect(body.tools).toEqual(tc.expected);
      });
    }
  }
});

// -----------------------------------------------------------------------
// OpenAI
// -----------------------------------------------------------------------

describe('OpenAI adapter cross-lang compat', () => {
  const fixture = loadFixture('openai.json');

  for (const tc of fixture.tests) {
    it(tc.name, () => {
      const body = buildBody(openaiAdapter, tc.input.messages as Message[], tc.input.tools);
      const messages = body.messages as any[];
      const expected = tc.ts_expected || tc.expected;

      expect(messages).toEqual(expected);

      if (tc.expected_no_field) {
        for (const m of messages) {
          expect(m).not.toHaveProperty(tc.expected_no_field);
        }
      }
    });
  }

  if (fixture.tool_tests) {
    for (const tc of fixture.tool_tests) {
      it(`tools: ${tc.name}`, () => {
        const body = buildBody(
          openaiAdapter,
          [{ role: 'user', content: 'test' }],
          tc.input as ToolDefinition[],
        );
        expect(body.tools).toEqual(tc.expected);
      });
    }
  }
});

// -----------------------------------------------------------------------
// Google
// -----------------------------------------------------------------------

describe('Google adapter cross-lang compat', () => {
  const fixture = loadFixture('google.json');

  for (const tc of fixture.tests) {
    it(tc.name, () => {
      const body = buildBody(googleAdapter, tc.input.messages as Message[], tc.input.tools);
      const expected = tc.ts_expected || tc.expected;

      if (expected.contents) {
        expect(body.contents).toEqual(expected.contents);
      }

      if (expected.system_parts) {
        expect(body.system_instruction).toEqual({ parts: expected.system_parts });
      } else if (!tc.ts_expected) {
        expect(body.system_instruction).toBeUndefined();
      }
    });
  }

  if (fixture.tool_tests) {
    for (const tc of fixture.tool_tests) {
      it(`tools: ${tc.name}`, () => {
        const body = buildBody(
          googleAdapter,
          [{ role: 'user', content: 'test' }],
          tc.input as ToolDefinition[],
        );
        const tools = body.tools as any[];
        expect(tools).toEqual(tc.expected);
      });
    }
  }
});
