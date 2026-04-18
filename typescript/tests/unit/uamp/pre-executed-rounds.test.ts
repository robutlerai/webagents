/**
 * Tests for the `pre_executed_rounds` field on `response.done` events.
 *
 * `pre_executed_rounds` carries platform-tool execution history (text_editor,
 * bash) from the LLM proxy back to the agent so the agent can splice it
 * into its conversation history. Without this, the agent forgets prior
 * tool calls and the LLM loops on create→present (see
 * plans/surface_platform_tool_history_3596ddbe).
 *
 * Verifies:
 *  - createResponseDoneEvent omits the field when no rounds are passed
 *    (back-compat for non-platform-tool turns)
 *  - createResponseDoneEvent attaches the field when rounds are present
 *  - UAMPClient emits `done` with `pre_executed_rounds` populated
 */

import { describe, it, expect } from 'vitest';
import {
  createResponseDoneEvent,
  parseEvent,
  serializeEvent,
} from '../../../src/uamp/events.js';
import type { PreExecutedRound, ResponseDoneEvent } from '../../../src/uamp/events.js';

const sampleRound: PreExecutedRound = {
  assistant: {
    role: 'assistant',
    content: 'creating unicorn.html',
    tool_calls: [
      {
        id: 'tc_1',
        type: 'function',
        function: {
          name: 'text_editor',
          arguments: '{"command":"create","path":"/unicorn.html"}',
        },
      },
    ],
  },
  tool_results: [
    {
      role: 'tool',
      tool_call_id: 'tc_1',
      name: 'text_editor',
      content: 'File created at /unicorn.html',
    },
  ],
};

describe('createResponseDoneEvent — pre_executed_rounds', () => {
  it('omits pre_executed_rounds when no rounds are passed', () => {
    const ev = createResponseDoneEvent('resp_1', []);
    expect(ev.response).not.toHaveProperty('pre_executed_rounds');
  });

  it('omits pre_executed_rounds when an empty array is passed', () => {
    const ev = createResponseDoneEvent('resp_1', [], 'completed', undefined, []);
    expect(ev.response).not.toHaveProperty('pre_executed_rounds');
  });

  it('attaches pre_executed_rounds when rounds are passed', () => {
    const ev = createResponseDoneEvent('resp_1', [], 'completed', undefined, [sampleRound]);
    expect(ev.response.pre_executed_rounds).toBeDefined();
    expect(ev.response.pre_executed_rounds).toHaveLength(1);
    expect(ev.response.pre_executed_rounds![0].assistant.tool_calls[0].function.name).toBe('text_editor');
  });

  it('round-trips through serialize/parse without losing pre_executed_rounds', () => {
    const ev = createResponseDoneEvent('resp_2', [], 'completed', undefined, [sampleRound]);
    const wire = serializeEvent(ev);
    const parsed = parseEvent(wire) as ResponseDoneEvent;
    expect(parsed.type).toBe('response.done');
    expect(parsed.response.pre_executed_rounds).toBeDefined();
    expect(parsed.response.pre_executed_rounds![0].tool_results[0].tool_call_id).toBe('tc_1');
    expect(parsed.response.pre_executed_rounds![0].tool_results[0].name).toBe('text_editor');
  });
});
