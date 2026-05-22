/**
 * Tests for native/built-in tool handling across all adapters.
 */
import { describe, it, expect } from 'vitest';
import { getAdapter } from '../../../src/adapters/index.js';
import type { ToolDefinition } from '../../../src/adapters/types.js';

const functionTool: ToolDefinition = {
  type: 'function',
  function: { name: 'get_weather', description: 'Get weather', parameters: { type: 'object', properties: {} } },
};

const baseParams = {
  messages: [{ role: 'user', content: 'Hello' }],
  apiKey: 'test-key',
  stream: true,
} as const;

describe('Google adapter native tools', () => {
  const adapter = getAdapter('google');

  it('emits function_declarations for function tools', () => {
    const req = adapter.buildRequest({ ...baseParams, model: 'google/gemini-2.0-flash', tools: [functionTool] });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(1);
    expect(body.tools[0]).toHaveProperty('function_declarations');
    expect(body.tools[0].function_declarations[0].name).toBe('get_weather');
  });

  it('emits native tool objects alongside function_declarations', () => {
    const nativeTool: ToolDefinition = { type: 'google_search' };
    const req = adapter.buildRequest({
      ...baseParams, model: 'google/gemini-2.0-flash',
      tools: [functionTool, nativeTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(2);
    expect(body.tools[0]).toHaveProperty('function_declarations');
    expect(body.tools[1]).toHaveProperty('google_search');
  });

  it('emits native-only tools without function_declarations', () => {
    const req = adapter.buildRequest({
      ...baseParams, model: 'google/gemini-2.0-flash',
      tools: [
        { type: 'code_execution' } as ToolDefinition,
        { type: 'url_context' } as ToolDefinition,
      ],
    });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(2);
    expect(body.tools[0]).toHaveProperty('code_execution');
    expect(body.tools[1]).toHaveProperty('url_context');
  });
});

describe('Anthropic adapter native tools', () => {
  const adapter = getAdapter('anthropic');

  it('converts function tools to Anthropic format', () => {
    const req = adapter.buildRequest({ ...baseParams, model: 'anthropic/claude-3-haiku-20240307', tools: [functionTool] });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(1);
    expect(body.tools[0].name).toBe('get_weather');
    expect(body.tools[0]).toHaveProperty('input_schema');
  });

  it('does NOT crash on non-function tools', () => {
    const nativeTool: ToolDefinition = { type: 'web_search' };
    expect(() => {
      adapter.buildRequest({
        ...baseParams, model: 'anthropic/claude-3-haiku-20240307',
        tools: [nativeTool],
      });
    }).not.toThrow();
  });

  it('passes native tool objects through with their type', () => {
    const nativeTool: ToolDefinition = { type: 'web_search' };
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-3-haiku-20240307',
      tools: [functionTool, nativeTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(2);
    expect(body.tools[0].name).toBe('get_weather');
    expect(body.tools[1].type).toBe('web_search');
  });

  it('handles Anthropic computer_use tool with dimensions', () => {
    const computerTool: ToolDefinition = {
      type: 'computer_20250124',
      display_width_px: 1024,
      display_height_px: 768,
    };
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-3-haiku-20240307',
      tools: [computerTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools[0].type).toBe('computer_20250124');
    expect(body.tools[0].display_width_px).toBe(1024);
  });

  it('passes through bash_20250124 native tool', () => {
    const bashTool: ToolDefinition = { type: 'bash_20250124', name: 'bash' };
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-sonnet-4-20250514',
      tools: [bashTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools[0].type).toBe('bash_20250124');
  });

  it('passes through memory_20250818 native tool', () => {
    const memoryTool: ToolDefinition = { type: 'memory_20250818' };
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-sonnet-4-20250514',
      tools: [memoryTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools[0].type).toBe('memory_20250818');
  });

  it('passes through computer_20250124 native tool with name and emits beta header', () => {
    // computer_use is the last remaining beta-gated native tool as of May
    // 2026 (memory / web_fetch / code_execution all graduated to GA and
    // Anthropic now 400s on their old beta IDs). Use it as the canonical
    // fixture for asserting the adapter's beta-collection mechanism.
    const computerTool: ToolDefinition = {
      type: 'computer_20250124',
      name: 'computer',
      beta: 'computer-use-2025-01-24',
    } as ToolDefinition;
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-sonnet-4-20250514',
      tools: [computerTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools[0].type).toBe('computer_20250124');
    expect(body.tools[0].name).toBe('computer');
    // `beta` is a registry-only marker; Anthropic 400s on unknown fields inside
    // the tool body, so the adapter must strip it before serialising.
    expect(body.tools[0]).not.toHaveProperty('beta');
    expect(req.headers['anthropic-beta']).toBe('computer-use-2025-01-24');
  });

  it('collects and dedupes anthropic-beta from multiple native tools', () => {
    // Mechanism-only test: ensures the adapter collects duplicate `beta`
    // markers across multiple tools and emits each value exactly once.
    // The exact strings here are intentionally synthetic — only
    // `computer-use-2025-01-24` is a real Anthropic beta in May 2026; the
    // other is a placeholder for whatever future beta-gated server tool
    // we add next. Keep them distinct so the dedupe path is exercised.
    const tools: ToolDefinition[] = [
      { type: 'computer_20250124',   name: 'computer',   beta: 'computer-use-2025-01-24' } as ToolDefinition,
      { type: 'future_tool_20991231', name: 'future_tool', beta: 'future-tool-2099-12-31' } as ToolDefinition,
      { type: 'computer_20250124',   name: 'computer',   beta: 'computer-use-2025-01-24' } as ToolDefinition, // duplicate
      { type: 'web_search_20250305', name: 'web_search' } as ToolDefinition, // GA, no beta
    ];
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-sonnet-4-20250514', tools,
    });
    const header = req.headers['anthropic-beta'] ?? '';
    const parts = header.split(',').filter(Boolean).sort();
    expect(parts).toEqual(['computer-use-2025-01-24', 'future-tool-2099-12-31']);
    const body = JSON.parse(req.body);
    for (const t of body.tools) {
      expect(t).not.toHaveProperty('beta');
    }
  });

  it('omits anthropic-beta header when only GA tools are sent', () => {
    // Regression guard for the May 2026 Anthropic GA cutover: shipping any
    // of these tools used to require a beta header that Anthropic now 400s
    // on with `Unexpected value(s) … for the anthropic-beta header`. None
    // of them carry a `beta` marker in `lib/models/tool-support.ts` so the
    // adapter must NOT synthesise the header.
    const tools: ToolDefinition[] = [
      { type: 'web_search_20250305',  name: 'web_search' } as ToolDefinition,
      { type: 'bash_20250124',        name: 'bash' } as ToolDefinition,
      { type: 'memory_20250818',      name: 'memory' } as ToolDefinition,
      { type: 'web_fetch_20260209',   name: 'web_fetch' } as ToolDefinition,
    ];
    const req = adapter.buildRequest({
      ...baseParams, model: 'anthropic/claude-sonnet-4-20250514', tools,
    });
    expect(req.headers['anthropic-beta']).toBeUndefined();
  });

  // Canonical native marker resolution. Callers (lib/llm/platform-tools.ts) emit
  // a model-agnostic { type: 'native', name: 'text_editor' | 'bash' } marker,
  // and the adapter is the only place that knows the per-model variant table.
  describe('canonical { type: "native" } marker resolution', () => {
    it('resolves text_editor to the modern variant for claude-sonnet-4', () => {
      const req = adapter.buildRequest({
        ...baseParams, model: 'anthropic/claude-sonnet-4-20250514',
        tools: [{ type: 'native', name: 'text_editor' } as unknown as ToolDefinition],
      });
      const body = JSON.parse(req.body);
      expect(body.tools[0]).toEqual({ type: 'text_editor_20250728', name: 'str_replace_based_edit_tool' });
    });

    it('resolves text_editor to the modern variant for claude-opus-4-7', () => {
      const req = adapter.buildRequest({
        ...baseParams, model: 'anthropic/claude-opus-4-7',
        tools: [{ type: 'native', name: 'text_editor' } as unknown as ToolDefinition],
      });
      const body = JSON.parse(req.body);
      expect(body.tools[0]).toEqual({ type: 'text_editor_20250728', name: 'str_replace_based_edit_tool' });
    });

    it('resolves text_editor to the legacy variant for claude-3-5-sonnet', () => {
      const req = adapter.buildRequest({
        ...baseParams, model: 'anthropic/claude-3-5-sonnet-20241022',
        tools: [{ type: 'native', name: 'text_editor' } as unknown as ToolDefinition],
      });
      const body = JSON.parse(req.body);
      expect(body.tools[0]).toEqual({ type: 'text_editor_20250124', name: 'str_replace_editor' });
    });

    it('resolves text_editor to the legacy variant for claude-3-7-sonnet', () => {
      const req = adapter.buildRequest({
        ...baseParams, model: 'anthropic/claude-3-7-sonnet-20250219',
        tools: [{ type: 'native', name: 'text_editor' } as unknown as ToolDefinition],
      });
      const body = JSON.parse(req.body);
      expect(body.tools[0]).toEqual({ type: 'text_editor_20250124', name: 'str_replace_editor' });
    });

    it('resolves bash to bash_20250124', () => {
      const req = adapter.buildRequest({
        ...baseParams, model: 'anthropic/claude-sonnet-4-20250514',
        tools: [{ type: 'native', name: 'bash' } as unknown as ToolDefinition],
      });
      const body = JSON.parse(req.body);
      expect(body.tools[0]).toEqual({ type: 'bash_20250124', name: 'bash' });
    });
  });

  // The proxy stores assistant tool calls with the canonical UAMP name
  // ("text_editor"). When that history is replayed to Anthropic on the next
  // round, convertMessages must rewrite the name back to the per-model variant
  // — otherwise Anthropic rejects the request because the (registered tool,
  // referenced name) pair doesn't match.
  describe('canonical → Anthropic name rewrite in assistant tool_calls', () => {
    const assistantWithTextEditorCall = {
      role: 'assistant' as const,
      content: '',
      tool_calls: [{
        id: 'call_1',
        type: 'function' as const,
        function: { name: 'text_editor', arguments: '{"command":"view","path":"/x.md"}' },
      }],
    };

    it('rewrites canonical text_editor to str_replace_based_edit_tool for claude-4-x', () => {
      const req = adapter.buildRequest({
        ...baseParams,
        model: 'anthropic/claude-sonnet-4-20250514',
        messages: [{ role: 'user', content: 'go' }, assistantWithTextEditorCall, {
          role: 'tool', tool_call_id: 'call_1', content: 'ok',
        }],
      });
      const body = JSON.parse(req.body);
      const assistantMsg = body.messages.find((m: any) => m.role === 'assistant');
      const toolUse = assistantMsg.content.find((b: any) => b.type === 'tool_use');
      expect(toolUse.name).toBe('str_replace_based_edit_tool');
    });

    it('rewrites canonical text_editor to str_replace_editor for claude-3-x', () => {
      const req = adapter.buildRequest({
        ...baseParams,
        model: 'anthropic/claude-3-5-sonnet-20241022',
        messages: [{ role: 'user', content: 'go' }, assistantWithTextEditorCall, {
          role: 'tool', tool_call_id: 'call_1', content: 'ok',
        }],
      });
      const body = JSON.parse(req.body);
      const assistantMsg = body.messages.find((m: any) => m.role === 'assistant');
      const toolUse = assistantMsg.content.find((b: any) => b.type === 'tool_use');
      expect(toolUse.name).toBe('str_replace_editor');
    });
  });
});

describe('OpenAI Responses adapter native tools', () => {
  const adapter = getAdapter('openai');

  it('flattens function tools (Responses API shape) and passes natives through', () => {
    const nativeTool: ToolDefinition = { type: 'web_search' };
    const req = adapter.buildRequest({
      ...baseParams, model: 'openai/gpt-5.5',
      tools: [functionTool, nativeTool],
    });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(2);
    // Function tools are flat (no nested `function` wrapper) in Responses.
    expect(body.tools[0]).toMatchObject({
      type: 'function',
      name: 'get_weather',
      description: 'Get weather',
    });
    expect(body.tools[0]).not.toHaveProperty('function');
    expect(body.tools[1]).toEqual(nativeTool);
  });

  it('handles native-only tools', () => {
    const req = adapter.buildRequest({
      ...baseParams, model: 'openai/gpt-5.5',
      tools: [{ type: 'code_interpreter', container: { type: 'auto' } } as ToolDefinition],
    });
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(1);
    expect(body.tools[0].type).toBe('code_interpreter');
  });

  it('passes through file_search native tool', () => {
    const req = adapter.buildRequest({
      ...baseParams, model: 'openai/gpt-5.5',
      tools: [{ type: 'file_search' } as ToolDefinition],
    });
    const body = JSON.parse(req.body);
    expect(body.tools[0].type).toBe('file_search');
  });

  it('passes through image_generation native tool', () => {
    const req = adapter.buildRequest({
      ...baseParams, model: 'openai/gpt-5.5',
      tools: [{ type: 'image_generation' } as ToolDefinition],
    });
    const body = JSON.parse(req.body);
    expect(body.tools[0].type).toBe('image_generation');
  });
});
