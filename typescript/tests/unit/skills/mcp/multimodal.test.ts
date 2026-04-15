/**
 * MCPSkill Multimodal Tests
 *
 * Tests that MCPSkill returns { text, content_items } StructuredToolResult
 * when an MCP server returns ImageContent in its tool call response.
 */

import { describe, it, expect, vi } from 'vitest';
import { MCPSkill } from '../../../../src/skills/mcp/skill.js';
import type { Context, StructuredToolResult } from '../../../../src/core/types.js';

function makeContext(): Context {
  const store = new Map<string, unknown>();
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    session: { id: 'test', created_at: 0, last_activity: 0, data: {} },
    auth: { authenticated: false },
    payment: { valid: false },
    metadata: {},
    hasScope: () => false,
    hasScopes: () => false,
  } as unknown as Context;
}

describe('MCPSkill multimodal tool results', () => {
  it('returns StructuredToolResult with content_items for MCP image results', async () => {
    const skill = new MCPSkill({
      mcp: { test_server: { command: 'echo', args: ['test'] } },
    });

    const mockSession = {
      listTools: vi.fn().mockResolvedValue({
        tools: [{ name: 'screenshot', description: 'Take a screenshot', inputSchema: { type: 'object' } }],
      }),
      listResources: vi.fn().mockResolvedValue({ resources: [] }),
      listPrompts: vi.fn().mockResolvedValue({ prompts: [] }),
      callTool: vi.fn().mockResolvedValue({
        content: [
          { type: 'text', text: 'Screenshot taken' },
          { type: 'image', data: 'iVBORw0KGgoAAAA==', mimeType: 'image/png' },
        ],
      }),
      close: vi.fn(),
    };

    (skill as any).sessions.set('test_server', mockSession);
    (skill as any)._initialized = true;
    (skill as any).toolsRegistry.set('test_server__screenshot', {
      server: 'test_server', originalName: 'screenshot',
      description: 'Take a screenshot', inputSchema: { type: 'object' },
    });
    (skill as any)._registerDynamicTool('test_server__screenshot', {
      name: 'screenshot', description: 'Take a screenshot', inputSchema: { type: 'object' },
    }, 'test_server');

    const screenshotTool = skill.tools.find(t => t.name === 'test_server__screenshot');
    expect(screenshotTool).toBeDefined();

    const result = await screenshotTool!.handler({}, makeContext());
    const structured = result as StructuredToolResult;
    expect(structured.text).toMatch(/^Screenshot taken/);
    expect(structured.content_items).toHaveLength(1);
    expect(structured.content_items![0].type).toBe('image');
  });

  it('returns plain text when MCP result has no image content', async () => {
    const skill = new MCPSkill({
      mcp: { text_server: { command: 'echo' } },
    });

    const mockSession = {
      listTools: vi.fn().mockResolvedValue({
        tools: [{ name: 'search', description: 'Search', inputSchema: { type: 'object' } }],
      }),
      listResources: vi.fn().mockResolvedValue({ resources: [] }),
      listPrompts: vi.fn().mockResolvedValue({ prompts: [] }),
      callTool: vi.fn().mockResolvedValue({
        content: [{ type: 'text', text: 'Found 3 results' }],
      }),
      close: vi.fn(),
    };

    (skill as any).sessions.set('text_server', mockSession);
    (skill as any)._initialized = true;
    (skill as any).toolsRegistry.set('text_server__search', {
      server: 'text_server', originalName: 'search',
      description: 'Search', inputSchema: { type: 'object' },
    });
    (skill as any)._registerDynamicTool('text_server__search', {
      name: 'search', description: 'Search', inputSchema: { type: 'object' },
    }, 'text_server');

    const searchTool = skill.tools.find(t => t.name === 'text_server__search');
    const result = await searchTool!.handler({}, makeContext());
    expect(typeof result).toBe('string');
    expect(result).toBe('Found 3 results');
  });
});

describe('MCPSkill enabledTools filtering', () => {
  it('only registers tools that are in the enabledTools allowlist', async () => {
    const skill = new MCPSkill({
      mcp: {
        filtered_server: {
          command: 'echo',
          enabledTools: ['allowed_tool'],
        },
      },
    });

    const mockSession = {
      listTools: vi.fn().mockResolvedValue({
        tools: [
          { name: 'allowed_tool', description: 'Allowed', inputSchema: { type: 'object' } },
          { name: 'blocked_tool', description: 'Blocked', inputSchema: { type: 'object' } },
          { name: 'another_blocked', description: 'Also blocked', inputSchema: { type: 'object' } },
        ],
      }),
      listResources: vi.fn().mockResolvedValue({ resources: [] }),
      listPrompts: vi.fn().mockResolvedValue({ prompts: [] }),
      close: vi.fn(),
    };

    (skill as any).sessions.set('filtered_server', mockSession);
    (skill as any).serverConfigs.set('filtered_server', { command: 'echo', enabledTools: ['allowed_tool'] });
    await (skill as any)._discoverCapabilities('filtered_server', mockSession);

    const toolNames = skill.tools.map(t => t.name);
    expect(toolNames).toContain('filtered_server__allowed_tool');
    expect(toolNames).not.toContain('filtered_server__blocked_tool');
    expect(toolNames).not.toContain('filtered_server__another_blocked');
  });

  it('registers all tools when enabledTools is not specified', async () => {
    const skill = new MCPSkill({
      mcp: {
        open_server: { command: 'echo' },
      },
    });

    const mockSession = {
      listTools: vi.fn().mockResolvedValue({
        tools: [
          { name: 'tool_a', description: 'A', inputSchema: { type: 'object' } },
          { name: 'tool_b', description: 'B', inputSchema: { type: 'object' } },
        ],
      }),
      listResources: vi.fn().mockResolvedValue({ resources: [] }),
      listPrompts: vi.fn().mockResolvedValue({ prompts: [] }),
      close: vi.fn(),
    };

    (skill as any).sessions.set('open_server', mockSession);
    (skill as any).serverConfigs.set('open_server', { command: 'echo' });
    await (skill as any)._discoverCapabilities('open_server', mockSession);

    const toolNames = skill.tools.map(t => t.name);
    expect(toolNames).toContain('open_server__tool_a');
    expect(toolNames).toContain('open_server__tool_b');
  });
});
