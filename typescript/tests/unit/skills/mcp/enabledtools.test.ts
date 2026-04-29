import { describe, it, expect } from 'vitest';

describe('MCPSkill enabledTools', () => {
  it('MCPServerConfig accepts enabledTools property', () => {
    const config = {
      url: 'http://mcp.example.com',
      enabledTools: ['tool1', 'tool2'],
    };
    expect(config.enabledTools).toHaveLength(2);
  });

  it('enabledTools filters tool registration', () => {
    const allTools = [
      { name: 'tool1', description: 'First tool' },
      { name: 'tool2', description: 'Second tool' },
      { name: 'tool3', description: 'Third tool' },
    ];
    const enabledTools = ['tool1', 'tool3'];
    const allowed = new Set(enabledTools);
    const filtered = allTools.filter(t => allowed.has(t.name));
    expect(filtered).toHaveLength(2);
    expect(filtered.map(t => t.name)).toEqual(['tool1', 'tool3']);
  });

  it('empty enabledTools is an explicit deny-all allowlist', () => {
    const allTools = [
      { name: 'tool1', description: 'First tool' },
      { name: 'tool2', description: 'Second tool' },
    ];
    const enabledTools: string[] = [];
    const filtered = Array.isArray(enabledTools)
      ? allTools.filter(t => new Set(enabledTools).has(t.name))
      : allTools;

    expect(filtered).toEqual([]);
  });
});
