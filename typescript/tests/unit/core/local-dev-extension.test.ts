/**
 * LocalDevExtension Unit Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { writeFile, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  LocalDevExtension,
  LocalFileSource,
  LocalDevSkillFactory,
} from '../../../src/core/extensions/local-dev.js';
import { DefaultAgentRuntime } from '../../../src/core/runtime.js';
import { Skill } from '../../../src/core/skill.js';
import { handoff } from '../../../src/core/decorators.js';
import type { Context } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import { createResponseDoneEvent } from '../../../src/uamp/events.js';

// ============================================================================
// Helpers
// ============================================================================

class MockLLM extends Skill {
  @handoff({ name: 'mock-llm' })
  async *processUAMP(_e: ClientEvent[], _c: Context): AsyncGenerator<ServerEvent> {
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'local response' }]);
  }
}

let testDir: string;

async function setupTestDir(): Promise<string> {
  testDir = join(tmpdir(), `webagents-test-${Date.now()}`);
  await mkdir(testDir, { recursive: true });
  return testDir;
}

async function cleanupTestDir(): Promise<void> {
  if (testDir) {
    await rm(testDir, { recursive: true, force: true });
  }
}

async function writeAgentFile(dir: string, filename: string, content: string): Promise<void> {
  await writeFile(join(dir, filename), content, 'utf-8');
}

// ============================================================================
// Tests
// ============================================================================

describe('LocalFileSource', () => {
  beforeEach(async () => { await setupTestDir(); });
  afterEach(async () => { await cleanupTestDir(); });

  it('lists agents from AGENT*.md files', async () => {
    await writeAgentFile(testDir, 'AGENT.md', '---\nname: default\n---\nBe helpful');
    await writeAgentFile(testDir, 'AGENT-search.md', '---\nname: search\ndescription: Search agent\n---\nSearch the web');

    const source = new LocalFileSource({ directory: testDir });
    const agents = await source.listAgents();

    expect(agents).toHaveLength(2);
    const names = agents.map(a => a.name);
    expect(names).toContain('default');
    expect(names).toContain('search');
  });

  it('ignores non-agent files', async () => {
    await writeAgentFile(testDir, 'AGENT.md', '---\nname: agent\n---\nHi');
    await writeAgentFile(testDir, 'README.md', '# Not an agent');
    await writeAgentFile(testDir, 'notes.txt', 'Random notes');

    const source = new LocalFileSource({ directory: testDir });
    const agents = await source.listAgents();

    expect(agents).toHaveLength(1);
    expect(agents[0].name).toBe('agent');
  });

  it('parses frontmatter and instructions', async () => {
    await writeAgentFile(testDir, 'AGENT.md', [
      '---',
      'name: helper',
      'description: A helpful assistant',
      'model: gpt-4o',
      '---',
      'You are a helpful assistant.',
      'Always be kind.',
    ].join('\n'));

    const source = new LocalFileSource({ directory: testDir });
    const agents = await source.listAgents();

    expect(agents[0].name).toBe('helper');
    expect(agents[0].description).toBe('A helpful assistant');
    expect(agents[0].model).toBe('gpt-4o');
  });

  it('resolves agent by name', async () => {
    await writeAgentFile(testDir, 'AGENT-math.md', '---\nname: math\n---\nDo math');

    const source = new LocalFileSource({ directory: testDir });
    const runtime = new DefaultAgentRuntime();
    source.setRuntime(runtime);

    const agent = await source.getAgent('math');
    expect(agent).not.toBeNull();
    expect(agent!.name).toBe('math');
  });

  it('returns null for unknown agent', async () => {
    await writeAgentFile(testDir, 'AGENT.md', '---\nname: existing\n---\nHi');

    const source = new LocalFileSource({ directory: testDir });
    const agent = await source.getAgent('nonexistent');
    expect(agent).toBeNull();
  });

  it('caches resolved agents', async () => {
    await writeAgentFile(testDir, 'AGENT.md', '---\nname: cached\n---\nHi');

    const source = new LocalFileSource({ directory: testDir });
    const runtime = new DefaultAgentRuntime();
    source.setRuntime(runtime);

    const agent1 = await source.getAgent('cached');
    const agent2 = await source.getAgent('cached');
    expect(agent1).toBe(agent2);
  });

  it('invalidates single agent cache', async () => {
    await writeAgentFile(testDir, 'AGENT.md', '---\nname: inv\n---\nHi');

    const source = new LocalFileSource({ directory: testDir });
    const runtime = new DefaultAgentRuntime();
    source.setRuntime(runtime);

    const agent1 = await source.getAgent('inv');
    source.invalidate('inv');
    const agent2 = await source.getAgent('inv');

    expect(agent1).not.toBe(agent2);
  });

  it('handles empty directory', async () => {
    const source = new LocalFileSource({ directory: testDir });
    const agents = await source.listAgents();
    expect(agents).toHaveLength(0);
  });

  it('handles nonexistent directory', async () => {
    const source = new LocalFileSource({ directory: '/nonexistent/path' });
    const agents = await source.listAgents();
    expect(agents).toHaveLength(0);
  });
});

describe('LocalDevSkillFactory', () => {
  it('returns default skills for any agent config', () => {
    const mockSkill = new MockLLM();
    const factory = new LocalDevSkillFactory([mockSkill]);

    const skills = factory.createSkills({ name: 'test' }, {} as any);
    expect(skills).toHaveLength(1);
    expect(skills[0]).toBe(mockSkill);
  });

  it('returns empty array when no default skills', () => {
    const factory = new LocalDevSkillFactory();
    const skills = factory.createSkills({ name: 'test' }, {} as any);
    expect(skills).toHaveLength(0);
  });
});

describe('LocalDevExtension', () => {
  beforeEach(async () => { await setupTestDir(); });
  afterEach(async () => { await cleanupTestDir(); });

  it('integrates with DefaultAgentRuntime', async () => {
    await writeAgentFile(testDir, 'AGENT-hello.md', '---\nname: hello\n---\nSay hello');

    const ext = new LocalDevExtension({
      directory: testDir,
      defaultSkills: [new MockLLM()],
    });

    const runtime = new DefaultAgentRuntime();
    runtime.registerExtension(ext);
    await runtime.initialize();

    const agents = await runtime.listAgents();
    expect(agents).toHaveLength(1);
    expect(agents[0].name).toBe('hello');

    const agent = await runtime.resolveAgent('hello');
    expect(agent).not.toBeNull();
    expect(agent!.name).toBe('hello');

    // Execute the agent
    const response = await runtime.execute('hello', [{ role: 'user', content: 'hi' }]);
    expect(response.content).toBeDefined();

    await runtime.cleanup();
  });

  it('provides sources and factories', () => {
    const ext = new LocalDevExtension({ directory: '/tmp' });
    expect(ext.getAgentSources()).toHaveLength(1);
    expect(ext.getSkillFactories()).toHaveLength(1);
    expect(ext.name).toBe('local-dev');
  });
});
