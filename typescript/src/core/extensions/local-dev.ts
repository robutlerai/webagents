/**
 * LocalDevExtension: file-system based agent loading for local development.
 *
 * Watches a directory for AGENT*.md files (agent definitions), parses them,
 * and provides them to the runtime via LocalFileSource. Composes skills
 * from the LocalDevSkillFactory (filesystem, shell, web, MCP).
 *
 * Usage:
 *   const runtime = new DefaultAgentRuntime();
 *   runtime.registerExtension(new LocalDevExtension({ directory: './agents' }));
 *   await runtime.initialize();
 */

import { readFile, readdir } from 'node:fs/promises';
import { join, resolve, basename } from 'node:path';
import { BaseAgent } from '../agent';
import type { AgentConfig, ISkill } from '../types';
import type {
  Extension,
  AgentSource,
  AgentInfo,
  SkillFactory,
  AgentRuntime,
  Middleware,
  RuntimeHooks,
} from '../runtime';

// ============================================================================
// Agent Markdown Parser
// ============================================================================

interface ParsedAgentFile {
  name: string;
  description?: string;
  instructions?: string;
  model?: string;
  skills?: string[];
  config: Record<string, unknown>;
}

/**
 * Parse an AGENT*.md file into agent configuration.
 * Supports YAML frontmatter for metadata + markdown body for instructions.
 */
function parseAgentMarkdown(content: string, filename: string): ParsedAgentFile {
  const result: ParsedAgentFile = {
    name: basename(filename, '.md').replace(/^AGENT[-_.]?/i, '').toLowerCase() || 'agent',
    config: {},
  };

  // Extract YAML frontmatter
  const fmMatch = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
  if (fmMatch) {
    const frontmatter = fmMatch[1];
    for (const line of frontmatter.split('\n')) {
      const colonIdx = line.indexOf(':');
      if (colonIdx === -1) continue;
      const key = line.slice(0, colonIdx).trim();
      const value = line.slice(colonIdx + 1).trim();

      switch (key) {
        case 'name':
          result.name = value;
          break;
        case 'description':
          result.description = value;
          break;
        case 'model':
          result.model = value;
          break;
        case 'skills':
          result.skills = value.split(',').map(s => s.trim()).filter(Boolean);
          break;
        default:
          result.config[key] = value;
          break;
      }
    }

    // Body after frontmatter is the system instructions
    result.instructions = content.slice(fmMatch[0].length).trim();
  } else {
    // No frontmatter: entire content is instructions
    result.instructions = content.trim();
  }

  return result;
}

// ============================================================================
// LocalFileSource
// ============================================================================

export interface LocalFileSourceOptions {
  /** Directory to scan for agent files */
  directory: string;
  /** File pattern (default: AGENT*.md) */
  pattern?: RegExp;
}

export class LocalFileSource implements AgentSource {
  readonly type = 'local-file';
  private directory: string;
  private pattern: RegExp;
  private agentCache: Map<string, BaseAgent> = new Map();
  private infoCache: AgentInfo[] | null = null;
  private runtime: AgentRuntime | null = null;

  constructor(options: LocalFileSourceOptions) {
    this.directory = resolve(options.directory);
    this.pattern = options.pattern ?? /^AGENT.*\.md$/i;
  }

  /** Set the runtime reference (called during extension init) */
  setRuntime(runtime: AgentRuntime): void {
    this.runtime = runtime;
  }

  async getAgent(name: string): Promise<BaseAgent | null> {
    if (this.agentCache.has(name)) {
      return this.agentCache.get(name)!;
    }

    // Scan for the agent file
    const files = await this.scanAgentFiles();
    for (const file of files) {
      const parsed = await this.parseFile(file);
      if (parsed.name === name) {
        const agent = await this.createAgent(parsed);
        this.agentCache.set(name, agent);
        return agent;
      }
    }

    return null;
  }

  async listAgents(): Promise<AgentInfo[]> {
    if (this.infoCache) return this.infoCache;

    const files = await this.scanAgentFiles();
    const infos: AgentInfo[] = [];

    for (const file of files) {
      const parsed = await this.parseFile(file);
      infos.push({
        name: parsed.name,
        displayName: parsed.name,
        description: parsed.description,
        source: this.type,
        loaded: this.agentCache.has(parsed.name),
        model: parsed.model,
      });
    }

    this.infoCache = infos;
    return infos;
  }

  invalidate(name: string): void {
    this.agentCache.delete(name);
    this.infoCache = null;
  }

  invalidateAll(): void {
    this.agentCache.clear();
    this.infoCache = null;
  }

  private async scanAgentFiles(): Promise<string[]> {
    try {
      const entries = await readdir(this.directory);
      return entries
        .filter(entry => this.pattern.test(entry))
        .map(entry => join(this.directory, entry));
    } catch {
      return [];
    }
  }

  private async parseFile(filePath: string): Promise<ParsedAgentFile> {
    const content = await readFile(filePath, 'utf-8');
    return parseAgentMarkdown(content, basename(filePath));
  }

  private async createAgent(parsed: ParsedAgentFile): Promise<BaseAgent> {
    const config: AgentConfig = {
      name: parsed.name,
      description: parsed.description,
      instructions: parsed.instructions,
      model: parsed.model,
      skills: [],
    };

    // Compose skills from registered factories
    if (this.runtime && 'getSkillFactories' in this.runtime) {
      const factories = (this.runtime as { getSkillFactories(): SkillFactory[] }).getSkillFactories();
      for (const factory of factories) {
        const skills = factory.createSkills(config, this.runtime);
        config.skills!.push(...skills);
      }
    }

    const agent = new BaseAgent(config);
    await agent.initialize();
    return agent;
  }
}

// ============================================================================
// LocalDevSkillFactory
// ============================================================================

/**
 * Skill factory for local development. Adds filesystem, shell, and web skills
 * to locally-loaded agents.
 */
export class LocalDevSkillFactory implements SkillFactory {
  readonly name = 'local-dev';
  private defaultSkills: ISkill[];

  constructor(defaultSkills: ISkill[] = []) {
    this.defaultSkills = defaultSkills;
  }

  createSkills(_agentConfig: AgentConfig, _runtime: AgentRuntime): ISkill[] {
    return [...this.defaultSkills];
  }
}

// ============================================================================
// LocalDevExtension
// ============================================================================

export interface LocalDevExtensionOptions {
  /** Directory to scan for agent files */
  directory: string;
  /** File pattern (default: AGENT*.md) */
  pattern?: RegExp;
  /** Default skills to add to every local agent */
  defaultSkills?: ISkill[];
}

export class LocalDevExtension implements Extension {
  readonly name = 'local-dev';
  private fileSource: LocalFileSource;
  private skillFactory: LocalDevSkillFactory;

  constructor(options: LocalDevExtensionOptions) {
    this.fileSource = new LocalFileSource({
      directory: options.directory,
      pattern: options.pattern,
    });
    this.skillFactory = new LocalDevSkillFactory(options.defaultSkills);
  }

  async initialize(runtime: AgentRuntime): Promise<void> {
    this.fileSource.setRuntime(runtime);
  }

  async cleanup(): Promise<void> {
    this.fileSource.invalidateAll();
  }

  getAgentSources(): AgentSource[] {
    return [this.fileSource];
  }

  getSkillFactories(): SkillFactory[] {
    return [this.skillFactory];
  }

  getMiddleware(): Middleware[] {
    return [];
  }

  getHooks(): RuntimeHooks {
    return {};
  }
}
