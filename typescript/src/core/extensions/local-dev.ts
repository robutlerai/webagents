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
import { parseAgentMarkdown } from '../../agents/index';

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
 *
 * Delegates to the package's one loader (2026-09-23). This file used to carry
 * its own line scanner that split `skills` on COMMAS, so the documented block
 * form
 *
 *     skills:
 *       - memory
 *       - mcp
 *
 * produced an empty list, and every other frontmatter key was swept into a
 * `config` bag as a raw string that nothing ever read. `config` is kept in the
 * shape below because it is part of this module's interface, and now carries
 * properly typed YAML values instead of strings.
 */
function parseAgentFile(content: string, filename: string): ParsedAgentFile {
  const parsed = parseAgentMarkdown(content, filename);
  return {
    name: parsed.name,
    description: parsed.description || undefined,
    instructions: parsed.instructions,
    model: parsed.model,
    skills: parsed.skills,
    config: {
      ...parsed.extra,
      ...(parsed.namespace ? { namespace: parsed.namespace } : {}),
      ...(parsed.intents.length ? { intents: parsed.intents } : {}),
    },
  };
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
    return parseAgentFile(content, basename(filePath));
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
