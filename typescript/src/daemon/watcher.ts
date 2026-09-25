/**
 * File Watcher
 * 
 * Watches for AGENT*.md files to auto-register agents.
 */

import { EventEmitter } from 'events';

import { parseAgentMarkdown } from '../agents/index';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Agent definition from markdown file
 */
export interface AgentDefinition {
  /** Agent name from frontmatter */
  name: string;
  /** Agent description */
  description?: string;
  /** System instructions */
  instructions?: string;
  /** Skills to load */
  skills?: string[];
  /** The skills as written, config included (`- rest: {sign: always}`). */
  skillEntries?: Array<string | Record<string, unknown>>;
  /** The `access:` block as written (ADR-0045), when the file has one. */
  access?: unknown;
  /** Model to use */
  model?: string;
  /** Source file path */
  filePath: string;
  /** Raw markdown content */
  content: string;
}

/**
 * Watcher events
 */
export interface WatcherEvents {
  'agent:added': (definition: AgentDefinition) => void;
  'agent:updated': (definition: AgentDefinition) => void;
  'agent:removed': (filePath: string) => void;
  'error': (error: Error) => void;
}

/**
 * File watcher for AGENT*.md files
 */
export class AgentWatcher extends EventEmitter {
  private watchDir: string;
  private watcher: fs.FSWatcher | null = null;
  private agents: Map<string, AgentDefinition> = new Map();
  
  constructor(watchDir: string) {
    super();
    this.watchDir = watchDir;
  }
  
  /**
   * Start watching directory
   */
  start(): void {
    if (this.watcher) {
      return;
    }
    
    // Initial scan
    this.scanDirectory();
    
    // Watch for changes
    try {
      this.watcher = fs.watch(this.watchDir, (_eventType, filename) => {
        if (filename && this.isAgentFile(filename)) {
          this.handleFileChange(filename);
        }
      });
      
      console.log(`Watching for agents in: ${this.watchDir}`);
    } catch (error) {
      this.emit('error', error as Error);
    }
  }
  
  /**
   * Stop watching
   */
  stop(): void {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }
  }
  
  /**
   * Get all discovered agents
   */
  getAgents(): AgentDefinition[] {
    return Array.from(this.agents.values());
  }
  
  /**
   * Check if filename matches AGENT*.md pattern.
   *
   * `AGENTS.md` IS EXCLUDED (2026-09-23). The pattern is case-insensitive and
   * `.*` matches `S`, so a repository carrying the cross-vendor `AGENTS.md`
   * (the Agentic AI Foundation standard, written for coding agents) had it
   * parsed as an agent definition and registered under the name `S`, since
   * `filename.replace(/^AGENT[_-]?/i, '')` leaves exactly that. That file
   * belongs to another tool. webagents' own inherited context lives in
   * `WEBAGENTS.md`, which this pattern does not match either, and which the
   * daemon reads through the loader rather than as an agent.
   */
  private isAgentFile(filename: string): boolean {
    if (/^AGENTS\.md$/i.test(filename)) return false;
    return /^AGENT.*\.md$/i.test(filename);
  }
  
  /**
   * Scan directory for agent files
   */
  private scanDirectory(): void {
    try {
      const files = fs.readdirSync(this.watchDir);
      
      for (const file of files) {
        if (this.isAgentFile(file)) {
          this.loadAgentFile(file);
        }
      }
    } catch (error) {
      this.emit('error', error as Error);
    }
  }
  
  /**
   * Handle file change event
   */
  private handleFileChange(filename: string): void {
    const filePath = path.join(this.watchDir, filename);
    
    if (fs.existsSync(filePath)) {
      const existingAgent = this.agents.get(filePath);
      this.loadAgentFile(filename);
      
      const agent = this.agents.get(filePath);
      if (agent) {
        if (existingAgent) {
          this.emit('agent:updated', agent);
        } else {
          this.emit('agent:added', agent);
        }
      }
    } else {
      // File was deleted
      if (this.agents.has(filePath)) {
        this.agents.delete(filePath);
        this.emit('agent:removed', filePath);
      }
    }
  }
  
  /**
   * Load and parse agent file
   */
  private loadAgentFile(filename: string): void {
    const filePath = path.join(this.watchDir, filename);
    
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const definition = this.toAgentDefinition(content, filePath);
      
      if (definition) {
        this.agents.set(filePath, definition);
      }
    } catch (error) {
      this.emit('error', new Error(`Failed to load ${filename}: ${(error as Error).message}`));
    }
  }
  
  /**
   * Parse an agent markdown file.
   *
   * Delegates to the package's one loader (2026-09-23). This used to be a
   * hand-rolled line scanner matching `^(\w+):\s*(.*)$`, which meant the
   * documented block form
   *
   *     skills:
   *       - memory
   *       - mcp
   *
   * matched the `skills:` line with an EMPTY value and produced `['']`: one
   * unnamed skill, and the real two silently gone. It also could not see
   * `namespace`, `intents`, `cron` or anything else outside its four cases.
   */
  private toAgentDefinition(content: string, filePath: string): AgentDefinition | null {
    const parsed = parseAgentMarkdown(content, filePath);

    return {
      name: parsed.name,
      description: parsed.description || undefined,
      instructions: parsed.instructions || content,
      skills: parsed.skills,
      skillEntries: parsed.skillEntries,
      ...(parsed.access !== undefined ? { access: parsed.access } : {}),
      model: parsed.model,
      filePath,
      content,
    };
  }
}
