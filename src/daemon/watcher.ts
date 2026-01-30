/**
 * File Watcher
 * 
 * Watches for AGENT*.md files to auto-register agents.
 */

import { EventEmitter } from 'events';
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
   * Check if filename matches AGENT*.md pattern
   */
  private isAgentFile(filename: string): boolean {
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
      const definition = this.parseAgentMarkdown(content, filePath);
      
      if (definition) {
        this.agents.set(filePath, definition);
      }
    } catch (error) {
      this.emit('error', new Error(`Failed to load ${filename}: ${(error as Error).message}`));
    }
  }
  
  /**
   * Parse agent markdown file
   */
  private parseAgentMarkdown(content: string, filePath: string): AgentDefinition | null {
    // Extract frontmatter
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
    
    if (!frontmatterMatch) {
      // Try to extract name from filename
      const filename = path.basename(filePath, '.md');
      const name = filename.replace(/^AGENT[_-]?/i, '') || 'agent';
      
      return {
        name,
        instructions: content,
        filePath,
        content,
      };
    }
    
    // Parse YAML frontmatter (simple parser)
    const frontmatter = frontmatterMatch[1];
    const instructions = content.slice(frontmatterMatch[0].length).trim();
    
    const definition: AgentDefinition = {
      name: '',
      filePath,
      content,
      instructions,
    };
    
    // Parse frontmatter fields
    const lines = frontmatter.split('\n');
    for (const line of lines) {
      const match = line.match(/^(\w+):\s*(.*)$/);
      if (match) {
        const [, key, value] = match;
        switch (key.toLowerCase()) {
          case 'name':
            definition.name = value.trim();
            break;
          case 'description':
            definition.description = value.trim();
            break;
          case 'model':
            definition.model = value.trim();
            break;
          case 'skills':
            // Handle array format
            if (value.startsWith('[')) {
              definition.skills = JSON.parse(value);
            } else {
              definition.skills = value.split(',').map(s => s.trim());
            }
            break;
        }
      }
    }
    
    // Default name from filename if not specified
    if (!definition.name) {
      const filename = path.basename(filePath, '.md');
      definition.name = filename.replace(/^AGENT[_-]?/i, '') || 'agent';
    }
    
    return definition;
  }
}
