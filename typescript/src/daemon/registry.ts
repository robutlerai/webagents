/**
 * Agent Registry
 * 
 * Manages registered agents from various sources.
 */

import type { IAgent } from '../core/types.js';
import type { Capabilities } from '../uamp/types.js';

/**
 * Registered agent entry
 */
export interface RegisteredAgent {
  /** Agent name */
  name: string;
  /** Agent instance (if local) */
  agent?: IAgent;
  /** Remote agent URL (if remote) */
  url?: string;
  /** Source of registration */
  source: 'file' | 'api' | 'websocket' | 'manual';
  /** Agent capabilities */
  capabilities: Capabilities;
  /** Registration timestamp */
  registeredAt: number;
  /** Last activity timestamp */
  lastActivity: number;
  /** Whether agent is healthy */
  healthy: boolean;
}

/**
 * Agent registry for managing multiple agents
 */
export class AgentRegistry {
  private agents: Map<string, RegisteredAgent> = new Map();
  private healthCheckInterval: NodeJS.Timeout | null = null;
  
  /**
   * Register a local agent
   */
  registerLocal(agent: IAgent): void {
    const entry: RegisteredAgent = {
      name: agent.name,
      agent,
      source: 'manual',
      capabilities: agent.getCapabilities(),
      registeredAt: Date.now(),
      lastActivity: Date.now(),
      healthy: true,
    };
    
    this.agents.set(agent.name, entry);
    console.log(`Registered local agent: ${agent.name}`);
  }
  
  /**
   * Register a remote agent
   */
  registerRemote(name: string, url: string, capabilities: Capabilities, source: 'api' | 'websocket' = 'api'): void {
    const entry: RegisteredAgent = {
      name,
      url,
      source,
      capabilities,
      registeredAt: Date.now(),
      lastActivity: Date.now(),
      healthy: true,
    };
    
    this.agents.set(name, entry);
    console.log(`Registered remote agent: ${name} at ${url}`);
  }
  
  /**
   * Unregister an agent
   */
  unregister(name: string): boolean {
    const removed = this.agents.delete(name);
    if (removed) {
      console.log(`Unregistered agent: ${name}`);
    }
    return removed;
  }
  
  /**
   * Get an agent by name
   */
  get(name: string): RegisteredAgent | undefined {
    return this.agents.get(name);
  }
  
  /**
   * Get all registered agents
   */
  getAll(): RegisteredAgent[] {
    return Array.from(this.agents.values());
  }
  
  /**
   * Find agents by capability
   */
  findByCapability(capability: string): RegisteredAgent[] {
    return this.getAll().filter(entry => 
      entry.capabilities.provides?.includes(capability)
    );
  }
  
  /**
   * Update agent activity timestamp
   */
  updateActivity(name: string): void {
    const entry = this.agents.get(name);
    if (entry) {
      entry.lastActivity = Date.now();
    }
  }
  
  /**
   * Mark agent as unhealthy
   */
  markUnhealthy(name: string): void {
    const entry = this.agents.get(name);
    if (entry) {
      entry.healthy = false;
    }
  }
  
  /**
   * Mark agent as healthy
   */
  markHealthy(name: string): void {
    const entry = this.agents.get(name);
    if (entry) {
      entry.healthy = true;
    }
  }
  
  /**
   * Start health check interval
   */
  startHealthChecks(intervalMs = 30000): void {
    if (this.healthCheckInterval) {
      return;
    }
    
    this.healthCheckInterval = setInterval(async () => {
      for (const entry of this.agents.values()) {
        if (entry.url) {
          try {
            const response = await fetch(`${entry.url}/health`, {
              method: 'GET',
              signal: AbortSignal.timeout(5000),
            });
            if (response.ok) {
              this.markHealthy(entry.name);
            } else {
              this.markUnhealthy(entry.name);
            }
          } catch {
            this.markUnhealthy(entry.name);
          }
        }
      }
    }, intervalMs);
  }
  
  /**
   * Stop health checks
   */
  stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }
  
  /**
   * Get registry stats
   */
  getStats(): { total: number; healthy: number; local: number; remote: number } {
    const all = this.getAll();
    return {
      total: all.length,
      healthy: all.filter(a => a.healthy).length,
      local: all.filter(a => a.agent).length,
      remote: all.filter(a => a.url).length,
    };
  }
}
