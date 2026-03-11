/**
 * Daemon Module
 * 
 * WebAgents daemon for managing agents.
 */

export { AgentRegistry } from './registry.js';
export type { RegisteredAgent } from './registry.js';

export { AgentWatcher } from './watcher.js';
export type { AgentDefinition } from './watcher.js';

export { CronScheduler } from './cron.js';
export type { ScheduledJob } from './cron.js';

export { WebAgentsDaemon } from './server.js';
export type { DaemonConfig } from './server.js';

export { installService, uninstallService, generateLaunchdPlist, generateSystemdUnit } from './service.js';
export type { ServiceConfig } from './service.js';
