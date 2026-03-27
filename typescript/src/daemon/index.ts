/**
 * Daemon Module
 * 
 * WebAgents daemon for managing agents.
 */

export { AgentRegistry } from './registry';
export type { RegisteredAgent } from './registry';

export { AgentWatcher } from './watcher';
export type { AgentDefinition } from './watcher';

export { CronScheduler } from './cron';
export type { ScheduledJob } from './cron';

export { WebAgentsDaemon } from './server';
export type { DaemonConfig } from './server';

export { installService, uninstallService, generateLaunchdPlist, generateSystemdUnit } from './service';
export type { ServiceConfig } from './service';
