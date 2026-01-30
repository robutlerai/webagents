/**
 * Cron Scheduler
 * 
 * Schedule agent tasks using cron expressions.
 */

import { EventEmitter } from 'events';

/**
 * Scheduled job
 */
export interface ScheduledJob {
  /** Job ID */
  id: string;
  /** Cron expression */
  cron: string;
  /** Agent name */
  agentName: string;
  /** Task to execute */
  task: string;
  /** Task parameters */
  params?: Record<string, unknown>;
  /** Whether job is enabled */
  enabled: boolean;
  /** Last run time */
  lastRun?: number;
  /** Next run time */
  nextRun?: number;
}

/**
 * Cron scheduler for agent tasks
 */
export class CronScheduler extends EventEmitter {
  private jobs: Map<string, ScheduledJob> = new Map();
  private timers: Map<string, NodeJS.Timeout> = new Map();
  private running = false;
  
  /**
   * Add a scheduled job
   */
  addJob(job: Omit<ScheduledJob, 'nextRun'>): void {
    const fullJob: ScheduledJob = {
      ...job,
      nextRun: this.calculateNextRun(job.cron) ?? undefined,
    };
    
    this.jobs.set(job.id, fullJob);
    
    if (this.running && job.enabled) {
      this.scheduleJob(fullJob);
    }
    
    console.log(`Added cron job: ${job.id} (${job.cron})`);
  }
  
  /**
   * Remove a scheduled job
   */
  removeJob(id: string): boolean {
    const timer = this.timers.get(id);
    if (timer) {
      clearTimeout(timer);
      this.timers.delete(id);
    }
    
    return this.jobs.delete(id);
  }
  
  /**
   * Enable a job
   */
  enableJob(id: string): void {
    const job = this.jobs.get(id);
    if (job) {
      job.enabled = true;
      if (this.running) {
        this.scheduleJob(job);
      }
    }
  }
  
  /**
   * Disable a job
   */
  disableJob(id: string): void {
    const job = this.jobs.get(id);
    if (job) {
      job.enabled = false;
      
      const timer = this.timers.get(id);
      if (timer) {
        clearTimeout(timer);
        this.timers.delete(id);
      }
    }
  }
  
  /**
   * Start the scheduler
   */
  start(): void {
    if (this.running) return;
    
    this.running = true;
    
    for (const job of this.jobs.values()) {
      if (job.enabled) {
        this.scheduleJob(job);
      }
    }
    
    console.log('Cron scheduler started');
  }
  
  /**
   * Stop the scheduler
   */
  stop(): void {
    this.running = false;
    
    for (const timer of this.timers.values()) {
      clearTimeout(timer);
    }
    this.timers.clear();
    
    console.log('Cron scheduler stopped');
  }
  
  /**
   * Get all jobs
   */
  getJobs(): ScheduledJob[] {
    return Array.from(this.jobs.values());
  }
  
  /**
   * Get a specific job
   */
  getJob(id: string): ScheduledJob | undefined {
    return this.jobs.get(id);
  }
  
  /**
   * Schedule a job to run at next time
   */
  private scheduleJob(job: ScheduledJob): void {
    if (!job.enabled || !this.running) return;
    
    const nextRun = this.calculateNextRun(job.cron);
    if (!nextRun) return;
    
    job.nextRun = nextRun;
    
    const delay = nextRun - Date.now();
    if (delay <= 0) return;
    
    const timer = setTimeout(() => {
      this.executeJob(job);
    }, delay);
    
    this.timers.set(job.id, timer);
  }
  
  /**
   * Execute a job
   */
  private executeJob(job: ScheduledJob): void {
    job.lastRun = Date.now();
    
    this.emit('job:execute', {
      id: job.id,
      agentName: job.agentName,
      task: job.task,
      params: job.params,
    });
    
    // Schedule next run
    this.scheduleJob(job);
  }
  
  /**
   * Calculate next run time from cron expression
   * 
   * Simplified cron parser supporting:
   * - * * * * * (minute hour day month weekday)
   * - @hourly, @daily, @weekly, @monthly
   */
  private calculateNextRun(cron: string): number | null {
    const now = new Date();
    const nextRun = new Date(now);
    
    // Handle special expressions
    if (cron.startsWith('@')) {
      switch (cron) {
        case '@hourly':
          nextRun.setHours(nextRun.getHours() + 1, 0, 0, 0);
          return nextRun.getTime();
        case '@daily':
          nextRun.setDate(nextRun.getDate() + 1);
          nextRun.setHours(0, 0, 0, 0);
          return nextRun.getTime();
        case '@weekly':
          nextRun.setDate(nextRun.getDate() + (7 - nextRun.getDay()));
          nextRun.setHours(0, 0, 0, 0);
          return nextRun.getTime();
        case '@monthly':
          nextRun.setMonth(nextRun.getMonth() + 1, 1);
          nextRun.setHours(0, 0, 0, 0);
          return nextRun.getTime();
        default:
          return null;
      }
    }
    
    // Parse standard cron (simplified)
    const parts = cron.split(' ');
    if (parts.length !== 5) {
      return null;
    }
    
    const [minute, hour, _day, _month, _weekday] = parts;
    
    // For simplicity, just handle the most common case: * * * * *
    // In a real implementation, use a cron library
    if (minute === '*' && hour === '*') {
      // Every minute
      nextRun.setMinutes(nextRun.getMinutes() + 1, 0, 0);
      return nextRun.getTime();
    }
    
    // Basic handling for specific minute/hour
    const targetMinute = minute === '*' ? now.getMinutes() : parseInt(minute, 10);
    const targetHour = hour === '*' ? now.getHours() : parseInt(hour, 10);
    
    nextRun.setHours(targetHour, targetMinute, 0, 0);
    
    if (nextRun.getTime() <= now.getTime()) {
      // Move to next occurrence
      if (hour === '*') {
        nextRun.setHours(nextRun.getHours() + 1);
      } else {
        nextRun.setDate(nextRun.getDate() + 1);
      }
    }
    
    return nextRun.getTime();
  }
}
