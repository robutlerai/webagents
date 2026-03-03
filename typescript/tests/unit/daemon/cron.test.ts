/**
 * CronScheduler Unit Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CronScheduler } from '../../../src/daemon/cron.js';

describe('CronScheduler', () => {
  let scheduler: CronScheduler;

  beforeEach(() => {
    scheduler = new CronScheduler();
    vi.useFakeTimers();
    // Suppress console.log for tests
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    scheduler.stop();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe('addJob', () => {
    it('adds a cron job', () => {
      scheduler.addJob({
        id: 'test-job',
        cron: '* * * * *',
        agentName: 'test-agent',
        task: 'test-task',
        enabled: true,
      });
      
      const jobs = scheduler.getJobs();
      expect(jobs).toHaveLength(1);
      expect(jobs[0].id).toBe('test-job');
    });

    it('stores cron expression', () => {
      scheduler.addJob({
        id: 'cron-test',
        cron: '0 * * * *',
        agentName: 'test-agent',
        task: 'test-task',
        enabled: true,
      });
      
      const jobs = scheduler.getJobs();
      expect(jobs[0].cron).toBe('0 * * * *');
    });

    it('stores agent name and task', () => {
      scheduler.addJob({
        id: 'agent-test',
        cron: '* * * * *',
        agentName: 'my-agent',
        task: 'run_analysis',
        params: { depth: 3 },
        enabled: true,
      });
      
      const jobs = scheduler.getJobs();
      expect(jobs[0].agentName).toBe('my-agent');
      expect(jobs[0].task).toBe('run_analysis');
      expect(jobs[0].params?.depth).toBe(3);
    });
  });

  describe('removeJob', () => {
    it('removes an existing job', () => {
      scheduler.addJob({
        id: 'to-remove',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      expect(scheduler.getJobs()).toHaveLength(1);
      
      const removed = scheduler.removeJob('to-remove');
      
      expect(removed).toBe(true);
      expect(scheduler.getJobs()).toHaveLength(0);
    });

    it('returns false for unknown job', () => {
      const removed = scheduler.removeJob('nonexistent');
      expect(removed).toBe(false);
    });
  });

  describe('start / stop', () => {
    it('starts the scheduler without error', () => {
      scheduler.addJob({
        id: 'start-test',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      expect(() => scheduler.start()).not.toThrow();
    });

    it('stops the scheduler', () => {
      scheduler.addJob({
        id: 'stop-test',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      scheduler.start();
      expect(() => scheduler.stop()).not.toThrow();
    });

    it('can restart after stop', () => {
      scheduler.addJob({
        id: 'restart-test',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      scheduler.start();
      scheduler.stop();
      expect(() => scheduler.start()).not.toThrow();
    });
  });

  describe('enable/disable jobs', () => {
    it('enables a disabled job', () => {
      scheduler.addJob({
        id: 'enable-test',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: false,
      });
      
      expect(scheduler.getJob('enable-test')?.enabled).toBe(false);
      
      scheduler.enableJob('enable-test');
      
      expect(scheduler.getJob('enable-test')?.enabled).toBe(true);
    });

    it('disables an enabled job', () => {
      scheduler.addJob({
        id: 'disable-test',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      scheduler.disableJob('disable-test');
      
      expect(scheduler.getJob('disable-test')?.enabled).toBe(false);
    });
  });

  describe('cron expressions', () => {
    it('calculates next run for every minute (* * * * *)', () => {
      vi.setSystemTime(new Date('2024-01-01T12:00:00'));
      
      scheduler.addJob({
        id: 'every-minute',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      const job = scheduler.getJob('every-minute');
      expect(job?.nextRun).toBeDefined();
      
      // Should be within the next minute
      const diff = (job?.nextRun || 0) - Date.now();
      expect(diff).toBeGreaterThan(0);
      expect(diff).toBeLessThanOrEqual(60000);
    });

    it('supports @hourly expression', () => {
      vi.setSystemTime(new Date('2024-01-01T12:30:00'));
      
      scheduler.addJob({
        id: 'hourly',
        cron: '@hourly',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      const job = scheduler.getJob('hourly');
      expect(job?.nextRun).toBeDefined();
      
      // Should be at the next hour (13:00)
      const nextRun = new Date(job!.nextRun!);
      expect(nextRun.getMinutes()).toBe(0);
    });

    it('supports @daily expression', () => {
      vi.setSystemTime(new Date('2024-01-01T12:30:00'));
      
      scheduler.addJob({
        id: 'daily',
        cron: '@daily',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      const job = scheduler.getJob('daily');
      expect(job?.nextRun).toBeDefined();
      
      // Should be next day at midnight
      const nextRun = new Date(job!.nextRun!);
      expect(nextRun.getHours()).toBe(0);
      expect(nextRun.getMinutes()).toBe(0);
    });
  });

  describe('job execution events', () => {
    it('emits job:execute event when job runs', () => {
      const executeFn = vi.fn();
      scheduler.on('job:execute', executeFn);
      
      scheduler.addJob({
        id: 'event-test',
        cron: '* * * * *',
        agentName: 'test-agent',
        task: 'run_task',
        params: { key: 'value' },
        enabled: true,
      });
      
      scheduler.start();
      
      // Advance to trigger execution
      vi.advanceTimersByTime(60000);
      
      expect(executeFn).toHaveBeenCalledWith(expect.objectContaining({
        id: 'event-test',
        agentName: 'test-agent',
        task: 'run_task',
        params: { key: 'value' },
      }));
    });
  });

  describe('getJobs', () => {
    it('returns all jobs with metadata', () => {
      scheduler.addJob({
        id: 'job1',
        cron: '* * * * *',
        agentName: 'agent1',
        task: 'task1',
        enabled: true,
      });
      scheduler.addJob({
        id: 'job2',
        cron: '0 * * * *',
        agentName: 'agent2',
        task: 'task2',
        enabled: true,
      });
      
      const jobs = scheduler.getJobs();
      
      expect(jobs).toHaveLength(2);
      expect(jobs[0]).toHaveProperty('id');
      expect(jobs[0]).toHaveProperty('cron');
      expect(jobs[0]).toHaveProperty('nextRun');
      expect(jobs[0]).toHaveProperty('agentName');
    });

    it('returns empty array when no jobs', () => {
      expect(scheduler.getJobs()).toEqual([]);
    });
  });

  describe('getJob', () => {
    it('returns specific job by id', () => {
      scheduler.addJob({
        id: 'find-me',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      const job = scheduler.getJob('find-me');
      expect(job).toBeDefined();
      expect(job?.id).toBe('find-me');
    });

    it('returns undefined for unknown id', () => {
      expect(scheduler.getJob('unknown')).toBeUndefined();
    });
  });

  describe('edge cases', () => {
    it('handles multiple rapid start calls', () => {
      scheduler.addJob({
        id: 'rapid-start',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      scheduler.start();
      scheduler.start();
      scheduler.start();
      
      // Should not throw
      vi.advanceTimersByTime(60000);
    });

    it('handles stop without start', () => {
      expect(() => scheduler.stop()).not.toThrow();
    });

    it('handles job added after start', () => {
      const executeFn = vi.fn();
      scheduler.on('job:execute', executeFn);
      
      scheduler.start();
      
      scheduler.addJob({
        id: 'late-add',
        cron: '* * * * *',
        agentName: 'test',
        task: 'test',
        enabled: true,
      });
      
      vi.advanceTimersByTime(60000);
      
      expect(executeFn).toHaveBeenCalled();
    });
  });
});
