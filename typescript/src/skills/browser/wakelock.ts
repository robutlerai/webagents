/**
 * Wake Lock Skill
 * 
 * Prevents screen from sleeping during long-running operations
 * like WebLLM inference.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

/**
 * Wake Lock Skill for preventing screen sleep
 */
export class WakeLockSkill extends Skill {
  private wakeLock: WakeLockSentinel | null = null;
  
  constructor() {
    super({ name: 'wakelock' });
  }
  
  /**
   * Check if Wake Lock API is available
   */
  isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'wakeLock' in navigator;
  }
  
  /**
   * Request a wake lock
   */
  @tool({ name: 'acquire_wakelock', description: 'Prevent screen from sleeping', provides: 'wakelock' })
  async acquireWakeLock(_params: Record<string, never>, _context: Context): Promise<{ success: boolean; error?: string }> {
    if (!this.isAvailable()) {
      return { success: false, error: 'Wake Lock API not available' };
    }
    
    try {
      this.wakeLock = await navigator.wakeLock.request('screen');
      
      // Re-acquire on visibility change
      document.addEventListener('visibilitychange', this.handleVisibilityChange);
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Release the wake lock
   */
  @tool({ name: 'release_wakelock', description: 'Allow screen to sleep again' })
  async releaseWakeLock(_params: Record<string, never>, _context: Context): Promise<{ success: boolean }> {
    if (this.wakeLock) {
      await this.wakeLock.release();
      this.wakeLock = null;
      document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    }
    return { success: true };
  }
  
  /**
   * Check wake lock status
   */
  @tool({ name: 'wakelock_status', description: 'Check if wake lock is active' })
  async getStatus(_params: Record<string, never>, _context: Context): Promise<{ active: boolean; available: boolean }> {
    return {
      active: this.wakeLock !== null && !this.wakeLock.released,
      available: this.isAvailable(),
    };
  }
  
  private handleVisibilityChange = async () => {
    if (this.wakeLock !== null && document.visibilityState === 'visible') {
      try {
        this.wakeLock = await navigator.wakeLock.request('screen');
      } catch {
        // Ignore errors on re-acquire
      }
    }
  };
  
  async cleanup(): Promise<void> {
    await this.releaseWakeLock({}, {} as Context);
  }
}
