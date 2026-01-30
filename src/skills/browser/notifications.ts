/**
 * Notifications Skill
 * 
 * Web Notifications API for alerting users.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';

interface NotifyParams {
  title: string;
  body?: string;
  icon?: string;
  tag?: string;
  requireInteraction?: boolean;
}

/**
 * Notifications Skill for Web Notifications API
 */
export class NotificationsSkill extends Skill {
  constructor() {
    super({ name: 'notifications' });
  }
  
  /**
   * Check if Notifications API is available
   */
  isAvailable(): boolean {
    return typeof window !== 'undefined' && 'Notification' in window;
  }
  
  /**
   * Request notification permission
   */
  @tool({ name: 'request_notification_permission', description: 'Request permission to show notifications' })
  async requestPermission(_params: Record<string, never>, _context: Context): Promise<{ permission: NotificationPermission | 'unavailable' }> {
    if (!this.isAvailable()) {
      return { permission: 'unavailable' };
    }
    
    const permission = await Notification.requestPermission();
    return { permission };
  }
  
  /**
   * Show a notification
   */
  @tool({
    name: 'show_notification',
    description: 'Show a browser notification',
    provides: 'notification',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Notification title' },
        body: { type: 'string', description: 'Notification body text' },
        icon: { type: 'string', description: 'URL to notification icon' },
        tag: { type: 'string', description: 'Tag to group notifications' },
      },
      required: ['title'],
    },
  })
  async showNotification(params: NotifyParams, _context: Context): Promise<{ success: boolean; error?: string }> {
    if (!this.isAvailable()) {
      return { success: false, error: 'Notifications API not available' };
    }
    
    if (Notification.permission !== 'granted') {
      return { success: false, error: 'Notification permission not granted' };
    }
    
    try {
      new Notification(params.title, {
        body: params.body,
        icon: params.icon,
        tag: params.tag,
        requireInteraction: params.requireInteraction,
      });
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Get current permission status
   */
  @tool({ name: 'notification_permission_status', description: 'Get current notification permission status' })
  async getPermissionStatus(_params: Record<string, never>, _context: Context): Promise<{ permission: NotificationPermission | 'unavailable' }> {
    if (!this.isAvailable()) {
      return { permission: 'unavailable' };
    }
    return { permission: Notification.permission };
  }
}
