/**
 * Storage Skill
 * 
 * LocalStorage and IndexedDB access for persistence.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

/**
 * Storage Skill for browser storage APIs
 */
export class StorageSkill extends Skill {
  constructor() {
    super({ name: 'storage' });
  }
  
  /**
   * Check if LocalStorage is available
   */
  isLocalStorageAvailable(): boolean {
    try {
      const test = '__storage_test__';
      localStorage.setItem(test, test);
      localStorage.removeItem(test);
      return true;
    } catch {
      return false;
    }
  }
  
  /**
   * Get item from localStorage
   */
  @tool({
    name: 'storage_get',
    description: 'Get item from browser storage',
    provides: 'storage',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Storage key' },
      },
      required: ['key'],
    },
  })
  async get(params: { key: string }, _context: Context): Promise<{ value: string | null; error?: string }> {
    if (!this.isLocalStorageAvailable()) {
      return { value: null, error: 'LocalStorage not available' };
    }
    
    try {
      const value = localStorage.getItem(params.key);
      return { value };
    } catch (error) {
      return { value: null, error: (error as Error).message };
    }
  }
  
  /**
   * Set item in localStorage
   */
  @tool({
    name: 'storage_set',
    description: 'Store item in browser storage',
    provides: 'storage',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Storage key' },
        value: { type: 'string', description: 'Value to store' },
      },
      required: ['key', 'value'],
    },
  })
  async set(params: { key: string; value: string }, _context: Context): Promise<{ success: boolean; error?: string }> {
    if (!this.isLocalStorageAvailable()) {
      return { success: false, error: 'LocalStorage not available' };
    }
    
    try {
      localStorage.setItem(params.key, params.value);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Remove item from localStorage
   */
  @tool({
    name: 'storage_remove',
    description: 'Remove item from browser storage',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Storage key to remove' },
      },
      required: ['key'],
    },
  })
  async remove(params: { key: string }, _context: Context): Promise<{ success: boolean }> {
    if (!this.isLocalStorageAvailable()) {
      return { success: false };
    }
    
    localStorage.removeItem(params.key);
    return { success: true };
  }
  
  /**
   * Get JSON object from localStorage
   */
  @tool({
    name: 'storage_get_json',
    description: 'Get JSON object from browser storage',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Storage key' },
      },
      required: ['key'],
    },
  })
  async getJSON(params: { key: string }, _context: Context): Promise<{ value: unknown; error?: string }> {
    if (!this.isLocalStorageAvailable()) {
      return { value: null, error: 'LocalStorage not available' };
    }
    
    try {
      const raw = localStorage.getItem(params.key);
      if (raw === null) {
        return { value: null };
      }
      const value = JSON.parse(raw);
      return { value };
    } catch (error) {
      return { value: null, error: (error as Error).message };
    }
  }
  
  /**
   * Set JSON object in localStorage
   */
  @tool({
    name: 'storage_set_json',
    description: 'Store JSON object in browser storage',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Storage key' },
        value: { description: 'JSON-serializable value to store' },
      },
      required: ['key', 'value'],
    },
  })
  async setJSON(params: { key: string; value: unknown }, _context: Context): Promise<{ success: boolean; error?: string }> {
    if (!this.isLocalStorageAvailable()) {
      return { success: false, error: 'LocalStorage not available' };
    }
    
    try {
      localStorage.setItem(params.key, JSON.stringify(params.value));
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
}
