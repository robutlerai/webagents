/**
 * Geolocation Skill
 * 
 * Access device location via Geolocation API.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

interface GeolocationOptions {
  enableHighAccuracy?: boolean;
  timeout?: number;
  maximumAge?: number;
}

interface LocationResult {
  latitude: number;
  longitude: number;
  accuracy: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
  heading?: number | null;
  speed?: number | null;
  timestamp: number;
}

/**
 * Geolocation Skill for location access
 */
export class GeolocationSkill extends Skill {
  constructor() {
    super({ name: 'geolocation' });
  }
  
  /**
   * Check if Geolocation API is available
   */
  isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'geolocation' in navigator;
  }
  
  /**
   * Get current position
   */
  @tool({
    name: 'get_location',
    description: 'Get current device location',
    provides: 'geolocation',
    parameters: {
      type: 'object',
      properties: {
        enableHighAccuracy: { type: 'boolean', description: 'Use GPS for higher accuracy' },
        timeout: { type: 'number', description: 'Timeout in milliseconds' },
      },
    },
  })
  async getCurrentPosition(
    params: GeolocationOptions,
    _context: Context
  ): Promise<{ success: boolean; location?: LocationResult; error?: string }> {
    if (!this.isAvailable()) {
      return { success: false, error: 'Geolocation API not available' };
    }
    
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            success: true,
            location: {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              accuracy: position.coords.accuracy,
              altitude: position.coords.altitude,
              altitudeAccuracy: position.coords.altitudeAccuracy,
              heading: position.coords.heading,
              speed: position.coords.speed,
              timestamp: position.timestamp,
            },
          });
        },
        (error) => {
          resolve({
            success: false,
            error: error.message,
          });
        },
        {
          enableHighAccuracy: params.enableHighAccuracy ?? false,
          timeout: params.timeout ?? 10000,
          maximumAge: params.maximumAge ?? 0,
        }
      );
    });
  }
}
