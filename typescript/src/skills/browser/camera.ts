/**
 * Camera Skill
 * 
 * Video capture via getUserMedia API.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';

interface CameraConstraints {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
}

/**
 * Camera Skill for video capture
 */
export class CameraSkill extends Skill {
  private stream: MediaStream | null = null;
  
  constructor() {
    super({ name: 'camera' });
  }
  
  /**
   * Check if camera API is available
   */
  isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 
           'mediaDevices' in navigator && 
           'getUserMedia' in navigator.mediaDevices;
  }
  
  /**
   * Start camera stream
   */
  @tool({
    name: 'start_camera',
    description: 'Start camera video capture',
    provides: 'camera',
    parameters: {
      type: 'object',
      properties: {
        width: { type: 'number', description: 'Desired video width' },
        height: { type: 'number', description: 'Desired video height' },
        facingMode: { type: 'string', enum: ['user', 'environment'], description: 'Camera facing mode' },
      },
    },
  })
  async startCamera(
    params: CameraConstraints,
    _context: Context
  ): Promise<{ success: boolean; streamId?: string; error?: string }> {
    if (!this.isAvailable()) {
      return { success: false, error: 'Camera API not available' };
    }
    
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: params.width,
          height: params.height,
          facingMode: params.facingMode,
        },
      });
      
      return { success: true, streamId: this.stream.id };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Stop camera stream
   */
  @tool({ name: 'stop_camera', description: 'Stop camera video capture' })
  async stopCamera(_params: Record<string, never>, _context: Context): Promise<{ success: boolean }> {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    return { success: true };
  }
  
  /**
   * Capture frame as base64 image
   */
  @tool({ name: 'capture_frame', description: 'Capture current video frame as image' })
  async captureFrame(
    params: { format?: 'jpeg' | 'png'; quality?: number },
    _context: Context
  ): Promise<{ success: boolean; image?: string; error?: string }> {
    if (!this.stream) {
      return { success: false, error: 'Camera not started' };
    }
    
    try {
      const track = this.stream.getVideoTracks()[0];
      if (!track) {
        return { success: false, error: 'No video track available' };
      }
      
      // Use ImageCapture if available
      if ('ImageCapture' in window) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const ImageCaptureClass = (window as any).ImageCapture;
        const imageCapture = new ImageCaptureClass(track);
        const bitmap = await imageCapture.grabFrame();
        
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          return { success: false, error: 'Could not create canvas context' };
        }
        
        ctx.drawImage(bitmap, 0, 0);
        const mimeType = params.format === 'png' ? 'image/png' : 'image/jpeg';
        const dataUrl = canvas.toDataURL(mimeType, params.quality ?? 0.92);
        const base64 = dataUrl.split(',')[1];
        
        return { success: true, image: base64 };
      }
      
      return { success: false, error: 'ImageCapture not available' };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Get current stream
   */
  getStream(): MediaStream | null {
    return this.stream;
  }
  
  async cleanup(): Promise<void> {
    await this.stopCamera({}, {} as Context);
  }
}
