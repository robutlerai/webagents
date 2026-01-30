/**
 * Microphone Skill
 * 
 * Audio capture via getUserMedia API.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';

interface MicrophoneConstraints {
  sampleRate?: number;
  channelCount?: number;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
}

/**
 * Microphone Skill for audio capture
 */
export class MicrophoneSkill extends Skill {
  private stream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  
  constructor() {
    super({ name: 'microphone' });
  }
  
  /**
   * Check if microphone API is available
   */
  isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 
           'mediaDevices' in navigator && 
           'getUserMedia' in navigator.mediaDevices;
  }
  
  /**
   * Start microphone stream
   */
  @tool({
    name: 'start_microphone',
    description: 'Start microphone audio capture',
    provides: 'microphone',
    parameters: {
      type: 'object',
      properties: {
        sampleRate: { type: 'number', description: 'Audio sample rate' },
        channelCount: { type: 'number', description: 'Number of audio channels' },
        echoCancellation: { type: 'boolean', description: 'Enable echo cancellation' },
        noiseSuppression: { type: 'boolean', description: 'Enable noise suppression' },
      },
    },
  })
  async startMicrophone(
    params: MicrophoneConstraints,
    _context: Context
  ): Promise<{ success: boolean; streamId?: string; error?: string }> {
    if (!this.isAvailable()) {
      return { success: false, error: 'Microphone API not available' };
    }
    
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: params.sampleRate,
          channelCount: params.channelCount,
          echoCancellation: params.echoCancellation ?? true,
          noiseSuppression: params.noiseSuppression ?? true,
        },
      });
      
      return { success: true, streamId: this.stream.id };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Stop microphone stream
   */
  @tool({ name: 'stop_microphone', description: 'Stop microphone audio capture' })
  async stopMicrophone(_params: Record<string, never>, _context: Context): Promise<{ success: boolean }> {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    
    this.mediaRecorder = null;
    this.audioChunks = [];
    
    return { success: true };
  }
  
  /**
   * Start recording audio
   */
  @tool({
    name: 'start_recording',
    description: 'Start recording audio from microphone',
    parameters: {
      type: 'object',
      properties: {
        mimeType: { type: 'string', description: 'Recording MIME type (e.g., audio/webm)' },
      },
    },
  })
  async startRecording(
    params: { mimeType?: string },
    _context: Context
  ): Promise<{ success: boolean; error?: string }> {
    if (!this.stream) {
      return { success: false, error: 'Microphone not started' };
    }
    
    try {
      this.audioChunks = [];
      const mimeType = params.mimeType || 'audio/webm;codecs=opus';
      
      this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };
      
      this.mediaRecorder.start(100); // Collect data every 100ms
      
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
  
  /**
   * Stop recording and get audio as base64
   */
  @tool({ name: 'stop_recording', description: 'Stop recording and get audio data' })
  async stopRecording(
    _params: Record<string, never>,
    _context: Context
  ): Promise<{ success: boolean; audio?: string; mimeType?: string; error?: string }> {
    if (!this.mediaRecorder) {
      return { success: false, error: 'Not recording' };
    }
    
    return new Promise((resolve) => {
      this.mediaRecorder!.onstop = async () => {
        try {
          const blob = new Blob(this.audioChunks, { type: this.mediaRecorder!.mimeType });
          const buffer = await blob.arrayBuffer();
          const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
          
          resolve({
            success: true,
            audio: base64,
            mimeType: this.mediaRecorder!.mimeType,
          });
        } catch (error) {
          resolve({ success: false, error: (error as Error).message });
        }
      };
      
      this.mediaRecorder!.stop();
    });
  }
  
  /**
   * Get current stream
   */
  getStream(): MediaStream | null {
    return this.stream;
  }
  
  async cleanup(): Promise<void> {
    await this.stopMicrophone({}, {} as Context);
  }
}
