/**
 * Speech-to-Speech Browser Agent Example
 * 
 * Demonstrates a full speech pipeline:
 * 1. Speech-to-Text (STT): Whisper/Moonshine
 * 2. LLM Processing: WebLLM
 * 3. Text-to-Speech (TTS): SpeechT5/Kokoro
 * 
 * Uses @handoff decorators for seamless pipeline orchestration.
 */

import { BaseAgent } from '../src/core/agent.js';
import { Skill } from '../src/core/skill.js';
import { tool, handoff } from '../src/core/decorators.js';
import { WebLLMSkill } from '../src/skills/llm/webllm/skill.js';
import { SpeechToTextSkill } from '../src/skills/speech/stt.js';
import { TextToSpeechSkill } from '../src/skills/speech/tts.js';

/**
 * Speech Pipeline Skill
 * 
 * Orchestrates the full speech-to-speech flow using @handoff decorators.
 */
class SpeechPipelineSkill extends Skill {
  private stt: SpeechToTextSkill;
  private tts: TextToSpeechSkill;
  private llm: WebLLMSkill;

  constructor() {
    super();
    this.stt = new SpeechToTextSkill({
      model: 'Xenova/whisper-tiny.en',
    });
    this.tts = new TextToSpeechSkill({
      model: 'Xenova/speecht5_tts',
    });
    this.llm = new WebLLMSkill({
      model: 'SmolLM2-360M-Instruct-q4f16_1-MLC', // Small model for speed
    });
  }

  get id(): string {
    return 'speech-pipeline';
  }

  get name(): string {
    return 'Speech Pipeline';
  }

  get description(): string {
    return 'Full speech-to-speech conversation pipeline';
  }

  async initialize(): Promise<void> {
    console.log('[SpeechPipeline] Initializing components...');
    
    // Initialize in parallel for speed
    await Promise.all([
      this.stt.initialize(),
      this.tts.initialize(),
      this.llm.initialize(),
    ]);
    
    console.log('[SpeechPipeline] All components ready');
  }

  /**
   * Transcribe speech to text
   * @handoff to processText after transcription
   */
  @tool({
    name: 'transcribe',
    description: 'Transcribe speech to text',
    parameters: {
      audio: { type: 'object', description: 'Audio blob or array buffer' },
    },
  })
  @handoff('processText')
  async transcribe(audio: Blob | ArrayBuffer): Promise<string> {
    console.log('[SpeechPipeline] Transcribing audio...');
    const result = await this.stt.transcribe(audio);
    console.log('[SpeechPipeline] Transcribed:', result.text);
    return result.text;
  }

  /**
   * Process text through LLM
   * @handoff to synthesize after LLM response
   */
  @tool({
    name: 'processText',
    description: 'Process text through LLM',
    parameters: {
      text: { type: 'string', description: 'Input text' },
    },
  })
  @handoff('synthesize')
  async processText(text: string): Promise<string> {
    console.log('[SpeechPipeline] Processing with LLM:', text);
    const response = await this.llm.chat([
      { role: 'system', content: 'You are a helpful voice assistant. Keep responses brief and conversational.' },
      { role: 'user', content: text },
    ]);
    console.log('[SpeechPipeline] LLM response:', response);
    return response;
  }

  /**
   * Synthesize text to speech
   */
  @tool({
    name: 'synthesize',
    description: 'Synthesize text to speech',
    parameters: {
      text: { type: 'string', description: 'Text to synthesize' },
    },
  })
  async synthesize(text: string): Promise<{ audio: Float32Array; sampleRate: number }> {
    console.log('[SpeechPipeline] Synthesizing speech...');
    const result = await this.tts.synthesize(text);
    console.log('[SpeechPipeline] Audio generated:', result.duration, 'seconds');
    return result;
  }

  /**
   * Full speech-to-speech pipeline
   * 
   * This method chains:
   * transcribe -> processText -> synthesize
   */
  @tool({
    name: 'speechToSpeech',
    description: 'Full speech-to-speech pipeline',
    parameters: {
      audio: { type: 'object', description: 'Audio input' },
    },
  })
  async speechToSpeech(audio: Blob | ArrayBuffer): Promise<{
    transcription: string;
    response: string;
    audio: Float32Array;
    sampleRate: number;
  }> {
    // Step 1: Transcribe
    const transcription = await this.transcribe(audio);
    
    // Step 2: Process with LLM
    const response = await this.processText(transcription);
    
    // Step 3: Synthesize
    const audioResult = await this.synthesize(response);
    
    return {
      transcription,
      response,
      audio: audioResult.audio,
      sampleRate: audioResult.sampleRate,
    };
  }

  async cleanup(): Promise<void> {
    await Promise.all([
      this.stt.cleanup(),
      this.tts.cleanup(),
      this.llm.cleanup(),
    ]);
  }
}

/**
 * Create a speech-enabled browser agent
 */
async function createSpeechAgent() {
  const agent = new BaseAgent({
    id: 'speech-browser-agent',
    name: 'Speech Browser Agent',
    description: 'A voice-enabled agent with speech-to-speech capabilities',
  });

  const speechPipeline = new SpeechPipelineSkill();
  agent.addSkill(speechPipeline);

  await agent.initialize();

  return agent;
}

/**
 * Browser UI Helper
 * 
 * Handles microphone input and audio playback
 */
class SpeechUI {
  private agent: BaseAgent;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];

  constructor(agent: BaseAgent) {
    this.agent = agent;
  }

  /**
   * Start recording from microphone
   */
  async startRecording(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mediaRecorder = new MediaRecorder(stream);
    this.audioChunks = [];

    this.mediaRecorder.ondataavailable = (event) => {
      this.audioChunks.push(event.data);
    };

    this.mediaRecorder.start();
    console.log('[SpeechUI] Recording started');
  }

  /**
   * Stop recording and process
   */
  async stopRecording(): Promise<{
    transcription: string;
    response: string;
  }> {
    if (!this.mediaRecorder) {
      throw new Error('No recording in progress');
    }

    return new Promise((resolve, reject) => {
      this.mediaRecorder!.onstop = async () => {
        console.log('[SpeechUI] Recording stopped');
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        try {
          // Get speech pipeline skill
          const skill = this.agent.getSkill('speech-pipeline') as SpeechPipelineSkill;
          
          // Run full pipeline
          const result = await skill.speechToSpeech(audioBlob);
          
          // Play response audio
          await this.playAudio(result.audio, result.sampleRate);
          
          resolve({
            transcription: result.transcription,
            response: result.response,
          });
        } catch (error) {
          reject(error);
        }
      };

      this.mediaRecorder!.stop();
      this.mediaRecorder!.stream.getTracks().forEach(track => track.stop());
    });
  }

  /**
   * Play audio using Web Audio API
   */
  private async playAudio(audio: Float32Array, sampleRate: number): Promise<void> {
    const audioContext = new AudioContext({ sampleRate });
    const buffer = audioContext.createBuffer(1, audio.length, sampleRate);
    buffer.copyToChannel(audio, 0);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);

    return new Promise((resolve) => {
      source.onended = () => {
        audioContext.close();
        resolve();
      };
      source.start();
    });
  }
}

/**
 * Example usage
 */
async function main() {
  console.log('Creating speech-enabled browser agent...');
  
  const agent = await createSpeechAgent();
  console.log('Agent created:', agent.id);

  // Create UI helper
  const ui = new SpeechUI(agent);

  // Example: Process a text-to-speech request directly
  const skill = agent.getSkill('speech-pipeline') as SpeechPipelineSkill;
  
  // Text-only demo (microphone requires user interaction)
  const llmResponse = await skill.processText('Hello! How are you?');
  console.log('LLM Response:', llmResponse);

  const audioResult = await skill.synthesize(llmResponse);
  console.log('Generated audio:', audioResult.duration, 'seconds');

  // Cleanup
  await agent.cleanup();
}

// Export for browser use
export { createSpeechAgent, SpeechPipelineSkill, SpeechUI, main };

// Run if executed directly
if (typeof window === 'undefined') {
  main().catch(console.error);
}
