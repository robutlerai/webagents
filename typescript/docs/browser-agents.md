# Browser Agents

webagents-ts supports running agents entirely in the browser using WebGPU-accelerated LLMs via WebLLM and Transformers.js.

## Quick Start

### Simple Text Agent

```typescript
import { BaseAgent } from 'webagents-ts';
import { WebLLMSkill } from 'webagents-ts/skills/llm/webllm';

// Create agent
const agent = new BaseAgent({
  id: 'browser-agent',
  name: 'Browser Agent',
});

// Add WebLLM skill
agent.addSkill(new WebLLMSkill({
  model: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
}));

// Initialize
await agent.initialize();

// Chat
const response = await agent.chat([
  { role: 'user', content: 'Hello!' }
]);
```

## Available Models

### WebLLM Models (WebGPU)

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | 360M | ⚡⚡⚡ | ★★☆ |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | 500M | ⚡⚡⚡ | ★★★ |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | 1B | ⚡⚡ | ★★★★ |
| `Qwen3-0.6B-q4f16_1-MLC` | 600M | ⚡⚡⚡ | ★★★ |
| `Qwen3-1.7B-q4f16_1-MLC` | 1.7B | ⚡⚡ | ★★★★ |
| `gemma-3-1b-it-q4f16_1-MLC` | 1B | ⚡⚡ | ★★★★ |
| `Llama-3.2-3B-Instruct-q4f16_1-MLC` | 3B | ⚡ | ★★★★★ |
| `Phi-3.5-mini-instruct-q4f16_1-MLC` | 3.8B | ⚡ | ★★★★★ |

### Transformers.js Models (WASM/WebGPU)

| Model | Parameters | Backend | Notes |
|-------|------------|---------|-------|
| `onnx-community/gemma-3-1b-it-ONNX-GQA` | 1B | WebGPU/WASM | Gemma 3 |
| `onnx-community/Llama-3.2-1B-Instruct-ONNX` | 1B | WebGPU/WASM | Llama 3.2 |
| `onnx-community/Qwen2.5-0.5B-Instruct` | 500M | WASM | Qwen 2.5 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | WASM | SmolLM2 |

## Speech Pipeline

webagents-ts includes speech skills for voice-enabled agents:

### Speech-to-Text (STT)

```typescript
import { SpeechToTextSkill } from 'webagents-ts/skills/speech';

const stt = new SpeechToTextSkill({
  model: 'Xenova/whisper-tiny.en', // or 'onnx-community/moonshine-tiny'
});

await stt.initialize();
const result = await stt.transcribe(audioBlob);
console.log(result.text);
```

**Available STT Models:**

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| `Xenova/whisper-tiny.en` | 39M | ⚡⚡⚡ | ★★★ |
| `Xenova/whisper-small.en` | 244M | ⚡⚡ | ★★★★ |
| `onnx-community/moonshine-tiny` | 27M | ⚡⚡⚡⚡ | ★★★ |
| `onnx-community/moonshine-base` | 61M | ⚡⚡⚡ | ★★★★ |

### Text-to-Speech (TTS)

```typescript
import { TextToSpeechSkill } from 'webagents-ts/skills/speech';

const tts = new TextToSpeechSkill({
  model: 'Xenova/speecht5_tts',
});

await tts.initialize();
const result = await tts.synthesize('Hello, world!');
// result.audio is a Float32Array
```

**Available TTS Models:**

| Model | Parameters | Quality | Notes |
|-------|------------|---------|-------|
| `Xenova/speecht5_tts` | 80M | ★★★ | English, requires speaker embeddings |
| `onnx-community/Kokoro-82M-v1.0-ONNX` | 82M | ★★★★ | Multiple voices, style control |
| `Xenova/mms-tts-eng` | 41M | ★★★ | Meta MMS, English |

## Full Speech-to-Speech Pipeline

Use the `@handoff` decorator to chain skills:

```typescript
import { BaseAgent, Skill, tool, handoff } from 'webagents-ts';
import { WebLLMSkill } from 'webagents-ts/skills/llm/webllm';
import { SpeechToTextSkill, TextToSpeechSkill } from 'webagents-ts/skills/speech';

class SpeechPipelineSkill extends Skill {
  private stt = new SpeechToTextSkill();
  private tts = new TextToSpeechSkill();
  private llm = new WebLLMSkill({ model: 'SmolLM2-360M-Instruct-q4f16_1-MLC' });

  get id() { return 'speech-pipeline'; }

  async initialize() {
    await Promise.all([
      this.stt.initialize(),
      this.tts.initialize(),
      this.llm.initialize(),
    ]);
  }

  @tool({ name: 'transcribe', description: 'STT' })
  @handoff('processText') // Automatically calls processText next
  async transcribe(audio: Blob) {
    return (await this.stt.transcribe(audio)).text;
  }

  @tool({ name: 'processText', description: 'LLM' })
  @handoff('synthesize') // Automatically calls synthesize next
  async processText(text: string) {
    return await this.llm.chat([{ role: 'user', content: text }]);
  }

  @tool({ name: 'synthesize', description: 'TTS' })
  async synthesize(text: string) {
    return await this.tts.synthesize(text);
  }
}
```

## Browser Integration

### HTML Example

```html
<!DOCTYPE html>
<html>
<head>
  <title>Voice Agent</title>
</head>
<body>
  <button id="record">🎤 Record</button>
  <div id="output"></div>

  <script type="module">
    import { createSpeechAgent, SpeechUI } from './browser-agent-speech.js';

    const agent = await createSpeechAgent();
    const ui = new SpeechUI(agent);

    let recording = false;
    document.getElementById('record').onclick = async () => {
      if (!recording) {
        await ui.startRecording();
        recording = true;
      } else {
        const result = await ui.stopRecording();
        document.getElementById('output').textContent = 
          `You: ${result.transcription}\nAgent: ${result.response}`;
        recording = false;
      }
    };
  </script>
</body>
</html>
```

## Performance Tips

1. **Use smaller models** for faster loading and inference
2. **Enable WebGPU** in your browser (Chrome 113+, Edge 113+)
3. **Cache models** - Transformers.js caches models in IndexedDB
4. **Use Web Workers** for non-blocking inference
5. **Quantize models** - Use q4 quantization for 4x smaller models

## Benchmarking

Run the benchmark suite:

```bash
npm run benchmark
# Open http://localhost:3456 in Chrome
```

Available benchmarks:
- **LLM Comparison**: WebLLM vs Transformers.js tok/s
- **UAMP Protocol**: Overhead measurement

## Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- ~2GB RAM for small models, ~8GB for large models
- GPU recommended for best performance (falls back to WASM/CPU)
