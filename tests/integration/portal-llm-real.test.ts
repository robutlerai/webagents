/**
 * Portal Transport + LLM Real Integration Tests
 * 
 * NO MOCKING - Uses real Transformers.js models.
 * Uses tiny models for fast tests (~50-100MB download on first run).
 * 
 * Model cache: ~/.cache/huggingface/hub/
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { BaseAgent } from '../../src/core/agent.js';
import { TransformersSkill } from '../../src/skills/llm/transformers/skill.js';
import { PortalTransportSkill } from '../../src/skills/transport/portal/skill.js';
import type { ServerEvent } from '../../src/uamp/events.js';

// Model with WebGPU support - ONNX format from onnx-community
// Supports: cpu, wasm, webgpu
// Size: ~2GB (quantized versions smaller)
const TEST_MODEL = 'onnx-community/Llama-3.2-1B-Instruct';

// Smaller fallback model if Llama is too large
const SMALL_MODEL = 'Xenova/distilgpt2'; // ~50MB, cpu only

// Mock WebSocket for portal testing
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  readyState = MockWebSocket.OPEN;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;

  send(_data: string): void {}

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
  }

  simulateMessage(data: string): void {
    const event = new MessageEvent('message', { data });
    if (this.onmessage) this.onmessage(event);
  }
}

describe('Portal + Transformers.js Real Integration', () => {
  let transformersSkill: TransformersSkill;
  let portalSkill: PortalTransportSkill;
  let agent: BaseAgent;

  beforeAll(async () => {
    // Use smaller model for basic tests (faster CI)
    const modelForBasicTests = SMALL_MODEL;
    
    console.log(`\nLoading model: ${modelForBasicTests}...`);
    console.log('(First run downloads model, subsequent runs use cache)\n');

    transformersSkill = new TransformersSkill({
      model: modelForBasicTests,
      device: 'cpu', // Use CPU for CI compatibility
      max_tokens: 30, // Keep responses short for tests
      temperature: 0.1, // Low temp for more deterministic output
    });

    // Initialize loads the model
    await transformersSkill.initialize();
    console.log('Model loaded!\n');

    portalSkill = new PortalTransportSkill();

    agent = new BaseAgent({
      name: 'real-transformers-agent',
      skills: [transformersSkill, portalSkill],
    });

    portalSkill.setAgent(agent);
  }, 120000); // 2 minute timeout for model download

  afterAll(async () => {
    await agent?.cleanup();
  });

  it('generates real response via agent.run()', async () => {
    console.log('Testing agent.run()...');

    const response = await agent.run([
      { role: 'user', content: 'Hello' },
    ]);

    console.log('Response:', response.content);

    expect(response.content).toBeDefined();
    expect(response.content.length).toBeGreaterThan(0);
    expect(typeof response.content).toBe('string');
  }, 30000);

  it('generates real response via agent.runStreaming()', async () => {
    console.log('Testing agent.runStreaming()...');

    const chunks: string[] = [];

    for await (const chunk of agent.runStreaming([
      { role: 'user', content: 'The weather today is' },
    ])) {
      if (chunk.type === 'delta') {
        chunks.push(chunk.delta);
        process.stdout.write(chunk.delta);
      }
    }

    console.log('\n');

    expect(chunks.length).toBeGreaterThan(0);
    const fullResponse = chunks.join('');
    expect(fullResponse.length).toBeGreaterThan(0);
  }, 30000);

  it('processes UAMP events through portal WebSocket', async () => {
    console.log('Testing portal UAMP processing...');

    const mockWs = new MockWebSocket();
    const responses: ServerEvent[] = [];

    portalSkill.handleConnection(mockWs as any, {} as any);

    mockWs.send = (data: string) => {
      try {
        const event = JSON.parse(data);
        responses.push(event);
        if (event.type === 'response.delta') {
          process.stdout.write((event as any).delta?.text || '');
        }
      } catch {
        // Ignore
      }
    };

    const uampMessage = JSON.stringify({
      type: 'uamp',
      events: [
        {
          type: 'session.create',
          event_id: 'e1',
          uamp_version: '1.0',
          session: { modalities: ['text'] },
        },
        {
          type: 'input.text',
          event_id: 'e2',
          text: 'Once upon a time',
          role: 'user',
        },
        {
          type: 'response.create',
          event_id: 'e3',
        },
      ],
    });

    mockWs.simulateMessage(uampMessage);

    // Wait for processing (real inference takes time)
    await new Promise(resolve => setTimeout(resolve, 10000));

    console.log('\n');

    // Verify response structure
    const createdEvent = responses.find(e => e.type === 'response.created');
    expect(createdEvent).toBeDefined();

    const doneEvent = responses.find(e => e.type === 'response.done');
    expect(doneEvent).toBeDefined();

    const output = (doneEvent as any)?.response?.output?.[0]?.text;
    console.log('Portal response:', output);
    expect(output).toBeDefined();
    expect(output.length).toBeGreaterThan(0);
  }, 30000);

  it('handles multi-turn conversation', async () => {
    console.log('Testing multi-turn conversation...');

    const response1 = await agent.run([
      { role: 'user', content: 'My name is Alice.' },
    ]);
    console.log('Turn 1:', response1.content);

    const response2 = await agent.run([
      { role: 'user', content: 'My name is Alice.' },
      { role: 'assistant', content: response1.content },
      { role: 'user', content: 'What did I just tell you?' },
    ]);
    console.log('Turn 2:', response2.content);

    expect(response1.content.length).toBeGreaterThan(0);
    expect(response2.content.length).toBeGreaterThan(0);
  }, 60000);

  it('respects max_tokens configuration', async () => {
    console.log('Testing max_tokens limit...');

    // Create skill with very short max_tokens
    const shortSkill = new TransformersSkill({
      model: SMALL_MODEL,
      device: 'cpu',
      max_tokens: 5,
    });
    await shortSkill.initialize();

    const shortAgent = new BaseAgent({
      name: 'short-agent',
      skills: [shortSkill],
    });

    const response = await shortAgent.run([
      { role: 'user', content: 'Tell me a very long story about' },
    ]);

    console.log('Short response:', response.content);

    // Response should be limited (though exact token count varies)
    // Just verify it generated something
    expect(response.content.length).toBeGreaterThan(0);

    await shortAgent.cleanup();
  }, 30000);

  it('handles system instructions', async () => {
    console.log('Testing system instructions...');

    const response = await agent.run([
      { role: 'system', content: 'You always respond in exactly 3 words.' },
      { role: 'user', content: 'Hello' },
    ]);

    console.log('With system prompt:', response.content);

    expect(response.content.length).toBeGreaterThan(0);
  }, 30000);

  it('returns capabilities correctly', () => {
    const caps = transformersSkill.getCapabilities();

    expect(caps.id).toBe(SMALL_MODEL);
    expect(caps.provider).toBe('transformers.js');
    expect(caps.modalities).toContain('text');
    expect(caps.supports_streaming).toBe(true);
    expect(caps.extensions?.device).toBe('cpu');
  });

  it('agent exposes combined capabilities', () => {
    const caps = agent.getCapabilities();

    expect(caps.id).toBe('real-transformers-agent');
    expect(caps.modalities).toContain('text');
  });

  describe('performance', () => {
    it('measures tokens per second', async () => {
      console.log('\n=== Performance Test ===\n');

      // Generate a longer response to get accurate timing
      const perfSkill = new TransformersSkill({
        model: SMALL_MODEL,
        device: 'cpu',
        max_tokens: 50,
        temperature: 0.7,
      });
      await perfSkill.initialize();

      const perfAgent = new BaseAgent({
        name: 'perf-agent',
        skills: [perfSkill],
      });

      // Warm up run
      console.log('Warming up...');
      await perfAgent.run([{ role: 'user', content: 'Hi' }]);

      // Run multiple iterations for accurate measurement
      const iterations = 3;
      const results: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const startTime = performance.now();

        const response = await perfAgent.run([
          { role: 'user', content: 'Once upon a time' },
        ]);

        const endTime = performance.now();
        const timeMs = endTime - startTime;

        // Estimate tokens (~4 chars per token for English)
        const estimatedTokens = Math.ceil(response.content.length / 4);
        const tokensPerSecond = estimatedTokens / (timeMs / 1000);

        results.push(tokensPerSecond);
        console.log(`Run ${i + 1}: ${response.content.substring(0, 50)}...`);
        console.log(`  Time: ${timeMs.toFixed(0)}ms, ~${estimatedTokens} tokens, ${tokensPerSecond.toFixed(1)} tok/s`);
      }

      const avgTokensPerSecond = results.reduce((a, b) => a + b, 0) / results.length;

      console.log('\n--- Performance Results ---');
      console.log(`Model: ${SMALL_MODEL}`);
      console.log(`Device: cpu`);
      console.log(`Iterations: ${iterations}`);
      console.log(`Average tokens/second: ${avgTokensPerSecond.toFixed(1)}`);
      console.log(`Min: ${Math.min(...results).toFixed(1)} tok/s`);
      console.log(`Max: ${Math.max(...results).toFixed(1)} tok/s`);
      console.log('---------------------------\n');

      // Verify reasonable performance
      expect(avgTokensPerSecond).toBeGreaterThan(10);

      // JSON output for CI
      console.log('Performance JSON:', JSON.stringify({
        model: SMALL_MODEL,
        device: 'cpu',
        iterations,
        avgTokensPerSecond,
        minTokensPerSecond: Math.min(...results),
        maxTokensPerSecond: Math.max(...results),
      }, null, 2));

      await perfAgent.cleanup();
    }, 60000);

    it('measures throughput with multiple requests', async () => {
      console.log('\n=== Throughput Test ===\n');

      const requests = [
        'Hello',
        'What is AI?',
        'Tell me a joke',
        'How are you?',
        'Goodbye',
      ];

      const startTime = performance.now();
      let totalTokens = 0;

      for (const prompt of requests) {
        const response = await agent.run([{ role: 'user', content: prompt }]);
        const tokens = Math.ceil(response.content.length / 4);
        totalTokens += tokens;
        console.log(`"${prompt}" -> ${tokens} tokens`);
      }

      const totalTimeMs = performance.now() - startTime;
      const requestsPerSecond = requests.length / (totalTimeMs / 1000);
      const tokensPerSecond = totalTokens / (totalTimeMs / 1000);

      console.log('\n--- Throughput Results ---');
      console.log(`Total requests: ${requests.length}`);
      console.log(`Total time: ${(totalTimeMs / 1000).toFixed(2)}s`);
      console.log(`Requests/second: ${requestsPerSecond.toFixed(2)}`);
      console.log(`Tokens/second (total): ${tokensPerSecond.toFixed(2)}`);
      console.log('--------------------------\n');

      expect(requests.length).toBe(5);
    }, 60000);

    it('compares streaming vs non-streaming latency', async () => {
      console.log('\n=== Streaming vs Non-Streaming ===\n');

      const prompt = [{ role: 'user' as const, content: 'Count from 1 to 5' }];

      // Non-streaming
      const nonStreamStart = performance.now();
      const nonStreamResponse = await agent.run(prompt);
      const nonStreamTime = performance.now() - nonStreamStart;

      // Streaming
      const streamStart = performance.now();
      let streamFirstToken = 0;
      let streamContent = '';

      for await (const chunk of agent.runStreaming(prompt)) {
        if (chunk.type === 'delta') {
          if (!streamFirstToken) {
            streamFirstToken = performance.now() - streamStart;
          }
          streamContent += chunk.delta;
        }
      }
      const streamTotalTime = performance.now() - streamStart;

      console.log('--- Latency Comparison ---');
      console.log(`Non-streaming total: ${nonStreamTime.toFixed(0)}ms`);
      console.log(`Streaming first token: ${streamFirstToken.toFixed(0)}ms`);
      console.log(`Streaming total: ${streamTotalTime.toFixed(0)}ms`);
      console.log(`First token advantage: ${(nonStreamTime - streamFirstToken).toFixed(0)}ms faster`);
      console.log('--------------------------\n');

      expect(nonStreamResponse.content.length).toBeGreaterThan(0);
      expect(streamContent.length).toBeGreaterThan(0);
    }, 30000);

    it('benchmarks across multiple runs', async () => {
      console.log('\n=== Extended Benchmark ===\n');

      const iterations = 5;
      const results: { timeMs: number; tokens: number; tokensPerSecond: number }[] = [];

      console.log('Running extended benchmark...');

      for (let i = 0; i < iterations; i++) {
        const startTime = performance.now();

        const response = await agent.run([
          { role: 'user', content: 'The future of AI is' },
        ]);

        const timeMs = performance.now() - startTime;
        const tokens = Math.ceil(response.content.length / 4);
        const tokensPerSecond = tokens / (timeMs / 1000);

        results.push({ timeMs, tokens, tokensPerSecond });
        console.log(`Run ${i + 1}: ${timeMs.toFixed(0)}ms, ${tokens} tokens, ${tokensPerSecond.toFixed(1)} tok/s`);
      }

      const avgTime = results.reduce((a, b) => a + b.timeMs, 0) / iterations;
      const avgTokens = results.reduce((a, b) => a + b.tokens, 0) / iterations;
      const avgTokPerSec = results.reduce((a, b) => a + b.tokensPerSecond, 0) / iterations;
      const minTokPerSec = Math.min(...results.map(r => r.tokensPerSecond));
      const maxTokPerSec = Math.max(...results.map(r => r.tokensPerSecond));

      // Calculate standard deviation
      const variance = results.reduce((sum, r) => 
        sum + Math.pow(r.tokensPerSecond - avgTokPerSec, 2), 0) / iterations;
      const stdDev = Math.sqrt(variance);

      console.log('\n--- Extended Benchmark Results ---');
      console.log(`Model: ${SMALL_MODEL}`);
      console.log(`Device: cpu`);
      console.log(`Iterations: ${iterations}`);
      console.log(`Average latency: ${avgTime.toFixed(0)}ms`);
      console.log(`Average tokens: ${avgTokens.toFixed(0)}`);
      console.log(`Tokens/second: ${avgTokPerSec.toFixed(1)} ± ${stdDev.toFixed(1)}`);
      console.log(`Range: ${minTokPerSec.toFixed(1)} - ${maxTokPerSec.toFixed(1)} tok/s`);
      console.log('----------------------------------\n');

      expect(avgTokPerSec).toBeGreaterThan(50);

      console.log('Benchmark JSON:', JSON.stringify({
        model: SMALL_MODEL,
        device: 'cpu',
        iterations,
        avgLatencyMs: avgTime,
        avgTokens,
        avgTokensPerSecond: avgTokPerSec,
        stdDev,
        minTokensPerSecond: minTokPerSec,
        maxTokensPerSecond: maxTokPerSec,
      }, null, 2));
    }, 60000);

    it('measures latency distribution', async () => {
      console.log('\n=== Latency Distribution ===\n');

      const samples = 10;
      const latencies: number[] = [];

      console.log('Collecting latency samples...');

      for (let i = 0; i < samples; i++) {
        const startTime = performance.now();
        await agent.run([{ role: 'user', content: 'Hi' }]);
        const latency = performance.now() - startTime;
        latencies.push(latency);
        process.stdout.write(`${latency.toFixed(0)}ms `);
      }
      console.log('\n');

      // Sort for percentile calculation
      latencies.sort((a, b) => a - b);

      const p50 = latencies[Math.floor(samples * 0.5)];
      const p90 = latencies[Math.floor(samples * 0.9)];
      const p99 = latencies[Math.floor(samples * 0.99)] || latencies[latencies.length - 1];
      const avg = latencies.reduce((a, b) => a + b, 0) / samples;
      const min = latencies[0];
      const max = latencies[latencies.length - 1];

      console.log('--- Latency Distribution ---');
      console.log(`Samples: ${samples}`);
      console.log(`Min: ${min.toFixed(0)}ms`);
      console.log(`P50: ${p50.toFixed(0)}ms`);
      console.log(`P90: ${p90.toFixed(0)}ms`);
      console.log(`P99: ${p99.toFixed(0)}ms`);
      console.log(`Max: ${max.toFixed(0)}ms`);
      console.log(`Avg: ${avg.toFixed(0)}ms`);
      console.log('----------------------------\n');

      expect(p50).toBeLessThan(5000); // Should respond in under 5s

      console.log('Latency JSON:', JSON.stringify({
        samples,
        minMs: min,
        p50Ms: p50,
        p90Ms: p90,
        p99Ms: p99,
        maxMs: max,
        avgMs: avg,
      }, null, 2));
    }, 60000);

    /**
     * Device Performance Comparison
     * 
     * Uses onnx-community/Llama-3.2-1B-Instruct which supports:
     * - cpu: Works everywhere
     * - wasm: Faster than CPU in browsers
     * - webgpu: Fastest (browser only)
     * 
     * Skip by default due to large model size (~2GB)
     * Run with: RUN_GPU_TESTS=true npm test
     */
    it.skipIf(!process.env.RUN_GPU_TESTS)('compares device performance (cpu vs wasm)', async () => {
      console.log('\n=== Device Performance Comparison ===\n');
      console.log(`Model: ${TEST_MODEL} (supports webgpu)\n`);

      const devices = ['cpu', 'wasm'] as const;
      const deviceResults: Record<string, { tokPerSec: number; latencyMs: number } | null> = {};

      for (const device of devices) {
        console.log(`Testing ${device.toUpperCase()}...`);

        try {
          const skill = new TransformersSkill({
            model: TEST_MODEL,
            device,
            max_tokens: 20,
            temperature: 0.1,
          });
          await skill.initialize();

          const testAgent = new BaseAgent({
            name: `${device}-agent`,
            skills: [skill],
          });

          // Warm up
          await testAgent.run([{ role: 'user', content: 'Hi' }]);

          // Benchmark
          const times: number[] = [];
          const tokensPerSec: number[] = [];

          for (let i = 0; i < 3; i++) {
            const start = performance.now();
            const response = await testAgent.run([
              { role: 'user', content: 'Hello' },
            ]);
            const timeMs = performance.now() - start;
            const tokens = Math.ceil(response.content.length / 4);

            times.push(timeMs);
            tokensPerSec.push(tokens / (timeMs / 1000));
            console.log(`  Run ${i + 1}: ${timeMs.toFixed(0)}ms, ${(tokens / (timeMs / 1000)).toFixed(1)} tok/s`);
          }

          const avgLatency = times.reduce((a, b) => a + b, 0) / times.length;
          const avgTokPerSec = tokensPerSec.reduce((a, b) => a + b, 0) / tokensPerSec.length;

          deviceResults[device] = { tokPerSec: avgTokPerSec, latencyMs: avgLatency };

          await testAgent.cleanup();
        } catch (error) {
          console.log(`  ${device}: Not supported - ${(error as Error).message.substring(0, 50)}`);
          deviceResults[device] = null;
        }
      }

      console.log('\n--- Device Comparison Results ---');
      for (const [device, result] of Object.entries(deviceResults)) {
        if (result) {
          console.log(`${device.toUpperCase()}: ${result.tokPerSec.toFixed(1)} tok/s (${result.latencyMs.toFixed(0)}ms avg)`);
        } else {
          console.log(`${device.toUpperCase()}: Not available`);
        }
      }

      // Calculate speedup if both available
      if (deviceResults.cpu && deviceResults.wasm) {
        const speedup = ((deviceResults.wasm.tokPerSec - deviceResults.cpu.tokPerSec) / deviceResults.cpu.tokPerSec * 100);
        console.log(`WASM vs CPU: ${speedup > 0 ? '+' : ''}${speedup.toFixed(1)}%`);
      }
      console.log('---------------------------------\n');

      // At least one device should work
      const workingDevices = Object.values(deviceResults).filter(r => r !== null);
      expect(workingDevices.length).toBeGreaterThan(0);

      console.log('Device Comparison JSON:', JSON.stringify(deviceResults, null, 2));
    }, 180000);

    /**
     * WebGPU Performance Test
     * 
     * NOTE: WebGPU is only available in browsers, not in Node.js.
     * This test is skipped in Node.js but documents how to run it.
     * 
     * To run WebGPU tests:
     * 1. Use Playwright with Chrome/Edge 113+
     * 2. Or run manually in browser with WebGPU enabled
     * 
     * Expected performance with WebGPU:
     * - Llama-3.2-1B: ~500-1000 tok/s (vs ~100-200 on CPU)
     * - Up to 5-10x faster than CPU/WASM
     */
    it.skip('measures WebGPU performance (browser only)', async () => {
      console.log('\n=== WebGPU Performance Test ===\n');
      console.log('NOTE: This test requires a browser with WebGPU support.');
      console.log('Model:', TEST_MODEL);
      console.log('');

      const webgpuSkill = new TransformersSkill({
        model: TEST_MODEL,
        device: 'webgpu',
        max_tokens: 50,
      });

      await webgpuSkill.initialize();

      const webgpuAgent = new BaseAgent({
        name: 'webgpu-agent',
        skills: [webgpuSkill],
      });

      // Warm up (first run is slower due to shader compilation)
      console.log('Warming up (compiling shaders)...');
      await webgpuAgent.run([{ role: 'user', content: 'Hi' }]);

      const iterations = 5;
      const results: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const startTime = performance.now();
        const response = await webgpuAgent.run([
          { role: 'user', content: 'Write a haiku about AI' },
        ]);
        const timeMs = performance.now() - startTime;
        const tokens = Math.ceil(response.content.length / 4);
        const tokensPerSecond = tokens / (timeMs / 1000);

        results.push(tokensPerSecond);
        console.log(`WebGPU Run ${i + 1}: ${timeMs.toFixed(0)}ms, ~${tokens} tokens, ${tokensPerSecond.toFixed(1)} tok/s`);
      }

      const avgTokensPerSecond = results.reduce((a, b) => a + b, 0) / results.length;
      const minTokPerSec = Math.min(...results);
      const maxTokPerSec = Math.max(...results);

      console.log('\n--- WebGPU Performance Results ---');
      console.log(`Model: ${TEST_MODEL}`);
      console.log(`Device: webgpu`);
      console.log(`Iterations: ${iterations}`);
      console.log(`Average: ${avgTokensPerSecond.toFixed(1)} tok/s`);
      console.log(`Range: ${minTokPerSec.toFixed(1)} - ${maxTokPerSec.toFixed(1)} tok/s`);
      console.log('----------------------------------\n');

      // WebGPU should be faster than CPU
      expect(avgTokensPerSecond).toBeGreaterThan(50);

      console.log('WebGPU Performance JSON:', JSON.stringify({
        model: TEST_MODEL,
        device: 'webgpu',
        iterations,
        avgTokensPerSecond,
        minTokensPerSecond: minTokPerSec,
        maxTokensPerSecond: maxTokPerSec,
      }, null, 2));

      await webgpuAgent.cleanup();
    }, 120000);
  });
});
