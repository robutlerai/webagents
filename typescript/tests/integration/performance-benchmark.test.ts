/**
 * CPU Performance Benchmark (Node.js)
 * 
 * Tests Transformers.js performance in Node.js environment.
 * This provides a baseline for CPU performance comparison.
 * 
 * Run: npm test -- tests/integration/performance-benchmark.test.ts
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TransformersSkill } from '../../src/skills/llm/transformers/skill.js';
import { BaseAgent } from '../../src/core/agent.js';

const SMALL_MODEL = 'Xenova/distilgpt2';

describe('Performance Benchmark (Node.js)', () => {
  let skill: TransformersSkill;
  let agent: BaseAgent;

  beforeAll(async () => {
    console.log('\n' + '='.repeat(60));
    console.log('  CPU PERFORMANCE BENCHMARK');
    console.log('='.repeat(60));
    console.log(`\nModel: ${SMALL_MODEL}`);
    console.log('Device: CPU (Node.js)\n');
    console.log('Loading model (first run downloads, ~50MB)...\n');

    skill = new TransformersSkill({
      model: SMALL_MODEL,
      device: 'cpu',
      max_tokens: 50,
      temperature: 0.7,
    });

    await skill.initialize();

    agent = new BaseAgent({
      name: 'benchmark-agent',
      skills: [skill],
    });

    console.log('Model loaded!\n');
  }, 120000);

  afterAll(async () => {
    await agent?.cleanup();
  });

  it('benchmarks CPU inference speed', async () => {
    console.log('=== CPU Inference Benchmark ===\n');
    
    // Warm up
    console.log('Warming up (2 runs)...');
    for (let i = 0; i < 2; i++) {
      await agent.run([{ role: 'user', content: 'Hi' }]);
    }
    console.log('Warm up complete\n');

    // Benchmark
    const iterations = 10;
    const results: { timeMs: number; tokens: number; tokPerSec: number }[] = [];

    console.log(`Running ${iterations} iterations...\n`);

    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now();
      const response = await agent.run([
        { role: 'user', content: 'Once upon a time in a magical kingdom' },
      ]);
      const timeMs = performance.now() - startTime;
      const tokens = Math.ceil(response.content.length / 4);
      const tokPerSec = tokens / (timeMs / 1000);

      results.push({ timeMs, tokens, tokPerSec });
      console.log(`  Run ${i + 1}: ${timeMs.toFixed(0)}ms, ${tokens} tokens, ${tokPerSec.toFixed(1)} tok/s`);
    }

    // Calculate statistics
    const times = results.map(r => r.timeMs);
    const tokPerSecs = results.map(r => r.tokPerSec);

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const avgTokPerSec = tokPerSecs.reduce((a, b) => a + b, 0) / tokPerSecs.length;
    const minTokPerSec = Math.min(...tokPerSecs);
    const maxTokPerSec = Math.max(...tokPerSecs);
    
    // Standard deviation
    const variance = tokPerSecs.reduce((sum, t) => sum + Math.pow(t - avgTokPerSec, 2), 0) / tokPerSecs.length;
    const stdDev = Math.sqrt(variance);

    // Percentiles
    const sorted = [...times].sort((a, b) => a - b);
    const p50 = sorted[Math.floor(iterations * 0.5)];
    const p90 = sorted[Math.floor(iterations * 0.9)];
    const p99 = sorted[Math.floor(iterations * 0.99)] || sorted[sorted.length - 1];

    console.log('\n' + '='.repeat(50));
    console.log('  RESULTS');
    console.log('='.repeat(50));
    console.log(`\nModel: ${SMALL_MODEL}`);
    console.log(`Device: CPU (Node.js)`);
    console.log(`Iterations: ${iterations}`);
    console.log(`\nThroughput:`);
    console.log(`  Average: ${avgTokPerSec.toFixed(1)} tok/s`);
    console.log(`  Range: ${minTokPerSec.toFixed(1)} - ${maxTokPerSec.toFixed(1)} tok/s`);
    console.log(`  Std Dev: ±${stdDev.toFixed(1)} tok/s`);
    console.log(`\nLatency:`);
    console.log(`  Average: ${avgTime.toFixed(0)}ms`);
    console.log(`  P50: ${p50.toFixed(0)}ms`);
    console.log(`  P90: ${p90.toFixed(0)}ms`);
    console.log(`  P99: ${p99.toFixed(0)}ms`);
    console.log('='.repeat(50) + '\n');

    // JSON output
    console.log('Performance JSON:');
    console.log(JSON.stringify({
      model: SMALL_MODEL,
      device: 'cpu',
      environment: 'node.js',
      iterations,
      throughput: {
        avgTokPerSec,
        minTokPerSec,
        maxTokPerSec,
        stdDev,
      },
      latency: {
        avgMs: avgTime,
        p50Ms: p50,
        p90Ms: p90,
        p99Ms: p99,
      },
    }, null, 2));

    // Assertions
    expect(avgTokPerSec).toBeGreaterThan(3);
  }, 120000);

  it('compares streaming vs non-streaming latency', async () => {
    console.log('\n=== Streaming vs Non-Streaming ===\n');

    const prompt = [{ role: 'user' as const, content: 'Count: 1, 2, 3, 4, 5' }];
    const iterations = 5;
    const nonStreamTimes: number[] = [];
    const streamFirstTokenTimes: number[] = [];
    const streamTotalTimes: number[] = [];

    for (let i = 0; i < iterations; i++) {
      // Non-streaming
      const nonStreamStart = performance.now();
      await agent.run(prompt);
      nonStreamTimes.push(performance.now() - nonStreamStart);

      // Streaming
      const streamStart = performance.now();
      let firstToken = 0;
      
      for await (const chunk of agent.runStreaming(prompt)) {
        if (chunk.type === 'delta' && !firstToken) {
          firstToken = performance.now() - streamStart;
        }
      }
      streamFirstTokenTimes.push(firstToken);
      streamTotalTimes.push(performance.now() - streamStart);
    }

    const avgNonStream = nonStreamTimes.reduce((a, b) => a + b, 0) / iterations;
    const avgFirstToken = streamFirstTokenTimes.reduce((a, b) => a + b, 0) / iterations;
    const avgStreamTotal = streamTotalTimes.reduce((a, b) => a + b, 0) / iterations;

    console.log('Non-streaming:');
    console.log(`  Average: ${avgNonStream.toFixed(0)}ms`);
    console.log('\nStreaming:');
    console.log(`  First token: ${avgFirstToken.toFixed(0)}ms`);
    console.log(`  Total: ${avgStreamTotal.toFixed(0)}ms`);
    console.log(`  Time to first token advantage: ${(avgNonStream - avgFirstToken).toFixed(0)}ms faster`);

    // Note: Transformers.js may not provide true streaming in all configurations
    // Just verify we got valid timing data
    expect(avgFirstToken).toBeGreaterThan(0);
    expect(avgStreamTotal).toBeGreaterThan(0);
  }, 60000);

  it('tests memory efficiency with multiple requests', async () => {
    console.log('\n=== Memory & Throughput Test ===\n');

    const requests = 20;
    const startMemory = process.memoryUsage();
    const startTime = performance.now();
    let totalTokens = 0;

    console.log(`Processing ${requests} sequential requests...\n`);

    for (let i = 0; i < requests; i++) {
      const response = await agent.run([
        { role: 'user', content: `Request ${i + 1}: Hello` },
      ]);
      totalTokens += Math.ceil(response.content.length / 4);
      
      if ((i + 1) % 5 === 0) {
        console.log(`  Completed ${i + 1}/${requests} requests`);
      }
    }

    const totalTimeMs = performance.now() - startTime;
    const endMemory = process.memoryUsage();
    const memoryDelta = (endMemory.heapUsed - startMemory.heapUsed) / 1024 / 1024;

    const requestsPerSecond = requests / (totalTimeMs / 1000);
    const tokensPerSecond = totalTokens / (totalTimeMs / 1000);

    console.log('\nResults:');
    console.log(`  Total requests: ${requests}`);
    console.log(`  Total time: ${(totalTimeMs / 1000).toFixed(2)}s`);
    console.log(`  Requests/sec: ${requestsPerSecond.toFixed(2)}`);
    console.log(`  Total tokens: ${totalTokens}`);
    console.log(`  Tokens/sec: ${tokensPerSecond.toFixed(1)}`);
    console.log(`  Memory delta: ${memoryDelta.toFixed(1)}MB`);

    expect(requestsPerSecond).toBeGreaterThan(0.1);
  }, 120000);
});

describe('Device Comparison (when available)', () => {
  it('lists available devices', async () => {
    console.log('\n=== Available Devices ===\n');
    console.log('Node.js Environment:');
    console.log(`  Platform: ${process.platform}`);
    console.log(`  Arch: ${process.arch}`);
    console.log(`  Node: ${process.version}`);
    
    // Check for GPU availability
    // Note: In Node.js, we primarily have CPU. GPU requires specific bindings.
    console.log('\nAvailable backends for Transformers.js in Node.js:');
    console.log('  - cpu: Always available ✅');
    console.log('  - wasm: Available in browsers');
    console.log('  - webgpu: Available in browsers with WebGPU support');
    
    console.log('\nNote: For GPU acceleration in Node.js, consider:');
    console.log('  - ONNX Runtime with CUDA/ROCm');
    console.log('  - TensorFlow.js with tfjs-node-gpu');
    console.log('  - Running in browser with WebGPU\n');

    expect(true).toBe(true);
  });
});
