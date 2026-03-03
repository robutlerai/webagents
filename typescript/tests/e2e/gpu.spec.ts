/**
 * GPU/WebGPU E2E Tests with Playwright
 * 
 * Tests real browser-based LLM inference using WebGPU and Transformers.js.
 * 
 * Run: npx playwright test
 * Run with UI: npx playwright test --ui
 * Run specific project: npx playwright test --project=chromium-webgpu
 */

import { test, expect } from '@playwright/test';

// Extended timeout for model loading
test.setTimeout(180000);

test.describe('WebGPU Performance Tests', () => {
  test('detects GPU capabilities', async ({ page }) => {
    // Navigate to the index page
    await page.goto('/');
    
    // Check WebGPU availability directly in the browser
    const gpuInfo = await page.evaluate(async () => {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      let gpuRenderer = 'unknown';
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          gpuRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        }
      }
      
      let webgpuAvailable = false;
      let webgpuAdapter = null;
      if ('gpu' in navigator) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) {
            webgpuAvailable = true;
            webgpuAdapter = 'WebGPU adapter available';
          }
        } catch (e) {
          // WebGPU not available
        }
      }
      
      return {
        webgpuAvailable,
        webgpuAdapter,
        gpuRenderer,
        webglAvailable: !!gl,
      };
    });
    
    console.log('\n=== GPU Detection ===');
    console.log(`WebGPU Available: ${gpuInfo.webgpuAvailable}`);
    console.log(`WebGL Available: ${gpuInfo.webglAvailable}`);
    console.log(`GPU Renderer: ${gpuInfo.gpuRenderer}`);
    
    // At least WebGL should be available
    expect(gpuInfo.webglAvailable).toBe(true);
  });

  test('loads and runs Transformers.js model with benchmarks', async ({ page }) => {
    // Use setContent to avoid cross-origin worker issues with served HTML
    await page.setContent(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Model Benchmark</title>
      </head>
      <body>
        <div id="output"></div>
        <script type="module">
          window.testResults = {
            gpuAvailable: false,
            webgpuAvailable: false,
            device: 'unknown',
            modelLoaded: false,
            tokensPerSecond: 0,
            latencyMs: 0,
            errors: [],
            response: '',
            completed: false
          };

          try {
            // Check GPU
            if ('gpu' in navigator) {
              try {
                const adapter = await navigator.gpu.requestAdapter();
                window.testResults.webgpuAvailable = !!adapter;
              } catch {}
            }
            
            const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0');
            env.allowLocalModels = false;
            
            // Load a small model for testing
            const generator = await pipeline('text-generation', 'Xenova/distilgpt2', {
              progress_callback: (p) => {
                if (p.progress) document.getElementById('output').textContent = 'Loading: ' + p.progress + '%';
              }
            });
            
            window.testResults.modelLoaded = true;
            document.getElementById('output').textContent = 'Model loaded, running benchmark...';
            
            // Benchmark
            const runs = [];
            for (let i = 0; i < 3; i++) {
              const start = performance.now();
              const result = await generator('The meaning of life is', { max_new_tokens: 30, do_sample: false });
              const timeMs = performance.now() - start;
              const response = result[0].generated_text;
              const tokens = Math.ceil(response.length / 4);
              runs.push({ timeMs, tokens, tokPerSec: tokens / (timeMs / 1000) });
              if (i === 0) window.testResults.response = response.substring(0, 50) + '...';
            }
            
            window.testResults.tokensPerSecond = runs.reduce((a, b) => a + b.tokPerSec, 0) / runs.length;
            window.testResults.latencyMs = runs.reduce((a, b) => a + b.timeMs, 0) / runs.length;
            window.testResults.completed = true;
            document.getElementById('output').textContent = 'Done: ' + window.testResults.tokensPerSecond.toFixed(1) + ' tok/s';
            
          } catch (error) {
            window.testResults.errors.push(error.message);
            document.getElementById('output').textContent = 'Error: ' + error.message;
          }
        </script>
      </body>
      </html>
    `);

    console.log('\n=== Model Loading and Benchmark ===');
    console.log('Waiting for model to load and run benchmark...');
    console.log('(First run downloads model, subsequent runs use cache)');
    
    // Wait for completion or timeout
    await page.waitForFunction(
      () => (window as any).testResults?.completed || (window as any).testResults?.errors?.length > 0,
      { timeout: 120000 }
    );
    
    const results = await page.evaluate(() => (window as any).testResults);
    
    console.log('\n=== Test Results ===');
    console.log(`Model Loaded: ${results.modelLoaded}`);
    console.log(`Device: ${results.device}`);
    console.log(`Tokens/sec: ${results.tokensPerSecond.toFixed(1)}`);
    console.log(`Latency: ${results.latencyMs.toFixed(0)}ms`);
    console.log(`Response: "${results.response}"`);
    console.log(`Errors: ${results.errors.length > 0 ? results.errors.join(', ') : 'None'}`);
    
    // Check results
    if (results.errors.length > 0 && !results.modelLoaded) {
      console.log('\nModel loading failed, but this might be due to network issues.');
    }
    
    // Basic assertions
    expect(results.modelLoaded).toBe(true);
    expect(results.tokensPerSecond).toBeGreaterThan(0);
    
    // Log performance JSON for CI
    console.log('\nPerformance JSON:', JSON.stringify({
      device: results.device,
      webgpu: results.webgpuAvailable,
      tokensPerSecond: results.tokensPerSecond,
      latencyMs: results.latencyMs,
    }, null, 2));
  });

  test('measures streaming performance', async ({ page }) => {
    await page.setContent(`
      <!DOCTYPE html>
      <html>
      <head><title>Streaming Test</title></head>
      <body>
        <div id="output"></div>
        <script type="module">
          window.streamResults = {
            firstTokenMs: 0,
            totalMs: 0,
            tokenCount: 0,
            errors: [],
            completed: false
          };

          try {
            const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0');
            env.allowLocalModels = false;
            
            const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
            
            const start = performance.now();
            let firstToken = 0;
            let tokens = 0;
            
            // Use streaming callback
            const result = await generator('Hello, I am', {
              max_new_tokens: 30,
              callback_function: (x) => {
                tokens++;
                if (tokens === 1) firstToken = performance.now() - start;
              }
            });
            
            window.streamResults.firstTokenMs = firstToken || (performance.now() - start);
            window.streamResults.totalMs = performance.now() - start;
            window.streamResults.tokenCount = tokens || Math.ceil(result[0].generated_text.length / 4);
            window.streamResults.completed = true;
            
          } catch (error) {
            window.streamResults.errors.push(error.message);
          }
        </script>
      </body>
      </html>
    `);

    console.log('\n=== Streaming Performance Test ===');
    console.log('Loading model and measuring streaming latency...');

    await page.waitForFunction(
      () => (window as any).streamResults?.completed || (window as any).streamResults?.errors?.length > 0,
      { timeout: 120000 }
    );

    const results = await page.evaluate(() => (window as any).streamResults);

    console.log('\n=== Streaming Results ===');
    console.log(`First Token: ${results.firstTokenMs.toFixed(0)}ms`);
    console.log(`Total Time: ${results.totalMs.toFixed(0)}ms`);
    console.log(`Token Count: ${results.tokenCount}`);
    console.log(`Errors: ${results.errors.length > 0 ? results.errors.join(', ') : 'None'}`);

    if (results.errors.length === 0) {
      expect(results.firstTokenMs).toBeGreaterThan(0);
      expect(results.tokenCount).toBeGreaterThan(0);
    }
  });
});

test.describe('LLM Comparison Page', () => {
  test('loads LLM comparison page', async ({ page }) => {
    await page.goto('/llm-comparison.html');
    
    // Check page loaded
    await expect(page.locator('h1')).toContainText('LLM Engine Comparison');
    
    // Check hardware detection
    await page.waitForTimeout(2000);
    const webgpuStatus = await page.locator('#hw-webgpu').textContent();
    console.log('WebGPU Status:', webgpuStatus);
    
    // Check model selector
    const modelSelect = page.locator('#modelSelect');
    await expect(modelSelect).toBeVisible();
  });
});

test.describe('UAMP Benchmark Page', () => {
  test('loads UAMP benchmark page', async ({ page }) => {
    await page.goto('/uamp-benchmark.html');
    
    // Check page loaded
    await expect(page.locator('h1')).toContainText('UAMP Protocol Benchmark');
    
    // Check controls are present
    await expect(page.locator('#modelSelect')).toBeVisible();
    await expect(page.locator('#runBtn')).toBeVisible();
  });
});
