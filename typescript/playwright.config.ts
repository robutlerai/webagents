import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for E2E tests with WebGPU support
 * 
 * GPU/WebGPU headless testing based on:
 * - https://developer.chrome.com/blog/supercharge-web-ai-testing (Chrome DevRel)
 * - https://blog.promaton.com/testing-3d-applications-with-playwright-on-gpu-1e9cfc8b54a9
 * 
 * Run: npx playwright test
 * Run with UI: npx playwright test --ui
 * Run headed: xvfb-run npx playwright test --headed (Linux)
 */

// Chromium flags for WebGPU on Linux (headless)
// Source: https://developer.chrome.com/blog/supercharge-web-ai-testing#enable-webgpu
const chromiumWebGPULinuxFlags = [
  '--no-sandbox',
  '--headless=new',
  '--use-angle=vulkan',
  '--enable-features=Vulkan',
  '--disable-vulkan-surface',
  '--enable-unsafe-webgpu',
];

// Chromium flags for macOS (Metal backend)
const chromiumWebGPUMacFlags = [
  '--enable-features=Vulkan,WebGPU,WebGPUDeveloperFeatures',
  '--enable-unsafe-webgpu',
  '--use-angle=metal',
  '--enable-gpu-rasterization',
  '--ignore-gpu-blocklist',
];

// Common WebGPU flags for all platforms
const commonWebGPUFlags = [
  '--enable-features=WebGPU',
  '--enable-unsafe-webgpu',
  '--ignore-gpu-blocklist',
];

export default defineConfig({
  testDir: './tests/e2e',
  testMatch: '**/*.spec.ts',
  fullyParallel: false, // GPU tests should run sequentially
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // Single worker for GPU resource management
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report' }],
  ],
  timeout: 300000, // 5 minutes for model loading
  
  use: {
    baseURL: 'http://localhost:3456',
    // Enable traces for stability (helps with Firefox on Linux)
    trace: 'retain-on-failure',
    video: process.env.CI ? 'retain-on-failure' : 'off',
  },

  projects: [
    // Chromium with WebGPU (auto-detects platform, supports headless)
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chromium', // New headless mode (v1.49+)
        launchOptions: {
          args: [
            ...(process.platform === 'linux' ? chromiumWebGPULinuxFlags : []),
            ...(process.platform === 'darwin' ? chromiumWebGPUMacFlags : []),
            ...commonWebGPUFlags,
          ],
        },
      },
    },
    
    // Chrome stable (uses installed Chrome, best WebGPU support)
    {
      name: 'chrome-webgpu',
      use: {
        channel: 'chrome',
        launchOptions: {
          args: [
            ...commonWebGPUFlags,
            '--use-gl=egl',
            ...(process.platform === 'darwin' ? ['--use-angle=metal'] : []),
          ],
        },
      },
    },
    
    // Firefox (requires xvfb-run on Linux for GPU)
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            'dom.webgpu.enabled': true,
            'webgl.force-enabled': true,
          },
        },
      },
    },
    
    // WebKit (requires --headed for GPU acceleration)
    {
      name: 'webkit',
      use: {
        ...devices['Desktop Safari'],
      },
    },
  ],

  // Local dev server for tests
  webServer: {
    command: 'npx tsx tests/e2e/server.ts',
    url: 'http://localhost:3456',
    reuseExistingServer: true, // Always reuse if running
    timeout: 30000,
  },
});
