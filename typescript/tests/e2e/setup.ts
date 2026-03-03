/**
 * E2E Test Setup
 * 
 * Configuration and utilities for end-to-end testing with Playwright.
 */

// This file sets up the E2E testing infrastructure.
// To run E2E tests, you need:
// 1. Install Playwright: npm install -D @playwright/test
// 2. Install browsers: npx playwright install

export interface TestAgent {
  url: string;
  name: string;
  port: number;
}

/**
 * Default test configuration
 */
export const testConfig = {
  /** Base URL for the test server */
  baseUrl: process.env.TEST_BASE_URL || 'http://localhost:3000',
  
  /** Timeout for requests */
  timeout: 30000,
  
  /** Whether to run in headless mode */
  headless: process.env.CI === 'true',
};

/**
 * Create a test server URL
 */
export function getTestUrl(path: string): string {
  return `${testConfig.baseUrl}${path}`;
}

/**
 * Wait for a condition with timeout
 */
export async function waitFor(
  condition: () => boolean | Promise<boolean>,
  timeout = 5000,
  interval = 100
): Promise<void> {
  const start = Date.now();
  
  while (Date.now() - start < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  throw new Error(`Timeout waiting for condition after ${timeout}ms`);
}

/**
 * Mock WebSocket for testing
 */
export class MockWebSocket {
  readyState = 1;
  messages: string[] = [];
  onmessage?: (event: { data: string }) => void;
  onopen?: () => void;
  onclose?: () => void;
  onerror?: (error: Error) => void;
  
  send(data: string): void {
    this.messages.push(data);
  }
  
  close(): void {
    this.readyState = 3;
    this.onclose?.();
  }
  
  simulateMessage(data: string): void {
    this.onmessage?.({ data });
  }
  
  simulateOpen(): void {
    this.readyState = 1;
    this.onopen?.();
  }
  
  simulateError(error: Error): void {
    this.onerror?.(error);
  }
}

/**
 * Create mock fetch responses
 */
export function createMockResponse(
  body: unknown,
  options: { status?: number; headers?: Record<string, string> } = {}
): Response {
  const { status = 200, headers = {} } = options;
  
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  });
}

/**
 * Create SSE stream from events
 */
export function createSSEStream(events: unknown[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  
  return new ReadableStream({
    pull(controller) {
      if (index < events.length) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(events[index])}\n\n`));
        index++;
      } else {
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      }
    },
  });
}
