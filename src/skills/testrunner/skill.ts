/**
 * TestRunnerSkill - Agentic compliance test runner.
 * 
 * This skill provides tools for an AI agent to run compliance tests
 * against WebAgents SDK implementations, with full browser automation support.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import { TestParser } from './parser.js';
import { StrictValidator } from './validator.js';
import { BrowserAutomationSkill } from '../browser/automation.js';
import type {
  TestRunnerConfig,
  TestSpec,
  TestResult,
  TestSuiteResult,
  HttpResponse,
  AssertionResult,
} from './types.js';
import type { ElementInfo, ScreenshotResult } from '../browser/automation.js';

/**
 * TestRunnerSkill
 * 
 * Provides tools for an AI agent to run compliance tests:
 * 1. Load test specifications from markdown files/content
 * 2. Execute HTTP requests against the SDK under test
 * 3. Validate responses against natural language and strict assertions
 * 4. Report detailed test results
 * 5. Browser automation for UI testing (click, type, screenshot, etc.)
 * 
 * @example
 * ```typescript
 * const skill = new TestRunnerSkill({
 *   baseUrl: 'http://localhost:8765',
 *   timeout: 60,
 *   enableBrowser: true, // Enable browser automation
 * });
 * 
 * // Load and parse a test spec
 * const spec = await skill.loadTestContent(markdownContent);
 * 
 * // Make HTTP requests
 * const response = await skill.httpRequest('POST', '/chat/completions', { ... });
 * 
 * // Browser automation for UI testing
 * await skill.browserClick('#submit-btn');
 * await skill.browserType('#email', 'user@example.com');
 * const screenshot = await skill.browserScreenshot();
 * 
 * // Validate responses
 * const result = await skill.validateStrict(response, { status: 200 });
 * ```
 */
export class TestRunnerSkill extends Skill {
  private testConfig: Required<TestRunnerConfig>;
  private results: TestSuiteResult[] = [];
  private parser: TestParser;
  private browser: BrowserAutomationSkill;

  /** Unique skill identifier */
  readonly id: string = 'testrunner';
  
  /** Skill description */
  readonly description: string = 'Run compliance tests against WebAgents SDK implementations with browser automation';

  constructor(config: TestRunnerConfig = {}) {
    super({ name: 'Test Runner' });
    this.testConfig = {
      baseUrl: config.baseUrl || 'http://localhost:8765',
      cacheDir: config.cacheDir || undefined,
      timeout: config.timeout || 60,
    } as Required<TestRunnerConfig>;
    this.parser = new TestParser();
    this.browser = new BrowserAutomationSkill();
  }

  /**
   * Get the browser automation skill for direct access
   */
  get browserAutomation(): BrowserAutomationSkill {
    return this.browser;
  }

  // ============================================================================
  // Tools
  // ============================================================================

  /**
   * Send HTTP request to the SDK server under test.
   */
  @tool({
    name: 'http_request',
    description: 'Send HTTP request to the SDK server under test',
    parameters: {
      method: {
        type: 'string',
        description: 'HTTP method (GET, POST, PUT, DELETE)',
      },
      path: {
        type: 'string',
        description: 'URL path (e.g., /chat/completions)',
      },
      body: {
        type: 'object',
        description: 'Request body (for POST/PUT)',
      },
      headers: {
        type: 'object',
        description: 'Additional headers',
      },
      stream: {
        type: 'boolean',
        description: 'Whether to expect SSE streaming response',
      },
    },
  })
  async httpRequest(
    method: string,
    path: string,
    body?: Record<string, unknown>,
    headers?: Record<string, string>,
    stream: boolean = false
  ): Promise<HttpResponse> {
    const url = `${this.testConfig.baseUrl}${path}`;

    const requestHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      ...headers,
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.testConfig.timeout * 1000);

      const response = await fetch(url, {
        method: method.toUpperCase(),
        headers: requestHeaders,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const contentType = response.headers.get('content-type') || '';

      if (stream && contentType.includes('text/event-stream')) {
        // Collect SSE chunks
        const chunks: unknown[] = [];
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (reader) {
          let done = false;
          while (!done) {
            const result = await reader.read();
            done = result.done;

            if (result.value) {
              const text = decoder.decode(result.value, { stream: true });
              const lines = text.split('\n');

              for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data: ')) {
                  const data = trimmed.slice(6);
                  if (data !== '[DONE]') {
                    try {
                      chunks.push(JSON.parse(data));
                    } catch {
                      chunks.push({ raw: data });
                    }
                  }
                }
              }
            }
          }
        }

        return {
          status: response.status,
          headers: Object.fromEntries(response.headers.entries()),
          body: chunks,
          format: 'sse',
        };
      } else {
        let responseBody: unknown;
        try {
          responseBody = await response.json();
        } catch {
          responseBody = await response.text();
        }

        return {
          status: response.status,
          headers: Object.fromEntries(response.headers.entries()),
          body: responseBody,
        };
      }
    } catch (error) {
      return {
        status: 0,
        headers: {},
        body: null,
        error: error instanceof Error ? error.message : String(error),
        errorType: error instanceof Error ? error.constructor.name : 'Error',
      };
    }
  }

  /**
   * Load and parse a compliance test specification from markdown content.
   */
  @tool({
    name: 'load_test',
    description: 'Load and parse a compliance test specification from markdown content',
    parameters: {
      content: {
        type: 'string',
        description: 'Markdown content of the test file',
      },
      filename: {
        type: 'string',
        description: 'Optional filename for reference',
      },
    },
  })
  async loadTestContent(content: string, filename: string = 'test.md'): Promise<TestSpec> {
    return this.parser.parse(content, filename);
  }

  /**
   * Validate a response against a natural language assertion using AI reasoning.
   */
  @tool({
    name: 'validate_assertion',
    description: 'Validate a response against a natural language assertion using AI reasoning',
    parameters: {
      response: {
        type: 'object',
        description: 'The HTTP response to validate',
      },
      assertion: {
        type: 'string',
        description: 'Natural language assertion (e.g., "Response contains a greeting")',
      },
    },
  })
  async validateAssertion(
    response: HttpResponse,
    assertion: string
  ): Promise<{
    assertion: string;
    responseSummary: string;
    instruction: string;
  }> {
    // This is the key method that uses LLM reasoning
    // The agent calling this tool will use its own judgment
    // to determine if the assertion passes

    return {
      assertion,
      responseSummary: this.summarizeResponse(response),
      instruction:
        'Evaluate whether the response satisfies this assertion. ' +
        'Consider the intent, not just exact matching. ' +
        'Return your assessment as passed: true/false with reasoning.',
    };
  }

  /**
   * Validate a response against strict deterministic assertions.
   */
  @tool({
    name: 'validate_strict',
    description: 'Validate a response against strict deterministic assertions',
    parameters: {
      response: {
        type: 'object',
        description: 'The HTTP response to validate',
      },
      assertions: {
        type: 'object',
        description: 'Strict assertion rules (YAML-like object)',
      },
    },
  })
  async validateStrict(
    response: HttpResponse,
    assertions: Record<string, unknown>
  ): Promise<{
    passed: boolean;
    results: AssertionResult[];
  }> {
    const validator = new StrictValidator();
    return validator.validate(response, assertions);
  }

  /**
   * Report the result of a test case.
   */
  @tool({
    name: 'report_result',
    description: 'Report the result of a test case',
    parameters: {
      testName: {
        type: 'string',
        description: 'Name of the test suite',
      },
      caseName: {
        type: 'string',
        description: 'Name of the specific test case',
      },
      passed: {
        type: 'boolean',
        description: 'Whether the test passed',
      },
      details: {
        type: 'string',
        description: 'Details about the result',
      },
      assertions: {
        type: 'array',
        description: 'Individual assertion results',
      },
    },
  })
  async reportResult(
    testName: string,
    caseName: string,
    passed: boolean,
    details: string,
    assertions: AssertionResult[] = []
  ): Promise<{
    recorded: boolean;
    test: string;
    case: string;
    passed: boolean;
  }> {
    const result: TestResult = {
      name: caseName,
      passed,
      assertions,
      details,
      durationMs: 0,
    };

    // Find or create suite result
    let suite = this.results.find(s => s.name === testName);
    if (!suite) {
      suite = {
        name: testName,
        testFile: '',
        cases: [],
        passed: 0,
        failed: 0,
        skipped: 0,
      };
      this.results.push(suite);
    }

    suite.cases.push(result);
    if (passed) {
      suite.passed += 1;
    } else {
      suite.failed += 1;
    }

    return {
      recorded: true,
      test: testName,
      case: caseName,
      passed,
    };
  }

  /**
   * Get a summary of all test results so far.
   */
  @tool({
    name: 'get_results_summary',
    description: 'Get a summary of all test results so far',
    parameters: {},
  })
  async getResultsSummary(): Promise<{
    total: number;
    passed: number;
    failed: number;
    skipped: number;
    suites: Array<{
      name: string;
      passed: number;
      failed: number;
      cases: Array<{
        name: string;
        passed: boolean;
        details: string;
      }>;
    }>;
  }> {
    const totalPassed = this.results.reduce((sum, s) => sum + s.passed, 0);
    const totalFailed = this.results.reduce((sum, s) => sum + s.failed, 0);
    const totalSkipped = this.results.reduce((sum, s) => sum + s.skipped, 0);

    return {
      total: totalPassed + totalFailed + totalSkipped,
      passed: totalPassed,
      failed: totalFailed,
      skipped: totalSkipped,
      suites: this.results.map(s => ({
        name: s.name,
        passed: s.passed,
        failed: s.failed,
        cases: s.cases.map(c => ({
          name: c.name,
          passed: c.passed,
          details: c.details,
        })),
      })),
    };
  }

  /**
   * Reset all test results.
   */
  resetResults(): void {
    this.results = [];
  }

  // ============================================================================
  // Browser Automation Tools
  // ============================================================================

  /**
   * Click on a DOM element
   */
  @tool({
    name: 'browser_click',
    description: 'Click on a DOM element by CSS selector',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to click',
      },
    },
  })
  async browserClick(selector: string): Promise<{ success: boolean; error?: string }> {
    return this.browser.click(selector);
  }

  /**
   * Type text into an input element
   */
  @tool({
    name: 'browser_type',
    description: 'Type text into an input element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of input element',
      },
      text: {
        type: 'string',
        description: 'Text to type',
      },
      clear: {
        type: 'boolean',
        description: 'Clear existing text first (default: true)',
      },
    },
  })
  async browserType(
    selector: string,
    text: string,
    clear: boolean = true
  ): Promise<{ success: boolean; error?: string }> {
    return this.browser.type(selector, text, clear);
  }

  /**
   * Query a DOM element and get its info
   */
  @tool({
    name: 'browser_query',
    description: 'Query a DOM element by CSS selector and get its information',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to find the element',
      },
    },
  })
  async browserQuery(selector: string): Promise<{ element: ElementInfo | null; error?: string }> {
    return this.browser.queryElement(selector);
  }

  /**
   * Wait for an element to appear
   */
  @tool({
    name: 'browser_wait_for',
    description: 'Wait for an element to appear in the DOM',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to wait for',
      },
      timeout: {
        type: 'number',
        description: 'Timeout in milliseconds (default: 10000)',
      },
      visible: {
        type: 'boolean',
        description: 'Wait for element to be visible (default: false)',
      },
    },
  })
  async browserWaitFor(
    selector: string,
    timeout: number = 10000,
    visible: boolean = false
  ): Promise<{ found: boolean; element?: ElementInfo; error?: string }> {
    return this.browser.waitForElement(selector, timeout, visible);
  }

  /**
   * Take a screenshot
   */
  @tool({
    name: 'browser_screenshot',
    description: 'Take a screenshot of the page or a specific element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element to capture (optional)',
      },
    },
  })
  async browserScreenshot(selector?: string): Promise<ScreenshotResult | { error: string }> {
    return this.browser.screenshot(selector);
  }

  /**
   * Get text content of an element
   */
  @tool({
    name: 'browser_get_text',
    description: 'Get text content of a DOM element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of element',
      },
    },
  })
  async browserGetText(selector: string): Promise<{ text: string; error?: string }> {
    return this.browser.getText(selector);
  }

  /**
   * Get value of an input element
   */
  @tool({
    name: 'browser_get_value',
    description: 'Get value of an input element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of input element',
      },
    },
  })
  async browserGetValue(selector: string): Promise<{ value: string; error?: string }> {
    return this.browser.getValue(selector);
  }

  /**
   * Execute JavaScript in page context
   */
  @tool({
    name: 'browser_evaluate',
    description: 'Execute JavaScript code in the page context',
    parameters: {
      script: {
        type: 'string',
        description: 'JavaScript code to evaluate',
      },
    },
  })
  async browserEvaluate(script: string): Promise<{ result: unknown; error?: string }> {
    return this.browser.evaluate(script);
  }

  /**
   * Select an option from a dropdown
   */
  @tool({
    name: 'browser_select',
    description: 'Select an option from a dropdown element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of select element',
      },
      value: {
        type: 'string',
        description: 'Value or text of option to select',
      },
      byText: {
        type: 'boolean',
        description: 'Select by visible text instead of value (default: false)',
      },
    },
  })
  async browserSelect(
    selector: string,
    value: string,
    byText: boolean = false
  ): Promise<{ success: boolean; selectedValue?: string; error?: string }> {
    return this.browser.select(selector, value, byText);
  }

  /**
   * Press a keyboard key
   */
  @tool({
    name: 'browser_press_key',
    description: 'Press a keyboard key (Enter, Tab, Escape, etc.)',
    parameters: {
      key: {
        type: 'string',
        description: 'Key to press (Enter, Tab, Escape, ArrowUp, etc.)',
      },
      selector: {
        type: 'string',
        description: 'Optional selector to focus before pressing',
      },
    },
  })
  async browserPressKey(
    key: string,
    selector?: string
  ): Promise<{ success: boolean; error?: string }> {
    return this.browser.pressKey(key, selector);
  }

  /**
   * Scroll the page or to an element
   */
  @tool({
    name: 'browser_scroll',
    description: 'Scroll the page or scroll to an element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector to scroll to (optional)',
      },
      x: {
        type: 'number',
        description: 'Horizontal scroll offset',
      },
      y: {
        type: 'number',
        description: 'Vertical scroll offset',
      },
    },
  })
  async browserScroll(
    selector?: string,
    x?: number,
    y?: number
  ): Promise<{ success: boolean; error?: string }> {
    return this.browser.scroll(selector, x, y);
  }

  /**
   * Get current page URL
   */
  @tool({
    name: 'browser_get_url',
    description: 'Get the current page URL',
    parameters: {},
  })
  async browserGetUrl(): Promise<{ url: string; origin: string; pathname: string; search: string; hash: string }> {
    return this.browser.getUrl();
  }

  /**
   * Check or uncheck a checkbox
   */
  @tool({
    name: 'browser_check',
    description: 'Check or uncheck a checkbox element',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector of checkbox element',
      },
      checked: {
        type: 'boolean',
        description: 'Whether to check (true) or uncheck (false)',
      },
    },
  })
  async browserCheck(
    selector: string,
    checked: boolean = true
  ): Promise<{ success: boolean; checked?: boolean; error?: string }> {
    return this.browser.check(selector, checked);
  }

  /**
   * Start capturing console logs
   */
  @tool({
    name: 'browser_start_console_capture',
    description: 'Start capturing console logs for debugging',
    parameters: {},
  })
  async browserStartConsoleCapture(): Promise<{ success: boolean }> {
    return this.browser.startConsoleCapture();
  }

  /**
   * Get captured console logs
   */
  @tool({
    name: 'browser_get_console_logs',
    description: 'Get captured console logs',
    parameters: {
      level: {
        type: 'string',
        description: 'Filter by level: log, warn, error, info, debug',
      },
      clear: {
        type: 'boolean',
        description: 'Clear logs after reading (default: false)',
      },
    },
  })
  async browserGetConsoleLogs(
    level?: string,
    clear: boolean = false
  ): Promise<{ logs: Array<{ level: string; message: string; timestamp: number }> }> {
    return this.browser.getConsoleLogs(level, clear);
  }

  /**
   * Stop capturing console logs
   */
  @tool({
    name: 'browser_stop_console_capture',
    description: 'Stop capturing console logs',
    parameters: {},
  })
  async browserStopConsoleCapture(): Promise<{ success: boolean }> {
    return this.browser.stopConsoleCapture();
  }

  /**
   * Get HTML content of an element
   */
  @tool({
    name: 'browser_get_html',
    description: 'Get HTML content of an element or the page',
    parameters: {
      selector: {
        type: 'string',
        description: 'CSS selector (optional, gets document.body)',
      },
      outer: {
        type: 'boolean',
        description: 'Include outer element HTML (default: true)',
      },
    },
  })
  async browserGetHtml(
    selector?: string,
    outer: boolean = true
  ): Promise<{ html: string; error?: string }> {
    return this.browser.getHtml(selector, outer);
  }

  /**
   * Get network performance entries
   */
  @tool({
    name: 'browser_get_network',
    description: 'Get network request entries from Performance API',
    parameters: {
      limit: {
        type: 'number',
        description: 'Maximum entries to return (default: 100)',
      },
    },
  })
  async browserGetNetwork(
    limit: number = 100
  ): Promise<{ entries: Array<{ name: string; duration: number; transferSize?: number }>; error?: string }> {
    return this.browser.getNetworkEntries(undefined, limit);
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  /**
   * Create a summary of a response for assertion validation.
   */
  private summarizeResponse(response: HttpResponse): string {
    const status = response.status || 'unknown';
    const body = response.body;

    if (Array.isArray(body)) {
      return `Status: ${status}, SSE stream with ${body.length} chunks`;
    } else if (typeof body === 'object' && body !== null) {
      const bodyObj = body as Record<string, unknown>;
      if ('choices' in bodyObj) {
        const choices = bodyObj.choices as Array<{ message?: { content?: string } }>;
        const content = choices?.[0]?.message?.content || '';
        return `Status: ${status}, Assistant response: ${content.slice(0, 200)}...`;
      } else if ('error' in bodyObj) {
        return `Status: ${status}, Error: ${bodyObj.error}`;
      } else {
        return `Status: ${status}, Body: ${JSON.stringify(body).slice(0, 200)}...`;
      }
    } else {
      return `Status: ${status}, Body: ${String(body).slice(0, 200)}...`;
    }
  }
}
