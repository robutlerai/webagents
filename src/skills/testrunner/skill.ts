/**
 * TestRunnerSkill - Agentic compliance test runner.
 * 
 * This skill provides tools for an AI agent to run compliance tests
 * against WebAgents SDK implementations.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import { TestParser } from './parser.js';
import { StrictValidator } from './validator.js';
import type {
  TestRunnerConfig,
  TestSpec,
  TestResult,
  TestSuiteResult,
  HttpResponse,
  AssertionResult,
} from './types.js';

/**
 * TestRunnerSkill
 * 
 * Provides tools for an AI agent to run compliance tests:
 * 1. Load test specifications from markdown files/content
 * 2. Execute HTTP requests against the SDK under test
 * 3. Validate responses against natural language and strict assertions
 * 4. Report detailed test results
 * 
 * @example
 * ```typescript
 * const skill = new TestRunnerSkill({
 *   baseUrl: 'http://localhost:8765',
 *   timeout: 60,
 * });
 * 
 * // Load and parse a test spec
 * const spec = await skill.loadTestContent(markdownContent);
 * 
 * // Make HTTP requests
 * const response = await skill.httpRequest('POST', '/chat/completions', { ... });
 * 
 * // Validate responses
 * const result = await skill.validateStrict(response, { status: 200 });
 * ```
 */
export class TestRunnerSkill extends Skill {
  private testConfig: Required<TestRunnerConfig>;
  private results: TestSuiteResult[] = [];
  private parser: TestParser;

  /** Unique skill identifier */
  readonly id: string = 'testrunner';
  
  /** Skill description */
  readonly description: string = 'Run compliance tests against WebAgents SDK implementations';

  constructor(config: TestRunnerConfig = {}) {
    super({ name: 'Test Runner' });
    this.testConfig = {
      baseUrl: config.baseUrl || 'http://localhost:8765',
      cacheDir: config.cacheDir || undefined,
      timeout: config.timeout || 60,
    } as Required<TestRunnerConfig>;
    this.parser = new TestParser();
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
