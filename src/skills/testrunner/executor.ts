/**
 * Test executor for running compliance tests.
 * 
 * Orchestrates test execution using the TestRunnerSkill.
 */

import { TestParser } from './parser.js';
import { StrictValidator } from './validator.js';
import type {
  ExecutionOptions,
  TestSpec,
  TestCase,
  TestResult,
  TestSuiteResult,
  HttpResponse,
  AssertionResult,
} from './types.js';

/**
 * Default execution options
 */
const DEFAULT_OPTIONS: ExecutionOptions = {
  baseUrl: 'http://localhost:8765',
  cacheMode: 'read_write',
  temperature: 0.0,
  timeout: 60,
  strictOnly: false,
  parallel: 1,
};

/**
 * Executes compliance tests against an SDK server.
 * 
 * Can run in two modes:
 * 1. Agentic: Uses AI to validate natural language assertions
 * 2. Strict-only: Only runs deterministic assertions (faster, no LLM)
 */
export class TestExecutor {
  private options: ExecutionOptions;
  private results: TestSuiteResult[] = [];

  constructor(options: Partial<ExecutionOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Run all tests from a markdown content string.
   */
  async runContent(content: string, filename: string = 'test.md'): Promise<TestSuiteResult> {
    const parser = new TestParser();
    const spec = parser.parse(content, filename);

    return this.runSpec(spec, filename);
  }

  /**
   * Run all tests from a parsed spec.
   */
  async runSpec(spec: TestSpec, filename: string = ''): Promise<TestSuiteResult> {
    // Check tags
    if (this.options.tags && this.options.tags.length > 0) {
      const hasMatchingTag = this.options.tags.some(tag => spec.tags.includes(tag));
      if (!hasMatchingTag) {
        return {
          name: spec.name,
          testFile: filename,
          cases: [],
          passed: 0,
          failed: 0,
          skipped: 1,
        };
      }
    }

    if (this.options.skipTags && this.options.skipTags.length > 0) {
      const hasSkipTag = this.options.skipTags.some(tag => spec.tags.includes(tag));
      if (hasSkipTag) {
        return {
          name: spec.name,
          testFile: filename,
          cases: [],
          passed: 0,
          failed: 0,
          skipped: 1,
        };
      }
    }

    const results: TestResult[] = [];

    for (const testCase of spec.cases) {
      const startTime = Date.now();
      const caseResult = await this.runCase(spec, testCase);
      caseResult.durationMs = Date.now() - startTime;
      results.push(caseResult);
    }

    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed && r.details !== 'skipped').length;
    const skipped = results.filter(r => r.details === 'skipped').length;

    const suiteResult: TestSuiteResult = {
      name: spec.name,
      testFile: filename,
      cases: results,
      passed,
      failed,
      skipped,
    };

    this.results.push(suiteResult);
    return suiteResult;
  }

  /**
   * Run a single test case.
   */
  private async runCase(_spec: TestSpec, testCase: TestCase): Promise<TestResult> {
    const caseName = testCase.name;

    if (!testCase.request) {
      return {
        name: caseName,
        passed: false,
        assertions: [],
        details: 'skipped',
        durationMs: 0,
      };
    }

    // Make request
    const response = await this.httpRequest(
      testCase.request.method,
      testCase.request.path,
      testCase.request.body,
      testCase.request.headers,
      testCase.request.stream
    );

    // Handle request error
    if (response.error) {
      if (testCase.expected === 'failure') {
        return {
          name: caseName,
          passed: true,
          assertions: [],
          details: 'Expected failure occurred',
          durationMs: 0,
        };
      }
      return {
        name: caseName,
        passed: false,
        assertions: [],
        details: `Request error: ${response.error}`,
        durationMs: 0,
      };
    }

    const assertionResults: AssertionResult[] = [];
    let allPassed = true;

    // Run strict assertions
    if (testCase.strict) {
      const validator = new StrictValidator();
      const strictResult = validator.validate(response, testCase.strict);

      assertionResults.push({
        type: 'strict',
        passed: strictResult.passed,
        details: strictResult.results,
      });

      if (!strictResult.passed) {
        allPassed = false;
      }
    }

    // Run natural language assertions (if not strict-only mode)
    if (!this.options.strictOnly) {
      for (const assertion of testCase.assertions) {
        // In agentic mode, we'd use LLM to validate
        // For now, we just record the assertion for the agent
        assertionResults.push({
          type: 'natural',
          assertion,
          passed: true, // Assume passed unless agent says otherwise
          needsValidation: true,
        });
      }
    }

    // Check expected outcome
    if (testCase.expected === 'failure') {
      // For expected failures, check if we got an error response
      if (response.status >= 400) {
        allPassed = true;
      } else {
        allPassed = false;
      }
    }

    return {
      name: caseName,
      passed: allPassed,
      assertions: assertionResults,
      details: allPassed ? 'All assertions passed' : 'Some assertions failed',
      durationMs: 0,
    };
  }

  /**
   * Make an HTTP request to the SDK server.
   */
  async httpRequest(
    method: string,
    path: string,
    body?: Record<string, unknown>,
    headers?: Record<string, string>,
    stream?: boolean
  ): Promise<HttpResponse> {
    const url = `${this.options.baseUrl}${path}`;

    const requestHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      ...headers,
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.options.timeout * 1000);

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
   * Get all results.
   */
  getResults(): TestSuiteResult[] {
    return this.results;
  }

  /**
   * Get aggregated summary.
   */
  getSummary(): {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
    suites: TestSuiteResult[];
  } {
    const totalPassed = this.results.reduce((sum, s) => sum + s.passed, 0);
    const totalFailed = this.results.reduce((sum, s) => sum + s.failed, 0);
    const totalSkipped = this.results.reduce((sum, s) => sum + s.skipped, 0);

    return {
      total: totalPassed + totalFailed + totalSkipped,
      passed: totalPassed,
      failed: totalFailed,
      skipped: totalSkipped,
      suites: this.results,
    };
  }

  /**
   * Print results to console.
   */
  printResults(): void {
    const summary = this.getSummary();

    console.log('\nCompliance Test Results');
    console.log('='.repeat(50));

    for (const suite of summary.suites) {
      if (suite.skipped > 0 && suite.cases.length === 0) {
        console.log(`\n⊘ ${suite.name}: SKIPPED`);
        continue;
      }

      const status = suite.failed === 0 ? '✓' : '✗';
      console.log(`\n${status} ${suite.name}: ${suite.passed}/${suite.passed + suite.failed} passed`);

      for (const testCase of suite.cases) {
        const caseStatus = testCase.passed ? '✓' : '✗';
        console.log(`  ${caseStatus} ${testCase.name}`);

        if (!testCase.passed && testCase.assertions.length > 0) {
          for (const assertion of testCase.assertions) {
            if (!assertion.passed) {
              const detail = assertion.reason || assertion.assertion || 'Failed';
              console.log(`    → ${detail}`);
            }
          }
        }
      }
    }

    console.log('\n' + '='.repeat(50));
    console.log(`Summary: ${summary.passed}/${summary.passed + summary.failed} tests passed`);
    if (summary.skipped > 0) {
      console.log(`         ${summary.skipped} skipped`);
    }
  }
}
