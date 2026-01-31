/**
 * Type definitions for TestRunnerSkill
 */

/**
 * Result of a single test case
 */
export interface TestResult {
  /** Test case name */
  name: string;
  /** Whether the test passed */
  passed: boolean;
  /** Individual assertion results */
  assertions: AssertionResult[];
  /** Details about the result */
  details: string;
  /** Duration in milliseconds */
  durationMs: number;
}

/**
 * Result of a full test suite
 */
export interface TestSuiteResult {
  /** Suite name */
  name: string;
  /** Path to the test file */
  testFile: string;
  /** Individual test case results */
  cases: TestResult[];
  /** Number of passed tests */
  passed: number;
  /** Number of failed tests */
  failed: number;
  /** Number of skipped tests */
  skipped: number;
}

/**
 * A single test case parsed from markdown
 */
export interface TestCase {
  /** Test case name */
  name: string;
  /** HTTP request configuration */
  request?: HttpRequestConfig;
  /** Flow steps for multi-agent tests */
  flow?: string[];
  /** Natural language assertions */
  assertions: string[];
  /** Strict (deterministic) assertions */
  strict?: Record<string, unknown>;
  /** Expected outcome */
  expected: 'success' | 'failure';
}

/**
 * HTTP request configuration
 */
export interface HttpRequestConfig {
  /** HTTP method */
  method: string;
  /** Request path */
  path: string;
  /** Request body */
  body?: Record<string, unknown>;
  /** Request headers */
  headers?: Record<string, string>;
  /** Whether to expect streaming response */
  stream?: boolean;
}

/**
 * Parsed test specification
 */
export interface TestSpec {
  /** Test name */
  name: string;
  /** Version */
  version: string;
  /** Transport type (completions, realtime, etc.) */
  transport: string;
  /** Test type (single-agent, multi-agent) */
  type: 'single-agent' | 'multi-agent';
  /** Tags for filtering */
  tags: string[];
  /** Timeout in seconds */
  timeout: number;
  /** Dependencies on other tests */
  dependsOn: string[];
  /** Description */
  description: string;
  /** Setup instructions */
  setup: string;
  /** Agent definitions for multi-agent tests */
  agents: AgentDefinition[];
  /** Test cases */
  cases: TestCase[];
}

/**
 * Agent definition for multi-agent tests
 */
export interface AgentDefinition {
  /** Agent ID */
  id: string;
  /** Agent name */
  name?: string;
  /** Agent instructions */
  instructions?: string;
  /** Handoff targets */
  handoffs?: string[];
  /** Tools */
  tools?: string[];
}

/**
 * HTTP response from the SDK under test
 */
export interface HttpResponse {
  /** HTTP status code */
  status: number;
  /** Response headers */
  headers: Record<string, string>;
  /** Response body (parsed JSON or raw text) */
  body: unknown;
  /** Format (e.g., 'sse' for streaming) */
  format?: string;
  /** Error message if request failed */
  error?: string;
  /** Error type if request failed */
  errorType?: string;
}

/**
 * Result of a single assertion
 */
export interface AssertionResult {
  /** Type of assertion */
  type: 'strict' | 'natural';
  /** Whether the assertion passed */
  passed: boolean;
  /** Path being checked */
  path?: string;
  /** Details/reason */
  reason?: string;
  /** The assertion text (for natural language assertions) */
  assertion?: string;
  /** Whether this needs agent validation */
  needsValidation?: boolean;
  /** Sub-results for nested checks */
  details?: AssertionResult[];
}

/**
 * Validation result
 */
export interface ValidationResult {
  /** Whether all validations passed */
  passed: boolean;
  /** Individual results */
  results: AssertionResult[];
}

/**
 * Test execution options
 */
export interface ExecutionOptions {
  /** Base URL of the SDK server under test */
  baseUrl: string;
  /** Cache mode for test results */
  cacheMode: 'read_write' | 'write_only' | 'disabled';
  /** LLM temperature for validation */
  temperature: number;
  /** Request timeout in seconds */
  timeout: number;
  /** Only run strict (deterministic) assertions */
  strictOnly: boolean;
  /** Tags to include */
  tags?: string[];
  /** Tags to skip */
  skipTags?: string[];
  /** Parallel execution count */
  parallel: number;
}

/**
 * TestRunnerSkill configuration
 */
export interface TestRunnerConfig {
  /** Base URL of the SDK server under test */
  baseUrl?: string;
  /** Directory for caching test results */
  cacheDir?: string;
  /** Request timeout in seconds */
  timeout?: number;
}
