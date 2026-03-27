/**
 * Strict assertion validator
 * 
 * Validates responses against deterministic YAML assertions.
 * 
 * Supports:
 * - Exact matching
 * - Regex matching (prefix with /)
 * - Type checking (type(string), type(number), etc.)
 * - Existence checks (exists, not_exists)
 * - Length checks (length(N))
 * - Contains checks (contains("str"))
 * - JSONPath-like access (body.choices[0].message.content)
 */

import type { HttpResponse, ValidationResult, AssertionResult } from './types';

export class StrictValidator {
  /**
   * Validate response against strict assertions.
   */
  validate(response: HttpResponse, assertions: Record<string, unknown>): ValidationResult {
    const results: AssertionResult[] = [];
    let allPassed = true;

    for (const [key, expected] of Object.entries(assertions)) {
      let result: AssertionResult;

      switch (key) {
        case 'status':
          result = this.checkStatus(response.status, expected);
          break;
        case 'headers':
          result = this.checkHeaders(response.headers || {}, expected as Record<string, unknown>);
          break;
        case 'body':
          result = this.checkBody(response.body || {}, expected as Record<string, unknown>);
          break;
        case 'format':
          result = this.checkFormat(response, expected as string);
          break;
        case 'chunks':
          result = this.checkChunks(response.body as unknown[], expected as Record<string, unknown>[]);
          break;
        case 'final_chunk':
          result = this.checkFinalChunk(response.body as unknown[], expected as Record<string, unknown>);
          break;
        case 'events':
          result = this.checkEvents((response as unknown as { events: unknown[] }).events || [], expected as Record<string, unknown>[]);
          break;
        default:
          result = { type: 'strict', path: key, passed: false, reason: `Unknown assertion key: ${key}` };
      }

      results.push(result);
      if (!result.passed) {
        allPassed = false;
      }
    }

    return { passed: allPassed, results };
  }

  /**
   * Check HTTP status code.
   */
  private checkStatus(actual: number, expected: unknown): AssertionResult {
    if (Array.isArray(expected)) {
      const passed = expected.includes(actual);
      return {
        type: 'strict',
        path: 'status',
        passed,
        reason: passed ? `Status ${actual} in [${expected.join(', ')}]` : `Status ${actual} not in [${expected.join(', ')}]`,
      };
    }

    const passed = actual === expected;
    return {
      type: 'strict',
      path: 'status',
      passed,
      reason: passed ? `Status ${actual} == ${expected}` : `Status ${actual} != ${expected}`,
    };
  }

  /**
   * Check response headers.
   */
  private checkHeaders(actual: Record<string, string>, expected: Record<string, unknown>): AssertionResult {
    const details: AssertionResult[] = [];
    let allPassed = true;

    for (const [header, value] of Object.entries(expected)) {
      // Headers are case-insensitive
      const actualValue = actual[header] || actual[header.toLowerCase()];
      const result = this.checkValue(`headers.${header}`, actualValue, value);
      details.push(result);
      if (!result.passed) {
        allPassed = false;
      }
    }

    return {
      type: 'strict',
      path: 'headers',
      passed: allPassed,
      details,
    };
  }

  /**
   * Check response body using JSONPath-like assertions.
   */
  private checkBody(actual: unknown, expected: Record<string, unknown>): AssertionResult {
    const details: AssertionResult[] = [];
    let allPassed = true;

    for (const [path, value] of Object.entries(expected)) {
      const actualValue = this.getPath(actual, path);
      const result = this.checkValue(`body.${path}`, actualValue, value);
      details.push(result);
      if (!result.passed) {
        allPassed = false;
      }
    }

    return {
      type: 'strict',
      path: 'body',
      passed: allPassed,
      details,
    };
  }

  /**
   * Check response format (e.g., sse).
   */
  private checkFormat(response: HttpResponse, expected: string): AssertionResult {
    const actual = response.format;
    const passed = actual === expected;
    return {
      type: 'strict',
      path: 'format',
      passed,
      reason: passed ? `Format ${actual} == ${expected}` : `Format ${actual} != ${expected}`,
    };
  }

  /**
   * Check SSE chunks.
   */
  private checkChunks(chunks: unknown[], expected: Record<string, unknown>[]): AssertionResult {
    if (!Array.isArray(chunks)) {
      return { type: 'strict', path: 'chunks', passed: false, reason: 'Response is not SSE chunks' };
    }

    const details: AssertionResult[] = [];
    let allPassed = true;

    for (const assertion of expected) {
      // Check if any chunk matches the assertion
      let matched = false;
      for (const chunk of chunks) {
        if (this.chunkMatches(chunk as Record<string, unknown>, assertion)) {
          matched = true;
          break;
        }
      }

      if (!matched) {
        allPassed = false;
        details.push({
          type: 'strict',
          passed: false,
          reason: `No chunk matched assertion: ${JSON.stringify(assertion)}`,
        });
      } else {
        details.push({
          type: 'strict',
          passed: true,
          reason: 'Chunk matched',
        });
      }
    }

    return {
      type: 'strict',
      path: 'chunks',
      passed: allPassed,
      details,
    };
  }

  /**
   * Check the final chunk of an SSE stream.
   */
  private checkFinalChunk(chunks: unknown[], expected: Record<string, unknown>): AssertionResult {
    if (!chunks || chunks.length === 0) {
      return { type: 'strict', path: 'final_chunk', passed: false, reason: 'No chunks' };
    }

    const final = chunks[chunks.length - 1] as Record<string, unknown>;
    const details: AssertionResult[] = [];
    let allPassed = true;

    for (const [path, value] of Object.entries(expected)) {
      const actualValue = this.getPath(final, path);
      const result = this.checkValue(`final_chunk.${path}`, actualValue, value);
      details.push(result);
      if (!result.passed) {
        allPassed = false;
      }
    }

    return {
      type: 'strict',
      path: 'final_chunk',
      passed: allPassed,
      details,
    };
  }

  /**
   * Check events (for multi-agent tests).
   */
  private checkEvents(events: unknown[], expected: Record<string, unknown>[]): AssertionResult {
    const details: AssertionResult[] = [];
    let allPassed = true;

    for (const expEvent of expected) {
      let matched = false;
      for (const event of events) {
        if (this.eventMatches(event as Record<string, unknown>, expEvent)) {
          matched = true;
          break;
        }
      }

      if (!matched) {
        allPassed = false;
        details.push({
          type: 'strict',
          passed: false,
          reason: `Event not found: ${JSON.stringify(expEvent)}`,
        });
      } else {
        details.push({
          type: 'strict',
          passed: true,
          reason: 'Event found',
        });
      }
    }

    return {
      type: 'strict',
      path: 'events',
      passed: allPassed,
      details,
    };
  }

  /**
   * Check if a chunk matches an assertion.
   */
  private chunkMatches(chunk: Record<string, unknown>, assertion: Record<string, unknown>): boolean {
    for (const [path, value] of Object.entries(assertion)) {
      const actual = this.getPath(chunk, path);
      const result = this.checkValue(path, actual, value);
      if (!result.passed) {
        return false;
      }
    }
    return true;
  }

  /**
   * Check if an event matches expected criteria.
   */
  private eventMatches(event: Record<string, unknown>, expected: Record<string, unknown>): boolean {
    for (const [key, value] of Object.entries(expected)) {
      if (key === 'type') {
        if (event.type !== value) {
          return false;
        }
      } else {
        const actual = event[key];
        const result = this.checkValue(key, actual, value);
        if (!result.passed) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Get value at JSONPath-like path.
   */
  private getPath(obj: unknown, path: string): unknown {
    const parts = path.split(/\.|\[|\]/).filter(p => p !== '');
    let current: unknown = obj;

    for (const part of parts) {
      if (current === null || current === undefined) {
        return undefined;
      }

      if (part === '*') {
        // Wildcard - return all values
        if (Array.isArray(current)) {
          const remaining = parts.slice(parts.indexOf(part) + 1).join('.');
          return current.map(item => this.getPath(item, remaining));
        }
        return undefined;
      }

      if (/^\d+$/.test(part)) {
        const idx = parseInt(part, 10);
        if (Array.isArray(current) && idx < current.length) {
          current = current[idx];
        } else {
          return undefined;
        }
      } else {
        if (typeof current === 'object' && current !== null) {
          current = (current as Record<string, unknown>)[part];
        } else {
          return undefined;
        }
      }
    }

    return current;
  }

  /**
   * Check a single value against expected.
   */
  private checkValue(path: string, actual: unknown, expected: unknown): AssertionResult {
    // Handle special string assertions
    if (typeof expected === 'string') {
      // Regex
      if (expected.startsWith('/') && expected.endsWith('/')) {
        const pattern = expected.slice(1, -1);
        if (actual === null || actual === undefined) {
          return { type: 'strict', path, passed: false, reason: `Value is null/undefined, expected regex ${pattern}` };
        }
        const passed = new RegExp(pattern).test(String(actual));
        return {
          type: 'strict',
          path,
          passed,
          reason: passed ? 'Regex matched' : 'Regex did not match',
        };
      }

      // Type check
      if (expected.startsWith('type(') && expected.endsWith(')')) {
        const expectedType = expected.slice(5, -1);
        const typeMap: Record<string, (v: unknown) => boolean> = {
          string: (v) => typeof v === 'string',
          number: (v) => typeof v === 'number',
          boolean: (v) => typeof v === 'boolean',
          array: (v) => Array.isArray(v),
          object: (v) => typeof v === 'object' && v !== null && !Array.isArray(v),
          null: (v) => v === null,
        };
        const checker = typeMap[expectedType];
        const passed = checker ? checker(actual) : false;
        return {
          type: 'strict',
          path,
          passed,
          reason: `Type is ${typeof actual}, expected ${expectedType}`,
        };
      }

      // Exists check
      if (expected === 'exists') {
        const passed = actual !== undefined && actual !== null;
        return { type: 'strict', path, passed, reason: passed ? 'Value exists' : 'Value does not exist' };
      }

      if (expected === 'not_exists') {
        const passed = actual === undefined || actual === null;
        return { type: 'strict', path, passed, reason: passed ? 'Value does not exist' : 'Value exists' };
      }

      if (expected === 'not_null') {
        const passed = actual !== null && actual !== undefined;
        return { type: 'strict', path, passed, reason: passed ? 'Value is not null' : 'Value is null' };
      }

      // Length check
      if (expected.startsWith('length(') && expected.endsWith(')')) {
        const expectedLen = parseInt(expected.slice(7, -1), 10);
        const actualLen = Array.isArray(actual) || typeof actual === 'string' ? actual.length : 0;
        const passed = actualLen === expectedLen;
        return {
          type: 'strict',
          path,
          passed,
          reason: `Length is ${actualLen}, expected ${expectedLen}`,
        };
      }

      // Contains check
      if (expected.startsWith('contains(') && expected.endsWith(')')) {
        const substring = expected.slice(9, -1).replace(/^["']|["']$/g, '');
        const passed = actual ? String(actual).includes(substring) : false;
        return {
          type: 'strict',
          path,
          passed,
          reason: `Contains '${substring}': ${passed}`,
        };
      }
    }

    // List of acceptable values
    if (Array.isArray(expected)) {
      const passed = expected.includes(actual);
      return {
        type: 'strict',
        path,
        passed,
        reason: passed ? `Value ${actual} in [${expected.join(', ')}]` : `Value ${actual} not in [${expected.join(', ')}]`,
      };
    }

    // Exact match
    const passed = actual === expected;
    return {
      type: 'strict',
      path,
      passed,
      reason: passed ? `Value ${JSON.stringify(actual)} == ${JSON.stringify(expected)}` : `Value ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`,
    };
  }
}
