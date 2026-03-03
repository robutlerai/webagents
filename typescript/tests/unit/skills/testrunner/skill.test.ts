/**
 * TestRunnerSkill Unit Tests
 * 
 * Tests the main TestRunner skill with its tools and reporting.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { TestRunnerSkill } from '../../../../src/skills/testrunner/skill.js';

describe('TestRunnerSkill', () => {
  let skill: TestRunnerSkill;

  beforeEach(() => {
    skill = new TestRunnerSkill();
    skill.resetResults(); // Clear any previous results
  });

  describe('initialization', () => {
    it('should have correct id and description', () => {
      expect(skill.id).toBe('testrunner');
      expect(skill.description).toContain('compliance tests');
    });

    it('should accept custom configuration', () => {
      const customSkill = new TestRunnerSkill({
        baseUrl: 'http://localhost:9000',
        timeout: 30,
        cacheDir: '/tmp/cache',
      });

      expect(customSkill['testConfig'].baseUrl).toBe('http://localhost:9000');
      expect(customSkill['testConfig'].timeout).toBe(30);
    });

    it('should use default configuration', () => {
      expect(skill['testConfig'].baseUrl).toBe('http://localhost:8765');
      expect(skill['testConfig'].timeout).toBe(60);
    });
  });

  describe('loadTestContent', () => {
    it('should parse test content and return spec', async () => {
      const content = `---
name: test-spec
version: "1.0"
transport: completions
tags: [core]
---

# Test Spec

Description.
`;
      const result = await skill.loadTestContent(content, 'test.md');

      expect(result.name).toBe('test-spec');
      expect(result.version).toBe('1.0');
      expect(result.transport).toBe('completions');
    });
  });

  describe('validateStrict', () => {
    it('should validate response against strict assertions', async () => {
      const response = {
        status: 200,
        headers: {},
        body: {
          object: 'chat.completion',
          choices: [{ message: { role: 'assistant', content: 'Hello' } }],
        },
      };

      const assertions = {
        status: 200,
        body: {
          object: 'chat.completion',
          'choices[0].message.role': 'assistant',
        },
      };

      const result = await skill.validateStrict(response, assertions);

      expect(result.passed).toBe(true);
    });

    it('should report failures', async () => {
      const response = {
        status: 404,
        headers: {},
        body: {},
      };

      const assertions = {
        status: 200,
      };

      const result = await skill.validateStrict(response, assertions);

      expect(result.passed).toBe(false);
      expect(result.results[0].path).toBe('status');
    });
  });

  describe('validateAssertion', () => {
    it('should return instruction for LLM validation', async () => {
      const assertion = 'Response contains valid JSON';
      const response = { 
        status: 200,
        headers: {},
        body: { choices: [{ message: { content: 'Hello' } }] }
      };

      const result = await skill.validateAssertion(response, assertion);

      // validateAssertion returns guidance for the agent
      expect(result.assertion).toBe(assertion);
      expect(result.responseSummary).toBeDefined();
      expect(result.instruction).toContain('Evaluate');
    });
  });

  describe('reportResult', () => {
    it('should store test result', async () => {
      // reportResult takes: testName, caseName, passed, details, assertions
      const result = await skill.reportResult(
        'Test Suite',
        'Test Case 1',
        true,
        'Test passed',
        []
      );

      expect(result.recorded).toBe(true);
      expect(result.test).toBe('Test Suite');
      expect(result.case).toBe('Test Case 1');
      expect(result.passed).toBe(true);
    });

    it('should accumulate results for same suite', async () => {
      await skill.reportResult('Suite', 'Test 1', true, 'Passed', []);
      await skill.reportResult('Suite', 'Test 2', false, 'Failed', []);

      const summary = await skill.getResultsSummary();

      expect(summary.total).toBe(2);
      expect(summary.passed).toBe(1);
      expect(summary.failed).toBe(1);
    });
  });

  describe('getResultsSummary', () => {
    it('should return empty summary initially', async () => {
      const summary = await skill.getResultsSummary();

      expect(summary.total).toBe(0);
      expect(summary.passed).toBe(0);
      expect(summary.failed).toBe(0);
      expect(summary.suites).toHaveLength(0);
    });

    it('should aggregate results across suites', async () => {
      await skill.reportResult('Suite 1', 'Test 1', true, 'OK', []);
      await skill.reportResult('Suite 2', 'Test 2', true, 'OK', []);

      const summary = await skill.getResultsSummary();

      expect(summary.total).toBe(2);
      expect(summary.passed).toBe(2);
      expect(summary.suites).toHaveLength(2);
    });
  });

  describe('HTML report generation', () => {
    it('should generate HTML report', async () => {
      await skill.reportResult('Test Suite', 'Passing Test', true, 'OK', []);
      await skill.reportResult('Test Suite', 'Failing Test', false, 'Assertion failed', []);

      const { html, summary } = await skill.generateHtmlReport('Test Report');

      expect(html).toContain('<!DOCTYPE html>');
      expect(html).toContain('Test Report');
      expect(html).toContain('Test Suite');
      // The HTML shows case names from summary.cases
      expect(summary.total).toBe(2);
      expect(summary.passed).toBe(1);
      expect(summary.failed).toBe(1);
    });

    it('should show pass rate percentage', async () => {
      await skill.reportResult('Suite', 'Test 1', true, '', []);
      await skill.reportResult('Suite', 'Test 2', true, '', []);

      const { html } = await skill.generateHtmlReport();

      expect(html).toContain('100%');
    });
  });

  describe('JSON export', () => {
    it('should export results as JSON', async () => {
      await skill.reportResult('Suite', 'Test', true, 'OK', []);

      const { json } = await skill.exportResultsJson();
      const parsed = JSON.parse(json);

      expect(parsed.total).toBe(1);
      expect(parsed.passed).toBe(1);
      expect(parsed.suites).toHaveLength(1);
    });

    it('should support minified JSON', async () => {
      await skill.reportResult('Suite', 'Test', true, '', []);

      const { json } = await skill.exportResultsJson(false);

      // Minified JSON has no newlines
      expect(json).not.toContain('\n');
    });
  });

  describe('JUnit XML export', () => {
    it('should export results as JUnit XML', async () => {
      await skill.reportResult('Suite', 'Passing Test', true, '', []);
      await skill.reportResult('Suite', 'Failing Test', false, 'Expected 200, got 400', []);

      const { xml } = await skill.exportResultsJunit();

      expect(xml).toContain('<?xml version="1.0"');
      expect(xml).toContain('<testsuites');
      expect(xml).toContain('tests="2"');
      expect(xml).toContain('failures="1"');
      expect(xml).toContain('Passing Test');
      expect(xml).toContain('Failing Test');
      expect(xml).toContain('<failure');
    });
  });

  describe('browser automation integration', () => {
    it('should have browser automation skill', () => {
      expect(skill.browserAutomation).toBeDefined();
    });

    it('should have SoM tools', () => {
      expect(skill.browserMarkElements).toBeDefined();
      expect(skill.browserUnmarkElements).toBeDefined();
      expect(skill.browserMarkedScreenshot).toBeDefined();
      expect(skill.browserClickMarked).toBeDefined();
      expect(skill.browserTypeMarked).toBeDefined();
    });
  });
});

describe('TestRunnerSkill Integration', () => {
  describe('end-to-end validation flow', () => {
    it('should validate responses against strict assertions', async () => {
      const skill = new TestRunnerSkill({ baseUrl: 'http://localhost:8765' });
      skill.resetResults();

      // Validate a mock response
      const mockResponse = {
        status: 200,
        headers: {},
        body: {
          object: 'chat.completion',
          choices: [{ message: { role: 'assistant', content: 'Hello!' } }],
        },
      };

      const validation = await skill.validateStrict(mockResponse, {
        status: 200,
        body: {
          object: 'chat.completion',
          'choices[0].message.role': 'assistant',
        },
      });

      expect(validation.passed).toBe(true);

      // Report the result
      await skill.reportResult(
        'Completions',
        'Basic Response',
        validation.passed,
        'Test completed successfully',
        validation.results
      );

      // Generate report
      const { summary } = await skill.generateHtmlReport();

      expect(summary.total).toBe(1);
      expect(summary.passed).toBe(1);
    });
  });
});
