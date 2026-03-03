/**
 * TestRunner Skill
 * 
 * Provides tools for running compliance tests against WebAgents SDK implementations.
 * Includes integrated browser automation for UI testing.
 */

export { TestRunnerSkill } from './skill.js';
export { TestParser } from './parser.js';
export { TestExecutor } from './executor.js';
export { StrictValidator } from './validator.js';

// Re-export browser automation skill (dependency)
export { BrowserAutomationSkill } from '../browser/automation.js';
export type {
  ElementInfo,
  ScreenshotResult,
  NetworkEntry,
  AccessibilityInfo,
} from '../browser/automation.js';

// Export types
export type {
  TestResult,
  TestSuiteResult,
  TestCase,
  TestSpec,
  HttpRequestConfig,
  HttpResponse,
  AssertionResult,
  ValidationResult,
  ExecutionOptions,
  TestRunnerConfig,
  // Note: AgentDefinition (for test specs) is intentionally not exported
  // to avoid conflicts with daemon/AgentDefinition
} from './types.js';

// Export TestAgentDefinition as the agent definition type for test specs
export type { AgentDefinition as TestAgentDefinition } from './types.js';
