/**
 * TestRunner Skill
 * 
 * Provides tools for running compliance tests against WebAgents SDK implementations.
 * Includes integrated browser automation for UI testing.
 */

export { TestRunnerSkill } from './skill';
export { TestParser } from './parser';
export { TestExecutor } from './executor';
export { StrictValidator } from './validator';

// Re-export browser automation skill (dependency)
export { BrowserAutomationSkill } from '../browser/automation';
export type {
  ElementInfo,
  ScreenshotResult,
  NetworkEntry,
  AccessibilityInfo,
} from '../browser/automation';

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
} from './types';

// Export TestAgentDefinition as the agent definition type for test specs
export type { AgentDefinition as TestAgentDefinition } from './types';
