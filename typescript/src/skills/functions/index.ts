/**
 * Functions substrate skill.
 *
 * Public exports:
 *  - `FunctionRuntimeSkill` — substrate consumed by cron / custom_http /
 *    custom_tools / host-self-edit / manual.
 *  - `FunctionContext` and friends — typed context user code reads from.
 *  - `validateManifest` — pure schema validator for save-time use.
 *  - `ExecutorClient` interface + `StubExecutorClient` for tests.
 */

export * from './manifest';
export * from './validator';
export * from './context';
export * from './executor-client';
export {
  FunctionRuntimeSkill,
  bridgeFunctionContext,
  type FunctionRuntimeSkillConfig,
  type DeclaredFunction,
} from './skill';
