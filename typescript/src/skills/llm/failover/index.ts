/**
 * Model failover: the agent's model, then its `fallback_models:`, in order (plan item 2.8).
 */

export { FailoverLLMSkill, FAILOVER_SKILL_NAME, FAILOVER_STAGE, RETRYABLE_STATUSES, failoverNote, providerFailure } from './skill';
export type { FailoverLLMSkillConfig, FailoverMember } from './skill';
