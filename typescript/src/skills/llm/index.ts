/**
 * LLM Skills Module
 * 
 * Skills for LLM inference (both in-browser and cloud).
 */

// In-browser LLM skills
export { WebLLMSkill } from './webllm/index';
export type { WebLLMSkillConfig } from './webllm/index';

export { TransformersSkill } from './transformers/index';
export type { TransformersSkillConfig } from './transformers/index';

// Cloud LLM skills
export { OpenAISkill } from './openai/index';
export type { OpenAISkillConfig } from './openai/index';

export { AnthropicSkill } from './anthropic/index';
export type { AnthropicSkillConfig } from './anthropic/index';

export { GoogleSkill } from './google/index';
export type { GoogleSkillConfig } from './google/index';

export { XAISkill } from './xai/index';
export type { XAISkillConfig } from './xai/index';

// LLM Proxy (routes through UAMP to a portal-hosted LLM service)
export { LLMProxySkill } from './proxy/index';
export type { LLMProxySkillConfig } from './proxy/index';

// The provider registry: which providers exist, what credentials they need,
// and which env var carries each. Used by `webagents models` and by the
// API-key preflight, so that neither has to hardcode a list that rots.
export { LLM_PROVIDERS, findProvider, configuredProviders } from './providers';
export type { LLMProvider, ProviderCredential } from './providers';
