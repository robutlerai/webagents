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

// Local models through Ollama's OpenAI-compatible endpoint (plan item 2.8).
export { OllamaSkill, ollamaBaseUrl, probeOllama, servesModel, ollamaModelCheck } from './ollama/index';
export type { OllamaSkillConfig, OllamaProbe } from './ollama/index';

// Model failover: the agent's model, then its `fallback_models:` (plan item 2.8).
export { FailoverLLMSkill, failoverNote, providerFailure } from './failover/index';
export type { FailoverLLMSkillConfig, FailoverMember } from './failover/index';

// What a model call costs, in credits, for the chat footer (plan item 2.4).
export { PROVIDER_LIST_PRICES, estimateCostCredits, formatCredits, priceRowFor } from './pricing';

// The provider registry: which providers exist, what credentials they need,
// and which env var carries each. Used by `webagents models` and by the
// API-key preflight, so that neither has to hardcode a list that rots.
export { LLM_PROVIDERS, findProvider, configuredProviders } from './providers';
export type { LLMProvider, ProviderCredential } from './providers';
