/**
 * LLM Skills Module
 * 
 * Skills for LLM inference (both in-browser and cloud).
 */

// In-browser LLM skills
export { WebLLMSkill } from './webllm/index.js';
export type { WebLLMSkillConfig } from './webllm/index.js';

export { TransformersSkill } from './transformers/index.js';
export type { TransformersSkillConfig } from './transformers/index.js';

// Cloud LLM skills
export { OpenAISkill } from './openai/index.js';
export type { OpenAISkillConfig } from './openai/index.js';

export { AnthropicSkill } from './anthropic/index.js';
export type { AnthropicSkillConfig } from './anthropic/index.js';

export { GoogleSkill } from './google/index.js';
export type { GoogleSkillConfig } from './google/index.js';

export { XAISkill } from './xai/index.js';
export type { XAISkillConfig } from './xai/index.js';
