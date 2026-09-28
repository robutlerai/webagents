/**
 * Ollama Skill Module: local models through Ollama's OpenAI-compatible endpoint.
 */

export { OllamaSkill, OLLAMA_BASE_URL_VAR, OLLAMA_DEFAULT_BASE_URL, OLLAMA_DEFAULT_MODEL, OLLAMA_PLACEHOLDER_KEY, ollamaBaseUrl } from './skill';
export type { OllamaSkillConfig } from './skill';
export { probeOllama, servesModel, ollamaModelCheck } from './probe';
export type { OllamaProbe } from './probe';
