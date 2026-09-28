/**
 * Ollama (2026-09-26, gap-closure plan item 2.8): local models through
 * Ollama's OpenAI-compatible endpoint.
 *
 * `ollama/<model>` in an agent file, or `skills: [ollama]`, reaches
 * `OLLAMA_BASE_URL` (`http://localhost:11434/v1` unless set) with the
 * OpenAI chat-completions wire shape, which Ollama serves as it is. No key:
 * the client sends a placeholder (`ollama`), which Ollama ignores, because
 * the OpenAI skill refuses to call with none. A `base_url` in the entry's
 * config still wins over the variable, as for every model skill
 * (`skills/resolve.ts`).
 *
 * The Python SDK's `OllamaSkill` (`llm/ollama/skill.py`) does the same; the
 * provider row, the variable and the default address are pinned by the
 * shared fixture `python/tests/fixtures/w2ops/models.json`.
 */

import { handoff } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import type { ClientEvent, ServerEvent } from '../../../uamp/events';
import { OpenAISkill, type OpenAISkillConfig } from '../openai/skill';

export const OLLAMA_BASE_URL_VAR = 'OLLAMA_BASE_URL';
export const OLLAMA_DEFAULT_BASE_URL = 'http://localhost:11434/v1';
export const OLLAMA_DEFAULT_MODEL = 'llama3.2';
/** What the client sends as the key; Ollama does not read it. */
export const OLLAMA_PLACEHOLDER_KEY = 'ollama';

export interface OllamaSkillConfig extends OpenAISkillConfig {
  /** The loader's spelling of `base_url`; `baseURL` is the OpenAI skill's. */
  baseUrl?: string;
}

/** Where Ollama answers: the variable when set, else the default address. */
export function ollamaBaseUrl(env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {}): string {
  const set = env[OLLAMA_BASE_URL_VAR];
  return (set && set.trim()) || OLLAMA_DEFAULT_BASE_URL;
}

export class OllamaSkill extends OpenAISkill {
  static override readonly providerId = 'ollama';

  constructor(config: OllamaSkillConfig = {}) {
    const { baseUrl, ...rest } = config;
    super({
      ...rest,
      name: config.name || 'ollama',
      model: config.model || OLLAMA_DEFAULT_MODEL,
      baseURL: config.baseURL || baseUrl || ollamaBaseUrl(),
      apiKey: config.apiKey || OLLAMA_PLACEHOLDER_KEY,
    });
  }

  // Its own handoff name, so an agent can hold an OpenAI skill and this one
  // (a failover chain) without the two overwriting each other.
  @handoff({ name: 'ollama', priority: 10 })
  override async *processUAMP(events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent, void, unknown> {
    yield* super.processUAMP(events, context);
  }
}
