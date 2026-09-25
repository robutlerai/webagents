/**
 * `OPENAI_BASE_URL` reaches the OpenAI skill (2026-09-24).
 *
 * The skill read `OPENAI_API_KEY` from the environment but not the endpoint, so
 * an environment aimed at an OpenAI-compatible server sent the call, with that
 * key, to api.openai.com. Found on a first-time-developer walkthrough, where a
 * served agent answered with OpenAI's own 401 for a key meant for another
 * endpoint. The Python SDK's skill honours the variable through the official
 * client; these pin that the two agree.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { OpenAISkill } from '../../../../src/skills/llm/openai/skill';

interface BuildsRequests {
  adapter: {
    buildRequest(input: { messages: unknown[]; model: string; apiKey: string }): { url: string };
  };
}

function urlOf(skill: OpenAISkill): string {
  return (skill as unknown as BuildsRequests).adapter.buildRequest({
    messages: [{ role: 'user', content: 'hi' }],
    model: 'm',
    apiKey: 'k',
  }).url;
}

describe('the OpenAI skill endpoint', () => {
  const saved = process.env.OPENAI_BASE_URL;
  beforeEach(() => {
    delete process.env.OPENAI_BASE_URL;
  });
  afterEach(() => {
    if (saved === undefined) delete process.env.OPENAI_BASE_URL;
    else process.env.OPENAI_BASE_URL = saved;
  });

  it('follows OPENAI_BASE_URL when the config names none', () => {
    process.env.OPENAI_BASE_URL = 'http://127.0.0.1:9/v1';
    expect(urlOf(new OpenAISkill({ model: 'm' }))).toMatch(/^http:\/\/127\.0\.0\.1:9\/v1\//);
  });

  it('lets an explicit baseURL win over the environment', () => {
    process.env.OPENAI_BASE_URL = 'http://127.0.0.1:9/v1';
    expect(urlOf(new OpenAISkill({ model: 'm', baseURL: 'http://127.0.0.1:8/v1' }))).toMatch(
      /^http:\/\/127\.0\.0\.1:8\/v1\//,
    );
  });

  it('uses OpenAI itself when neither is set', () => {
    expect(urlOf(new OpenAISkill({ model: 'm' }))).toContain('api.openai.com');
  });
});
