/**
 * FireworksSkill Unit Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { FireworksSkill } from '../../../../src/skills/llm/fireworks/skill.js';

describe('FireworksSkill', () => {
  describe('constructor', () => {
    it('sets default name to "fireworks"', () => {
      const skill = new FireworksSkill();
      expect(skill.name).toBe('fireworks');
    });

    it('allows custom name', () => {
      const skill = new FireworksSkill({ name: 'my-fireworks' });
      expect(skill.name).toBe('my-fireworks');
    });
  });

  describe('getCapabilities', () => {
    it('returns fireworks provider with default model', () => {
      const skill = new FireworksSkill();
      const caps = skill.getCapabilities();
      expect(caps.provider).toBe('fireworks');
      expect(caps.id).toBe('deepseek-v3p2');
      expect(caps.supports_streaming).toBe(true);
    });

    it('reports vision capability for vision models', () => {
      const skill = new FireworksSkill({ model: 'kimi-k2p5' });
      const caps = skill.getCapabilities();
      expect(caps.modalities).toContain('image');
    });

    it('reports text-only for non-vision models', () => {
      const skill = new FireworksSkill({ model: 'deepseek-v3p2' });
      const caps = skill.getCapabilities();
      expect(caps.modalities).toEqual(['text']);
    });

    it('returns correct context window for known models', () => {
      const skill = new FireworksSkill({ model: 'qwen3-8b' });
      const caps = skill.getCapabilities();
      expect(caps.context_window).toBe(131072);
    });
  });

  describe('DEFAULT_MODELS', () => {
    it('contains expected models', () => {
      const models = FireworksSkill.DEFAULT_MODELS;
      expect(models['deepseek-v3p2']).toBeDefined();
      expect(models['glm-5']).toBeDefined();
      expect(models['kimi-k2p5']).toBeDefined();
      expect(models['minimax-m2p5']).toBeDefined();
      expect(models['qwen3-8b']).toBeDefined();
      expect(models['llama-v3p3-70b-instruct']).toBeDefined();
      expect(models['cogito-671b-v2']).toBeDefined();
    });

    it('all models have consistent schema', () => {
      for (const [key, def] of Object.entries(FireworksSkill.DEFAULT_MODELS)) {
        expect(def.name).toBe(key);
        expect(typeof def.maxOutputTokens).toBe('number');
        expect(typeof def.supportsTools).toBe('boolean');
        expect(typeof def.contextWindow).toBe('number');
      }
    });
  });

  describe('BASE_URL', () => {
    it('is the Fireworks inference endpoint', () => {
      expect(FireworksSkill.BASE_URL).toBe('https://api.fireworks.ai/inference/v1');
    });
  });

  describe('processUAMP (without API key)', () => {
    it('yields error when client not initialized', async () => {
      const skill = new FireworksSkill();
      // Don't call initialize — no API key

      const events: unknown[] = [];
      const mockContext = {
        get: vi.fn(() => undefined),
        set: vi.fn(),
        delete: vi.fn(),
      };

      for await (const event of (skill as any).processUAMP(
        [
          { type: 'session.create', event_id: 'e1', session: { modalities: ['text'] } },
          { type: 'input.text', event_id: 'e2', text: 'Hello' },
        ],
        mockContext,
      )) {
        events.push(event);
      }

      const errorEvent = events.find((e: any) => e.type === 'response.error') as any;
      expect(errorEvent).toBeDefined();
      expect(errorEvent.error.message).toContain('not configured');
    });
  });
});
