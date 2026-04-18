/**
 * Multimodal History Round-Trip Tests
 *
 * Regression coverage for the "Want me to add music?" bug where the assistant
 * text portion of a prior multimodal turn was being silently dropped before
 * it reached the LLM. The portal-runtime layer used to overwrite `content`
 * (string) with `content_items` (array) for any message that had media; the
 * Google and Anthropic adapters then skipped their text-prepend branch
 * (which is gated on `typeof m.content === 'string'`), and OpenAI never
 * combined the two at all. Net effect: the LLM saw image-only and lost the
 * antecedent question.
 *
 * Wire shape after the fix (single source of truth for text):
 *   { role: 'assistant', content: "<text>", content_items: [<media-only>] }
 *
 *   - Google + Anthropic: their adapters already prepend `m.content` next to
 *     the rendered media parts when both are present.
 *   - OpenAI: a backstop in `convertMessages` prepends `m.content` as a text
 *     part when items are present (deduped against any leading text item, so
 *     there's no risk of double-rendering if items happen to carry the text).
 *
 * Each test asserts the prior-turn text + media both reach the provider
 * payload AND that the text appears exactly once (no duplication).
 */

import { describe, it, expect } from 'vitest';
import { openaiAdapter } from '../../../src/adapters/openai.js';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import { googleAdapter } from '../../../src/adapters/google.js';

const IMAGE_UUID = '9e23908d-0b2a-4015-92b4-7cdf8c4796fe';
const IMAGE_URL = `/api/content/${IMAGE_UUID}`;
const RESOLVED_MEDIA = new Map([
  [IMAGE_URL, { kind: 'binary' as const, mimeType: 'image/png', base64: 'iVBORw0KGgo...' }],
]);

const ASSISTANT_QUESTION = 'I generated a unicorn for you. Want me to add music?';

describe('multimodal history round-trip — prior assistant turn text + media', () => {
  describe('OpenAI adapter', () => {
    it('emits text exactly once + image when text is in `content` string and items are media-only', () => {
      // This is the canonical post-fix wire shape produced by
      // chatHistoryToOpenAIMessages: text in `content`, items hold media only.
      const req = openaiAdapter.buildRequest({
        messages: [
          { role: 'user', content: 'create a unicorn webpage' },
          {
            role: 'assistant',
            content: ASSISTANT_QUESTION,
            content_items: [{ type: 'image', image: { url: IMAGE_URL } }],
          },
          { role: 'user', content: 'yes' },
        ],
        model: 'gpt-4o',
        apiKey: 'test-key',
        resolvedMedia: RESOLVED_MEDIA,
      });
      const body = JSON.parse(req.body);
      const assistantMsg = body.messages.find((m: { role: string }) => m.role === 'assistant');
      expect(Array.isArray(assistantMsg.content)).toBe(true);
      const textParts = assistantMsg.content.filter((p: { type: string }) => p.type === 'text');
      const imgPart = assistantMsg.content.find((p: { type: string }) => p.type === 'image_url');
      expect(textParts).toHaveLength(1);
      expect(textParts[0].text).toBe(ASSISTANT_QUESTION);
      expect(imgPart).toBeDefined();
      expect(imgPart.image_url.url).toContain('data:image/png;base64,');
      // Text must lead so the LLM reads the question before the image.
      expect(assistantMsg.content[0].type).toBe('text');
    });

    it('does not duplicate text when items happen to start with the same text item', () => {
      // Defense: if an upstream caller already baked the text into items, the
      // backstop dedup in convertMessages must NOT prepend a second copy.
      const req = openaiAdapter.buildRequest({
        messages: [
          {
            role: 'assistant',
            content: ASSISTANT_QUESTION,
            content_items: [
              { type: 'text', text: ASSISTANT_QUESTION },
              { type: 'image', image: { url: IMAGE_URL } },
            ],
          },
        ],
        model: 'gpt-4o',
        apiKey: 'test-key',
        resolvedMedia: RESOLVED_MEDIA,
      });
      const body = JSON.parse(req.body);
      const assistantMsg = body.messages[0];
      const textParts = assistantMsg.content.filter((p: { type: string }) => p.type === 'text');
      expect(textParts).toHaveLength(1);
      expect(textParts[0].text).toBe(ASSISTANT_QUESTION);
    });
  });

  describe('Anthropic adapter', () => {
    it('emits text exactly once + image when text is in `content` string and items are media-only', () => {
      const req = anthropicAdapter.buildRequest({
        messages: [
          { role: 'user', content: 'create a unicorn webpage' },
          {
            role: 'assistant',
            content: ASSISTANT_QUESTION,
            content_items: [{ type: 'image', image: { url: IMAGE_URL } }],
          },
          { role: 'user', content: 'yes' },
        ],
        model: 'claude-sonnet-4-7',
        apiKey: 'test-key',
        resolvedMedia: RESOLVED_MEDIA,
      });
      const body = JSON.parse(req.body);
      const assistantMsg = body.messages.find((m: { role: string }) => m.role === 'assistant');
      expect(assistantMsg).toBeDefined();
      expect(Array.isArray(assistantMsg.content)).toBe(true);

      const textBlocks = assistantMsg.content.filter((b: { type: string }) => b.type === 'text');
      const imgBlock = assistantMsg.content.find((b: { type: string }) => b.type === 'image');
      expect(textBlocks).toHaveLength(1);
      expect(textBlocks[0].text).toBe(ASSISTANT_QUESTION);
      expect(imgBlock).toBeDefined();
      expect(imgBlock.source.type).toBe('base64');
      expect(imgBlock.source.media_type).toBe('image/png');
      // Text must lead so the LLM reads the question before the image.
      expect(assistantMsg.content[0].type).toBe('text');
    });
  });

  describe('Google adapter', () => {
    it('emits text exactly once + image when text is in `content` string and items are media-only', () => {
      const req = googleAdapter.buildRequest({
        messages: [
          { role: 'user', content: 'create a unicorn webpage' },
          {
            role: 'assistant',
            content: ASSISTANT_QUESTION,
            content_items: [{ type: 'image', image: IMAGE_URL }],
          },
          { role: 'user', content: 'yes' },
        ],
        model: 'gemini-2.5-flash',
        apiKey: 'test-key',
        resolvedMedia: RESOLVED_MEDIA,
      });
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      expect(modelTurn).toBeDefined();
      expect(Array.isArray(modelTurn.parts)).toBe(true);

      const textParts = modelTurn.parts.filter((p: { text?: string }) =>
        typeof p.text === 'string' && p.text === ASSISTANT_QUESTION
      );
      const imgPart = modelTurn.parts.find((p: { inlineData?: unknown }) => p.inlineData);
      expect(textParts).toHaveLength(1);
      expect(imgPart).toBeDefined();
      // Text must lead so the LLM reads the question before the image.
      expect(modelTurn.parts[0].text).toBe(ASSISTANT_QUESTION);
    });
  });
});
