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
 *   - OpenAI Responses: assistant turns become `output_text` items; the
 *     adapter stitches `m.content` together with describe-style markers
 *     for any media `content_items`.
 *
 * Each test asserts the prior-turn text reaches the provider payload exactly
 * once (no duplication).
 */

import { describe, it, expect } from 'vitest';
import { openaiAdapter } from '../../../src/adapters/responses.js';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import { googleAdapter } from '../../../src/adapters/google.js';

const IMAGE_UUID = '9e23908d-0b2a-4015-92b4-7cdf8c4796fe';
const IMAGE_URL = `/api/content/${IMAGE_UUID}`;
const RESOLVED_MEDIA = new Map([
  [IMAGE_URL, { kind: 'binary' as const, mimeType: 'image/png', base64: 'iVBORw0KGgo...' }],
]);

const ASSISTANT_QUESTION = 'I generated a unicorn for you. Want me to add music?';

describe('multimodal history round-trip — prior assistant turn text + media', () => {
  describe('OpenAI Responses adapter', () => {
    it('emits text exactly once + image marker when text is in `content` string and items are media-only', () => {
      // Responses API wire shape: assistant turns become message items with
      // `output_text` content. Media content_items render as describe-style
      // text markers — Responses doesn't accept inline image bytes inside
      // assistant `output_text`, and the historical "image" was a tool
      // side-effect that doesn't need byte-perfect replay.
      const req = openaiAdapter.buildRequest({
        messages: [
          { role: 'user', content: 'create a unicorn webpage' },
          {
            role: 'assistant',
            content: ASSISTANT_QUESTION,
            content_items: [{ type: 'image', image: { url: IMAGE_URL }, content_id: IMAGE_UUID }],
          },
          { role: 'user', content: 'yes' },
        ],
        model: 'gpt-5.5',
        apiKey: 'test-key',
        resolvedMedia: RESOLVED_MEDIA,
      });
      const body = JSON.parse(req.body);
      const assistantItem = body.input.find((i: { type: string; role?: string }) =>
        i.type === 'message' && i.role === 'assistant');
      expect(assistantItem).toBeDefined();
      const outputTextParts = assistantItem.content.filter((p: { type: string }) => p.type === 'output_text');
      expect(outputTextParts).toHaveLength(1);
      const stitched = outputTextParts[0].text as string;
      // Question text appears exactly once.
      const occurrences = (stitched.match(new RegExp(ASSISTANT_QUESTION, 'g')) ?? []).length;
      expect(occurrences).toBe(1);
      // Image marker is appended.
      expect(stitched).toContain('[Available image:');
      // Question precedes the image marker (text leads).
      const qIdx = stitched.indexOf(ASSISTANT_QUESTION);
      const imgIdx = stitched.indexOf('[Available image:');
      expect(qIdx).toBeGreaterThanOrEqual(0);
      expect(imgIdx).toBeGreaterThan(qIdx);
    });

    it('does not duplicate text when items happen to also carry the same text item', () => {
      // If an upstream caller already baked the text into items, the dedup
      // in convertMessagesToInput must NOT emit it twice.
      const req = openaiAdapter.buildRequest({
        messages: [
          {
            role: 'assistant',
            content: ASSISTANT_QUESTION,
            content_items: [
              { type: 'text', text: ASSISTANT_QUESTION },
              { type: 'image', image: { url: IMAGE_URL }, content_id: IMAGE_UUID },
            ],
          },
        ],
        model: 'gpt-5.5',
        apiKey: 'test-key',
        resolvedMedia: RESOLVED_MEDIA,
      });
      const body = JSON.parse(req.body);
      const assistantItem = body.input.find((i: { type: string; role?: string }) =>
        i.type === 'message' && i.role === 'assistant');
      const stitched = assistantItem.content[0].text as string;
      const occurrences = (stitched.match(new RegExp(ASSISTANT_QUESTION, 'g')) ?? []).length;
      expect(occurrences).toBe(1);
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
