import { describe, it, expect } from 'vitest';
import type { ContentItem, FileContent } from '../../../src/uamp/types.js';

// Replicates the (now reverse-walked) `indexAvailableContent` logic from
// `core/agent.ts`. The agent's `present` tool resolves the content_id to a
// content_item via this index, then re-emits it as a `response.delta { file }`.
// The DocumentChip badge ("Created" vs "Edited") reads `metadata.command` off
// that re-emitted item, so the index MUST surface the latest known version of
// the file, not the original create-time version.
function indexAvailableContent(
  conversation: Array<{ content_items?: ContentItem[] }>,
  collectedContentItems: ContentItem[],
): Map<string, ContentItem> {
  const out = new Map<string, ContentItem>();
  for (let i = collectedContentItems.length - 1; i >= 0; i--) {
    const ci = collectedContentItems[i]!;
    const cid = (ci as { content_id?: string }).content_id;
    if (cid && !out.has(cid)) out.set(cid, ci);
  }
  for (let mi = conversation.length - 1; mi >= 0; mi--) {
    const items = conversation[mi]!.content_items;
    if (Array.isArray(items)) {
      for (let ci = items.length - 1; ci >= 0; ci--) {
        const item = items[ci]!;
        const cid = (item as { content_id?: string }).content_id;
        if (cid && !out.has(cid)) out.set(cid, item);
      }
    }
  }
  return out;
}

const fileItem = (contentId: string, command: string): FileContent =>
  ({
    type: 'file',
    file: { url: `/api/content/${contentId}` },
    content_id: contentId,
    filename: 'unicorn.html',
    mime_type: 'text/html',
    metadata: { command },
  } as unknown as FileContent);

describe('indexAvailableContent (latest-wins ordering)', () => {
  it('returns the latest content_item when the same content_id appears in multiple turns', () => {
    // Reproduces the bug the user observed: turn 1 created `unicorn.html`
    // (command='create'), turn 2 delegated to a sub-agent which str_replace'd
    // it. The sub-agent's tool_result row carries command='str_replace'.
    // Without latest-wins, present() picked up the older create-time entry
    // and the DocumentChip in the parent's final assistant message was
    // mis-labelled "Created" instead of "Edited".
    const conversation = [
      { content_items: [fileItem('uuid-1', 'create')] }, // turn 1 assistant
      { content_items: [fileItem('uuid-1', 'str_replace')] }, // turn 2 tool_result
    ];
    const index = indexAvailableContent(conversation, []);
    const item = index.get('uuid-1') as { metadata?: { command?: string } };
    expect(item?.metadata?.command).toBe('str_replace');
  });

  it('prefers collectedContentItems (current turn) over conversation history', () => {
    const conversation = [
      { content_items: [fileItem('uuid-1', 'create')] },
    ];
    const collected = [fileItem('uuid-1', 'str_replace')];
    const index = indexAvailableContent(conversation, collected);
    expect((index.get('uuid-1') as any).metadata.command).toBe('str_replace');
  });

  it('within a single message, latest content_item wins', () => {
    const conversation = [
      {
        content_items: [
          fileItem('uuid-1', 'create'),
          fileItem('uuid-1', 'str_replace'),
        ],
      },
    ];
    const index = indexAvailableContent(conversation, []);
    expect((index.get('uuid-1') as any).metadata.command).toBe('str_replace');
  });

  it('within collectedContentItems, latest entry wins', () => {
    const collected = [
      fileItem('uuid-1', 'create'),
      fileItem('uuid-1', 'str_replace'),
      fileItem('uuid-1', 'str_replace'),
    ];
    const index = indexAvailableContent([], collected);
    expect((index.get('uuid-1') as any).metadata.command).toBe('str_replace');
  });

  it('still indexes single-occurrence items unchanged', () => {
    const conversation = [
      { content_items: [fileItem('uuid-only', 'create')] },
    ];
    const index = indexAvailableContent(conversation, []);
    expect(index.size).toBe(1);
    expect((index.get('uuid-only') as any).metadata.command).toBe('create');
  });

  it('skips items without content_id', () => {
    const conversation = [
      {
        content_items: [
          { type: 'file', file: { url: '/x' } } as unknown as ContentItem,
          fileItem('uuid-1', 'str_replace'),
        ],
      },
    ];
    const index = indexAvailableContent(conversation, []);
    expect(index.size).toBe(1);
    expect(index.has('uuid-1')).toBe(true);
  });
});
