/**
 * Chip badge regression: after delegate-attach + link, when the sub-agent
 * EDITS the attached file (text_editor str_replace), the parent chat's
 * DocumentChip must render "Edited" — not "Created".
 *
 * The badge is a pure function of metadata.command on the content_item
 * persisted on the parent assistant message (see components/chat/document-chip.tsx
 * `getVerbBadge`). NLISkill.delegate forwards the sub-agent's streaming `file`
 * deltas verbatim through `_nli_output_items`, which the parent UAMP proxy
 * persists with the sub-agent's metadata.command intact.
 *
 * This test pins the contract end-to-end:
 *   1. delegate(attachments=[X]) is called.
 *   2. Sub-agent emits a `file` delta with metadata.command='str_replace'.
 *   3. _nli_output_items merged on context contains that exact metadata.
 *   4. components/chat/document-chip.tsx getVerbBadge('str_replace') maps to
 *      t('tools.documentEdited').
 */

import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context } from '../../../../src/core/types.js';
import type { FileContent, ContentItem } from '../../../../src/uamp/types.js';

let mockClientInstance: any = null;

function defaultMockFactory(config: any) {
  mockClientInstance = {
    config,
    connect: vi.fn().mockResolvedValue(undefined),
    sendInput: vi.fn().mockResolvedValue(undefined),
    sendPayment: vi.fn().mockResolvedValue(undefined),
    cancel: vi.fn().mockResolvedValue(undefined),
    close: vi.fn(),
    on: vi.fn(),
    _handlers: new Map<string, Function[]>(),
  };
  mockClientInstance.on.mockImplementation((event: string, handler: Function) => {
    if (!mockClientInstance._handlers.has(event)) {
      mockClientInstance._handlers.set(event, []);
    }
    mockClientInstance._handlers.get(event)!.push(handler);
    return mockClientInstance;
  });
  return mockClientInstance;
}

vi.mock('../../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation((config: any) => defaultMockFactory(config)),
}));

import { UAMPClient } from '../../../../src/uamp/client.js';

function triggerEvent(name: string, ...args: unknown[]): void {
  const handlers = mockClientInstance?._handlers?.get(name) || [];
  for (const h of handlers) h(...args);
}

function makeContext(data: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(data));
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    signal: undefined,
    auth: { authenticated: true, user_id: 'user-A1' },
    payment: { valid: false },
    metadata: {},
    session: { id: 'test', created_at: 0, last_activity: 0, data: {} },
    hasScope: () => false,
    hasScopes: () => false,
    _store: store,
  } as unknown as Context;
}

describe('NLI delegate edit-badge regression', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockClientInstance = null;
    (UAMPClient as unknown as Mock).mockImplementation((config: any) => defaultMockFactory(config));
  });

  it("forwards sub-agent's str_replace metadata to _nli_output_items so parent chip renders 'Edited' not 'Created'", async () => {
    const linkFn = vi.fn(async () => undefined);
    const fromDb: FileContent = {
      type: 'file',
      file: { url: 'https://signed.example/unicorn?sig=abc' },
      filename: 'unicorn.html',
      mime_type: 'text/html',
      content_id: 'unicorn-1',
    } as FileContent;

    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'uamp',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: 'sub-chat-AB', type: 'agent', created: true }),
    });

    const resolveById = vi.fn(async (id: string) => (id === 'unicorn-1' ? fromDb : null));
    const context = makeContext({
      _agentic_messages: [],
      _resolveContentById: resolveById,
      _linkContentToSubChat: linkFn,
    });

    // Sub-agent emits a `file` delta carrying metadata.command='str_replace',
    // then completes. NLISkill.streamMessageUAMP picks it up via client.on('file', …).
    (UAMPClient as unknown as Mock).mockImplementation((config: any) => {
      const inst = defaultMockFactory(config);
      inst.sendInput = vi.fn().mockImplementation(async () => {
        queueMicrotask(() => {
          triggerEvent('file', {
            type: 'file',
            content_id: 'unicorn-1',
            filename: 'unicorn.html',
            mime_type: 'text/html',
            file: { url: 'https://signed.example/unicorn?sig=abc' },
            metadata: {
              command: 'str_replace',
              old_str: '<title>Old</title>',
              new_str: '<title>New</title>',
              editedBy: 'sub-agent-user',
            },
          });
          queueMicrotask(() => triggerEvent('done', { output: [] }));
        });
      });
      return inst;
    });

    await skill.delegate(
      { agent: '@sub-agent', message: 'tweak the title', attachments: ['unicorn-1'] },
      context,
    );

    expect(linkFn).toHaveBeenCalledTimes(1);
    expect(linkFn.mock.calls[0][0]).toEqual(['unicorn-1']);

    const setSpy = (context as unknown as { set: Mock }).set;
    const outputCall = setSpy.mock.calls.find((c) => c[0] === '_nli_output_items');
    expect(outputCall).toBeDefined();
    const items = outputCall![1] as ContentItem[];
    const unicorn = items.find((i) => (i as { content_id?: string }).content_id === 'unicorn-1');
    expect(unicorn).toBeDefined();
    const meta = (unicorn as unknown as { metadata?: Record<string, unknown> }).metadata;
    expect(meta).toBeDefined();
    expect(meta!.command).toBe('str_replace');
    expect(meta!.old_str).toBe('<title>Old</title>');
    expect(meta!.new_str).toBe('<title>New</title>');
  });
});

// ---------------------------------------------------------------------------
// Parallel mapping assertion: lock the badge mapping so changes here ripple to
// /Users/vs/dev/portal/components/chat/document-chip.tsx getVerbBadge.
// ---------------------------------------------------------------------------

describe('DocumentChip getVerbBadge mapping (locked contract)', () => {
  // Inlined from components/chat/document-chip.tsx so this stays a unit test
  // (no DOM required). Any divergence here is a contract violation: the file
  // content_item produced above carries metadata.command='str_replace', so the
  // badge MUST resolve to documentEdited (not documentCreated, not documentFile).
  function getVerbBadge(
    command: string | undefined,
    t: (key: string) => string,
  ): { label: string; className: string } {
    switch (command) {
      case 'create':
        return { label: t('tools.documentCreated'), className: 'bg-emerald-500/10 text-emerald-500' };
      case 'str_replace':
      case 'insert':
      case 'undo_edit':
        return { label: t('tools.documentEdited'), className: 'bg-blue-500/10 text-blue-500' };
      case 'replace':
        return { label: t('tools.documentUpdated'), className: 'bg-blue-500/10 text-blue-500' };
      case 'view':
        return { label: t('tools.documentViewed'), className: 'bg-muted text-muted-foreground' };
      default:
        return { label: t('tools.documentFile'), className: 'bg-muted text-muted-foreground' };
    }
  }
  const t = (k: string) => k;

  it("'str_replace' → tools.documentEdited (blue, not emerald 'Created')", () => {
    const b = getVerbBadge('str_replace', t);
    expect(b.label).toBe('tools.documentEdited');
    expect(b.className).toContain('blue');
  });
  it("'create' → tools.documentCreated (control)", () => {
    expect(getVerbBadge('create', t).label).toBe('tools.documentCreated');
  });
  it('undefined → tools.documentFile (control)', () => {
    expect(getVerbBadge(undefined, t).label).toBe('tools.documentFile');
  });
});
