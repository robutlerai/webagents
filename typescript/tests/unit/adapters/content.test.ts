/**
 * Content Helpers Unit Tests
 *
 * Tests for the generic UAMP content utility functions.
 */

import { describe, it, expect } from 'vitest';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, isTextDecodableMime } from '../../../src/adapters/content.js';

describe('extractContentRef', () => {
  it('extracts URL from a plain string', () => {
    expect(extractContentRef('/api/content/abc-123')).toBe('/api/content/abc-123');
  });

  it('extracts URL from a { url } object', () => {
    expect(extractContentRef({ url: '/api/content/abc-123' })).toBe('/api/content/abc-123');
  });

  it('returns null for null/undefined', () => {
    expect(extractContentRef(null)).toBeNull();
    expect(extractContentRef(undefined)).toBeNull();
  });

  it('returns null for non-string/non-object', () => {
    expect(extractContentRef(42)).toBeNull();
    expect(extractContentRef(true)).toBeNull();
  });

  it('returns null for object without url property', () => {
    expect(extractContentRef({ path: '/foo' })).toBeNull();
  });
});

describe('isUAMPContentArray', () => {
  it('detects image items', () => {
    expect(isUAMPContentArray([{ type: 'image', image: '/api/content/abc' }])).toBe(true);
  });

  it('detects audio items', () => {
    expect(isUAMPContentArray([{ type: 'audio', audio: '/api/content/abc' }])).toBe(true);
  });

  it('detects video items', () => {
    expect(isUAMPContentArray([{ type: 'video', video: '/api/content/abc' }])).toBe(true);
  });

  it('detects file items', () => {
    expect(isUAMPContentArray([{ type: 'file', file: '/api/content/abc' }])).toBe(true);
  });

  it('rejects text-only arrays', () => {
    expect(isUAMPContentArray([{ type: 'text', text: 'hello' }])).toBe(false);
  });

  it('rejects OpenAI-format arrays (image_url)', () => {
    expect(isUAMPContentArray([{ type: 'image_url', image_url: { url: 'http://...' } }])).toBe(false);
  });

  it('rejects empty arrays', () => {
    expect(isUAMPContentArray([])).toBe(false);
  });

  it('rejects non-arrays', () => {
    expect(isUAMPContentArray('not an array')).toBe(false);
    expect(isUAMPContentArray(null)).toBe(false);
    expect(isUAMPContentArray(undefined)).toBe(false);
  });

  it('detects mixed text + media arrays', () => {
    expect(isUAMPContentArray([
      { type: 'text', text: 'look at this' },
      { type: 'image', image: { url: '/api/content/abc' } },
    ])).toBe(true);
  });
});

describe('canonicalContentUrl', () => {
  it('extracts canonical URL from /api/content/ path', () => {
    const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
    expect(canonicalContentUrl(`/api/content/${uuid}`)).toBe(`/api/content/${uuid}`);
  });

  it('extracts canonical URL from full URL', () => {
    const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
    expect(canonicalContentUrl(`https://example.com/api/content/${uuid}?token=abc`)).toBe(`/api/content/${uuid}`);
  });

  it('returns null for non-content URLs', () => {
    expect(canonicalContentUrl('https://example.com/image.png')).toBeNull();
  });

  it('returns null for empty string', () => {
    expect(canonicalContentUrl('')).toBeNull();
  });
});

describe('describeContentItem', () => {
  it('includes path= in marker when meta.path is set', () => {
    const item = {
      type: 'file',
      content_id: 'abc-123',
      filename: 'unicorn.html',
      mime_type: 'text/html',
      metadata: { command: 'create', path: '/home/user/unicorn.html', by: 'claude-agent' },
    };
    const marker = describeContentItem(item);
    expect(marker).toContain('path=/home/user/unicorn.html');
    expect(marker).toContain('command=create');
    expect(marker).toContain('by=@claude-agent');
  });

  it('omits path= when meta.path is absent', () => {
    const item = {
      type: 'file',
      content_id: 'def-456',
      filename: 'notes.md',
      metadata: { command: 'create' },
    };
    const marker = describeContentItem(item);
    expect(marker).not.toContain('path=');
    expect(marker).toContain('command=create');
  });

  it('omits path= when metadata is missing entirely', () => {
    const item = {
      type: 'image',
      content_id: 'img-001',
    };
    const marker = describeContentItem(item);
    expect(marker).not.toContain('path=');
  });

  it('marks text-decodable files as analysable when textDecodableMime is set, even if supportedDocMimes rejects', () => {
    const item = {
      type: 'file',
      content_id: 'html-1',
      filename: 'unicorn.html',
      mime_type: 'text/html',
    };
    const marker = describeContentItem(item, {
      supportedModalities: new Set(['image']),
      supportedDocMimes: new Set(['application/pdf']),
      textDecodableMime: isTextDecodableMime,
    });
    expect(marker).toContain('read_content');
    expect(marker).not.toContain('NOT analysable');
  });

  it('marks non-PDF files as NOT analysable when neither supportedDocMimes nor textDecodableMime accept them', () => {
    const item = {
      type: 'file',
      content_id: 'docx-1',
      filename: 'notes.docx',
      mime_type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    };
    const marker = describeContentItem(item, {
      supportedModalities: new Set(['image']),
      supportedDocMimes: new Set(['application/pdf']),
      textDecodableMime: isTextDecodableMime,
    });
    expect(marker).toContain('NOT analysable');
  });

  it('still marks PDF as analysable via supportedDocMimes', () => {
    const item = {
      type: 'file',
      content_id: 'pdf-1',
      filename: 'report.pdf',
      mime_type: 'application/pdf',
    };
    const marker = describeContentItem(item, {
      supportedModalities: new Set(['image']),
      supportedDocMimes: new Set(['application/pdf']),
      textDecodableMime: isTextDecodableMime,
    });
    expect(marker).toContain('read_content');
    expect(marker).not.toContain('NOT analysable');
  });
});

describe('isTextDecodableMime', () => {
  it('accepts all text/* mimes', () => {
    expect(isTextDecodableMime('text/plain')).toBe(true);
    expect(isTextDecodableMime('text/html')).toBe(true);
    expect(isTextDecodableMime('text/css')).toBe(true);
    expect(isTextDecodableMime('text/javascript')).toBe(true);
    expect(isTextDecodableMime('text/x-python')).toBe(true);
  });

  it('accepts text-shaped application/* allowlist', () => {
    expect(isTextDecodableMime('application/json')).toBe(true);
    expect(isTextDecodableMime('application/xml')).toBe(true);
    expect(isTextDecodableMime('application/yaml')).toBe(true);
    expect(isTextDecodableMime('application/x-sh')).toBe(true);
  });

  it('rejects binary doc mimes', () => {
    expect(isTextDecodableMime('application/pdf')).toBe(false);
    expect(isTextDecodableMime('application/vnd.openxmlformats-officedocument.wordprocessingml.document')).toBe(false);
    expect(isTextDecodableMime('image/png')).toBe(false);
    expect(isTextDecodableMime('application/octet-stream')).toBe(false);
  });

  it('handles empty/null/undefined safely', () => {
    expect(isTextDecodableMime('')).toBe(false);
    expect(isTextDecodableMime(undefined)).toBe(false);
    expect(isTextDecodableMime(null)).toBe(false);
  });
});
