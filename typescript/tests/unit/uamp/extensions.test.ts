/**
 * UAMP Extension envelope tests
 *
 * Covers the `extension.message` frame that carries opaque sub-protocol
 * payloads (e.g. `workspace.terminal`). Asserts:
 *   - factory shape with and without `extension_version`
 *   - parseEvent / serializeEvent round-trip preserves every field
 *   - isExtensionMessage returns true on the envelope, false otherwise
 *   - isClientEvent and isServerEvent BOTH return false on the envelope
 *     (the legacy guards intentionally do not classify it)
 */

import { describe, it, expect } from 'vitest';
import {
  createBaseEvent,
  createExtensionMessage,
  createInputTextEvent,
  createResponseDeltaEvent,
  isClientEvent,
  isExtensionMessage,
  isServerEvent,
  parseEvent,
  serializeEvent,
} from '../../../src/uamp/events.js';
import type { ExtensionMessageEvent } from '../../../src/uamp/events.js';

describe('UAMP extension envelope', () => {
  describe('createExtensionMessage', () => {
    it('omits extension_version when not provided', () => {
      const env = createExtensionMessage('workspace.terminal', { type: 'open' });
      expect(env.type).toBe('extension.message');
      expect(env.namespace).toBe('workspace.terminal');
      expect(env.payload).toEqual({ type: 'open' });
      expect(env.event_id).toBeDefined();
      expect('extension_version' in env).toBe(false);
    });

    it('emits extension_version when provided', () => {
      const env = createExtensionMessage(
        'workspace.terminal',
        { type: 'open', session_id: 't1' },
        { version: 2 },
      );
      expect(env.extension_version).toBe(2);
    });

    it('treats opts.version === 0 as a real value (not stripped)', () => {
      // Opt-in numeric value, not a falsy default.
      const env = createExtensionMessage('x.y', null, { version: 0 });
      expect(env.extension_version).toBe(0);
    });

    it('preserves arbitrary payload shape (opaque to UAMP)', () => {
      const payload = {
        type: 'in',
        session_id: 'abc',
        data: 'aGVsbG8=',
        nested: { count: 3, list: [1, 2, 3] },
      };
      const env = createExtensionMessage('workspace.terminal', payload);
      expect(env.payload).toEqual(payload);
    });
  });

  describe('parseEvent / serializeEvent round-trip', () => {
    it('preserves every envelope field', () => {
      const env = createExtensionMessage(
        'workspace.terminal',
        { type: 'ready', session_id: 't1' },
        { version: 1 },
      );
      const parsed = parseEvent(serializeEvent(env));
      expect(parsed).toEqual(env);
    });

    it('round-trips without extension_version', () => {
      const env = createExtensionMessage('acme.audit', { kind: 'login' });
      const parsed = parseEvent(serializeEvent(env));
      expect(parsed.type).toBe('extension.message');
      if (isExtensionMessage(parsed)) {
        expect(parsed.namespace).toBe('acme.audit');
        expect(parsed.payload).toEqual({ kind: 'login' });
        expect(parsed.extension_version).toBeUndefined();
      } else {
        throw new Error('expected envelope');
      }
    });
  });

  describe('isExtensionMessage', () => {
    it('returns true for an envelope', () => {
      const env = createExtensionMessage('workspace.terminal', {});
      expect(isExtensionMessage(env)).toBe(true);
    });

    it('returns false for a legacy client event', () => {
      const ev = createInputTextEvent('hello');
      expect(isExtensionMessage(ev)).toBe(false);
    });

    it('returns false for a legacy server event', () => {
      const ev = createResponseDeltaEvent('resp_1', 'hi');
      expect(isExtensionMessage(ev)).toBe(false);
    });
  });

  describe('legacy guards do NOT classify the envelope', () => {
    it('isClientEvent returns false for an envelope', () => {
      const env = createExtensionMessage('workspace.terminal', {});
      expect(isClientEvent(env)).toBe(false);
    });

    it('isServerEvent returns false for an envelope', () => {
      const env = createExtensionMessage('workspace.terminal', {});
      expect(isServerEvent(env)).toBe(false);
    });

    it('legacy guards still classify legacy events correctly', () => {
      const client = createInputTextEvent('hi');
      const server = createResponseDeltaEvent('resp_1', 'hi');
      expect(isClientEvent(client)).toBe(true);
      expect(isServerEvent(client)).toBe(false);
      expect(isClientEvent(server)).toBe(false);
      expect(isServerEvent(server)).toBe(true);
    });
  });

  describe('parseEvent accepts envelope wire format', () => {
    it('parses a hand-crafted envelope with extension_version', () => {
      const wire = JSON.stringify({
        ...createBaseEvent('extension.message'),
        type: 'extension.message',
        namespace: 'workspace.terminal',
        extension_version: 1,
        payload: { type: 'open', session_id: 't1', cols: 80, rows: 24 },
      } satisfies ExtensionMessageEvent);
      const parsed = parseEvent(wire);
      expect(isExtensionMessage(parsed)).toBe(true);
    });
  });
});
