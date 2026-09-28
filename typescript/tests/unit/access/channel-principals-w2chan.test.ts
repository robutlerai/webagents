/**
 * A channel sender's identity on a relayed turn (the channel relay, plan item
 * 2.2, 2026-09-26), from the table both SDKs run
 * (`python/tests/fixtures/access/channel_caller.json`; Python
 * tests/access/test_channel_principals_w2chan.py): what `portalCallerAuth`
 * makes of the platform's `caller.channel`, the principals the access skill
 * collects (the channel first, so it is the memory namespace), and the
 * sentence the model is told. The access block's decisions over channel
 * principals are in `policy.test.ts`, from `policy.json`.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { portalCallerAuth } from '../../../src/portal/connect';
import { channelIdentityOf } from '../../../src/access/caller';
import { userPrincipals, whoIsCalling } from '../../../src/skills/access/skill';
import { namespaceOf } from '../../../src/skills/memory/namespace';
import type { AuthInfo } from '../../../src/core/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/access/channel_caller.json'), 'utf8')) as {
  types: Record<string, string>;
  type_rule: string;
  assertions: Array<{
    case: string;
    caller: Record<string, unknown>;
    auth: { scope: string; user_id: string; username?: string | null; channel: { type: string; sender_id: string } | null };
    principals: string[];
    namespace: string | null;
    who: string;
  }>;
};

describe('the platform assertion with a channel', () => {
  for (const c of TABLE.assertions) {
    it(c.case, () => {
      const auth = portalCallerAuth(c.caller) as Record<string, unknown>;
      expect(auth).toBeDefined();
      expect(auth.scope).toBe(c.auth.scope);
      expect(auth.user_id).toBe(c.auth.user_id);
      expect(auth.provider).toBe('portal');
      if (c.auth.username) expect(auth.username).toBe(c.auth.username);
      else expect(auth).not.toHaveProperty('username');
      if (c.auth.channel) expect(auth.channel).toEqual(c.auth.channel);
      else expect(auth).not.toHaveProperty('channel');

      const principals = userPrincipals(auth as Partial<AuthInfo>);
      expect(principals).toEqual(c.principals);
      expect(namespaceOf(auth as Partial<AuthInfo>)).toBe(c.namespace);
      // The sentence, as the access skill leaves the auth after admitting the caller.
      expect(whoIsCalling({ ...(auth as Partial<AuthInfo>), principals, scopes: [] } as Partial<AuthInfo>)).toBe(c.who);
    });
  }
});

describe('channelIdentityOf', () => {
  it('lower-cases the type and keeps the sender id as written', () => {
    expect(channelIdentityOf({ type: 'Slack', sender_id: 'U9ABC' })).toEqual({ type: 'slack', sender_id: 'U9ABC' });
  });

  it('reads no channel from anything else', () => {
    for (const bad of [null, undefined, 'telegram:8842', [], { type: 'telegram' }, { sender_id: '8842' }, { type: 3, sender_id: '8842' }, { type: 'telegram', sender_id: 8842 }, { type: '-telegram', sender_id: '1' }, { type: 'telegram', sender_id: 'a'.repeat(201) }]) {
      expect(channelIdentityOf(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  it('accepts every type the platform names', () => {
    for (const type of Object.values(TABLE.types)) {
      expect(channelIdentityOf({ type, sender_id: 'x' })).toEqual({ type, sender_id: 'x' });
    }
  });
});
