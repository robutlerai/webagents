/**
 * BlueskySkill — AT Protocol post publishing.
 *
 * Token resolution: ResolvedToken.token = AT Protocol session JWT
 * (`accessJwt`). `metadata.did` MUST be present so the skill can author
 * records on the correct repo.
 *
 * Reference: https://docs.bsky.app/docs/api/com-atproto-repo-create-record
 */
import { prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  MessagingSkill,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'bluesky';
const PUBLISH_TOOL = 'bluesky_post';

interface BlueskyMetadata {
  did?: string;
  pdsHost?: string;
}

export class BlueskySkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('bluesky', opts);
  }

  @tool({
    name: PUBLISH_TOOL,
    description:
      'Publish a Bluesky (AT Protocol) post. Subject to the host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Post text (max 300 graphemes per AT Protocol).' },
        langs: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional BCP47 language tags (e.g. ["en"]).',
        },
      },
      required: ['text'],
    },
  })
  async post(args: { text: string; langs?: string[] }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    const gate = await this.maybeRequestApproval('bluesky_post', args as Record<string, unknown>);
    if (gate) return { ok: true as const, data: gate };

    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'post',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const meta = (t.metadata ?? {}) as BlueskyMetadata;
        if (!meta.did) throw new Error('bluesky_did_missing');
        const host = meta.pdsHost ?? 'https://bsky.social';
        const body = {
          repo: meta.did,
          collection: 'app.bsky.feed.post',
          record: {
            $type: 'app.bsky.feed.post',
            text: args.text,
            createdAt: new Date().toISOString(),
            ...(args.langs?.length ? { langs: args.langs } : {}),
          },
        };
        const r = await fetch(`${host}/xrpc/com.atproto.repo.createRecord`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${t.token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const text = await r.text();
        if (!r.ok) {
          const err = new Error(text.slice(0, 300)) as Error & { status?: number };
          err.status = r.status;
          throw err;
        }
        const json = text ? (JSON.parse(text) as { uri?: string; cid?: string }) : {};
        return { externalMessageId: json.uri, cid: json.cid };
      },
    );
  }

  @prompt({ name: 'bluesky-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: PUBLISH_TOOL,
    });
  }
}
