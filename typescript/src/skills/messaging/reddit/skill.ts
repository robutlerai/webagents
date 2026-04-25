/**
 * RedditSkill — submit posts to a subreddit using OAuth bearer tokens.
 *
 * Token resolution: ResolvedToken.token = OAuth2 bearer token (`submit`
 * scope minimum). Reddit requires a unique `User-Agent`; the skill reads
 * it from `metadata.userAgent` then falls back to `REDDIT_USER_AGENT`.
 *
 * Reference: https://www.reddit.com/dev/api/#POST_api_submit
 */
import { http, prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  handleOAuthCallback,
  MessagingSkill,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'reddit';
const PUBLISH_TOOL = 'reddit_post';

interface RedditMetadata {
  userAgent?: string;
}

export class RedditSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('reddit', opts);
  }

  @tool({
    name: PUBLISH_TOOL,
    description:
      'Submit a post to a subreddit. Use kind="self" for text posts (text in `text`) ' +
      'and kind="link" for URL posts (URL in `url`). Subject to the host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        subreddit: { type: 'string', description: 'Subreddit name without the leading r/.' },
        title: { type: 'string' },
        kind: { type: 'string', enum: ['self', 'link'], default: 'self' },
        text: { type: 'string' },
        url: { type: 'string' },
        nsfw: { type: 'boolean' },
        spoiler: { type: 'boolean' },
      },
      required: ['subreddit', 'title'],
    },
  })
  async post(args: {
    subreddit: string;
    title: string;
    kind?: 'self' | 'link';
    text?: string;
    url?: string;
    nsfw?: boolean;
    spoiler?: boolean;
  }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    const gate = await this.maybeRequestApproval('reddit_post', args as Record<string, unknown>);
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
        const meta = (t.metadata ?? {}) as RedditMetadata;
        const userAgent = meta.userAgent ?? process.env.REDDIT_USER_AGENT ?? 'webagents-reddit-skill/1.0';
        const kind = args.kind ?? 'self';
        const body = new URLSearchParams({
          sr: args.subreddit,
          title: args.title,
          kind,
          api_type: 'json',
        });
        if (kind === 'self' && args.text) body.set('text', args.text);
        if (kind === 'link' && args.url) body.set('url', args.url);
        if (args.nsfw) body.set('nsfw', 'true');
        if (args.spoiler) body.set('spoiler', 'true');

        const r = await fetch('https://oauth.reddit.com/api/submit', {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${t.token}`,
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': userAgent,
          },
          body: body.toString(),
        });
        const text = await r.text();
        if (!r.ok) {
          const err = new Error(text.slice(0, 300)) as Error & { status?: number };
          err.status = r.status;
          throw err;
        }
        const parsed = text
          ? (JSON.parse(text) as {
              json?: { data?: { url?: string; id?: string }; errors?: unknown[] };
            })
          : {};
        const errors = parsed.json?.errors ?? [];
        if (errors.length > 0) {
          throw new Error(`reddit_submit_errors:${JSON.stringify(errors)}`);
        }
        return {
          externalMessageId: parsed.json?.data?.id,
          url: parsed.json?.data?.url,
        };
      },
    );
  }

  @prompt({ name: 'reddit-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: PUBLISH_TOOL,
    });
  }

  /** Standalone OAuth callback. Portal hosts use the central route. */
  @http({
    path: '/messaging/reddit/oauth/callback',
    method: 'GET',
    auth: 'public',
    description: 'Reddit OAuth2 redirect target for non-portal hosts.',
  })
  async oauthCallback(req: Request): Promise<Response> {
    return handleOAuthCallback({
      request: req,
      provider: PROVIDER,
      redirectUri: `${this.httpBaseUrl}/api/agents/${this.agentId ?? 'agent'}/messaging/reddit/oauth/callback`,
      tokenUrl: 'https://www.reddit.com/api/v1/access_token',
      tokenWriter: this.tokenWriter,
      agentId: this.agentId,
      integrationId: this.integrationId,
    });
  }
}
