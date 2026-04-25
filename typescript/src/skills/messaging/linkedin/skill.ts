/**
 * LinkedInSkill — Posts API publishing on LinkedIn.
 *
 * Token resolution: ResolvedToken.token = OAuth2 access token (`w_member_social`
 * scope minimum). Metadata MUST carry `personUrn` (e.g. `urn:li:person:abc`)
 * or `organizationUrn` (e.g. `urn:li:organization:123`). When the integration
 * configures `requirePostApproval=true` and the host supplies an
 * `approvalGate`, the skill defers publishing to the host.
 *
 * The legacy `/v2/ugcPosts` endpoint has been replaced by the versioned
 * Posts API at `/rest/posts`. We pin the `LinkedIn-Version` to a known
 * supported month (overridable via `metadata.linkedinApiVersion` so hosts
 * can roll forward without an SDK release).
 *
 * Reference:
 *   https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api
 *   https://learn.microsoft.com/en-us/linkedin/marketing/community-management/contentapi-migration-guide
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

const PROVIDER = 'linkedin';
const PUBLISH_TOOL = 'linkedin_post';

interface LinkedInMetadata {
  personUrn?: string;
  organizationUrn?: string;
  /**
   * Override for the `LinkedIn-Version` request header. LinkedIn requires
   * a YYYYMM-format version on every Posts API call. Defaults to
   * `LINKEDIN_DEFAULT_VERSION` below; hosts can roll forward by setting
   * this on the connected_accounts.metadata row.
   */
  linkedinApiVersion?: string;
}

/**
 * Default `LinkedIn-Version` header value. LinkedIn supports a rolling
 * window of monthly versions; bump this constant when the previous version
 * approaches sunset (typically every ~12 months).
 */
const LINKEDIN_DEFAULT_VERSION = '202604';

export class LinkedInSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('linkedin', opts);
  }

  @tool({
    name: PUBLISH_TOOL,
    description:
      'Publish a LinkedIn UGC post (text, with optional article link). Subject to the ' +
      'host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string' },
        link: { type: 'string', description: 'Optional article URL.' },
        link_title: { type: 'string' },
        link_description: { type: 'string' },
        as_organization: {
          type: 'boolean',
          description: 'Publish as the connected organization instead of the personal profile.',
        },
      },
      required: ['text'],
    },
  })
  async post(args: {
    text: string;
    link?: string;
    link_title?: string;
    link_description?: string;
    as_organization?: boolean;
  }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    const gate = await this.maybeRequestApproval('linkedin_post', args as Record<string, unknown>);
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
        const meta = (t.metadata ?? {}) as LinkedInMetadata;
        const author = args.as_organization ? meta.organizationUrn : meta.personUrn;
        if (!author) throw new Error('linkedin_author_urn_missing');
        // Posts API uses a flat schema. Article content is optional; a plain
        // text post omits the `content` block entirely. See:
        //   https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api
        const body: Record<string, unknown> = {
          author,
          commentary: args.text,
          visibility: 'PUBLIC',
          distribution: {
            feedDistribution: 'MAIN_FEED',
            targetEntities: [],
            thirdPartyDistributionChannels: [],
          },
          lifecycleState: 'PUBLISHED',
          isReshareDisabledByAuthor: false,
          ...(args.link
            ? {
                content: {
                  article: {
                    source: args.link,
                    ...(args.link_title ? { title: args.link_title } : {}),
                    ...(args.link_description ? { description: args.link_description } : {}),
                  },
                },
              }
            : {}),
        };
        const linkedinVersion = meta.linkedinApiVersion ?? LINKEDIN_DEFAULT_VERSION;
        const r = await fetch('https://api.linkedin.com/rest/posts', {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${t.token}`,
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0',
            'LinkedIn-Version': linkedinVersion,
          },
          body: JSON.stringify(body),
        });
        const text = await r.text();
        if (!r.ok) {
          const err = new Error(text.slice(0, 300)) as Error & { status?: number };
          err.status = r.status;
          throw err;
        }
        // Posts API returns 201 + the post URN in the `x-restli-id` response
        // header; the body is empty. Fall back to a parsed `id` for forward
        // compatibility if LinkedIn ever ships a body.
        const externalMessageId =
          r.headers.get('x-restli-id') ??
          (text ? (JSON.parse(text) as { id?: string }).id : undefined);
        return { externalMessageId };
      },
    );
  }

  @prompt({ name: 'linkedin-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: PUBLISH_TOOL,
    });
  }

  /**
   * Standalone OAuth callback. Portal hosts use the central
   * /api/auth/connect/linkedin/callback route and ignore this endpoint.
   */
  @http({
    path: '/messaging/linkedin/oauth/callback',
    method: 'GET',
    auth: 'public',
    description: 'LinkedIn OAuth2 redirect target for non-portal hosts.',
  })
  async oauthCallback(req: Request): Promise<Response> {
    return handleOAuthCallback({
      request: req,
      provider: PROVIDER,
      redirectUri: `${this.httpBaseUrl}/api/agents/${this.agentId ?? 'agent'}/messaging/linkedin/oauth/callback`,
      tokenUrl: 'https://www.linkedin.com/oauth/v2/accessToken',
      tokenWriter: this.tokenWriter,
      agentId: this.agentId,
      integrationId: this.integrationId,
    });
  }
}
