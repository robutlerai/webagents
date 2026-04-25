/**
 * TikTokSkill - publish-only TikTok Content Posting API.
 *
 * Token resolution: ResolvedToken.token = OAuth2 access token granted via the
 * Content Posting API scopes (`video.upload` for the FILE_UPLOAD path,
 * `video.publish` to actually publish). `metadata.openId` is the TikTok user's
 * stable identifier returned by `/v2/user/info/`.
 *
 * The skill never holds large media in memory: callers pass a public URL and
 * TikTok pulls it via the `PULL_FROM_URL` source type. Direct file uploads
 * (`FILE_UPLOAD`) require multi-part chunked PUTs and are out of scope.
 *
 * Publishing is approval-gated through `PostApprovalGate.requestApproval`
 * just like LinkedIn / Bluesky / Reddit so the portal's pending-draft store
 * can mediate per-agent approval workflows.
 *
 * Reference: https://developers.tiktok.com/doc/content-posting-api-reference-direct-post/
 */
import { prompt, tool } from '../../../core/decorators';
import type { Context } from '../../../core/types';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  MessagingSkill,
  type MessagingSkillOptions,
} from '../shared';

const PROVIDER = 'tiktok';
const VIDEO_TOOL = 'tiktok_publish_video';
const PHOTO_TOOL = 'tiktok_publish_photo';

const TIKTOK_API = 'https://open.tiktokapis.com';
const POLL_INTERVAL_MS = 2_000;
const POLL_TIMEOUT_MS = 60_000;

interface TikTokMetadata {
  openId?: string;
}

type PrivacyLevel =
  | 'PUBLIC_TO_EVERYONE'
  | 'MUTUAL_FOLLOW_FRIENDS'
  | 'SELF_ONLY';

interface InitResponse {
  data?: { publish_id?: string };
  error?: { code?: string; message?: string };
}

interface StatusResponse {
  data?: {
    status?: 'PROCESSING_DOWNLOAD' | 'PROCESSING_UPLOAD' | 'PUBLISH_COMPLETE' | 'FAILED';
    fail_reason?: string;
    publicaly_available_post_id?: string[];
  };
  error?: { code?: string; message?: string };
}

export class TikTokSkill extends MessagingSkill {
  readonly provider = PROVIDER;

  constructor(opts: MessagingSkillOptions = {}) {
    super('tiktok', opts);
  }

  @tool({
    name: VIDEO_TOOL,
    description:
      'Publish a video to TikTok via the Content Posting API (PULL_FROM_URL). ' +
      'Subject to the host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        videoUrl: { type: 'string', description: 'Publicly fetchable URL TikTok will pull from.' },
        caption: { type: 'string' },
        privacyLevel: {
          type: 'string',
          enum: ['PUBLIC_TO_EVERYONE', 'MUTUAL_FOLLOW_FRIENDS', 'SELF_ONLY'],
          description: 'TikTok privacy level. Sandbox apps are forced to SELF_ONLY.',
        },
        disableComment: { type: 'boolean' },
        disableDuet: { type: 'boolean' },
        disableStitch: { type: 'boolean' },
      },
      required: ['videoUrl'],
    },
  })
  async publishVideo(args: {
    videoUrl: string;
    caption?: string;
    privacyLevel?: PrivacyLevel;
    disableComment?: boolean;
    disableDuet?: boolean;
    disableStitch?: boolean;
  }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    if (!args.videoUrl) return this.invalidInput('videoUrl required');
    const gate = await this.maybeRequestApproval(VIDEO_TOOL, args as Record<string, unknown>);
    if (gate) return { ok: true as const, data: gate };

    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'publish_video',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const body = {
          post_info: {
            title: args.caption ?? '',
            privacy_level: args.privacyLevel ?? 'SELF_ONLY',
            disable_comment: args.disableComment ?? false,
            disable_duet: args.disableDuet ?? false,
            disable_stitch: args.disableStitch ?? false,
          },
          source_info: {
            source: 'PULL_FROM_URL' as const,
            video_url: args.videoUrl,
          },
        };
        const init = await tiktokFetch<InitResponse>(
          `${TIKTOK_API}/v2/post/publish/video/init/`,
          t.token,
          body,
        );
        const publishId = init.data?.publish_id;
        if (!publishId) {
          throw new Error(init.error?.message ?? 'tiktok_init_returned_no_publish_id');
        }
        const final = await pollUntilTerminal(t.token, publishId);
        if (final.data?.status !== 'PUBLISH_COMPLETE') {
          throw new Error(final.data?.fail_reason ?? final.error?.message ?? 'tiktok_publish_failed');
        }
        return {
          externalMessageId: publishId,
          publishedPostIds: final.data.publicaly_available_post_id ?? [],
        };
      },
    );
  }

  @tool({
    name: PHOTO_TOOL,
    description:
      'Publish a photo carousel to TikTok via the Content Posting API. ' +
      'Subject to the host approval gate when configured.',
    parameters: {
      type: 'object',
      properties: {
        photoUrls: {
          type: 'array',
          items: { type: 'string' },
          description: 'Up to 35 publicly fetchable image URLs.',
        },
        caption: { type: 'string' },
        privacyLevel: {
          type: 'string',
          enum: ['PUBLIC_TO_EVERYONE', 'MUTUAL_FOLLOW_FRIENDS', 'SELF_ONLY'],
        },
        disableComment: { type: 'boolean' },
        photoCoverIndex: {
          type: 'integer',
          description:
            'Zero-based index of the cover image in photoUrls. Required by the TikTok ' +
            'Content Posting API; defaults to 0 (first photo) when omitted.',
          minimum: 0,
        },
      },
      required: ['photoUrls'],
    },
  })
  async publishPhoto(args: {
    photoUrls: string[];
    caption?: string;
    privacyLevel?: PrivacyLevel;
    disableComment?: boolean;
    photoCoverIndex?: number;
  }) {
    if (!this.capabilityEnabled('publish_posts')) return this.capabilityDisabled('publish_posts');
    if (!args.photoUrls?.length) return this.invalidInput('photoUrls required');
    const coverIndex = args.photoCoverIndex ?? 0;
    if (coverIndex < 0 || coverIndex >= args.photoUrls.length) {
      return this.invalidInput('photoCoverIndex out of range');
    }
    const gate = await this.maybeRequestApproval(PHOTO_TOOL, args as Record<string, unknown>);
    if (gate) return { ok: true as const, data: gate };

    return this.wrapApiCall(
      {
        provider: PROVIDER,
        type: 'publish_photo',
        agentId: this.agentId,
        integrationId: this.integrationId,
      },
      async () => {
        const t = await this.resolveToken();
        const meta = (t.metadata ?? {}) as TikTokMetadata;
        if (!meta.openId) throw new Error('tiktok_open_id_missing');
        const body = {
          post_info: {
            title: args.caption ?? '',
            description: args.caption ?? '',
            privacy_level: args.privacyLevel ?? 'SELF_ONLY',
            disable_comment: args.disableComment ?? false,
          },
          source_info: {
            source: 'PULL_FROM_URL' as const,
            photo_images: args.photoUrls,
            photo_cover_index: coverIndex,
          },
          post_mode: 'DIRECT_POST',
          media_type: 'PHOTO',
        };
        const init = await tiktokFetch<InitResponse>(
          `${TIKTOK_API}/v2/post/publish/content/init/`,
          t.token,
          body,
        );
        const publishId = init.data?.publish_id;
        if (!publishId) {
          throw new Error(init.error?.message ?? 'tiktok_init_returned_no_publish_id');
        }
        const final = await pollUntilTerminal(t.token, publishId);
        if (final.data?.status !== 'PUBLISH_COMPLETE') {
          throw new Error(final.data?.fail_reason ?? final.error?.message ?? 'tiktok_publish_failed');
        }
        return {
          externalMessageId: publishId,
          publishedPostIds: final.data.publicaly_available_post_id ?? [],
        };
      },
    );
  }

  @prompt({ name: 'tiktok-bridge-awareness' })
  async bridgePrompt(ctx?: Context): Promise<string | null> {
    const b = bridgeMatches(ctx, PROVIDER);
    if (!b) return null;
    return buildBridgeAwarenessPrompt({
      provider: PROVIDER,
      contactDisplayName: b.contactDisplayName,
      sendToolName: VIDEO_TOOL,
    });
  }
}

async function tiktokFetch<T>(url: string, token: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) {
    const err = new Error(text.slice(0, 300)) as Error & { status?: number };
    err.status = r.status;
    throw err;
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

async function pollUntilTerminal(token: string, publishId: string): Promise<StatusResponse> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const r = await tiktokFetch<StatusResponse>(
      `${TIKTOK_API}/v2/post/publish/status/fetch/`,
      token,
      { publish_id: publishId },
    );
    const s = r.data?.status;
    if (s === 'PUBLISH_COMPLETE' || s === 'FAILED') return r;
    await new Promise((res) => setTimeout(res, POLL_INTERVAL_MS));
  }
  throw new Error('tiktok_publish_timeout');
}
