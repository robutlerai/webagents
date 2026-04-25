/**
 * Default env-backed token resolver used when a host does not supply one.
 *
 * Each provider declares the env vars it expects to read. When the host
 * sets `getToken`, this fallback is bypassed entirely — only standalone
 * usage (e.g. an OSS agent that imports `TelegramSkill` directly) hits it.
 *
 * Keeping the fallback inside the SDK lets a developer do:
 *
 * ```ts
 * import { TelegramSkill } from 'webagents/skills/messaging/telegram';
 * agent.use(new TelegramSkill());
 * // → reads TELEGRAM_BOT_TOKEN from process.env
 * ```
 *
 * with no portal connection.
 */
import type { ResolvedToken, TokenResolver } from './options';

export interface ProviderEnvSpec {
  /** Primary credential env var (token / api key / bot token). */
  tokenEnv: string;
  /** Optional refresh-token env var. */
  refreshTokenEnv?: string;
  /** Additional metadata env vars → metadata key. */
  metadataEnv?: Record<string, string>;
}

export const PROVIDER_ENV: Record<string, ProviderEnvSpec> = {
  telegram: {
    tokenEnv: 'TELEGRAM_BOT_TOKEN',
    metadataEnv: { botUsername: 'TELEGRAM_BOT_USERNAME' },
  },
  twilio: {
    tokenEnv: 'TWILIO_AUTH_TOKEN',
    metadataEnv: {
      accountSid: 'TWILIO_ACCOUNT_SID',
      fromNumber: 'TWILIO_FROM_NUMBER',
      messagingServiceSid: 'TWILIO_MESSAGING_SERVICE_SID',
    },
  },
  slack: {
    tokenEnv: 'SLACK_BOT_TOKEN',
    refreshTokenEnv: 'SLACK_REFRESH_TOKEN',
    metadataEnv: {
      teamId: 'SLACK_TEAM_ID',
      botUserId: 'SLACK_BOT_USER_ID',
      signingSecret: 'SLACK_SIGNING_SECRET',
    },
  },
  discord: {
    tokenEnv: 'DISCORD_BOT_TOKEN',
    metadataEnv: {
      applicationId: 'DISCORD_APPLICATION_ID',
      publicKey: 'DISCORD_PUBLIC_KEY',
    },
  },
  whatsapp: {
    tokenEnv: 'WHATSAPP_ACCESS_TOKEN',
    metadataEnv: {
      phoneNumberId: 'WHATSAPP_PHONE_NUMBER_ID',
      wabaId: 'WHATSAPP_WABA_ID',
    },
  },
  messenger: {
    tokenEnv: 'MESSENGER_PAGE_TOKEN',
    metadataEnv: { pageId: 'MESSENGER_PAGE_ID' },
  },
  instagram: {
    tokenEnv: 'INSTAGRAM_ACCESS_TOKEN',
    metadataEnv: { igUserId: 'INSTAGRAM_USER_ID' },
  },
  linkedin: {
    tokenEnv: 'LINKEDIN_ACCESS_TOKEN',
    refreshTokenEnv: 'LINKEDIN_REFRESH_TOKEN',
    metadataEnv: { memberUrn: 'LINKEDIN_MEMBER_URN' },
  },
  bluesky: {
    tokenEnv: 'BLUESKY_APP_PASSWORD',
    metadataEnv: { handle: 'BLUESKY_HANDLE', did: 'BLUESKY_DID' },
  },
  reddit: {
    tokenEnv: 'REDDIT_ACCESS_TOKEN',
    refreshTokenEnv: 'REDDIT_REFRESH_TOKEN',
    metadataEnv: { username: 'REDDIT_USERNAME' },
  },
};

export function defaultEnvTokenResolver(): TokenResolver {
  return {
    async getToken({ provider }) {
      const spec = PROVIDER_ENV[provider];
      if (!spec) return null;
      const token = process.env[spec.tokenEnv];
      if (!token) return null;
      const metadata: Record<string, string> = {};
      if (spec.metadataEnv) {
        for (const [key, env] of Object.entries(spec.metadataEnv)) {
          const v = process.env[env];
          if (v) metadata[key] = v;
        }
      }
      const out: ResolvedToken = {
        token,
        metadata,
      };
      if (spec.refreshTokenEnv && process.env[spec.refreshTokenEnv]) {
        out.refreshToken = process.env[spec.refreshTokenEnv];
      }
      return out;
    },
  };
}
