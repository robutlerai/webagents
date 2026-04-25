/**
 * Messaging Skills — bridges (Telegram, Twilio, Slack, Discord,
 * WhatsApp, Messenger, Instagram) and publish-only platforms (LinkedIn,
 * Bluesky, Reddit).
 *
 * Each skill is DB-agnostic. Hosts wire credential resolution and
 * persistence through `MessagingSkillOptions.getToken` / `setToken`
 * (preferred) or rely on the env-resolver fallback for standalone usage.
 *
 * Reference architecture: docs/architecture/messaging-bridge.md.
 */

// Shared primitives (options, env-resolver, env-writer, bridge-context,
// signatures, refreshable-token-resolver, messaging-skill-base, meta-graph).
export * from './shared';

// Per-provider skills.
export { TelegramSkill } from './telegram';
export { TwilioSkill } from './twilio';
export { SlackSkill } from './slack';
export { DiscordSkill } from './discord';
export { WhatsAppSkill } from './whatsapp';
export { MessengerSkill } from './messenger';
export { InstagramSkill } from './instagram';
export { LinkedInSkill } from './linkedin';
export { BlueskySkill } from './bluesky';
export { RedditSkill } from './reddit';
export { TikTokSkill } from './tiktok';
export { SendGridSkill } from './sendgrid';
export { XSkill } from './x';
export { GoogleChatSkill } from './google-chat';
