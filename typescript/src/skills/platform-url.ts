/**
 * The platform's base URL, for every skill that calls the platform
 * (2026-09-25). The Python twin is
 * `python/webagents/agents/skills/robutler/platform_url.py`.
 *
 * ONE ANSWER: the skill's configured URL, else `ROBUTLER_API_URL`, else
 * `ROBUTLER_INTERNAL_API_URL`, else the CLI's `platform.url` (the portal
 * `webagents login` signed in to), else https://robutler.ai. The discovery
 * skill resolved it this way first (its file comment says why the public
 * variable comes before the in-cluster one); the lookup moved here when the
 * NLI skill needed it too. NLI defaulted to https://portal.webagents.ai, a host
 * that does not resolve, so `@name` reached nothing unless `baseUrl` was set.
 *
 * The CLI's configuration is read through a dynamic import, so a host with no
 * file system (a browser) gets the default.
 */

/** The platform when nothing names another. */
export const DEFAULT_PLATFORM_URL = 'https://robutler.ai';

export function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

/** `value` as a base URL: trimmed, no trailing slash, `undefined` when blank. */
export function trimmedUrl(value: string | undefined): string | undefined {
  const url = value?.trim().replace(/\/+$/, '');
  return url || undefined;
}

/** The configured URL, else the two variables; `undefined` when none names one. */
export function configuredPlatformUrl(configured?: string): string | undefined {
  return (
    trimmedUrl(configured) ??
    trimmedUrl(envVar('ROBUTLER_API_URL')) ??
    trimmedUrl(envVar('ROBUTLER_INTERNAL_API_URL'))
  );
}

/** `configuredPlatformUrl`, else the CLI's `platform.url`, else the default. */
export async function resolveSkillPlatformUrl(configured?: string): Promise<string> {
  const named = configuredPlatformUrl(configured);
  if (named) return named;
  try {
    const { resolvePlatformUrl } = await import('../cli/config-store.js');
    return trimmedUrl(resolvePlatformUrl()[0]) ?? DEFAULT_PLATFORM_URL;
  } catch {
    // No CLI configuration to read: the default stands.
    return DEFAULT_PLATFORM_URL;
  }
}
