/**
 * Who you are on Robutler, and which agent this folder publishes to
 * (2026-09-24): `whoami`, `link` and `unlink`, in the words the Python CLI
 * uses (`python/webagents/cli/account.py`), so the two CLIs answer alike.
 *
 * `link` binds the folder to an agent you already have, so `publish` updates
 * it rather than creating a second one: a platform username is minted once,
 * and a duplicate cannot be taken back. The binding is `link.agentId` and
 * `link.agentName` in the folder's own `.webagents/config.json`.
 */

import { ConfigStore, cliCommand, resolvePlatformUrl } from './config-store';
import { projectLink } from './publish';

export type WhoAmI =
  | { ok: true; username: string; platform: string; message: string }
  | { ok: false; code: string; message: string; fix: string };

function hostOf(url: string): string {
  return url.replace(/^https?:\/\//, '');
}

/** The signed-in account, asked of the platform (`GET /api/users/me`). */
export async function whoAmI(): Promise<WhoAmI> {
  const { getToken } = await import('./credentials.js');
  const [portalUrl] = resolvePlatformUrl();
  const host = hostOf(portalUrl);
  const token = await getToken();
  if (!token) return noToken(host);
  let res: Response;
  try {
    res = await fetch(`${portalUrl}/api/users/me`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(8000),
    });
  } catch (error) {
    return {
      ok: false,
      code: 'unreachable',
      message: `Could not reach ${host}: ${(error as Error).message}`,
      fix: 'Check the network, or `webagents config get platform.url`.',
    };
  }
  if (res.status === 401) {
    return { ok: false, code: 'expired', message: `Your sign-in on ${host} has expired.`, fix: `Run \`${cliCommand('login')}\`.` };
  }
  if (!res.ok) return { ok: false, code: 'http_error', message: `${host} answered ${res.status}.`, fix: '' };
  const data = (await res.json().catch(() => ({}))) as { user?: { username?: string }; username?: string };
  const username = String(data.user?.username ?? data.username ?? '');
  return { ok: true, username, platform: portalUrl, message: `Signed in as @${username} on ${host}.` };
}

/**
 * No token to use, and why (keychain-ux, 2026-09-27): a sign-in this run could
 * not read without a macOS dialog nobody can answer is the one sentence, not
 * "not signed in"; and when the Python CLI is signed in here, the answer says
 * each CLI keeps its own sign-in. The Python `account._no_token`.
 */
async function noToken(host: string): Promise<WhoAmI> {
  const { CLI_NAMESPACE, tokenBlocked } = await import('./credentials');
  const { KeychainRecord, blockedSentence, otherRuntimeItems, otherSignedInSentence, recordPath, serviceName } = await import('../skills/secrets/keychain-ux');
  const { globalDir, profileName, scopedNamespace } = await import('./config-store');
  const path = await import('node:path');
  const namespace = scopedNamespace(CLI_NAMESPACE, profileName());
  if (await tokenBlocked()) {
    return { ok: false, code: 'keychain_needs_terminal', message: await blockedSentence(serviceName(namespace)), fix: `Run \`${cliCommand('whoami')}\` in a terminal.` };
  }
  let message = `Not signed in to ${host}.`;
  const record = new KeychainRecord(recordPath(path.join(globalDir(), 'secrets')));
  if ((await otherRuntimeItems(record, namespace)).includes('platform_token')) message = `${message} ${otherSignedInSentence()}`;
  return { ok: false, code: 'not_signed_in', message, fix: `Run \`${cliCommand('login')}\`.` };
}

/**
 * `webagents whoami` in a terminal (keychain-ux, 2026-09-27): read what a run
 * with nobody to answer could not (the daemon, `serve`, a script), so macOS
 * asks now, once, after the explanation, and that run can read it next time.
 * Values are read and dropped. The Python `account.settle_keychain`.
 */
export async function settleKeychain(): Promise<number> {
  const { dialogsAllowed, recordPath, settlePending } = await import('../skills/secrets/keychain-ux');
  if (!dialogsAllowed()) return 0;
  const { openSecretStore } = await import('../skills/secrets/store');
  const { globalDir } = await import('./config-store');
  const path = await import('node:path');
  const os = await import('node:os');
  const defaultDir = process.env.WEBAGENTS_SECRETS_DIR || path.join(os.homedir(), '.webagents', 'secrets');
  const paths = [recordPath(path.join(globalDir(), 'secrets')), recordPath(defaultDir)];
  return settlePending(paths, (namespace, secretsDir) => openSecretStore({ namespace, secretsDir, quiet: true }));
}

/** What `logout` and `secrets remove` say about an old `webagents:` item only a macOS dialog could remove. */
export async function leftBehindLines(items: Array<{ item: string; account: string }>): Promise<string[]> {
  const { leftBehindSentence } = await import('../skills/secrets/keychain-ux');
  return items.map(({ item, account }) => leftBehindSentence(item, account));
}

/**
 * `webagents logout`: the stored token goes, from the keystore and the file.
 * An old `webagents:cli` item only a macOS dialog could remove is named, with
 * where to remove it; a token this run could not remove without a dialog
 * nobody can answer is the one sentence and exit 1, never "Signed out"
 * (keychain-ux, 2026-09-27). The Python `account.logout_command`.
 */
export async function logoutCommand(
  say: (line: string) => void,
  error: (line: string) => void,
  afterClear?: () => void,
): Promise<number> {
  const { clearToken, leftBehind } = await import('./credentials');
  const { KeychainDialogBlocked } = await import('../skills/secrets/keychain-ux');
  try {
    await clearToken();
  } catch (problem) {
    if (!(problem instanceof KeychainDialogBlocked)) throw problem;
    error(problem.message);
    return 1;
  }
  afterClear?.();
  const [portalUrl] = resolvePlatformUrl();
  say(`Signed out of ${hostOf(portalUrl)}.`);
  for (const line of await leftBehindLines(leftBehind())) say(line);
  return 0;
}

/**
 * Where this CLI finds no keys and the Python CLI has some in the keychain,
 * say that each CLI keeps its own (keychain-ux, 2026-09-27). The file fallback
 * is shared, so this is said for the keychain only. The Python
 * `commands.secrets._other_runtime_keys_line`.
 */
export async function otherRuntimeKeysLine(open: () => Promise<{ status(): { backend: string } }>): Promise<string> {
  try {
    if ((await open()).status().backend !== 'keystore') return '';
    const { KeychainRecord, otherKeysSentence, otherRuntimeItems, recordPath } = await import('../skills/secrets/keychain-ux');
    const { PROVIDERS_NAMESPACE } = await import('./provider-keys');
    const { globalDir, profileName, scopedNamespace } = await import('./config-store');
    const path = await import('node:path');
    const profile = profileName();
    const record = new KeychainRecord(recordPath(path.join(globalDir(profile), 'secrets')));
    return (await otherRuntimeItems(record, scopedNamespace(PROVIDERS_NAMESPACE, profile))).length ? otherKeysSentence() : '';
  } catch {
    // A note, never a failure.
    return '';
  }
}

/** `link --show`: what this folder publishes to. */
export function showLink(folder: string, say: (line: string) => void): boolean {
  const link = projectLink(folder);
  if (!link.agentId) {
    say('This folder is not linked to an agent on Robutler.');
    say(`\`${cliCommand('link <name>')}\` links it to one of yours; \`${cliCommand('publish')}\` creates one.`);
    return true;
  }
  say(`Linked to ${link.agentName ?? link.agentId} (${link.agentId}).`);
  return true;
}

interface RemoteAgent {
  id?: string;
  username?: string | null;
  displayName?: string | null;
  name?: string | null;
}

/**
 * `link [name]`: bind this folder to the agent of yours called `name` (the
 * local agent file's name when omitted). Matched on the username first
 * (`me.helper`, or its `helper` part), then on the display name.
 */
export async function linkFolder(
  folder: string,
  name: string | undefined,
  say: (line: string) => void,
  error: (line: string) => void,
): Promise<boolean> {
  const { getToken } = await import('./credentials.js');
  const token = await getToken();
  const [portalUrl] = resolvePlatformUrl();
  if (!token) {
    error(`Not signed in. Run \`${cliCommand('login')}\` first.`);
    return false;
  }
  let wanted = name?.trim();
  if (!wanted) {
    const { loadAgentProject, toPlatformPayload } = await import('./agent-project.js');
    let project;
    try {
      project = loadAgentProject(folder);
    } catch (err) {
      error((err as Error).message);
      return false;
    }
    if (!project) {
      error(`No agent file here, so no name to look for. Pass one: \`${cliCommand('link <name>')}\`.`);
      return false;
    }
    wanted = String(toPlatformPayload(project).name ?? '');
  }

  let agents: RemoteAgent[];
  try {
    const res = await fetch(`${portalUrl}/api/agents`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) {
      error(`Could not list your agents: ${res.status}.`);
      error(`Check \`${cliCommand('whoami')}\`, then \`${cliCommand('login')}\`.`);
      return false;
    }
    agents = ((await res.json()) as { agents?: RemoteAgent[] }).agents ?? [];
  } catch (err) {
    error(`Could not list your agents: ${(err as Error).message}`);
    return false;
  }

  const match =
    agents.find((a) => {
      const username = String(a.username ?? '');
      return username === wanted || username.endsWith(`.${wanted}`);
    }) ?? agents.find((a) => (a.displayName ?? a.name) === wanted);
  if (!match || !match.id) {
    error(`None of your agents is called ${wanted}.`);
    if (agents.length) {
      error('You have:');
      for (const a of agents.slice(0, 10)) error(`  ${a.username ?? a.displayName ?? a.id}`);
    }
    error(`\`${cliCommand('publish')}\` creates it.`);
    return false;
  }
  const username = String(match.username ?? wanted);
  const store = new ConfigStore({ cwd: folder });
  store.set('link.agentId', String(match.id), 'project');
  store.set('link.agentName', username, 'project');
  say(`Linked this folder to ${username}.`);
  return true;
}

/** `unlink`: forget the binding; the agent itself is untouched. */
export function unlinkFolder(folder: string, say: (line: string) => void): void {
  const link = projectLink(folder);
  if (!link.agentId) {
    say('This folder is not linked.');
    return;
  }
  const store = new ConfigStore({ cwd: folder });
  store.unset('link.agentId', 'project');
  store.unset('link.agentName', 'project');
  say(`Unlinked from ${link.agentName ?? link.agentId}.`);
}
