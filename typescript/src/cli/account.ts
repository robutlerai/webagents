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
  if (!token) return { ok: false, code: 'not_signed_in', message: `Not signed in to ${host}.`, fix: `Run \`${cliCommand('login')}\`.` };
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
