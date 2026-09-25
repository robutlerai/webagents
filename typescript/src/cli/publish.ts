/**
 * Sending the agent to the platform: `webagents publish` and the chat's
 * `/publish` (2026-09-24). One implementation, so the two cannot drift.
 *
 * Creates the agent (`POST /api/agents`) and links the folder to it, or
 * updates the linked one (`PATCH /api/agents/{id}`), the routes and the link
 * keys (`link.agentId`, `link.agentName` in the folder's
 * `.webagents/config.json`) the Python CLI's `deploy` uses too.
 *
 * Two rules the old command broke:
 *
 *  - THE LINK COMES FROM THE PROJECT ONLY. It was read through every config
 *    layer, global included, so one `config set link.agentId X` (global by
 *    default) made every unlinked folder on the machine update agent X.
 *  - AN UPDATE NEVER SENDS `name`. The portal renames an agent whose display
 *    name differs from the one sent, and a platform username is minted once:
 *    after a rename in the portal, the next publish moved `me.shipper` to
 *    `me.shipper-2` for good and retired the old handle.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { ConfigStore, cliCommand, globalDir, profileName, scopedNamespace } from './config-store';

export interface PublishIO {
  /** What happened, first: "Published me.helper (id)." or "Updated me.helper." */
  ok(line: string): void;
  print(line: string): void;
  error(line: string): void;
  /** Asked before creating a new agent; false when nobody can answer. */
  confirm(question: string): Promise<boolean>;
}

export interface PublishResult {
  ok: boolean;
  /** The agent's platform username, when known. */
  username?: string;
  agentId?: string;
  created?: boolean;
}

/** `link.agentId` / `link.agentName` from the folder's own config, never another layer. */
export function projectLink(projectRoot: string): { agentId?: string; agentName?: string } {
  const store = new ConfigStore({ cwd: projectRoot });
  const project = store.layers().find(([name]) => name === 'project')?.[1] ?? {};
  const agentId = project['link.agentId'];
  const agentName = project['link.agentName'];
  return {
    ...(typeof agentId === 'string' && agentId ? { agentId } : {}),
    ...(typeof agentName === 'string' && agentName ? { agentName } : {}),
  };
}

export async function publishAgent(
  agentPath: string,
  io: PublishIO,
  options: { yes?: boolean; dryRun?: boolean } = {},
): Promise<PublishResult> {
  const { getToken } = await import('./credentials.js');
  const { resolvePlatformUrl } = await import('./config-store.js');
  const [portalUrl] = resolvePlatformUrl();

  const { loadAgentProject, toPlatformPayload, AgentProjectError } = await import('./agent-project.js');
  let project;
  try {
    project = loadAgentProject(agentPath);
  } catch (err) {
    io.error(err instanceof AgentProjectError ? err.message : String(err));
    return { ok: false };
  }
  if (!project) {
    io.error(`No AGENT.md or agent.json at ${path.resolve(agentPath)}. Create one with \`webagents init\`.`);
    return { ok: false };
  }
  const payload = toPlatformPayload(project);

  const projectRoot =
    fs.existsSync(agentPath) && fs.statSync(agentPath).isFile() ? path.dirname(path.resolve(agentPath)) : path.resolve(agentPath);
  const link = projectLink(projectRoot);

  const body = link.agentId ? { ...payload } : payload;
  if (link.agentId) delete (body as Record<string, unknown>).name;
  const method = link.agentId ? 'PATCH' : 'POST';
  const url = link.agentId ? `${portalUrl}/api/agents/${encodeURIComponent(link.agentId)}` : `${portalUrl}/api/agents`;

  // Before the sign-in check: seeing what would be sent needs no account.
  if (options.dryRun) {
    io.print(`Would ${method} ${url}:`);
    io.print(JSON.stringify(body, null, 2));
    return { ok: true, created: !link.agentId };
  }

  const token = await getToken();
  if (!token) {
    io.error(`Not signed in. Run \`${cliCommand('login')}\` (or /login in the chat) first.`);
    return { ok: false };
  }

  if (!link.agentId) {
    io.print('This folder is not linked to an agent on Robutler.');
    io.print(
      `Publishing creates a new agent named ${String(payload.name ?? 'agent')}. ` +
        'Its username is set once and cannot be changed.',
    );
    if (!options.yes && !(await io.confirm('Create it?'))) {
      io.error('Not created.');
      return { ok: false };
    }
  }

  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    });
  } catch (err) {
    io.error(`Could not reach ${portalUrl}: ${(err as Error).message}`);
    return { ok: false };
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    io.error(`Publish failed: ${res.status}${detail ? ` ${detail}` : ''}`);
    if (res.status === 404 && link.agentId) {
      io.error('The linked agent is gone. Remove `link.agentId` from .webagents/config.json to create a new one.');
    }
    return { ok: false };
  }
  const data = (await res.json().catch(() => ({}))) as {
    agent?: { id?: string; username?: string };
    id?: string;
    username?: string;
    rawApiKey?: string;
  };

  if (link.agentId) {
    const username = data.agent?.username ?? data.username ?? link.agentName ?? String(payload.name ?? 'agent');
    io.ok(`Updated ${username}.`);
    return { ok: true, username, agentId: link.agentId, created: false };
  }

  // THE KEY IS NEVER PRINTED (S-223). `POST /api/agents` answers
  // `{agent, rawApiKey}` and returns that key once, so it goes to the store
  // under the name the Python CLI's `deploy` uses.
  const username = data.agent?.username ?? String(payload.name ?? 'agent');
  io.ok(`Published ${username}${data.agent?.id ? ` (${data.agent.id})` : ''}.`);
  if (data.agent?.id) {
    const store = new ConfigStore({ cwd: projectRoot });
    store.set('link.agentId', String(data.agent.id), 'project');
    store.set('link.agentName', username, 'project');
    io.print('This folder is now linked to it; the next publish updates it.');
  }
  if (data.rawApiKey) {
    const keyName = `AGENT_KEY_${username.toUpperCase().replace(/[.-]/g, '_')}`;
    try {
      const { openSecretStore } = await import('../skills/secrets/store.js');
      const resolved = profileName();
      const secrets = await openSecretStore({
        namespace: scopedNamespace('providers', resolved),
        secretsDir: path.join(globalDir(resolved), 'secrets'),
        quiet: true,
      });
      const backend = await secrets.set(keyName, data.rawApiKey);
      io.print(`Stored its API key as ${keyName} (${backend === 'keystore' ? 'your keychain' : 'an owner-only file'}). The platform returns it once.`);
      io.print(`Read it with \`webagents secrets get ${keyName} --show\`.`);
    } catch (err) {
      io.error(`Could not store the agent's API key: ${(err as Error).message}`);
      io.error('It was not printed and cannot be recovered; regenerate it in Settings.');
    }
  }
  return { ok: true, username, ...(data.agent?.id ? { agentId: String(data.agent.id) } : {}), created: true };
}
