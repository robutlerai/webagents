/**
 * The `webagents acp` command's action (gap-closure plan item 1.6,
 * 2026-09-26), out of `cli/index.ts` for the reason `serve-action.ts` is:
 * that module parses argv at import time, so a command can only be tested
 * from its own module.
 *
 * It builds THE SAME AGENT `serve` builds (`createServedAgent`: the file's
 * skills and model, the stored keys, the folder, the access block) and
 * serves it to the code editor that spawned this process over the Agent
 * Client Protocol, on stdin and stdout (`skills/transport/acp/skill.ts`).
 * The caller is the person whose editor this is, the owner, as in the local
 * chat. STDOUT IS THE WIRE, so it is reserved before the agent file is even
 * read (`loadAgentConfigFile` prints a line when there is no file, and a
 * skill may print while starting). Sessions are kept under the profile
 * directory (`~/.webagents`, or `~/.webagents-<profile>`), in
 * `acp/sessions`, unless the agent file's own `- acp: {sessions_dir: ...}`
 * says where. The Python twin is `python/webagents/cli/acp_serve.py`.
 */

import * as path from 'node:path';
import type { IAgent, ISkill } from '../core/types';
import type { ACPTransportSkill } from '../skills/transport/acp/skill';

/** Seams for the unit test. Every one defaults to what the CLI really does. */
export interface AcpCommandDeps {
  createAgent?: (agentPath: string) => Promise<IAgent> | IAgent;
  serve?: (skill: ACPTransportSkill, agent: IAgent) => Promise<unknown>;
  sessionsDir?: () => string;
}

/** Where sessions go when the agent file names no `sessions_dir`: the profile directory's `acp/sessions`. */
export async function defaultSessionsDir(): Promise<string> {
  const { globalDir } = await import('./config-store.js');
  return path.join(globalDir(), 'acp', 'sessions');
}

export async function acpAction(agentPath: string, deps: AcpCommandDeps = {}): Promise<void> {
  const { reserveStdoutForMcp } = await import('../server/mcp.js');
  reserveStdoutForMcp();

  const createAgent =
    deps.createAgent ??
    (async (target: string) => {
      // The owner's own editor, over stdio: the owner's turns, so the
      // sign-in may pay, as the ACP `login` auth method promises (S-327
      // keeps it off `serve` and the daemon only).
      const { createServedAgent } = await import('./serve-action.js');
      return createServedAgent(target, { forCallers: false });
    });
  const agent = await createAgent(agentPath);

  const { ACPTransportSkill } = await import('../skills/transport/acp/skill.js');
  const skills = ((agent as unknown as { skills?: ISkill[] }).skills ?? []) as ISkill[];
  let skill = skills.find((s): s is ISkill & ACPTransportSkill => s instanceof ACPTransportSkill) as ACPTransportSkill | undefined;
  if (!skill) {
    skill = new ACPTransportSkill({});
    (agent as unknown as { addSkill?: (s: ISkill) => void }).addSkill?.(skill as unknown as ISkill);
  }
  if (!skill.settings.sessions_dir) {
    skill.settings.sessions_dir = deps.sessionsDir ? deps.sessionsDir() : await defaultSessionsDir();
  }

  const serve = deps.serve ?? (async (transport: ACPTransportSkill, built: IAgent) => transport.serveStdio(built));
  await serve(skill, agent);
}
