/**
 * `webagents doctor` (2026-09-24): what stands between this folder and a
 * running agent, and the one command that fixes each thing.
 *
 * The same checks, names and words as the Python CLI's doctor
 * (`python/webagents/cli/commands/doctor.py`), so the two CLIs diagnose alike:
 * runtime, agent, model, sign-in, keys, sandbox, config. The model check is
 * the chat's own decision (`InteractiveREPL.doctorFacts`), so doctor and the
 * chat never disagree about whether the agent can run.
 */

import { cliCommand } from './config-store';

export type CheckStatus = 'ok' | 'warn' | 'fail';

export interface Check {
  name: string;
  status: CheckStatus;
  detail: string;
  fix?: string;
}

export async function runChecks(): Promise<Check[]> {
  const checks: Check[] = [];
  const major = Number(process.versions.node.split('.')[0]);
  checks.push({
    name: 'runtime',
    // The package's own `engines.node`.
    status: major >= 22 ? 'ok' : 'warn',
    detail: `Node ${process.versions.node}`,
    ...(major >= 22 ? {} : { fix: 'Install Node 22 or later.' }),
  });

  const { InteractiveREPL } = await import('./app.js');
  const repl = new InteractiveREPL({});
  let facts: ReturnType<InstanceType<typeof InteractiveREPL>['doctorFacts']> | undefined;
  try {
    await repl.initialize();
    facts = repl.doctorFacts();
  } catch (error) {
    checks.push({ name: 'agent', status: 'fail', detail: (error as Error).message, fix: 'Fix the agent file, or start over with `webagents init`.' });
  }
  if (facts) {
    checks.push({
      name: 'agent',
      status: 'ok',
      detail: facts.agentFile ? `${facts.agent} (${facts.agentFile})` : 'none here; the built-in assistant runs',
    });
    checks.push({
      name: 'model',
      status: facts.modelOk ? 'ok' : 'fail',
      detail: facts.model,
      ...(facts.modelOk ? {} : { fix: `\`${cliCommand('login')}\`, or \`${cliCommand('secrets set <NAME>')}\`` }),
    });
  }

  const { whoAmI } = await import('./account.js');
  const who = await whoAmI();
  checks.push(
    who.ok
      ? { name: 'sign-in', status: 'ok', detail: `@${who.username} on ${who.platform.replace(/^https?:\/\//, '')}` }
      : { name: 'sign-in', status: 'warn', detail: who.message, ...(who.fix ? { fix: who.fix.replace(/^Run /, '').replace(/\.$/, '') } : {}) },
  );

  try {
    const { providerKeyStore } = await import('./provider-keys.js');
    const keychain = (await providerKeyStore()).status().backend === 'keystore';
    checks.push(
      keychain
        ? { name: 'keys', status: 'ok', detail: 'stored in your keychain' }
        : { name: 'keys', status: 'warn', detail: 'stored in an owner-only file (no keychain on this machine)' },
    );
  } catch (error) {
    checks.push({ name: 'keys', status: 'warn', detail: `the key store cannot be opened: ${(error as Error).message}` });
  }

  checks.push(
    facts?.hasShell
      ? { name: 'sandbox', status: 'warn', detail: 'none in this SDK: shell commands run with your permissions' }
      : { name: 'sandbox', status: 'ok', detail: 'not needed: the agent cannot run commands' },
  );

  const { ConfigStore } = await import('./config-store.js');
  const problems = new ConfigStore().validate();
  checks.push(
    problems.length
      ? { name: 'config', status: 'fail', detail: `${problems.length} problem${problems.length === 1 ? '' : 's'}`, fix: '`webagents config validate`' }
      : { name: 'config', status: 'ok', detail: 'valid' },
  );
  return checks;
}

const MARKS: Record<CheckStatus, string> = { ok: '✓', warn: '▲', fail: '✗' };

/** The report as the Python doctor prints it. */
export function reportLines(checks: Check[]): string[] {
  const width = Math.max(...checks.map((c) => c.name.length)) + 3;
  const lines = ['Checks'];
  for (const c of checks) lines.push(`  ${MARKS[c.status]} ${c.name.padEnd(width)}${c.detail}`);
  const toFix = checks.filter((c) => c.status !== 'ok' && c.fix);
  if (toFix.length) {
    lines.push('', 'To fix');
    for (const c of toFix) lines.push(`  ${c.name.padEnd(width + 2)}${c.fix}`);
  }
  return lines;
}
