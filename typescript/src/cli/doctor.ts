/**
 * `webagents doctor` (2026-09-24): what stands between this folder and a
 * running agent, and the one command that fixes each thing.
 *
 * The same checks, names and words as the Python CLI's doctor
 * (`python/webagents/cli/doctor.py`), so the two CLIs diagnose alike:
 * runtime, agent, model, sign-in, keys, keychain, sandbox, skills, mcp, config. The
 * model check is the chat's own decision (`InteractiveREPL.doctorFacts`), so
 * doctor and the chat never disagree about whether the agent can run.
 */

import { cliCommand } from './config-store';
import { credentialSecretName } from '../skills/mcp/connect-errors';

/** The mcp check's words (S-292), the same in the Python doctor and pinned by `cli/secrets.json` (`doctor`). */
export const MCP_CHECK_WORDS = {
  notUsed: 'not used: the agent names no MCP servers',
  connected: '{count} connected: {names}',
  fixEntry: "Fix the server's entry in the agent file",
  fixLiteral: '${secret:{name}} in the agent file, then `{hint}`',
  // A `${env:NAME}` that is not set names the variable (2026-09-26): the fix
  // line said only "Fix the server's entry in the agent file".
  fixEnv: 'Set {name} in the environment, or store it with `{hint}` and write ${secret:{name}} in the agent file',
  // A server that answered 401 or 403 wants a bearer token (2026-09-29,
  // `skills/mcp/connect-errors.ts`): the recipe, never "fix the server's entry".
  fixCredential: "Authorization: Bearer ${secret:{name}} in {server}'s headers, then `{hint}`",
} as const;

export type CheckStatus = 'ok' | 'warn' | 'fail';

export interface Check {
  name: string;
  status: CheckStatus;
  detail: string;
  fix?: string;
}

export interface DoctorOptions {
  /** `-a <name>`: the folder's agent to check, as the chat's `-a` picks it (2026-09-26); the default file otherwise. */
  agent?: string;
}

export async function runChecks(options: DoctorOptions = {}): Promise<Check[]> {
  const checks: Check[] = [];
  // `-a <name>` names the file the checks are about (2026-09-26, the e2e
  // run: `doctor -a helper` was "unknown option"). An unknown name throws
  // `AgentNotFound` with its sentence, before anything is built.
  let agentFile: string | null | undefined;
  if (options.agent) {
    const { agentFileFor } = await import('./agent-files.js');
    agentFile = agentFileFor(process.cwd(), options.agent);
  }
  const major = Number(process.versions.node.split('.')[0]);
  checks.push({
    name: 'runtime',
    // The package's own `engines.node`.
    status: major >= 22 ? 'ok' : 'warn',
    detail: `Node ${process.versions.node}`,
    ...(major >= 22 ? {} : { fix: 'Install Node 22 or later.' }),
  });
  // Before anything reads the keychain (the agent's keys, the sign-in):
  // whether the NEXT use will ask is a fact about now, and the reads below
  // may answer it (keychain-ux, 2026-09-27).
  const keychainLine = await keychainCheck();

  const { InteractiveREPL } = await import('./app.js');
  const repl = new InteractiveREPL(agentFile !== undefined ? { agentFile } : {});
  let facts: ReturnType<InstanceType<typeof InteractiveREPL>['doctorFacts']> | undefined;
  // The skills say what they found while starting (an MCP server that did
  // not connect, a literal that looks like a key) on the console, which put
  // raw log lines above the report (2026-09-26, the e2e run). Every one of
  // those findings is in the checks below, so the console is held while the
  // agent starts; WEBAGENTS_DEBUG keeps the lines.
  const held = holdConsole();
  try {
    await repl.initialize();
    facts = repl.doctorFacts();
  } catch (error) {
    checks.push({ name: 'agent', status: 'fail', detail: (error as Error).message, fix: 'Fix the agent file, or start over with `webagents init`.' });
  } finally {
    // THE AGENT IS CLOSED HERE (2026-09-26, the e2e run's HIGH bug): this
    // built a chat, read its facts and left it running, so a connected
    // stdio MCP server's child process kept the event loop alive and
    // doctor never exited once a server had connected. The facts above were
    // read first: the report they carry does not need the connections.
    await repl.closeAgent();
    held.release();
  }
  if (facts) {
    checks.push({
      name: 'agent',
      status: 'ok',
      detail: facts.agentFile ? `${facts.agent} (${facts.agentFile})` : 'none here; the built-in assistant runs',
    });
    if (facts.modelOk && facts.localModel) {
      // A local model (Ollama, plan item 2.8): whether it answers at its
      // address and serves the model, in the words the Python doctor uses
      // (`ollamaModelCheck`, fixture `w2ops/models.json`).
      const { ollamaModelCheck, probeOllama } = await import('../skills/llm/ollama/probe.js');
      const local = ollamaModelCheck(facts.localModel.model, facts.localModel.baseUrl, await probeOllama(facts.localModel.baseUrl));
      checks.push({ name: 'model', status: local.status, detail: local.detail, ...(local.fix ? { fix: local.fix } : {}) });
    } else {
      checks.push({
        name: 'model',
        status: facts.modelOk ? 'ok' : 'fail',
        detail: facts.model,
        ...(facts.modelOk ? {} : { fix: `\`${cliCommand('login')}\`, or \`${cliCommand('secrets set <NAME>')}\`` }),
      });
    }
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
  checks.push(keychainLine);

  checks.push(await sandboxCheck(facts));
  const install = await installCheck(facts);
  if (install) checks.push(install);
  checks.push(await skillmdCheck(process.cwd(), agentFile ?? undefined));
  checks.push(mcpCheck(facts?.mcp));

  const { ConfigStore } = await import('./config-store.js');
  const problems = new ConfigStore().validate();
  checks.push(
    problems.length
      ? { name: 'config', status: 'fail', detail: `${problems.length} problem${problems.length === 1 ? '' : 's'}`, fix: '`webagents config validate`' }
      : { name: 'config', status: 'ok', detail: 'valid' },
  );
  return checks;
}

/**
 * What the `keychain` line and the chat's /status `Keychain` row say, found
 * without reading a value: where the sign-in and keys live, which program
 * last used this CLI's items, whether the next use will ask, an earlier
 * version's items, and what a run with nobody to answer could not read
 * (keychain-ux, 2026-09-27; the Python `keychain_facts`).
 */
export async function keychainFacts(): Promise<import('../skills/secrets/keychain-ux').KeychainFacts> {
  const path = await import('node:path');
  const { KeychainRecord, doctorFacts, legacyServiceName, recordPath } = await import('../skills/secrets/keychain-ux');
  const { CLI_NAMESPACE, TOKEN_ENV_VAR, TOKEN_KEY } = await import('./credentials');
  const { PROVIDERS_NAMESPACE, providerKeyStore } = await import('./provider-keys');
  const { openSecretStore } = await import('../skills/secrets/store');
  const { globalDir, profileName, scopedNamespace } = await import('./config-store');
  const profile = profileName();
  const tokenNamespace = scopedNamespace(CLI_NAMESPACE, profile);
  const tokenStore = await openSecretStore({ namespace: tokenNamespace, secretsDir: path.join(globalDir(profile), 'secrets'), quiet: true });
  const keysStore = await providerKeyStore();
  const facts = await doctorFacts(
    [tokenStore, keysStore],
    new KeychainRecord(recordPath(path.join(globalDir(profile), 'secrets'))),
    {
      [legacyServiceName(tokenNamespace)]: [TOKEN_KEY],
      [legacyServiceName(scopedNamespace(PROVIDERS_NAMESPACE, profile))]: await keysStore.legacyIndexNames(),
    },
    { envToken: Boolean(process.env[TOKEN_ENV_VAR]) },
  );
  facts.command = cliCommand('whoami');
  return facts;
}

/** The `keychain` line (`keychainFacts`), in the fixture's words. */
export async function keychainCheck(): Promise<Check> {
  const { doctorLine } = await import('../skills/secrets/keychain-ux');
  try {
    const line = doctorLine(await keychainFacts());
    return { name: line.name, status: line.status, detail: line.detail, ...(line.fix ? { fix: line.fix } : {}) };
  } catch (error) {
    return { name: 'keychain', status: 'warn', detail: `cannot be checked: ${(error as Error).message}` };
  }
}

/**
 * Hold the console while the agent starts (`runChecks`): nothing a skill
 * prints reaches the terminal until `release()`, unless WEBAGENTS_DEBUG is
 * set, in which case nothing is held. The Python doctor does the same by
 * sending the agent's log to the profile's log file for the run.
 */
function holdConsole(): { release(): void } {
  if (process.env.WEBAGENTS_DEBUG) return { release() {} };
  const held = { log: console.log, info: console.info, warn: console.warn, error: console.error };
  const silent = () => {};
  console.log = silent;
  console.info = silent;
  console.warn = silent;
  console.error = silent;
  return {
    release() {
      console.log = held.log;
      console.info = held.info;
      console.warn = held.warn;
      console.error = held.error;
    },
  };
}

/**
 * The sandbox check, the Python doctor's words (`cli/doctor.py`,
 * `_sandbox_check`), from the shell's own policy and the srt engine's status
 * (plan item 1.2). Until 2026-09-26 this SDK answered "none in this SDK".
 */
async function sandboxCheck(
  facts:
    | {
        hasShell: boolean;
        sandbox?: import('../sandbox/policy.js').SandboxPolicy | null;
        sandboxError?: string | null;
        sandboxOrigin?: import('../sandbox/policy.js').SandboxOrigin;
        /** The state SKILL.md scripts run under (`SkillMdSkill.scriptState`), when the agent has such skills. */
        skillScripts?: string;
      }
    | undefined,
): Promise<Check> {
  // The fix is what THIS machine lacks, as `webagents sandbox setup` says it
  // (the sandbox-engine lane, 2026-09-27; `unavailableFix`).
  const { backendStatus, sandboxState, unavailableFix } = await import('../sandbox/index.js');
  if (!facts?.hasShell) {
    // SKILL.md scripts are confined commands too: an agent that runs them
    // is not one that "cannot run commands" (brief item 3, 2026-09-27).
    if (!facts?.skillScripts) return { name: 'sandbox', status: 'ok', detail: 'not needed: the agent cannot run commands' };
    const status = backendStatus();
    if (!status.available) {
      return { name: 'sandbox', status: 'fail', detail: `${facts.skillScripts} for SKILL.md scripts, but ${status.reason || 'no sandbox backend here'}: scripts are refused`, fix: unavailableFix(status) };
    }
    return { name: 'sandbox', status: 'ok', detail: `${facts.skillScripts} for SKILL.md scripts, enforced by ${status.backend} ${status.version}${status.found ? ` (${status.found})` : ''}` };
  }
  if (facts.sandboxError) {
    return { name: 'sandbox', status: 'fail', detail: `invalid: ${facts.sandboxError}: shell commands are refused`, fix: 'Fix the `sandbox:` section of the agent file' };
  }
  const policy = facts.sandbox;
  const origin = facts.sandboxOrigin ?? 'default';
  const state = sandboxState(policy, origin);
  if (!policy || !policy.confined) {
    // An opt-out is reported as what it is, not as a sandbox.
    return origin === '--no-sandbox'
      ? { name: 'sandbox', status: 'warn', detail: `${state}: not confined; shell commands run with your permissions for this run`, fix: 'Run without --no-sandbox to confine them' }
      : { name: 'sandbox', status: 'warn', detail: `${state}: not confined; shell commands run with your permissions`, fix: 'Remove `sandbox: off` (or `preset: unrestricted`) from the agent file to confine them' };
  }
  const status = backendStatus();
  if (!status.available) {
    return { name: 'sandbox', status: 'fail', detail: `${state}, but ${status.reason || 'no sandbox backend here'}: shell commands are refused`, fix: unavailableFix(status) };
  }
  const found = status.found ? ` (${status.found})` : '';
  return { name: 'sandbox', status: 'ok', detail: `${state}, enforced by ${status.backend} ${status.version}${found}` };
}

/**
 * The `install` line (S-316, the ptypass-fixes lane, 2026-09-27): when
 * webagents, srt or the node srt runs on is installed inside a folder the
 * agent's confined commands may write, the sandbox write-denies it, and this
 * says so, with why a confined `npm install` into that `node_modules` fails.
 * The words are `installInsideCheck`'s, the Python doctor's too (fixture
 * `sandbox/srt.json` `sdk_install_deny.report`).
 */
async function installCheck(facts: { sandbox?: import('../sandbox/policy.js').SandboxPolicy | null } | undefined): Promise<Check | undefined> {
  const policy = facts?.sandbox;
  if (!policy || !policy.confined) return undefined;
  const { installInsideCheck } = await import('../sandbox/index.js');
  const found = installInsideCheck([policy.cwd, ...policy.writeRoots].filter((root) => root && root !== policy.scratch));
  return found ? { name: found.name, status: found.status, detail: found.detail, ...(found.fix ? { fix: found.fix } : {}) } : undefined;
}

/**
 * The SKILL.md skills check (plan item 1.4, 2026-09-26), the Python doctor's
 * words (`skillmd_loader.doctor_report`): what the folder's agent would load
 * from `.agents/skills` and its `agent_skills:`, and every folder that was
 * skipped or warned about, since a skipped skill is otherwise only a line
 * at start-up.
 */
async function skillmdCheck(folder: string, chosenFile?: string): Promise<Check> {
  const { discoverSkills, doctorReport } = await import('../skills/skillmd/skillmd-loader.js');
  const { findAgentFile, parseAgentMarkdown, readAgentFile } = await import('../agents/index.js');
  let explicit: string[] = [];
  // The file `-a` chose, else the folder's default one.
  const file = chosenFile ?? findAgentFile(folder);
  if (file) {
    try {
      explicit = parseAgentMarkdown(readAgentFile(file), file).agentSkills ?? [];
    } catch {
      // The agent check above already reports a broken file.
      explicit = [];
    }
  }
  const report = doctorReport(discoverSkills(folder, explicit));
  return { name: 'skills', status: report.status, detail: report.detail, ...(report.fix ? { fix: report.fix } : {}) };
}

/**
 * The MCP servers check (S-292, 2026-09-26), the Python doctor's words
 * (`cli/doctor.py`, `_mcp_check`), from the skill's own report: a server
 * whose `${secret:NAME}` is not stored, or whose entry was refused, fails
 * the check with the sentence and the `webagents secrets set` command; a
 * literal that looks like a key is a warning with the reference to use.
 * Never a value: the report masks them.
 */
export function mcpCheck(report: import('../skills/mcp/skill.js').McpServerReportRow[] | undefined): Check {
  if (!report || !report.length) return { name: 'mcp', status: 'ok', detail: MCP_CHECK_WORDS.notUsed };
  const broken = report.filter((row) => row.rejected || row.error);
  if (broken.length) {
    const missing = [...new Set(broken.flatMap((row) => row.missingSecrets))];
    const unset = [...new Set(broken.flatMap((row) => row.missingEnv ?? []))];
    const wantsToken = broken.filter((row) => row.needsCredential).map((row) => row.name);
    const fixes = [
      ...missing.map((name) => `\`${cliCommand(`secrets set ${name}`)}\``),
      ...unset.map((name) => MCP_CHECK_WORDS.fixEnv.replace(/\{name\}/g, name).replace('{hint}', cliCommand(`secrets set ${name}`))),
      ...wantsToken.map((server) => {
        const name = credentialSecretName(server);
        return MCP_CHECK_WORDS.fixCredential.replace('{name}', name).replace('{server}', server).replace('{hint}', cliCommand(`secrets set ${name}`));
      }),
    ];
    return {
      name: 'mcp',
      status: 'fail',
      detail: broken.map((row) => `${row.name}: ${row.rejected ?? row.error}`).join('; '),
      fix: fixes.length ? fixes.join(', ') : MCP_CHECK_WORDS.fixEntry,
    };
  }
  const warned = report.filter((row) => row.warnings.length);
  if (warned.length) {
    const suggested = [...new Set(warned.flatMap((row) => row.warnings.flatMap((w) => [...w.matchAll(/\$\{secret:([A-Za-z0-9_]+)\}/g)].map((m) => m[1]))))];
    return {
      name: 'mcp',
      status: 'warn',
      detail: warned.flatMap((row) => row.warnings.map((w) => `${row.name} ${w}`)).join('; '),
      fix: suggested
        .map((name) => MCP_CHECK_WORDS.fixLiteral.replace('{name}', name).replace('{hint}', cliCommand(`secrets set ${name}`)))
        .join('; '),
    };
  }
  const names = report.map((row) => row.name);
  const count = `${names.length} server${names.length === 1 ? '' : 's'}`;
  return { name: 'mcp', status: 'ok', detail: MCP_CHECK_WORDS.connected.replace('{count}', count).replace('{names}', names.join(', ')) };
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
