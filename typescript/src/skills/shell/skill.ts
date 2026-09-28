/**
 * Shell Skill: one command per call, in the working folder.
 *
 * OWNER-ONLY BY DEFAULT (2026-09-26, S-248). The command tool declared no
 * scopes, and a tool without scopes is offered to every caller, so anyone the
 * platform let message an agent connected through Portal Connect, or anyone
 * who could reach a served agent with any bearer, could run commands as the
 * developer's user on the machine serving it. The allow-list below is not a
 * boundary (`python -c` is allowed). The tool is `audience: 'owner'` now: the
 * owner and admins see it; nobody else does unless the agent file hands it
 * to a group with `access: tools:` (ADR-0045), which replaces the scope with
 * that group's. The Python `ShellSkill` declares the same.
 *
 * OS-LEVEL ENFORCEMENT, ON BY DEFAULT (2026-09-26, gap-closure plan item
 * 1.2; the engine is srt, `src/sandbox/`; on by default since 2026-09-27,
 * the sandbox-default lane, owner decision). The allow-list and the token
 * checks below stay, and they are NOT the boundary: they inspect the text
 * the model proposed while the shell runs something else (`echo $(id -un)`
 * passes an argv[0] check and runs `id`). `policy` is what holds: a kernel
 * sandbox the command cannot widen from inside, with secrets withheld from
 * its environment (S-220). An agent with no `sandbox:` block gets the
 * defaults (`development`: writes in its folder and a scratch folder, no
 * network, no local servers, `.env` and the credential folders unreadable).
 * The opt-out is explicit and said loudly: `sandbox: off` in the agent file
 * (`preset: unrestricted` stays accepted), or `--no-sandbox` for one run,
 * which lifts confinement from the OWNER'S commands only. A sandbox that
 * cannot be enforced here refuses the command rather than running it free,
 * declared or default, and a caller other than the owner is refused unless
 * the command is confined (defense in depth behind the owner-only default).
 * When a confined command's output shows a refusal, ONE sentence names the
 * switch that opens it (`refusalHint`), and in the interactive chat only,
 * a refused host is asked about by name (`askHost`), read from srt's own
 * proxy log. The Python `ShellSkill` does the same, pinned by
 * `python/tests/fixtures/sandbox/srt.json`.
 */

import { exec } from 'child_process';
import * as path from 'path';
import { Skill } from '../../core/skill';
import { tool, prompt } from '../../core/decorators';
import { scopeAllows, callerScopes } from '../../core/scopes';
import type { Context } from '../../core/types';
import {
  INTERRUPTED_RESULT,
  OFF_PRESET,
  SandboxUnavailable,
  describePolicy,
  noSandboxRequested,
  parseSandboxDeclaration,
  policyFromDeclaration,
  refusalHint,
  runSandboxed,
  sandboxRequiredReason,
  sandboxState,
  type SandboxDeclaration,
  type SandboxOrigin,
  type SandboxPolicy,
} from '../../sandbox/index';

/**
 * Said once when an agent loads with `sandbox: off` or `preset: unrestricted`
 * (fixture `unrestricted.warning`).
 *
 * ONE WORDING (the ptypass-fixes lane, 2026-09-27): this line said "Use
 * `development` or `strict`" while `/sandbox` and `doctor` said "Remove
 * `sandbox: off`". It is now `/sandbox`'s headline and fix, word for word
 * (`CHAT_WORDS` `sandboxOff`, `sandboxOffFix`, `sandboxFlagFix`).
 */
export const UNRESTRICTED_WARNING =
  'Sandbox: off (agent file): commands are not confined and run with your permissions. ' +
  'Remove `sandbox: off` (or `preset: unrestricted`) from the agent file to confine them.';

/** Said once when the run started with `--no-sandbox` (fixture `status.warnings.off_flag`). */
export const NO_SANDBOX_WARNING =
  'Sandbox: off (--no-sandbox): commands are not confined and run with your permissions. Run without --no-sandbox to confine them.';

/**
 * Where the opt-out is said when an agent loads (the ptypass-fixes lane,
 * 2026-09-27): on stderr by default (`-p`, `serve()`, the daemon). The chat
 * says it as one of its own notices, wrapped at words (the raw line wrapped
 * mid-word above the welcome card), and `doctor` holds it and says its own
 * line (`setOptOutAnnouncer`).
 */
let optOutAnnouncer: (message: string, origin: SandboxOrigin) => void = (message) => console.warn(message);

/** Say the opt-out through `announce` until the returned function is called, which puts the previous one back. */
export function setOptOutAnnouncer(announce: (message: string, origin: SandboxOrigin) => void): () => void {
  const previous = optOutAnnouncer;
  optOutAnnouncer = announce;
  return () => {
    optOutAnnouncer = previous;
  };
}

/** What the owner may answer when the chat asks about a refused host. */
export type HostAnswer = 'once' | 'always' | 'no';

/**
 * The chat's question about a host a confined command was refused: the host
 * and the command, by name. `once` re-runs the command with the host added
 * for that run; `always` writes the host into the agent file's
 * `network.hosts` (through `allowHostAlways`) and re-runs; `no` returns the
 * output with the hint. Only the interactive chat sets these; `serve()`, the
 * daemon and `-p` never ask, and nobody but the owner is asked for.
 */
export interface HostAsker {
  askHost(question: { host: string; command: string }): Promise<HostAnswer>;
  allowHostAlways(host: string): Promise<void>;
}

export interface ShellSkillConfig {
  baseDir?: string;
  allowedCommands?: string[];
  blockedCommands?: string[];
  /**
   * The agent file's spellings (2026-09-27, the final e2e re-run): a
   * `- shell: {allowed_commands: [...]}` block reaches this constructor as
   * written (`skills/resolve.ts` spreads it), and the Python `ShellSkill`
   * reads `allowed_commands`, so a file that worked there added nothing
   * here. Both spellings are read; the fixture `sandbox/srt.json`
   * (`shell_block`) pins it.
   */
  allowed_commands?: string[];
  blocked_commands?: string[];
  sandboxEnabled?: boolean;
  /**
   * The agent file's `sandbox:` block (checked, or as written): what the
   * kernel confines every command to. Absent means the file declared none,
   * and the DEFAULTS apply (file comment); `off` or `false` is the opt-out.
   */
  sandbox?: SandboxDeclaration | Record<string, unknown> | boolean | string | null;
  /** The environment the opt-out flag is read from; `process.env` when unset. */
  env?: Record<string, string | undefined>;
  /** The chat's ask-on-first-use hooks (`HostAsker`); unset everywhere else. */
  asker?: HostAsker;
}

/**
 * THE COMMAND LIST GATES UNCONFINED COMMANDS ONLY (the ptypass-fixes lane,
 * 2026-09-27). The PTY pass found `sleep` and `mkdir` refused in both SDKs,
 * `python3` here only and `rg` in Python only, before the sandbox was ever
 * reached, while `docs/cli/sandbox.md` says the kernel is the boundary. A
 * confined command now runs whatever it is, and the kernel decides what it
 * may touch; only a name the agent file lists under `blocked_commands` is
 * still refused. With `sandbox: off` (or `--no-sandbox`) the lists below are
 * the only gate there is, so they stay, as ONE list and ONE wording in both
 * SDKs (Python `DEFAULT_ALLOWED`, `DEFAULT_BLOCKED`), pinned by the fixture
 * `sandbox/srt.json` (`shell_allowlist`). A command's name is the base name
 * of the word in each command position (the first word, and the first after
 * `&&`, `||`, `|` or `;`).
 */
export const DEFAULT_ALLOWED = [
  'ls', 'cat', 'grep', 'find', 'head', 'tail', 'wc', 'echo', 'date',
  'pwd', 'which', 'whereis', 'git', 'npm', 'pip', 'python', 'python3', 'node',
  'uvx', 'curl', 'wget', 'rg', 'fd',
];

export const DEFAULT_BLOCKED = [
  'rm', 'rmdir', 'dd', 'mkfs', 'fdisk', 'kill', 'killall', 'pkill',
  'shutdown', 'reboot', 'halt', 'su', 'sudo', 'chmod', 'chown',
];

/** The refusals, the Python skill's words (fixture `shell_allowlist`). */
export const NOT_ALLOWED = "Command '{name}' is not in the allowlist";
export const IS_BLOCKED = "Command '{name}' is blocked";

const CHAIN_OPERATORS = new Set(['&&', '||', '|', ';']);

export class ShellSkill extends Skill {
  private workingDir: string;
  private sandboxEnabled: boolean;
  private allowedCommands: Set<string>;
  private blockedCommands: Set<string>;
  /** The names the agent file itself blocks: refused even when confined. */
  private declaredBlocked: Set<string>;
  /** The resolved policy: the file's `sandbox:`, the defaults, or the opt-out; null only when it could not be resolved. */
  readonly policy: SandboxPolicy | null = null;
  /** A declaration that could not be resolved; every command is refused with it. */
  readonly sandboxError: string | null = null;
  /** Where the policy came from, for the status row, `/sandbox` and `doctor`. */
  sandboxOrigin: SandboxOrigin;
  /** The chat's ask-on-first-use hooks, set by the chat alone. */
  asker: HostAsker | undefined;

  constructor(config: ShellSkillConfig = {}) {
    super({ name: 'ShellSkill' });
    this.workingDir = path.resolve(config.baseDir || process.cwd());
    this.sandboxEnabled = config.sandboxEnabled ?? !!config.baseDir;
    this.allowedCommands = new Set([...DEFAULT_ALLOWED, ...(config.allowedCommands || []), ...(config.allowed_commands || [])]);
    this.blockedCommands = new Set([...DEFAULT_BLOCKED, ...(config.blockedCommands || []), ...(config.blocked_commands || [])]);
    this.declaredBlocked = new Set([...(config.blockedCommands || []), ...(config.blocked_commands || [])]);
    this.asker = config.asker;

    // ON BY DEFAULT (file comment): no block means the defaults, not "no
    // sandbox". `--no-sandbox` (the environment the CLI's root option sets)
    // wins over the file for this run; it is the owner's explicit choice.
    let raw: unknown = config.sandbox === undefined || config.sandbox === null ? {} : config.sandbox;
    this.sandboxOrigin = config.sandbox === undefined || config.sandbox === null ? 'default' : 'agent file';
    if (noSandboxRequested(config.env ?? process.env)) {
      raw = OFF_PRESET;
      this.sandboxOrigin = '--no-sandbox';
    }
    try {
      this.policy = policyFromDeclaration(parseSandboxDeclaration(raw), { cwd: this.workingDir });
    } catch (err) {
      // A malformed declaration must not silently become "no sandbox".
      // Remembered and refused at execution time, where there is somewhere
      // to report it.
      this.sandboxError = (err as Error).message;
    }
    if (this.policy && !this.policy.confined) {
      // An opt-out nobody sees is S-217 again. Said once, where the agent loads (`setOptOutAnnouncer`).
      optOutAnnouncer(this.sandboxOrigin === '--no-sandbox' ? NO_SANDBOX_WARNING : UNRESTRICTED_WARNING, this.sandboxOrigin);
    }
  }

  /**
   * The chat wrote a `sandbox:` block into the agent file for the policy in
   * use (an `always` answer, the ptypass-fixes lane, 2026-09-27): the
   * defaults are now the file's own, so the state reads `(agent file)` at
   * once rather than after `/reload`. An opt-out stays what it was.
   */
  declaredInAgentFile(): void {
    if (this.sandboxOrigin === 'default') this.sandboxOrigin = 'agent file';
  }

  /** The state as the status row prints it: `development (default)`, `off (agent file)`, `off (--no-sandbox)`. */
  sandboxStateLine(): string {
    return sandboxState(this.policy, this.sandboxOrigin);
  }

  /**
   * Whether the command was asked for by the agent's owner (or an admin).
   * A context with no `auth` at all is code calling the skill directly,
   * outside any run: the process itself.
   */
  private callerIsOwner(context: Context | undefined): boolean {
    const auth = (context as { auth?: unknown } | undefined)?.auth;
    if (auth === undefined || auth === null) return true;
    return scopeAllows('owner', callerScopes(auth as Parameters<typeof callerScopes>[0]));
  }

  // --------------------------------------------------------------------------
  // Tokenizer
  // --------------------------------------------------------------------------

  private _tokenize(command: string): string[] {
    const tokens: string[] = [];
    let current = '';
    let inSingle = false;
    let inDouble = false;

    for (let i = 0; i < command.length; i++) {
      const ch = command[i];

      if (ch === "'" && !inDouble) {
        inSingle = !inSingle;
        continue;
      }
      if (ch === '"' && !inSingle) {
        inDouble = !inDouble;
        continue;
      }

      if (!inSingle && !inDouble && (ch === ' ' || ch === '\t')) {
        if (current.length > 0) {
          tokens.push(current);
          current = '';
        }
        continue;
      }

      current += ch;
    }
    if (current.length > 0) tokens.push(current);
    return tokens;
  }

  // --------------------------------------------------------------------------
  // Command checking
  // --------------------------------------------------------------------------

  /**
   * Whether `command` may run, before it is handed to the runner. Confined
   * (the comment above `DEFAULT_ALLOWED`): only a name the agent file blocks
   * is refused; the kernel decides the rest. Unconfined: the one list, then
   * the path checks below.
   */
  private _checkCommand(command: string, confined = false): { allowed: boolean; reason: string } {
    const tokens = this._tokenize(command);
    if (tokens.length === 0) return { allowed: false, reason: 'Empty command' };

    const commandPositions = this._extractCommandPositions(tokens);

    for (const cmd of commandPositions) {
      const base = path.basename(cmd);
      if (confined) {
        if (this.declaredBlocked.has(base)) return { allowed: false, reason: IS_BLOCKED.replace('{name}', base) };
        continue;
      }
      if (this.blockedCommands.has(base)) {
        return { allowed: false, reason: IS_BLOCKED.replace('{name}', base) };
      }
      if (!this.allowedCommands.has(base)) {
        return { allowed: false, reason: NOT_ALLOWED.replace('{name}', base) };
      }
    }
    if (confined) return { allowed: true, reason: '' };

    if (this.sandboxEnabled) {
      const sandboxCheck = this._checkSandbox(tokens);
      if (!sandboxCheck.allowed) return sandboxCheck;
    }

    return { allowed: true, reason: '' };
  }

  private _extractCommandPositions(tokens: string[]): string[] {
    const commands: string[] = [];
    let expectCommand = true;

    for (const token of tokens) {
      if (expectCommand) {
        commands.push(token);
        expectCommand = false;
      }
      if (CHAIN_OPERATORS.has(token)) {
        expectCommand = true;
      }
    }

    return commands;
  }

  private _checkSandbox(tokens: string[]): { allowed: boolean; reason: string } {
    for (const token of tokens) {
      if (CHAIN_OPERATORS.has(token)) continue;

      if (token.includes('..')) {
        return { allowed: false, reason: `Path traversal ('..') is not allowed in sandbox mode` };
      }

      if (token.startsWith('~')) {
        return { allowed: false, reason: `Home directory expansion ('~') is not allowed in sandbox mode` };
      }

      if (path.isAbsolute(token)) {
        const resolved = path.resolve(token);
        if (!resolved.startsWith(this.workingDir)) {
          return {
            allowed: false,
            reason: `Absolute path '${token}' is outside the sandbox directory`,
          };
        }
      }
    }

    return { allowed: true, reason: '' };
  }

  // --------------------------------------------------------------------------
  // Tool
  // --------------------------------------------------------------------------

  @prompt({ priority: 55, name: 'shellGuide', scope: 'all' })
  shellGuide(_ctx: Context): string {
    const confined = Boolean(this.policy?.confined);
    const allowed = [...this.allowedCommands].sort().join(', ');
    const blocked = [...this.blockedCommands].sort().join(', ');
    const declared = [...this.declaredBlocked].sort().join(', ');
    // Confined, a command is not refused for its name (the comment above
    // `DEFAULT_ALLOWED`): the list and the path rules below gate only
    // unconfined commands, so the model is told only what holds.
    const gate = confined
      ? [`- Any command may run: the operating system confines it (below), so none is refused for its name${declared ? `, except these, which the agent file blocks: ${declared}` : ''}.`]
      : [
          `- Allowed commands (the only ones that will execute): ${allowed}.`,
          `- Blocked commands (will be denied even if you spell them right): ${blocked}.`,
          '- Pipes (`|`), redirects (`>`, `<`), heredocs, command substitution (`$(...)` / backticks), background (`&`), and chaining (`&&`, `||`, `;`) split the command and require EVERY chained verb to be allowed. Keep to a single verb per call to stay predictable.',
        ];
    const confinement = this.policy
      ? this.policy.confined
        ? `- The operating system confines every command (sandbox ${this.sandboxStateLine()}): ${describePolicy(this.policy)}; secret-looking variables are withheld. A refusal from the kernel reads \`Operation not permitted\`; do not retry it. A sentence after the output names the agent-file switch that would allow what was refused.`
        : `- The sandbox is ${this.sandboxStateLine()}: commands are NOT confined and run with the owner's permissions.`
      : undefined;
    return [
      '## Shell access',
      '',
      `- This skill runs **one command per call** in a host process under ${this.sandboxEnabled ? `a sandboxed working directory (${this.workingDir})` : 'the working directory'}.`,
      ...gate,
      ...(confinement ? [confinement] : []),
      ...(confined
        ? []
        : [
            this.sandboxEnabled
              ? '- Sandbox mode is ON: paths containing `..`, `~`, or absolute paths outside the sandbox are rejected. Use relative paths.'
              : '- Sandbox mode is OFF: any path is reachable. Be deliberate about which directories you read/write.',
          ]),
      '- Default timeout 30s (max 30s). For long-running work, prefer the function executor (a function with `wallMs` set) over a long shell command.',
      '- Output combines stdout + stderr with a trailing exit code line on failure. Truncated at ~1MB. Treat `Access denied: …` and `Command timed out after Ns` as terminal: do NOT retry the same command verbatim.',
    ].join('\n');
  }

  @tool({
    audience: 'owner',
    description: 'Run a shell command in the working folder',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Shell command to execute' },
        timeout: { type: 'number', description: 'Timeout in seconds (default: 30)' },
      },
      required: ['command'],
    },
  })
  async runCommand(
    params: { command: string; timeout?: number },
    context: Context,
  ): Promise<string> {
    const { command, timeout = 30 } = params;

    if (this.sandboxError) {
      // Declared and unusable. Refusing is the only honest answer: running
      // unsandboxed is precisely what the declaration forbade.
      return `Access denied: invalid sandbox declaration: ${this.sandboxError}`;
    }

    // DEFENSE IN DEPTH behind the owner-only default (S-248): a command asked
    // for by anyone but the owner runs ONLY confined. An agent file that
    // hands `shell` to a group with `access: tools:` and declares no sandbox,
    // or declares `unrestricted`, or runs where srt is missing, refuses that
    // group rather than running as the owner.
    if (!this.callerIsOwner(context)) {
      const reason = sandboxRequiredReason(this.policy);
      if (reason) return `Access denied: commands from callers other than the owner run only in a sandbox, and ${reason}`;
    }

    const check = this._checkCommand(command, Boolean(this.policy?.confined));
    if (!check.allowed) return `Access denied: ${check.reason}`;

    if (this.policy) {
      // Ask on first use, in the interactive chat only, for the owner only:
      // a refused host is read from srt's own proxy log and asked about by
      // name; `once` re-runs with it for this command, `always` writes it
      // into the agent file first. Every other mode returns the hint.
      const asking = Boolean(this.asker) && this.policy.confined && this.callerIsOwner(context);
      let policy = this.policy;
      let result;
      for (;;) {
        try {
          // `signal` is the turn's: the chat's Esc or Ctrl+C kills the
          // command's whole process group, as its timeout does.
          result = await runSandboxed(command, policy, { timeout, maxBuffer: 1024 * 1024, captureRefusals: asking, signal: context?.signal });
        } catch (err) {
          // FAIL CLOSED. The file said sandboxed, or the defaults do; this
          // machine cannot enforce it; so the command does not run.
          if (err instanceof SandboxUnavailable) return `Access denied: ${err.message}`;
          return `Error executing command: ${(err as Error).message}`;
        }
        if (result.interrupted) return INTERRUPTED_RESULT;
        if (result.timedOut) return `Command timed out after ${timeout}s`;
        const refused = (result.refusedHosts ?? []).filter((host) => !policy.networkDomains.includes(host));
        if (!asking || !refused.length || !this.asker) break;
        const allowed: string[] = [];
        for (const host of refused) {
          const answer = await this.asker.askHost({ host, command });
          if (answer === 'no') continue;
          if (answer === 'always') await this.asker.allowHostAlways(host);
          allowed.push(host);
        }
        if (!allowed.length) break;
        policy = { ...policy, network: true, networkDomains: [...policy.networkDomains, ...allowed] };
      }
      let output = result.stdout || '';
      if (result.stderr) output += `\nStderr: ${result.stderr}`;
      if (result.exitCode !== 0) output += `\nExit code: ${result.exitCode}`;
      const hint = policy.confined ? refusalHint(command, `${result.stdout}\n${result.stderr}`) : undefined;
      if (hint) output += `\n${hint}`;
      return output || '(No output)';
    }

    return new Promise((resolve) => {
      exec(
        command,
        {
          cwd: this.workingDir,
          timeout: timeout * 1000,
          maxBuffer: 1024 * 1024,
        },
        (error, stdout, stderr) => {
          let output = stdout || '';
          if (stderr) output += `\nStderr: ${stderr}`;
          if (error) {
            if (error.killed) {
              resolve(`Command timed out after ${timeout}s`);
              return;
            }
            output += `\nExit code: ${error.code ?? 1}`;
          }
          resolve(output || '(No output)');
        },
      );
    });
  }
}
