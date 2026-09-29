/**
 * What a `sandbox:` declaration means, resolved into something a kernel can
 * hold (gap-closure plan item 1.2, 2026-09-26; the Python twin is
 * `python/webagents/sandbox/policy.py`).
 *
 * The declaration in an agent file is a wish. This module turns it into a
 * `SandboxPolicy`: absolute, symlink-resolved paths plus a network decision,
 * with the escalation set subtracted. `srt.ts` turns that into an srt
 * settings file (`@anthropic-ai/sandbox-runtime`) and runs the command under
 * it.
 *
 * WHAT THE OS CAN AND CANNOT ENFORCE, stated plainly because the schema
 * implies more than is deliverable:
 *
 *   - `allowed_folders`  -> ENFORCED. Becomes the write roots.
 *   - `preset`           -> ENFORCED, as defaults for folders and network.
 *                           `unrestricted` is the exception and says so: it
 *                           is NOT confined at all (see `PRESETS`).
 *   - `network`          -> ENFORCED, per host. srt runs a proxy outside the
 *                           sandbox and the kernel lets the command reach
 *                           only that proxy, which admits the listed hosts.
 *                           Empty means no network; there is no allow-all.
 *   - `allowed_commands` -> NOT ENFORCEMENT. Kept: it decides what runs
 *                           without prompting, and inspects text the shell
 *                           then expands (`echo $(id -un)` passes an argv[0]
 *                           check and runs `id`).
 *   - `allowed_imports`  -> NOT ENFORCEABLE HERE AT ALL: an OS sandbox
 *                           confines files and sockets, not `import`.
 *
 * UNKNOWN KEYS ARE REJECTED, with a did-you-mean (S-270): a `presets: strict`
 * that loaded without a word ran with the looser default. The sentence is
 * the Python loader's, pinned by `python/tests/fixtures/sandbox/srt.json`
 * (`unknown_key`), which both suites read.
 *
 * THE SHAPE IS GRANULAR, AND THE SANDBOX IS ON BY DEFAULT (owner decision,
 * 2026-09-27; the sandbox-default lane). Every key maps to something srt
 * enforces, and an agent with no `sandbox:` block gets exactly this:
 *
 *     sandbox:
 *       preset: development     # strict | development | off
 *       files:
 *         write: [.]            # the agent's folder; the private scratch folder is always added
 *         read: all             # development: all; strict: the write folders plus system folders
 *         deny: []              # more paths commands may never read, on top of the built-in list
 *       network:
 *         hosts: []             # host names, or the groups npm, pypi, github (`HOST_GROUPS`)
 *         local: false          # connecting to local servers and listening on a port (srt allowLocalBinding)
 *         sockets: []           # unix sockets by path (srt allowUnixSockets; macOS only)
 *       env: []                 # variables a command may see; none that look secret by default
 *
 * The flat keys keep working as aliases: `allowed_folders` is `files.write`,
 * a bare `network:` list is `network.hosts`, `env_passthrough` is `env`;
 * `allowed_commands` and `allowed_imports` stay as they are. `sandbox: off`
 * (also `false`; PyYAML reads `off` as a boolean) is the explicit opt-out and
 * means what `preset: unrestricted` means, which stays accepted. The
 * normalised form is `SandboxDeclaration`; `parseSandboxDeclaration` folds
 * the aliases in, and the fixture pins the mapping.
 *
 * THE BUILT-IN DENIES apply under every confined preset and no key removes
 * them (S-311, S-315, S-309):
 *   - reads of `CREDENTIAL_DIRS` under $HOME, `Library/Keychains` included:
 *     on macOS the keychain trusts the program that created an item, and
 *     every item this SDK stores was created by the interpreter a confined
 *     command can start, so with the file readable it read the CLI's
 *     platform token and the secrets store with no dialog (S-311, exercised);
 *   - reads of every per-profile folder `~/.webagents-<profile>`
 *     (`PROFILE_DIR_PATTERN`): srt's deny is a path prefix, so `.webagents`
 *     never covered its siblings, which hold history, sessions, checkpoints
 *     and, with the file backend, the token and secrets (S-315, exercised);
 *   - reads of `.env`, `.env.*` and `.webagents/` in the working folder and
 *     every write root (`ROOT_READ_DENY`): provider keys live in `.env`, and
 *     the daemon's agent signing key lived under `.webagents/` (S-309);
 *   - reads of every `.env` and `.env.*` under $HOME (`HOME_ENV_DENY`), and
 *     of the credential files people keep outside the first list: git's and
 *     GitHub's, shell histories, clouds', other agents' logins, browser
 *     profiles (S-343, `CREDENTIAL_DIRS`);
 *   - writes to `ESCALATION_DENY` and the agent-file patterns (S-283);
 *   - writes to the SDK's own install whenever it lies inside a write root
 *     (S-316, `installWriteDenies`): a project-local `node_modules` holding
 *     webagents or srt, or the node srt runs on.
 * macOS takes globs (`~/.webagents-*`, `.env.*`); Linux enumerates what
 * exists when the command starts, as `AGENT_FILE_PATTERNS` already does.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

/** A declaration that cannot be used as written, with a sentence for its author. */
export class SandboxDeclarationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SandboxDeclarationError';
  }
}

/** The keys of the normalised `sandbox:` block (the Python `SandboxConfig` fields), sorted. */
export const SANDBOX_KEYS = ['allowed_commands', 'allowed_imports', 'env', 'files', 'network', 'preset'] as const;

/** The flat spellings still accepted, and the key each folds into (fixture `schema.aliases`). */
export const SANDBOX_ALIASES: Readonly<Record<string, string>> = {
  allowed_folders: 'files.write',
  env_passthrough: 'env',
};

/** Every key a `sandbox:` block may carry, as the refusal lists them: the keys and the aliases, sorted. */
export const SANDBOX_ACCEPTED_KEYS = [...SANDBOX_KEYS, ...Object.keys(SANDBOX_ALIASES)].sort() as readonly string[];

export const SANDBOX_FILES_KEYS = ['deny', 'read', 'write'] as const;
export const SANDBOX_NETWORK_KEYS = ['hosts', 'local', 'sockets'] as const;

/** Sub-keys that are parsed but enforce nothing; reported rather than ignored. */
export const INERT_SANDBOX_FIELDS = ['allowed_imports'] as const;

export const DEFAULT_PRESET = 'development';

/** The spelling of the opt-out in an agent file: `sandbox: off`. Both parsers also take `false`. */
export const OFF_PRESET = 'off';

/**
 * `network.hosts` entries that name a group rather than a host, expanded to
 * the hosts each truly needs and nothing more (fixture `host_groups`).
 */
export const HOST_GROUPS: Readonly<Record<string, readonly string[]>> = {
  npm: ['registry.npmjs.org'],
  pypi: ['pypi.org', 'files.pythonhosted.org'],
  github: ['github.com', 'api.github.com', 'codeload.github.com', 'objects.githubusercontent.com', 'raw.githubusercontent.com'],
};

/**
 * The command-line opt-out: `webagents --no-sandbox` puts this in the
 * environment (as `--profile` puts `WEBAGENTS_PROFILE`), and the shell reads
 * it when it resolves its policy. One run, the owner's commands only:
 * callers other than the owner are refused under it (S-248), and SKILL.md
 * scripts keep running confined.
 */
export const ENV_NO_SANDBOX = 'WEBAGENTS_NO_SANDBOX';

export function noSandboxRequested(env: Record<string, string | undefined> = process.env): boolean {
  const raw = (env[ENV_NO_SANDBOX] ?? '').trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes';
}

/** Where a shell's policy came from, said wherever the sandbox is reported (fixture `status`). */
export type SandboxOrigin = 'default' | 'agent file' | '--no-sandbox';

/** The state as the status row and `/sandbox` print it: `development (default)`, `off (agent file)`, `off (--no-sandbox)`. */
export function sandboxState(policy: SandboxPolicy | null | undefined, origin: SandboxOrigin): string {
  if (!policy || !policy.confined) return `${OFF_PRESET} (${origin})`;
  return `${policy.preset} (${origin})`;
}

/**
 * `preset` -> writes the working folder?, scopes reads?, confined at all?
 *
 * Reads and writes are deliberately asymmetric, the way Claude Code's are:
 * writes are enumerated everywhere, reads are broad by default and scoped
 * only under `strict`, because module resolution and toolchains traverse far
 * more of the filesystem than anyone expects.
 *
 * `unrestricted` IS NOT CONFINED. It used to mean "writes as `development`,
 * network open". srt has no allow-all network entry (`*` is rejected), so
 * that cannot be expressed any more, and confining files under a name that
 * says otherwise would be the S-217 shape, while demanding a domain list
 * would make it a second spelling of `development` plus `network:`. It is
 * an explicit opt-out, said loudly wherever the sandbox is reported, and a
 * caller other than the owner is refused under it as under no declaration.
 */
export const PRESETS: Record<string, { writesCwd: boolean; scopedReads: boolean; confined: boolean }> = {
  strict: { writesCwd: false, scopedReads: true, confined: true },
  development: { writesCwd: true, scopedReads: false, confined: true },
  unrestricted: { writesCwd: true, scopedReads: false, confined: false },
};

/**
 * Paths that stay WRITE-DENIED even inside an allowed root: a command that
 * can write here can grant itself permissions for the NEXT command, which
 * would make the whole policy advisory. Relative to each write root.
 *
 * THE AGENT'S OWN FILES ARE IN THIS SET (S-283, 2026-09-26). Under
 * `development`, and under `strict` with the default `allowed_folders:
 * ["."]`, the agent's folder is a write root, and the review's probe wrote
 * `AGENT.md`, `WEBAGENTS.md`, `mcp.json` and another skill's `SKILL.md`
 * from a confined command: a persistent injection into every later turn,
 * a wider `sandbox: network:`, an open `access:` block or a new `cron:`
 * schedule, all reloaded by the daemon on the write itself. So the agent
 * definition (`AGENT.md`, and `AGENT-<name>.md` through
 * `AGENT_FILE_PATTERNS`), the inherited context (`WEBAGENTS.md`), the file
 * the MCP skill reads (`mcp.json`, undotted, beside the `.mcp.json` already
 * here) and the whole `.agents/skills` folder (every SKILL.md skill, the
 * ones not installed yet included, since the folder itself is denied) are
 * write-denied in every write root, on both engines. Literal entries hold
 * whether or not the file exists yet on macOS; on Linux srt takes only
 * paths that exist (`srt.ts`, `buildSettings`).
 */
export const ESCALATION_DENY = [
  '.git/hooks',
  '.git/config',
  '.claude',
  '.webagents',
  '.vscode',
  '.idea',
  '.bashrc',
  '.zshrc',
  '.profile',
  '.bash_profile',
  '.gitconfig',
  '.mcp.json',
  '.env',
  'AGENT.md',
  'WEBAGENTS.md',
  'mcp.json',
  '.agents/skills',
] as const;

/**
 * Agent-file NAME PATTERNS write-denied in every write root (S-283): a
 * planted `AGENT-<name>.md` is a new agent the daemon serves on its next
 * scan, with whatever `sandbox:`, `access:` and `cron:` the planter wrote.
 * The daemon's own rule for an agent file name (`daemon/watcher.ts`,
 * `AGENT.md` or `AGENT-<name>.md`) is what the pattern spells.
 *
 * HOW A PATTERN REACHES THE KERNEL, since the two engines differ:
 *
 *   - macOS: srt compiles a deny glob into a Seatbelt regex, so the pattern
 *     itself is passed and covers a file CREATED during the command too
 *     (probed 2026-09-26: `AGENT-evil.md` refused, `AGENT-helper.md` refused
 *     and not renamable).
 *   - Linux: bubblewrap binds concrete paths only, and srt strips a write
 *     glob silently (`stripWriteGlobs` in its `sandbox-manager`). So the
 *     files that exist when the settings are built are enumerated and denied
 *     by name, and a match that does not exist yet is NOT covered there.
 *     Stated in the fixture (`agent_file_deny.linux`) rather than papered
 *     over with a placeholder, which bubblewrap would create on the host.
 *
 * Both engines get the enumerated names, so the settings differ by exactly
 * the glob entry. Pinned by `python/tests/fixtures/sandbox/srt.json`.
 */
export const AGENT_FILE_PATTERNS = ['AGENT-*.md'] as const;

/** Whether `p` is `root` or lies beneath it (both absolute, realpath'd). */
function within(p: string, root: string): boolean {
  const trimmed = root.replace(/\/+$/, '') || '/';
  return trimmed === '/' || p === trimmed || p.startsWith(trimmed + '/');
}

/**
 * The parts of the SDK's own install that a command could otherwise write,
 * to be write-denied (S-316, the ptypass-fixes lane, 2026-09-27).
 *
 * `installs` are where the running CLI and its engine live
 * (`srt.sdkInstallPaths`): the `node_modules` holding webagents and srt, and
 * the folder holding the node that runs srt. Under `development` the
 * agent's folder is a write root, so a project-local `node_modules` holding
 * webagents was writable from a confined command, and srt, which runs
 * OUTSIDE the sandbox for every command, and the SDK code the next
 * `webagents` start loads, were both rewritable: the sandbox's guarantee
 * ended at the next command. An install inside a write root is denied
 * whole; a write root inside an install is denied whole too (`files.write`
 * naming a folder of the install). An install outside every write root adds
 * nothing, which is the usual case (a global install, a project elsewhere).
 * Nothing else is denied: a command may still write a DIFFERENT
 * `node_modules` or venv the project keeps. Pinned by the fixture's
 * `sdk_install_deny`; the Python twin is `install_write_denies`.
 */
export function installWriteDenies(writeRoots: readonly string[], installs: readonly string[]): string[] {
  const denied: string[] = [];
  for (const install of installs) {
    for (const root of writeRoots) {
      const hit = within(install, root) ? install : within(root, install) ? root : undefined;
      if (hit && !denied.includes(hit)) denied.push(hit);
    }
  }
  return denied;
}

/** Whether `name` (one path segment) matches one of `AGENT_FILE_PATTERNS`; `*` matches within the segment. */
export function matchesAgentFilePattern(name: string): boolean {
  return AGENT_FILE_PATTERNS.some((pattern) => matchesPattern(name, pattern));
}

/**
 * The agent-file denies for one write root: every existing file whose name
 * matches a pattern, by name, and on macOS the pattern itself. A root that
 * cannot be listed contributes only the pattern (macOS) or nothing.
 */
export function agentFileDenies(root: string, platform: string = process.platform): string[] {
  let names: string[] = [];
  try {
    names = fs.readdirSync(root);
  } catch {
    names = [];
  }
  const denied = names.filter(matchesAgentFilePattern).sort().map((name) => path.join(root, name));
  if (platform === 'darwin') for (const pattern of AGENT_FILE_PATTERNS) denied.push(path.join(root, pattern));
  return denied;
}

/**
 * Folders and files under the home directory a `development` command cannot
 * read: the credentials of the person running the agent. `strict` denies
 * every read outside the declared folders, so this only matters there.
 * `Library/Keychains` is the login keychain file (S-311, file comment).
 *
 * WIDENED FOR S-343 (2026-09-29, exercised). The first eleven entries left
 * most of what people actually keep in files readable: a confined `cat` read
 * a scratch HOME's `~/.config/gh/hosts.yml`, `~/.git-credentials`,
 * `~/.zsh_history`, `~/.codex/auth.json`, `~/.config/op/config` and another
 * project's `.env` (`homeEnvDenies`). The command itself has no network, but
 * its output is the tool result, so it reaches the model, and anything that
 * steers the model (a skill's instructions, a fetched page) can pass it on
 * through a tool that is not confined. A list will always miss something;
 * narrowing what `development` reads at all is the durable fix, and the
 * owner's call. Where another program keeps more than secrets in its folder
 * (`.claude`, `.codex`, `.cargo`, `.gem`, `.terraform.d`), only the secret
 * files are denied, so a skill or an agent kept there still runs.
 */
export const CREDENTIAL_DIRS = [
  '.ssh',
  '.aws',
  '.config/gcloud',
  '.gnupg',
  '.kube',
  '.docker',
  '.netrc',
  '.npmrc',
  '.pypirc',
  '.webagents',
  'Library/Keychains',
  // Git and GitHub.
  '.git-credentials',
  '.config/git/credentials',
  '.config/gh',
  '.config/hub',
  // Shell and REPL histories, where a key typed on a command line ends up.
  '.zsh_history',
  '.zsh_sessions',
  '.bash_history',
  '.local/share/fish',
  '.python_history',
  '.node_repl_history',
  '.psql_history',
  '.mysql_history',
  '.sqlite_history',
  '.rediscli_history',
  // Databases, clouds and hosting.
  '.pgpass',
  '.my.cnf',
  '.s3cfg',
  '.boto',
  '.azure',
  '.oci',
  '.config/doctl',
  '.terraform.d/credentials.tfrc.json',
  '.vault-token',
  '.fly',
  '.config/rclone',
  '.config/stripe',
  '.config/configstore',
  '.config/op',
  // Package registries.
  '.cargo/credentials',
  '.cargo/credentials.toml',
  '.gem/credentials',
  // Other agents' logins and conversations.
  '.claude.json',
  '.claude/.credentials.json',
  '.claude/projects',
  '.codex/auth.json',
  '.codex/sessions',
  'Library/Application Support/Claude',
  // Browser profiles: their cookies are signed-in sessions (Firefox keeps them in plain SQLite).
  'Library/Application Support/Firefox',
  'Library/Application Support/Google/Chrome',
  'Library/Application Support/BraveSoftware',
  'Library/Application Support/Microsoft Edge',
  'Library/Application Support/Arc',
  '.mozilla',
  '.config/google-chrome',
  '.config/chromium',
  '.config/BraveSoftware',
  '.config/microsoft-edge',
] as const;

/**
 * `.env` files anywhere under the home directory, relative to it (S-343): a
 * command in one project read another project's `.env`, since
 * `ROOT_READ_DENY` covers only the working folder and the write roots, and
 * only at their top. Under every preset that reads beyond its folders.
 *
 * HOW THEY REACH THE KERNEL (`homeEnvDenies`): on macOS the two globs
 * themselves (srt compiles a deny glob into a Seatbelt regex, so a file made
 * later is covered and nothing is walked). On Linux srt would expand a glob
 * by walking the whole home folder at every command start, so the SDK walks
 * it itself, bounded (`HOME_ENV_WALK`), caches the answer briefly, and denies
 * what it found by name. A `.env` deeper than the walk goes, inside a skipped
 * folder, past the budget, or made after the walk, stays readable there.
 */
export const HOME_ENV_DENY = ['**/.env', '**/.env.*'] as const;

/** The Linux walk for `HOME_ENV_DENY` (fixture `builtin_denies.home_env_deny.linux`). */
export const HOME_ENV_WALK = {
  /** Folder levels below $HOME it lists: `~/dev/project/app/pkg` is 4. */
  depth: 4,
  /** The most folders it lists before it stops. */
  budget: 5000,
  /** How long an answer is reused, since the shell builds settings per command. */
  cacheMs: 60_000,
  /** Folders it never enters: tool caches and installs, which hold no project's `.env`. */
  skip: [
    '.cache', '.cargo', '.docker', '.git', '.gradle', '.local', '.m2', '.npm', '.nvm', '.pnpm-store',
    '.pyenv', '.rbenv', '.rustup', '.Trash', '.venv', '.yarn', '__pycache__', 'go', 'Library',
    'node_modules', 'snap', 'venv',
  ],
} as const;

/** The per-profile folders beside `~/.webagents` (S-315): `~/.webagents-local` and its siblings. */
export const PROFILE_DIR_PATTERN = '.webagents-*';

/** Read-denied in the working folder and every write root, under every confined preset (S-309, S-312). */
export const ROOT_READ_DENY = ['.env', '.webagents'] as const;
export const ROOT_READ_DENY_PATTERNS = ['.env.*'] as const;

/** Whether `name` (one path segment) matches `pattern`, `*` matching within the segment. */
function matchesPattern(name: string, pattern: string): boolean {
  const regex = new RegExp(`^${pattern.split('*').map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('[^/]*')}$`);
  return regex.test(name);
}

/** The entries of `folder` whose names match `pattern`, sorted; none when it cannot be listed. */
function matchingEntries(folder: string, pattern: string): string[] {
  let names: string[] = [];
  try {
    names = fs.readdirSync(folder);
  } catch {
    return [];
  }
  return names.filter((name) => matchesPattern(name, pattern)).sort().map((name) => path.join(folder, name));
}

/**
 * The profile folders to deny under `home`: the glob itself on macOS
 * (Seatbelt takes it as a regex, so a profile made after the settings were
 * built is covered too), the folders that exist on Linux (bubblewrap binds
 * concrete paths only; a placeholder would be created on the host).
 */
export function profileDirDenies(home: string, platform: string = process.platform): string[] {
  if (platform === 'darwin') return [path.join(home, PROFILE_DIR_PATTERN)];
  return matchingEntries(home, PROFILE_DIR_PATTERN).filter((entry) => {
    try {
      return fs.statSync(entry).isDirectory();
    } catch {
      return false;
    }
  });
}

/**
 * The read denies for one folder a command may write (the working folder,
 * every write root): `.env`, `.webagents` and every `.env.*`. Literal entries
 * hold whether or not the file exists yet on macOS, and the pattern is passed
 * as a glob there; Linux gets only what exists when the command starts.
 */
export function rootReadDenies(root: string, platform: string = process.platform): string[] {
  const denied: string[] = [];
  if (platform === 'darwin') {
    for (const relative of ROOT_READ_DENY) denied.push(path.join(root, relative));
    for (const pattern of ROOT_READ_DENY_PATTERNS) denied.push(path.join(root, pattern));
    return denied;
  }
  for (const relative of ROOT_READ_DENY) if (fs.existsSync(path.join(root, relative))) denied.push(path.join(root, relative));
  for (const pattern of ROOT_READ_DENY_PATTERNS) for (const entry of matchingEntries(root, pattern)) if (!denied.includes(entry)) denied.push(entry);
  return denied;
}

const homeEnvCache = new Map<string, { at: number; found: string[] }>();

/**
 * The `.env` denies under `home` (`HOME_ENV_DENY`): the globs on macOS, and
 * on Linux the files a bounded walk finds (`HOME_ENV_WALK`), reused for a
 * minute. `now` is for tests.
 */
export function homeEnvDenies(home: string, platform: string = process.platform, now: number = Date.now()): string[] {
  if (platform === 'darwin') return HOME_ENV_DENY.map((glob) => path.join(home, glob));
  const cached = homeEnvCache.get(home);
  if (cached && now - cached.at < HOME_ENV_WALK.cacheMs) return [...cached.found];
  const found = walkHomeEnvFiles(home);
  homeEnvCache.set(home, { at: now, found });
  return [...found];
}

/** Every `.env` and `.env.*` file (or link) the Linux walk reaches under `home`, sorted. */
export function walkHomeEnvFiles(home: string): string[] {
  const found: string[] = [];
  const skip = new Set<string>(HOME_ENV_WALK.skip);
  let listed = 0;
  // Breadth first, so the budget spends itself near the top of the tree.
  let level: string[] = [home];
  for (let depth = 0; depth <= HOME_ENV_WALK.depth && level.length; depth += 1) {
    const next: string[] = [];
    for (const folder of level) {
      if (listed >= HOME_ENV_WALK.budget) return found.sort();
      listed += 1;
      let entries: fs.Dirent[];
      try {
        entries = fs.readdirSync(folder, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        const full = path.join(folder, entry.name);
        if ((entry.isFile() || entry.isSymbolicLink()) && (entry.name === '.env' || entry.name.startsWith('.env.'))) found.push(full);
        else if (entry.isDirectory() && !skip.has(entry.name)) next.push(full);
      }
    }
    level = next;
  }
  return found.sort();
}

/**
 * Read access every command needs before it can do anything at all, under
 * `strict`. `/private/var/select` is what Apple's `python3` and `git` shims
 * read to find the developer tools.
 */
export const SYSTEM_READ: Record<string, readonly string[]> = {
  darwin: [
    '/usr', '/bin', '/sbin', '/System', '/Library', '/private/etc',
    '/private/var/db', '/private/var/select', '/dev', '/opt/homebrew', '/opt/local',
  ],
  linux: ['/usr', '/etc', '/opt', '/bin', '/sbin', '/lib', '/lib64', '/lib32', '/dev', '/proc', '/sys', '/run'],
};

/** A PRIVATE scratch folder under the temp folder, never the temp folder itself. */
export const SCRATCH_DIR_NAME = 'webagents-sandbox';

/**
 * A `sandbox:` block as an agent file writes it, normalised: every key
 * present, the aliases folded in (file comment). `files.read` is `all`, or
 * the folders reads are scoped to (the write folders, the working folder and
 * the system folders are always readable beside them). `network.hosts` keeps
 * a group name as written; `policyFromDeclaration` expands it.
 */
export interface SandboxDeclaration {
  preset: string;
  files: { write: string[]; read: 'all' | string[]; deny: string[] };
  network: { hosts: string[]; local: boolean; sockets: string[] };
  env: string[];
  allowed_commands: string[];
  allowed_imports: string[];
}

/** A resolved, enforceable policy. */
export interface SandboxPolicy {
  /** Folders the command may write to. Already realpath'd. */
  writeRoots: string[];
  /** Folders the command may read when `scopedReads` is set. */
  readRoots: string[];
  /** Whether reads are confined to `readRoots`: `strict`, or a `files.read` list. */
  scopedReads: boolean;
  /** More paths the command may never read (`files.deny`), realpath'd. */
  readDeny: string[];
  /** The private scratch folder, exported to the command as TMPDIR. */
  scratch?: string;
  /** Outbound network: any host reachable (a `network:` list, or unconfined). */
  network: boolean;
  /** The hosts the command may reach, srt's `allowedDomains`, groups expanded. */
  networkDomains: string[];
  /** Connecting to local servers and listening on a port (`network.local`, srt `allowLocalBinding`). */
  localNetwork: boolean;
  /** Unix sockets the command may reach by path (`network.sockets`, srt `allowUnixSockets` on macOS). */
  unixSockets: string[];
  /** False for `unrestricted` and `off`: nothing is applied and the command runs as the agent. */
  confined: boolean;
  preset: string;
  /** Where the command runs. */
  cwd: string;
  /** Carried for reporting; NOT enforced. */
  advisoryCommands: string[];
  /** Declared but unenforceable, reported so it cannot be believed. */
  unenforceable: string[];
  /** Secret-looking variables the command may still see (S-220), the `env` list. */
  envPassthrough: string[];
  /**
   * Folders that stay write-denied even inside a write root, on top of the
   * escalation set: EVERY SKILL.md skill's folder while one skill's script
   * runs (plan item 1.4, 2026-09-26; widened from the running skill's own
   * folder for S-283, since a `pdf` script wrote `.agents/skills/other/
   * SKILL.md`), so third-party code cannot rewrite the instructions and
   * scripts the next activation of any skill will trust. Absolute.
   */
  readOnly?: string[];
}

// ---------------------------------------------------------------------------
// The did-you-mean, as difflib.get_close_matches computes it
// ---------------------------------------------------------------------------

/** Characters matched by Ratcliff/Obershelp: the longest common block, then the same on each side. */
function matchingCharacters(a: string, b: string): number {
  if (!a.length || !b.length) return 0;
  let bestI = 0;
  let bestJ = 0;
  let bestLen = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      let k = 0;
      while (i + k < a.length && j + k < b.length && a[i + k] === b[j + k]) k++;
      if (k > bestLen) {
        bestI = i;
        bestJ = j;
        bestLen = k;
      }
    }
  }
  if (bestLen === 0) return 0;
  return (
    bestLen +
    matchingCharacters(a.slice(0, bestI), b.slice(0, bestJ)) +
    matchingCharacters(a.slice(bestI + bestLen), b.slice(bestJ + bestLen))
  );
}

/** `difflib.SequenceMatcher(None, a, b).ratio()`. */
export function similarity(a: string, b: string): number {
  const total = a.length + b.length;
  return total === 0 ? 1 : (2 * matchingCharacters(a, b)) / total;
}

/** `difflib.get_close_matches(word, known, n=1, cutoff=0.6)[0]`, or undefined. */
export function closestKnown(word: string, known: readonly string[], cutoff = 0.6): string | undefined {
  let best: { score: number; name: string } | undefined;
  for (const name of known) {
    const score = similarity(word, name);
    if (score < cutoff) continue;
    // heapq.nlargest on (score, name): the higher score, then the greater name.
    if (!best || score > best.score || (score === best.score && name > best.name)) best = { score, name };
  }
  return best?.name;
}

/** The refusal for keys the schema does not define, naming the nearest match: the Python loader's sentence. */
export function unknownKeysMessage(unknown: readonly string[], known: readonly string[], what: string): string {
  const sorted = [...known].sort();
  const problems = unknown.map((key) => {
    const close = closestKnown(key, sorted);
    return close ? `unknown key '${key}' (did you mean '${close}'?)` : `unknown key '${key}'`;
  });
  return `${what}: ${problems.join('; ')}. Known keys: ${sorted.join(', ')}`;
}

// ---------------------------------------------------------------------------
// The declaration
// ---------------------------------------------------------------------------

function stringList(value: unknown, key: string): string[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || !value.every((item) => typeof item === 'string')) {
    throw new SandboxDeclarationError(`sandbox: ${key} must be a list of strings`);
  }
  return value as string[];
}

/**
 * A `network:` entry is a host (`github.com`), a wildcard under a domain
 * (`*.example.com`) or a host with a port (`127.0.0.1:8080`, `[::1]:80`).
 * srt rejects anything else at run time; checking here refuses it at load,
 * where the author of the file can see it. The sentences are Python's.
 */
const NETWORK_ENTRY =
  /^(\*\.)?(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*(?::\d{1,5})?$/;
const NETWORK_ENTRY_V6 = /^\[[0-9A-Fa-f:.]+\](?::\d{1,5})?$/;

export function checkNetworkEntry(entry: unknown): string {
  if (typeof entry !== 'string' || !entry.trim()) {
    throw new SandboxDeclarationError('sandbox: a network entry must be a host name');
  }
  const value = entry.trim();
  if (value === '*' || (value.startsWith('*.') && !value.slice(2).includes('.'))) {
    throw new SandboxDeclarationError(
      `sandbox: network entry '${value}' is not allowed: srt has no allow-all; list the hosts (\`*.example.com\` needs a domain under the wildcard)`,
    );
  }
  if (value.includes('://') || value.includes('/')) {
    throw new SandboxDeclarationError(`sandbox: network entry '${value}' must be a host name, not a URL or an address range`);
  }
  if (!NETWORK_ENTRY.test(value) && !NETWORK_ENTRY_V6.test(value)) {
    throw new SandboxDeclarationError(
      `sandbox: network entry '${value}' is not a host name (\`github.com\`, \`*.example.com\`, \`127.0.0.1:8080\`)`,
    );
  }
  return value;
}

/** Whether a `sandbox:` value spells the opt-out: `off`, or the boolean PyYAML reads `off` as. */
export function isSandboxOff(raw: unknown): boolean {
  return raw === false || (typeof raw === 'string' && raw.trim().toLowerCase() === OFF_PRESET);
}

/** The refusal for a `sandbox:` that is neither a mapping nor `off`. */
const SHAPE_MESSAGE = 'sandbox: must be a mapping of settings (preset, files, network, env, ...) or `off`';

/**
 * The declaration as written, checked and normalised: unknown keys refused
 * with the did-you-mean (at every level), aliases folded in, lists typed,
 * defaults filled. `off` (or `false`) is the opt-out and becomes `preset:
 * unrestricted`; `on` (or `true`, or an empty mapping) is the defaults.
 * `files.read` defaults by preset: `all` unless `strict`, which scopes reads
 * to the write folders. The preset's NAME is checked when the policy is
 * built, so a file with `preset: stirct` still loads and its shell refuses
 * every command with the reason, as the Python skill does.
 */
export function parseSandboxDeclaration(raw: unknown): SandboxDeclaration {
  if (isSandboxOff(raw)) raw = { preset: 'unrestricted' };
  if (raw === true || (typeof raw === 'string' && raw.trim().toLowerCase() === 'on')) raw = {};
  if (raw === null || raw === undefined || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new SandboxDeclarationError(SHAPE_MESSAGE);
  }
  const data = raw as Record<string, unknown>;
  const unknown = Object.keys(data).filter((key) => !SANDBOX_ACCEPTED_KEYS.includes(key));
  if (unknown.length) throw new SandboxDeclarationError(unknownKeysMessage(unknown, SANDBOX_ACCEPTED_KEYS, 'sandbox'));
  if ('allowed_folders' in data && data.files !== undefined && data.files !== null) {
    throw new SandboxDeclarationError('sandbox: allowed_folders is the old spelling of files.write; use one of them');
  }
  if ('env_passthrough' in data && data.env !== undefined && data.env !== null) {
    throw new SandboxDeclarationError('sandbox: env_passthrough is the old spelling of env; use one of them');
  }
  const rawPreset = data.preset === undefined || data.preset === null ? DEFAULT_PRESET : data.preset;
  if (typeof rawPreset !== 'string') throw new SandboxDeclarationError('sandbox: preset must be a string');
  const preset: string = rawPreset.trim().toLowerCase() === OFF_PRESET ? 'unrestricted' : rawPreset;

  // files: the mapping, or the old flat `allowed_folders`.
  let write: string[] = ['.'];
  let read: 'all' | string[] | undefined;
  let deny: string[] = [];
  if (data.files !== undefined && data.files !== null) {
    if (typeof data.files !== 'object' || Array.isArray(data.files)) {
      throw new SandboxDeclarationError('sandbox: files must be a mapping (write, read, deny)');
    }
    const files = data.files as Record<string, unknown>;
    const bad = Object.keys(files).filter((key) => !(SANDBOX_FILES_KEYS as readonly string[]).includes(key));
    if (bad.length) throw new SandboxDeclarationError(unknownKeysMessage(bad, SANDBOX_FILES_KEYS, 'sandbox.files'));
    if (files.write !== undefined) write = stringList(files.write, 'files.write');
    if (files.read !== undefined && files.read !== null) {
      if (typeof files.read === 'string' && files.read.trim().toLowerCase() === 'all') read = 'all';
      else read = stringList(files.read, 'files.read (`all`, or a list of folders)');
    }
    deny = stringList(files.deny, 'files.deny');
  } else if (data.allowed_folders !== undefined) {
    write = stringList(data.allowed_folders, 'allowed_folders');
  }
  if (read === undefined) read = preset.toLowerCase() === 'strict' ? [] : 'all';

  // network: the mapping, or the old bare list of hosts.
  let hosts: string[] = [];
  let local = false;
  let sockets: string[] = [];
  if (Array.isArray(data.network)) {
    hosts = stringList(data.network, 'network');
  } else if (data.network !== undefined && data.network !== null) {
    if (typeof data.network !== 'object') throw new SandboxDeclarationError('sandbox: network must be a list of hosts or a mapping (hosts, local, sockets)');
    const network = data.network as Record<string, unknown>;
    const bad = Object.keys(network).filter((key) => !(SANDBOX_NETWORK_KEYS as readonly string[]).includes(key));
    if (bad.length) throw new SandboxDeclarationError(unknownKeysMessage(bad, SANDBOX_NETWORK_KEYS, 'sandbox.network'));
    hosts = stringList(network.hosts, 'network.hosts');
    if (network.local !== undefined && network.local !== null) {
      if (typeof network.local !== 'boolean') throw new SandboxDeclarationError('sandbox: network.local must be true or false');
      local = network.local;
    }
    sockets = stringList(network.sockets, 'network.sockets');
  }

  return {
    preset,
    files: { write, read, deny },
    network: { hosts, local, sockets },
    env: data.env !== undefined ? stringList(data.env, 'env') : stringList(data.env_passthrough, 'env_passthrough'),
    allowed_commands: stringList(data.allowed_commands, 'allowed_commands'),
    allowed_imports: stringList(data.allowed_imports, 'allowed_imports'),
  };
}

/** The hosts a `network.hosts` list names: groups expanded in place, each entry checked, duplicates dropped. */
export function expandHosts(entries: readonly string[]): string[] {
  const hosts: string[] = [];
  for (const entry of entries) {
    const group = typeof entry === 'string' ? HOST_GROUPS[entry.trim().toLowerCase()] : undefined;
    for (const host of group ?? [checkNetworkEntry(entry)]) if (!hosts.includes(host)) hosts.push(host);
  }
  return hosts;
}

// ---------------------------------------------------------------------------
// The policy
// ---------------------------------------------------------------------------

/** Absolute and symlink-resolved: `/tmp`, `/etc` and `/var` are symlinks into `/private` on macOS. */
function real(p: string): string {
  const expanded = p.startsWith('~') ? path.join(os.homedir(), p.slice(1)) : p;
  const absolute = path.resolve(expanded);
  try {
    return fs.realpathSync(absolute);
  } catch {
    // Not there yet: resolve the nearest existing ancestor so the spelling still matches later.
    const parent = path.dirname(absolute);
    if (parent === absolute) return absolute;
    return path.join(real(parent), path.basename(absolute));
  }
}

/** Order-preserving, and drops a path already covered by an ancestor. */
function dedupePaths(paths: readonly string[]): string[] {
  let out: string[] = [];
  for (const p of paths) {
    if (out.some((kept) => p === kept || p.startsWith(kept.replace(/\/+$/, '') + '/'))) continue;
    out = out.filter((kept) => !kept.startsWith(p.replace(/\/+$/, '') + '/'));
    out.push(p);
  }
  return out;
}

/** Only paths that are there, once each, in order. */
function existing(paths: readonly string[]): string[] {
  const seen: string[] = [];
  for (const p of paths) if (!seen.includes(p) && fs.existsSync(p)) seen.push(p);
  return seen;
}

/**
 * Escalation paths to deny inside each write root (the literal set, then the
 * agent-file patterns as `agentFileDenies` resolves them for `platform`),
 * then `readOnly`. Built per command, so the enumeration sees the files as
 * they are when the command starts.
 */
export function denyWrites(policy: SandboxPolicy, platform: string = process.platform): string[] {
  const denied: string[] = [];
  for (const root of policy.writeRoots) {
    for (const relative of ESCALATION_DENY) denied.push(path.join(root, relative));
    for (const entry of agentFileDenies(root, platform)) if (!denied.includes(entry)) denied.push(entry);
  }
  for (const folder of policy.readOnly ?? []) if (!denied.includes(folder)) denied.push(folder);
  return denied;
}

/**
 * What the command cannot read. Scoped reads (`strict`, or a `files.read`
 * list): everything, with `allowReads` re-allowed beneath. Otherwise the
 * credential folders and files, the profile folders and every `.env` under
 * $HOME (S-343). In both cases the built-in root denies (`.env`, `.env.*`,
 * `.webagents` in the working folder and every write root but the scratch)
 * and `files.deny`, which srt re-emits after its allows so a deny nested
 * inside an allowed folder still holds.
 */
export function denyReads(policy: SandboxPolicy, platform: string = process.platform): string[] {
  const denied: string[] = [];
  const add = (entry: string) => {
    if (!denied.includes(entry)) denied.push(entry);
  };
  if (policy.scopedReads) {
    add('/');
  } else {
    const home = real(os.homedir());
    for (const relative of CREDENTIAL_DIRS) add(path.join(home, relative));
    for (const entry of profileDirDenies(home, platform)) add(entry);
    for (const entry of homeEnvDenies(home, platform)) add(entry);
  }
  const roots = [policy.cwd, ...policy.writeRoots].filter((root) => root && root !== policy.scratch);
  for (const root of roots) for (const entry of rootReadDenies(root, platform)) add(entry);
  for (const entry of policy.readDeny) if (platform === 'darwin' || fs.existsSync(entry)) add(entry);
  return denied;
}

/** Under scoped reads, the reads re-allowed beneath the denied root. */
export function allowReads(policy: SandboxPolicy): string[] {
  if (!policy.scopedReads) return [];
  const system = SYSTEM_READ[process.platform] ?? [];
  return dedupePaths(existing([...system, ...policy.readRoots, ...policy.writeRoots, policy.cwd]));
}

/** One line, for a refusal message or a log. */
export function describePolicy(policy: SandboxPolicy): string {
  if (!policy.confined) return `not confined (preset ${policy.preset})`;
  const roots = policy.writeRoots.join(', ') || '(nothing)';
  const network = policy.networkDomains.length ? policy.networkDomains.join(', ') : 'off';
  const switches = [policy.localNetwork ? 'local network on' : '', policy.unixSockets.length ? `sockets: ${policy.unixSockets.join(', ')}` : ''].filter(Boolean);
  return `writes: ${roots}; network: ${network}${switches.length ? `; ${switches.join('; ')}` : ''}`;
}

/**
 * Resolve a declaration into a policy. `cwd` is where the command runs (the
 * agent's folder); `tmpdir` is where the private scratch folder goes.
 */
export function policyFromDeclaration(
  declaration: SandboxDeclaration,
  options: { cwd?: string; tmpdir?: string } = {},
): SandboxPolicy {
  let preset = String(declaration.preset || DEFAULT_PRESET).toLowerCase();
  if (preset === OFF_PRESET) preset = 'unrestricted';
  const rules = PRESETS[preset];
  if (!rules) {
    throw new SandboxDeclarationError(
      `unknown sandbox preset '${preset}'; expected one of ${[...Object.keys(PRESETS), OFF_PRESET].sort().join(', ')}`,
    );
  }
  const networkDomains = expandHosts(declaration.network.hosts);
  const working = real(options.cwd ?? process.cwd());

  // A folder of ours inside the temp area, never the temp area itself.
  const scratchBase = real(options.tmpdir ?? process.env.TMPDIR ?? os.tmpdir());
  let scratch: string | undefined = path.join(scratchBase, SCRATCH_DIR_NAME);
  try {
    fs.mkdirSync(scratch, { recursive: true, mode: 0o700 });
  } catch {
    // A scratch folder we cannot create is one we must not promise.
    scratch = undefined;
  }

  // `files.write` is relative to the working folder when relative.
  const relativeTo = (entry: string) => real(path.isAbsolute(entry) || entry.startsWith('~') ? entry : path.join(working, entry));
  const declared = declaration.files.write.map(relativeTo);
  const writeRoots = [...declared];
  if (rules.writesCwd && !writeRoots.includes(working)) writeRoots.push(working);
  if (scratch && !writeRoots.includes(scratch)) writeRoots.push(scratch);

  // `files.read`: `all` reads broadly; a list scopes reads to it, the write
  // folders, the working folder and the system folders (what `strict` does).
  const scopedReads = declaration.files.read !== 'all';
  const readRoots = scopedReads ? dedupePaths([...(declaration.files.read as string[]).map(relativeTo), ...declared, working]) : [];

  const unenforceable: string[] = [];
  if (declaration.allowed_imports.length) unenforceable.push('allowed_imports');
  // srt takes unix socket paths on macOS only ("seccomp cannot filter by
  // path"); the safe direction is to grant nothing there, and to say so.
  const unixSockets = declaration.network.sockets.map((entry) => real(entry));
  if (unixSockets.length && process.platform !== 'darwin') unenforceable.push('network.sockets');

  return {
    writeRoots: dedupePaths(writeRoots),
    readRoots,
    scopedReads,
    readDeny: declaration.files.deny.map(relativeTo),
    scratch,
    network: !rules.confined || networkDomains.length > 0 || declaration.network.local,
    networkDomains,
    localNetwork: declaration.network.local,
    unixSockets,
    confined: rules.confined,
    preset,
    cwd: working,
    advisoryCommands: [...declaration.allowed_commands],
    unenforceable,
    envPassthrough: [...declaration.env],
  };
}

/** The policy an agent with no `sandbox:` block gets: the defaults, resolved against `cwd`. */
export function defaultPolicy(options: { cwd?: string; tmpdir?: string } = {}): SandboxPolicy {
  return policyFromDeclaration(parseSandboxDeclaration({}), options);
}
