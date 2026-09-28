/**
 * Running one command under srt, `@anthropic-ai/sandbox-runtime`
 * (gap-closure plan item 1.2, 2026-09-26; the Python twin is
 * `python/webagents/sandbox/srt.py`, and both are pinned by
 * `python/tests/fixtures/sandbox/srt.json`).
 *
 * WHY srt, AND WHY ITS CLI. This SDK had no kernel sandbox: `ShellSkill`'s
 * "sandbox mode" was a string check on the command's tokens, and a `sandbox:`
 * block in an agent file was dropped without a word (S-248 addendum). srt is
 * Seatbelt on macOS and bubblewrap on Linux, maintained for Claude Code, plus
 * the proxy outside the sandbox that lets `network:` name hosts; it is a
 * declared dependency of this package, and the Python SDK runs the same
 * engine, so one agent file is confined identically under either CLI.
 *
 * The CLI, not the library: srt's `SandboxManager` is a process-wide
 * singleton whose network lists are global, so two agents with different
 * `network:` lists in one daemon would share one policy. Running
 * `[node, cli.js, --settings <file>, -c <command>]` per command gives each
 * command its own policy and returns the command's own exit code.
 *
 * WHAT THIS MODULE DOES THAT srt DOES NOT:
 *
 * 1. Pins the binaries. `node` is `process.execPath` (or `WEBAGENTS_SRT_NODE`)
 *    and `cli.js` is resolved from the dependency, never from PATH; srt
 *    itself runs with a root-owned PATH, and the agent's PATH is restored
 *    INSIDE the sandbox, so a binary planted in a user-writable folder only
 *    ever runs confined.
 * 2. Scrubs the environment (S-220): secret-looking names are withheld, and
 *    the names that would change what srt ITSELF does are dropped
 *    (`DROPPED_ENV`): `NODE_OPTIONS` would run code in srt, outside the
 *    sandbox.
 * 3. Fails closed with the true reason: node and cli.js found, the package
 *    version exactly `SRT_VERSION` (its `package.json`; `srt --version`
 *    prints npm's variable or "1.0.0"), the Linux helpers present, and a
 *    one-time `-c true` preflight passed, before anything is called available.
 * 4. A private scratch folder through `CLAUDE_CODE_TMPDIR`, in `allowWrite`.
 * 5. Timeouts and interrupts: srt exits 0 when its child dies of SIGTERM,
 *    so the whole process group is killed here and the result says
 *    `timedOut`. An abort of `signal` (the chat's Esc or Ctrl+C, through the
 *    tool's context) kills the group the same way and the result says
 *    `interrupted` (the ptypass-fixes lane, 2026-09-27): before, the turn
 *    stopped and the command ran on for 13 to 18 seconds, until its timeout.
 *    Every command starts with stdin ignored (`/dev/null`), never the
 *    owner's terminal; the Python twin inherited it (S-317), this one never
 *    did, and a test pins it.
 * 6. The settings file is 0600 in a 0700 folder outside every writable root,
 *    written for one command and removed after it.
 *
 * THE ENGINE ARRIVES WITH THE PACKAGE, AND `webagents sandbox setup` SAYS
 * WHAT THIS MACHINE LACKS (the sandbox-engine lane, 2026-09-27). The sandbox
 * is on by default and fails closed, so a refusal names what is missing HERE
 * with its fix (`backendStatus().fix`, `unavailableMessage`): the Linux
 * programs with this distribution's install line, what a container must
 * allow, WSL 2 on Windows. `setupChecks` runs the same checks and a real
 * confined `true`; the Python twin vendors srt into its wheel and brings node
 * through nodejs-wheel-binaries. Pinned by `sandbox_engine.json`.
 *
 * THE PACKAGE IS NEVER IMPORTED STATICALLY (handbook rule 4): the portal
 * typechecks this source against a node_modules that may not carry it. Its
 * `cli.js` is located with `createRequire` and run as a child process, and
 * the load fails closed with a sentence when it is not there.
 *
 * Never set: `enableWeakerNestedSandbox` (binds the host `/proc`),
 * `allowAppleEvents`, `allowAllUnixSockets`, `allowLocalBinding`, `allowPty`.
 */

import { spawn, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SCRATCH_DIR_NAME, allowReads, denyReads, denyWrites, installWriteDenies, type SandboxPolicy } from './policy';

export const SRT_PACKAGE = '@anthropic-ai/sandbox-runtime';
export const SRT_VERSION = '0.0.77';

/**
 * What a refusal says after the reason when srt cannot run here, declared
 * sandbox or default (fixture `refusals.unavailable_tail`). The engine ships
 * inside both packages (the sandbox-engine lane, 2026-09-27), so the reason
 * before it says what THIS machine lacks, with its fix (`unavailableMessage`),
 * and the tail names the check and the opt-out.
 */
export const UNAVAILABLE_TAIL =
  '`webagents sandbox setup` checks this machine. To run commands with your permissions instead, ' +
  'pass --no-sandbox for this run or put `sandbox: off` in the agent file. The command was not run.';

/**
 * The one sentence the shell appends when a confined command's output shows
 * the sandbox refused something, naming the switch that opens it (fixture
 * `hints`). Detection reads the command's own output: srt's proxy answers a
 * host that is not listed with 403 (`CONNECT tunnel failed, response 403`),
 * a direct socket to a local port or a unix socket gets EPERM, and a DNS
 * lookup fails because the sandbox has no resolver (the proxy resolves the
 * hosts it admits).
 */
export const REFUSAL_HINTS = {
  hosts: 'The sandbox refused a network connection: list the host under `sandbox: network: hosts:` in the agent file (or a group: npm, pypi, github).',
  local: 'The sandbox refused a local connection or a listening port: set `sandbox: network: local: true` in the agent file to allow them.',
  sockets: 'The sandbox refused a unix socket: list its path under `sandbox: network: sockets:` in the agent file to allow it.',
  env: 'The sandbox keeps `.env` and secret-looking variables from commands: list a variable name under `sandbox: env:` in the agent file to pass it through.',
} as const;

export type RefusalKind = keyof typeof REFUSAL_HINTS;

const EPERM = /\bEPERM\b|Operation not permitted|Couldn't connect to server|Could not connect to server/i;
const PROXY_403 = /CONNECT tunnel failed, response 403|Tunnel connection failed: 403|403 Forbidden|blocked by network allowlist|Received HTTP code 403 from proxy/i;
const DNS_FAILURE = /Could not resolve host|getaddrinfo (?:ENOTFOUND|EAI_AGAIN)|nodename nor servname provided|Temporary failure in name resolution|Name or service not known|Name does not resolve|Failed to resolve/i;
const LOOPBACK = /\blocalhost\b|127\.0\.0\.1|\[?::1\]?|0\.0\.0\.0|\bbind\b|\blisten(?:ing)?\b|EADDRNOTAVAIL|http\.server|\bserver?\b|--port\b/i;
const UNIX_SOCKET = /\.sock\b|unix socket|unix:\/\/|ssh-agent|SSH_AUTH_SOCK|authentication agent|Docker daemon/i;
const DOTENV = /(?:^|[\s/'"=])\.env(?:\.[A-Za-z0-9_.-]+)?\b/;

/** Which refusal the output of `command` shows, if any (file comment on `REFUSAL_HINTS`). */
export function refusalKind(command: string, output: string): RefusalKind | undefined {
  const both = `${command}\n${output}`;
  if (EPERM.test(output) && DOTENV.test(command)) return 'env';
  if (UNIX_SOCKET.test(both) && (EPERM.test(output) || /Cannot connect to the Docker daemon|Could not open a connection to your authentication agent/i.test(output))) return 'sockets';
  if (PROXY_403.test(output) || DNS_FAILURE.test(output)) return 'hosts';
  if (EPERM.test(output) && LOOPBACK.test(both)) return 'local';
  return undefined;
}

/** The hint sentence for `command`'s output, or undefined when nothing was refused. */
export function refusalHint(command: string, output: string): string | undefined {
  const kind = refusalKind(command, output);
  return kind ? REFUSAL_HINTS[kind] : undefined;
}

/**
 * srt's own record of a refused host, on its stderr under `--debug`:
 * `[SandboxDebug] Connection blocked to <host>:<port>` from its proxy (the
 * same line for a CONNECT and a plain HTTP request). Read from srt's output,
 * never guessed from the command.
 */
const BLOCKED_LINE = /^\[SandboxDebug\] (?:Connection blocked to|HTTP request blocked to) (\S+):(\d+)\s*$/;

export function refusedHostsFromSrtLog(stderr: string): string[] {
  const hosts: string[] = [];
  for (const line of stderr.split('\n')) {
    const match = BLOCKED_LINE.exec(line.trim());
    if (!match) continue;
    const host = match[1].replace(/^\[|\]$/g, '').toLowerCase();
    if (!hosts.includes(host)) hosts.push(host);
  }
  return hosts;
}

/** The PATH srt itself runs with: root-owned directories only. */
export const SRT_PATH = '/usr/bin:/bin:/usr/sbin:/sbin';

/** An explicit install: the absolute path of srt's `dist/cli.js`, and the node to run it with. */
export const ENV_CLI = 'WEBAGENTS_SRT_CLI';
export const ENV_NODE = 'WEBAGENTS_SRT_NODE';

/** srt's own grace between SIGTERM and SIGKILL, matched here. */
export const KILL_GRACE_MS = 2000;

/**
 * What the shell tool and the SKILL.md script runner answer for a command an
 * interrupt stopped (fixture `srt.json` `interrupt.result`, the Python
 * `INTERRUPTED_RESULT`).
 */
export const INTERRUPTED_RESULT = 'Interrupted: the command and everything it started were stopped.';

const PREFLIGHT_TIMEOUT_MS = 60_000;

/** What srt needs on Linux, looked for in root-owned directories only. */
export const LINUX_DEPS = ['bwrap', 'socat', 'rg'] as const;
export const ROOT_OWNED_BIN_DIRS = ['/usr/bin', '/bin', '/usr/sbin', '/sbin', '/usr/local/bin'] as const;

/**
 * A variable whose NAME contains one of these is withheld from a sandboxed
 * command unless the agent lists it in `env_passthrough` (S-220). Matched
 * case-insensitively; the name is how every credential announces itself.
 */
export const SECRET_NAME_PARTS = [
  'KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'PASSWD', 'CREDENTIAL', 'PRIVATE', 'AUTH', 'SESSION', 'COOKIE',
] as const;

/**
 * Names dropped from srt's environment (case-insensitively), on top of the
 * secret scrub: they change what srt or the wrapped bash does, outside or
 * before the sandbox.
 */
export const DROPPED_ENV = [
  'SRT_DEBUG', 'CLAUDE_TMPDIR', 'CLAUDE_CODE_TMPDIR', 'NODE_OPTIONS', 'NODE_PATH', 'NODE_REPL_EXTERNAL_MODULE',
  'BASH_ENV', 'ENV', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'FTP_PROXY', 'GRPC_PROXY',
  'RSYNC_PROXY', 'SANDBOX_RUNTIME',
] as const;

/** No enforcement backend, and the policy asked for one. Raised rather than degraded. */
export class SandboxUnavailable extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SandboxUnavailable';
  }
}

export interface SrtLocation {
  node: string;
  cli: string;
  version: string;
  packageDir: string;
  nodeFrom: string;
  cliFrom: string;
}

export interface BackendStatus {
  platform: string;
  backend: 'srt' | null;
  path: string | null;
  node: string | null;
  version: string | null;
  available: boolean;
  reason: string;
  /** What to do on this machine when `available` is false; empty when nothing better than `webagents sandbox setup`. */
  fix: string;
  found: string;
}

export interface SandboxResult {
  stdout: string;
  stderr: string;
  exitCode: number;
  timedOut: boolean;
  /** The run's `signal` was aborted (Esc or Ctrl+C in the chat) and the command's whole process group killed. */
  interrupted?: boolean;
  /**
   * The hosts srt's proxy refused during the command, from srt's own log
   * (`captureRefusals`); absent otherwise. Lower-cased host names, no port.
   */
  refusedHosts?: string[];
}

type Env = Record<string, string | undefined>;

// ---------------------------------------------------------------------------
// The environment
// ---------------------------------------------------------------------------

/** The environment a sandboxed command gets, and the names withheld (S-220). */
export function scrubEnvironment(env: Env, passthrough: readonly string[] = []): { kept: Record<string, string>; withheld: string[] } {
  const allowed = new Set(passthrough.map((name) => name.toUpperCase()));
  const kept: Record<string, string> = {};
  const withheld: string[] = [];
  for (const [name, value] of Object.entries(env)) {
    if (value === undefined) continue;
    const upper = name.toUpperCase();
    if (!allowed.has(upper) && SECRET_NAME_PARTS.some((part) => upper.includes(part))) {
      withheld.push(name);
      continue;
    }
    kept[name] = value;
  }
  return { kept, withheld: withheld.sort() };
}

/**
 * The environment srt runs with, from an already secret-scrubbed one: the
 * `DROPPED_ENV` names gone, a root-owned PATH, and the scratch folder named
 * for srt to export as TMPDIR inside.
 */
export function srtEnvironment(scrubbed: Record<string, string>, scratch?: string): Record<string, string> {
  const dropped = new Set<string>(DROPPED_ENV.map((name) => name.toUpperCase()));
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(scrubbed)) if (!dropped.has(name.toUpperCase())) env[name] = value;
  env.PATH = SRT_PATH;
  if (scratch) {
    env.CLAUDE_CODE_TMPDIR = scratch;
    env.TMPDIR = scratch;
  }
  return env;
}

/** `'...'` with single quotes escaped, as `shlex.quote` does. */
function shellQuote(value: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'"'"'`)}'`;
}

/**
 * The NO_PROXY srt exports inside the sandbox (its `generateProxyEnvVars`,
 * the list of `SRT_VERSION` exactly): loopback and the private ranges are
 * sent direct rather than through srt's proxy. A LISTED LOOPBACK HOST WAS
 * UNREACHABLE (2026-09-27, the final e2e re-run, `g1b-netdebug`): every
 * ordinary client honours NO_PROXY, the sandbox refuses its direct socket
 * (`Operation not permitted`), and only the proxy, which the same list told
 * it to skip, could have reached the host `network:` named. So the command
 * starts with a NO_PROXY from which every entry covering a `network:` host
 * has been removed (`noProxyFor`), and those hosts go through the proxy,
 * which admits them. Pinned by the fixture's `no_proxy`.
 */
export const SRT_NO_PROXY = ['localhost', '127.0.0.1', '::1', '169.254.0.0/16', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'] as const;

/** The host a `network:` entry names: without a port, a leading `*.` or IPv6 brackets, lower-cased. */
export function networkHost(entry: string): string {
  let value = entry.trim().toLowerCase();
  if (value.startsWith('*.')) value = value.slice(2);
  const v6 = /^\[([^\]]+)\](?::\d{1,5})?$/.exec(value);
  if (v6) return v6[1];
  const colon = value.lastIndexOf(':');
  if (colon > 0 && value.indexOf(':') === colon && /^\d{1,5}$/.test(value.slice(colon + 1))) value = value.slice(0, colon);
  return value.replace(/\.$/, '');
}

/**
 * Whether one NO_PROXY entry covers `host`: a name covers itself and every
 * host under it, an address covers that address, a range covers an address
 * inside it (the matching every client does, so what a client would bypass
 * the proxy for).
 */
export function noProxyEntryCovers(entry: string, host: string): boolean {
  const target = networkHost(host);
  const family = net.isIP(target);
  if (entry.includes('/')) {
    if (!family) return false;
    const [address, prefix] = entry.split('/');
    const list = new net.BlockList();
    const entryFamily = net.isIP(address) === 6 ? 'ipv6' : 'ipv4';
    if ((entryFamily === 'ipv6') !== (family === 6)) return false;
    list.addSubnet(address, Number(prefix), entryFamily);
    return list.check(target, entryFamily);
  }
  if (net.isIP(entry)) {
    if (!family || (net.isIP(entry) === 6) !== (family === 6)) return false;
    const list = new net.BlockList();
    list.addAddress(entry, family === 6 ? 'ipv6' : 'ipv4');
    return list.check(target, family === 6 ? 'ipv6' : 'ipv4');
  }
  return !family && (target === entry || target.endsWith(`.${entry}`));
}

/** srt's NO_PROXY less every entry that covers a host in `network`, in srt's order. */
export function noProxyFor(networkDomains: readonly string[]): string[] {
  return SRT_NO_PROXY.filter((entry) => !networkDomains.some((host) => noProxyEntryCovers(entry, host)));
}

/**
 * The `-c` string: the agent's PATH restored inside, Node told to honour the
 * proxy variables (`NODE_USE_ENV_PROXY=1`: Node 24 and later ignore
 * HTTP_PROXY otherwise, so `fetch` to a listed host was stopped at the
 * socket while curl reached it), the NO_PROXY the `network:` list leaves
 * (only when it removed something; `unset` when nothing is left), then the
 * command on its own line. With `mergeStderr`, the command's stderr joins
 * its stdout, so srt's own stderr carries only srt's lines
 * (`refusedHostsFromSrtLog`).
 */
export function wrappedCommand(command: string, userPath: string, networkDomains: readonly string[] = [], mergeStderr = false): string {
  // npm's cache lives in `~/.npm`, which no policy makes writable, and npm
  // refuses to run without a writable cache (probed 2026-09-27: it asked for
  // a chown). The scratch folder is writable by construction.
  const lines = [`export PATH=${shellQuote(userPath)}`, 'export NODE_USE_ENV_PROXY=1', 'export npm_config_cache="${TMPDIR:-/tmp}/npm-cache"'];
  const kept = noProxyFor(networkDomains);
  if (kept.length !== SRT_NO_PROXY.length) {
    const value = shellQuote(kept.join(','));
    lines.push(kept.length ? `export NO_PROXY=${value} no_proxy=${value}` : 'unset NO_PROXY no_proxy');
  }
  if (mergeStderr) lines.push('exec 2>&1');
  return [...lines, command].join('\n');
}

/** `KEY=value` lines of one `.env` file, quotes stripped, as the CLI's config store reads them. */
function parseDotenv(file: string): Record<string, string> {
  const out: Record<string, string> = {};
  let text: string;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return out;
  }
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
    const eq = trimmed.indexOf('=');
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (value.length >= 2 && value[0] === value[value.length - 1] && (value[0] === '"' || value[0] === "'")) value = value.slice(1, -1);
    if (key) out[key] = value;
  }
  return out;
}

/**
 * The listed `env` names that the process environment does not have, read
 * from the `.env` files the CLI loads (`./.env`, then the profile folder's
 * `~/.webagents[-<profile>]/.env`), and nothing else from those files: the
 * command cannot read `.env` itself, and S-220's rule stands, only listed
 * names pass. The process environment wins where it has the name.
 */
export function envFromDotenv(names: readonly string[], cwd: string, env: Env = process.env): Record<string, string> {
  if (!names.length) return {};
  const profile = (env.WEBAGENTS_PROFILE ?? '').trim();
  const files = [path.join(cwd, '.env'), path.join(os.homedir(), profile ? `.webagents-${profile}` : '.webagents', '.env')];
  const found: Record<string, string> = {};
  for (const file of files) {
    const values = parseDotenv(file);
    for (const name of names) {
      if (env[name] !== undefined || found[name] !== undefined) continue;
      const hit = Object.keys(values).find((key) => key.toUpperCase() === name.toUpperCase());
      if (hit !== undefined) found[name] = values[hit];
    }
  }
  return found;
}

// ---------------------------------------------------------------------------
// Locating srt
// ---------------------------------------------------------------------------

function executable(p: string): boolean {
  try {
    fs.accessSync(p, fs.constants.X_OK);
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

/** A regular, executable file that neither it nor its folder lets other users replace. */
function safeBinary(p: string): { ok: true; path: string } | { ok: false; detail: string } {
  let realPath: string;
  try {
    realPath = fs.realpathSync(p);
  } catch {
    return { ok: false, detail: `${p} is not an executable file` };
  }
  if (!executable(realPath)) return { ok: false, detail: `${p} is not an executable file` };
  try {
    const mode = fs.statSync(realPath).mode;
    const dirMode = fs.statSync(path.dirname(realPath)).mode;
    if (mode & 0o002 || dirMode & 0o002) return { ok: false, detail: `${p} or its folder is world-writable` };
  } catch (err) {
    return { ok: false, detail: `${p} cannot be inspected: ${(err as Error).message}` };
  }
  return { ok: true, path: realPath };
}

function packageVersion(packageDir: string): string | undefined {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(packageDir, 'package.json'), 'utf8')) as { version?: unknown };
    return data && typeof data.version === 'string' && data.version ? data.version : undefined;
  } catch {
    return undefined;
  }
}

/** The dependency's own install, resolved from this module; undefined when it is not installed. */
function dependencyCli(): string | undefined {
  try {
    const pkg = createRequire(import.meta.url).resolve(`${SRT_PACKAGE}/package.json`);
    const cli = path.join(path.dirname(pkg), 'dist', 'cli.js');
    return fs.existsSync(cli) ? cli : undefined;
  } catch {
    return undefined;
  }
}

/** srt's `engines.node`: the oldest node it runs on. */
export const NODE_MINIMUM: readonly [number, number, number] = [20, 11, 0];

/**
 * How each piece was found, and the fixes the reasons come with (fixture
 * `sandbox_engine.json`, the sandbox-engine lane, 2026-09-27).
 */
export const CLI_FROM_BUNDLED = "the package's own install";
export const NODE_FROM_PROCESS = 'this process';
export const FIX_SETUP_POINTER = 'run `webagents sandbox setup` for the details';
export const FIX_WINDOWS = 'run webagents inside WSL 2 (Windows Subsystem for Linux), where the Linux sandbox works';
export const FIX_PLATFORM = 'run webagents on macOS, Linux or WSL 2';
export const FIX_TMPDIR = "point TMPDIR at a short path, such as /tmp/wa: srt's socket path must fit in 104 bytes on macOS";
export const FIX_CONTAINER =
  'run the container with `--security-opt seccomp=unconfined --security-opt apparmor=unconfined` ' +
  '(or `--privileged`) so bubblewrap can create its namespaces, and as a user other than root';
export const FIX_NAMESPACES = {
  userns_apparmor: '`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, or an AppArmor profile that grants bwrap `userns`',
  userns_clone: '`sudo sysctl -w kernel.unprivileged_userns_clone=1`',
  userns_max: '`sudo sysctl -w user.max_user_namespaces=15000`',
} as const;

const minimumText = (): string => NODE_MINIMUM.join('.');

/** `[major, minor, patch]` from `node --version`'s output, or undefined. */
export function parseNodeVersion(output: string): [number, number, number] | undefined {
  const match = /^v(\d+)\.(\d+)\.(\d+)/.exec((output ?? '').trim());
  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : undefined;
}

function atLeastMinimum(version: readonly number[]): boolean {
  for (let i = 0; i < 3; i += 1) if (version[i] !== NODE_MINIMUM[i]) return version[i] > NODE_MINIMUM[i];
  return true;
}

const NODE_VERSIONS = new Map<string, [number, number, number] | undefined>();

/**
 * The version a node binary reports. It runs with an EMPTY environment:
 * NODE_OPTIONS (`--require`) would otherwise run code in it, outside any
 * sandbox. The node running this process answers from `process.version`.
 */
function nodeVersion(node: string): [number, number, number] | undefined {
  if (node === safeRealPath(process.execPath)) return parseNodeVersion(process.version);
  if (!NODE_VERSIONS.has(node)) {
    const result = spawnSync(node, ['--version'], { encoding: 'utf8', env: {} as NodeJS.ProcessEnv, cwd: '/', timeout: 15_000 });
    NODE_VERSIONS.set(node, result.status === 0 ? parseNodeVersion(result.stdout ?? '') : undefined);
  }
  return NODE_VERSIONS.get(node);
}

function safeRealPath(p: string): string {
  try {
    return fs.realpathSync(p);
  } catch {
    return p;
  }
}

function engineEnvFix(): string {
  return `point ${ENV_CLI} at srt ${SRT_VERSION}'s dist/cli.js, or unset it to use the engine that ships with webagents`;
}

type Found<T> = ({ ok: true } & T) | { ok: false; reason: string; fix: string };

/**
 * srt's `cli.js`. `WEBAGENTS_SRT_CLI` wins when set (an explicit install is
 * never second-guessed); otherwise the dependency's own. Either must be
 * exactly `SRT_VERSION`, read from its `package.json`: srt's settings schema
 * strips unknown keys silently, so a version this SDK has not been checked
 * against could drop a rule unseen.
 */
export function locateCli(env: Env = process.env): Found<{ cli: string; cliFrom: string }> {
  const explicit = (env[ENV_CLI] ?? '').trim();
  let cli: string;
  let cliFrom: string;
  if (explicit) {
    const expanded = explicit.startsWith('~') ? path.join(os.homedir(), explicit.slice(1)) : explicit;
    if (!fs.existsSync(expanded) || !fs.statSync(expanded).isFile()) {
      return { ok: false, reason: `${SRT_PACKAGE}@${SRT_VERSION} was not found (${ENV_CLI}=${explicit} is not a file)`, fix: engineEnvFix() };
    }
    cli = fs.realpathSync(expanded);
    cliFrom = ENV_CLI;
  } else {
    const own = dependencyCli();
    if (!own) {
      return {
        ok: false,
        reason: `${SRT_PACKAGE}@${SRT_VERSION} was not found (it is not installed next to webagents)`,
        fix: `reinstall webagents, which depends on ${SRT_PACKAGE}@${SRT_VERSION}`,
      };
    }
    cli = fs.realpathSync(own);
    cliFrom = CLI_FROM_BUNDLED;
  }
  const version = packageVersion(path.dirname(path.dirname(cli)));
  if (version !== SRT_VERSION) {
    return {
      ok: false,
      reason: `${cli} is ${SRT_PACKAGE} ${version ?? 'of no readable version'}; this SDK requires exactly ${SRT_VERSION}`,
      fix: cliFrom === ENV_CLI ? engineEnvFix() : `reinstall webagents, which depends on ${SRT_PACKAGE}@${SRT_VERSION}`,
    };
  }
  return { ok: true, cli, cliFrom };
}

/**
 * The node srt runs with: `WEBAGENTS_SRT_NODE` when set (a safe binary of
 * srt's minimum), else the node running this process, which the package's
 * own `engines` already holds above srt's.
 */
export function chooseNode(env: Env = process.env): Found<{ node: string; nodeFrom: string }> {
  const minimum = minimumText();
  const explicit = (env[ENV_NODE] ?? '').trim();
  const checked = safeBinary(explicit ? (explicit.startsWith('~') ? path.join(os.homedir(), explicit.slice(1)) : explicit) : process.execPath);
  const label = explicit ? `${ENV_NODE}: ` : '';
  if (!checked.ok) {
    return {
      ok: false,
      reason: `node was not found for srt (${label}${checked.detail})`,
      fix: explicit ? `point ${ENV_NODE} at node ${minimum} or later, or unset it` : `set ${ENV_NODE} to node ${minimum} or later`,
    };
  }
  const version = nodeVersion(checked.path);
  if (!version || !atLeastMinimum(version)) {
    const shown = explicit || checked.path;
    const detail = version
      ? `${shown} is node v${version.join('.')}; srt needs ${minimum} or later`
      : `${shown} does not report a node version; srt needs ${minimum} or later`;
    return {
      ok: false,
      reason: `node was not found for srt (${label}${detail})`,
      fix: explicit ? `point ${ENV_NODE} at node ${minimum} or later, or unset it` : `run webagents with node ${minimum} or later`,
    };
  }
  return { ok: true, node: checked.path, nodeFrom: explicit ? ENV_NODE : NODE_FROM_PROCESS };
}

/**
 * Where srt is, or why it cannot be used, and what to do about it
 * (`locateCli`, then `chooseNode`).
 */
export function locateEngine(env: Env = process.env): { location: SrtLocation; reason: ''; fix: '' } | { location: null; reason: string; fix: string } {
  const cli = locateCli(env);
  if (!cli.ok) return { location: null, reason: cli.reason, fix: cli.fix };
  const node = chooseNode(env);
  if (!node.ok) return { location: null, reason: node.reason, fix: node.fix };
  return {
    location: { node: node.node, cli: cli.cli, version: SRT_VERSION, packageDir: path.dirname(path.dirname(cli.cli)), nodeFrom: node.nodeFrom, cliFrom: cli.cliFrom },
    reason: '',
    fix: '',
  };
}

// ---------------------------------------------------------------------------
// The SDK's own install (S-316)
// ---------------------------------------------------------------------------

/** This module's own file, or undefined where the bundler gives no file URL. */
function thisModuleFile(): string | undefined {
  try {
    return fileURLToPath(import.meta.url);
  } catch {
    return undefined;
  }
}

/** The nearest folder up from `moduleFile` whose package.json names `webagents`: the SDK's package root. */
function sdkPackageRoot(moduleFile: string): string | undefined {
  let dir = path.dirname(moduleFile);
  for (let i = 0; i < 8; i += 1) {
    try {
      const data = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8')) as { name?: unknown };
      if (data && data.name === 'webagents') return dir;
    } catch {
      // no package.json here
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return undefined;
}

/** The outermost `node_modules` folder `p` lies in, else `p`: npm and pnpm keep a package's dependencies under it. */
function outermostNodeModules(p: string): string {
  const parts = p.split(path.sep);
  const at = parts.indexOf('node_modules');
  return at >= 0 ? parts.slice(0, at + 1).join(path.sep) || path.sep : p;
}

const INSTALLS = new Map<string, string[]>();

/**
 * Where the running CLI and its sandbox engine live, realpath'd (S-316,
 * fixture `sdk_install_deny.typescript`): the outermost `node_modules`
 * holding the webagents package (every dependency the CLI loads), else its
 * package folder (a source checkout); the same for srt, from the install
 * `locateCli` found; and the folders holding `process.execPath` and the node
 * srt runs on. `buildSettings` write-denies the ones that lie inside a write
 * root (`installWriteDenies`). `moduleFile` is for tests: where this module
 * would be.
 */
export function sdkInstallPaths(env: Env = process.env, moduleFile?: string): string[] {
  const here = moduleFile ?? thisModuleFile();
  const key = `${env[ENV_CLI] ?? ''}\n${env[ENV_NODE] ?? ''}\n${here ?? ''}`;
  const cached = INSTALLS.get(key);
  if (cached) return [...cached];
  const found: string[] = [];
  const add = (p: string | undefined) => {
    if (!p) return;
    const resolved = safeRealPath(p);
    if (!found.includes(resolved)) found.push(resolved);
  };
  const root = here ? sdkPackageRoot(here) : undefined;
  if (root) add(outermostNodeModules(safeRealPath(root)));
  const cli = locateCli(env);
  if (cli.ok) add(outermostNodeModules(path.dirname(path.dirname(cli.cli))));
  add(path.dirname(safeRealPath(process.execPath)));
  const node = chooseNode(env);
  if (node.ok) add(path.dirname(node.node));
  // An install already covered by another one adds nothing.
  const installs = found.filter((p) => !found.some((other) => other !== p && (p === other || p.startsWith(other.replace(/\/+$/, '') + '/'))));
  INSTALLS.set(key, installs);
  return [...installs];
}

/**
 * What `doctor` and `webagents sandbox setup` say when the install lies
 * inside the agent's folder (fixture `sdk_install_deny.report`, the Python
 * words).
 */
export const INSTALL_INSIDE_DETAIL =
  'webagents runs from inside this folder ({where}): shell commands may not write there, ' +
  'so no command can change the code that confines the next one';
export const INSTALL_INSIDE_FIX = 'if commands here must install packages into that environment, install webagents outside this folder';
export const INSTALL_IS_THE_FOLDER = 'this folder itself';

/**
 * The `install` check for `doctor` and `sandbox setup`: a warning naming the
 * parts of the install inside `folders` (the agent's folder, or the
 * policy's write roots), relative to the first folder where they can be;
 * undefined when there are none. The sandbox protects them either way; the
 * warning says why a confined `npm install` into the CLI's own
 * `node_modules` fails.
 */
export function installInsideCheck(folders: readonly string[], env: Env = process.env, moduleFile?: string): SetupCheck | undefined {
  const roots = folders.filter(Boolean).map((folder) => safeRealPath(folder));
  const hits = installWriteDenies(roots, sdkInstallPaths(env, moduleFile));
  if (!hits.length) return undefined;
  const base = roots[0].replace(/\/+$/, '') || '/';
  const shown = hits.map((hit) => (hit === base ? INSTALL_IS_THE_FOLDER : hit.startsWith(base + '/') ? path.relative(base, hit) : hit));
  return setupCheck('install', 'warn', INSTALL_INSIDE_DETAIL.replace('{where}', shown.join(', ')), INSTALL_INSIDE_FIX);
}

/** Where srt is, or why it cannot be used (`locateEngine` without the fix). */
export function locateSrt(env: Env = process.env): { location: SrtLocation; reason: '' } | { location: null; reason: string } {
  const found = locateEngine(env);
  return found.location ? { location: found.location, reason: '' } : { location: null, reason: found.reason };
}

/** Absolute, root-owned paths of srt's Linux programs, and the ones missing. */
function linuxPrograms(): { found: Record<string, string>; missing: string[] } {
  const found: Record<string, string> = {};
  const missing: string[] = [];
  for (const dep of LINUX_DEPS) {
    const hit = ROOT_OWNED_BIN_DIRS.map((dir) => path.join(dir, dep)).find(executable);
    if (hit) found[dep] = hit;
    else missing.push(dep);
  }
  return { found, missing };
}

export function programsReason(missing: readonly string[]): string {
  return `${missing.join(', ')} not found in ${ROOT_OWNED_BIN_DIRS.join(', ')}; srt needs bubblewrap, socat and ripgrep on Linux`;
}

function linuxDeps(): { found: Record<string, string>; missing: string } {
  const { found, missing } = linuxPrograms();
  return { found, missing: missing.length ? programsReason(missing) : '' };
}

/** The Linux packages that hold srt's programs, and the install line per family (fixture `install`). */
export const LINUX_PACKAGES: Record<string, string> = { bwrap: 'bubblewrap', socat: 'socat', rg: 'ripgrep' };
const INSTALL_FAMILIES: ReadonlyArray<[readonly string[], string]> = [
  [['debian', 'ubuntu'], 'sudo apt-get install {packages}'],
  [['fedora', 'rhel', 'centos', 'rocky', 'almalinux'], 'sudo dnf install {packages}'],
  [['arch', 'manjaro'], 'sudo pacman -S {packages}'],
  [['alpine'], 'sudo apk add {packages}'],
  [['opensuse', 'suse', 'sles'], 'sudo zypper install {packages}'],
];

function osReleaseIds(text: string): string[] {
  const values: Record<string, string> = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    const eq = trimmed.indexOf('=');
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq);
    if (key === 'ID' || key === 'ID_LIKE') values[key] = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, '').toLowerCase();
  }
  return [values.ID ?? '', ...(values.ID_LIKE ?? '').split(/\s+/)].filter(Boolean);
}

/**
 * The line that installs the `missing` programs on the distribution
 * `osRelease` describes, in backquotes; a plain sentence when the family is
 * not known.
 */
export function linuxInstallFix(osRelease: string, missing: readonly string[]): string {
  const packages = LINUX_DEPS.filter((dep) => missing.includes(dep)).map((dep) => LINUX_PACKAGES[dep]).join(' ');
  for (const id of osReleaseIds(osRelease)) {
    for (const [ids, line] of INSTALL_FAMILIES) {
      if (ids.includes(id) || ids.some((known) => id.startsWith(`${known}-`))) return '`' + line.replace('{packages}', packages) + '`';
    }
  }
  return `install ${packages} with your package manager`;
}

function readText(p: string): string {
  try {
    return fs.readFileSync(p, 'utf8');
  } catch {
    return '';
  }
}

function osRelease(root = '/'): string {
  return readText(path.join(root, 'etc', 'os-release')) || readText(path.join(root, 'usr', 'lib', 'os-release'));
}

const CGROUP_WORDS = /docker|kubepods|containerd|lxc|podman/;

/** Whether this Linux process runs in a container (fixture `container`). */
export function inContainer(root = '/', env: Env = process.env): boolean {
  if (fs.existsSync(path.join(root, '.dockerenv')) || fs.existsSync(path.join(root, 'run', '.containerenv'))) return true;
  if (env.container) return true;
  return CGROUP_WORDS.test(readText(path.join(root, 'proc', '1', 'cgroup')));
}

const NAMESPACE_CHECKS: ReadonlyArray<[string, string, keyof typeof FIX_NAMESPACES]> = [
  ['kernel/apparmor_restrict_unprivileged_userns', '1', 'userns_apparmor'],
  ['kernel/unprivileged_userns_clone', '0', 'userns_clone'],
  ['user/max_user_namespaces', '0', 'userns_max'],
];

/** The first setting that stops bubblewrap from creating user namespaces here, or undefined. */
export function namespaceRestriction(root = '/'): { file: string; value: string; fix: string } | undefined {
  for (const [relative, restricts, fix] of NAMESPACE_CHECKS) {
    const value = readText(path.join(root, 'proc', 'sys', relative)).trim();
    if (value === restricts) return { file: `/proc/sys/${relative}`, value, fix: FIX_NAMESPACES[fix] };
  }
  return undefined;
}

/**
 * The longest unix socket path srt can listen on and connect to on macOS
 * (fixture `tmpdir.limit`): `sun_path` holds 104 bytes, and node uses all of
 * them. Measured with real srt on 2026-09-27: 104 bytes listened and carried
 * a request, 105 failed with `listen EINVAL`.
 */
export const SOCKET_PATH_LIMIT = 104;

/**
 * The socket name srt gives its proxy, `srt-mux-<pid>-<n>.sock` (its
 * `sandbox/mux-proxy.js`), with the widest pid macOS hands out (xnu
 * `PID_MAX` is 99999) and srt's first socket (fixture `tmpdir.socket`).
 */
export const SRT_SOCKET_NAME = 'srt-mux-99999-0.sock';

/** Realpath of `p`, resolving the nearest existing ancestor when `p` is not there yet (as the policy's `real` does). */
function realOrAncestor(p: string): string {
  const absolute = path.resolve(p);
  try {
    return fs.realpathSync(absolute);
  } catch {
    const parent = path.dirname(absolute);
    return parent === absolute ? absolute : path.join(realOrAncestor(parent), path.basename(absolute));
  }
}

/**
 * The longest socket path srt builds here (fixture `tmpdir`).
 *
 * srt listens on `join(os.tmpdir(), "srt-mux-<pid>-<n>.sock")`, and its
 * `os.tmpdir()` is the TMPDIR this module gives it, as spelled: the
 * preflight's `mkdtemp` folder under the temp folder (not realpath'd; the
 * kernel measures the string), or a command's scratch folder, which the
 * policy realpaths (`$TMPDIR/webagents-sandbox`). The longer of the two is
 * measured.
 *
 * MEASURED, NOT GUESSED (the ptypass-fixes lane, 2026-09-27). The first
 * version realpath'd the temp folder (`/private` added, which srt never
 * sees) and counted a 7-digit pid, so every stock Mac read 110 bytes and
 * `webagents sandbox setup` failed, while a confined `true` and a confined
 * curl both ran (the PTY pass, `07e-tsetup2`).
 */
export function srtSocketProbe(tmpdir?: string): string {
  const preflight = path.join(path.resolve(tmpdir ?? os.tmpdir()), 'webagents-srt-preflight-XXXXXX');
  const scratch = path.join(realOrAncestor(tmpdir ?? process.env.TMPDIR ?? os.tmpdir()), SCRATCH_DIR_NAME);
  const longest = Buffer.byteLength(preflight) >= Buffer.byteLength(scratch) ? preflight : scratch;
  return path.join(longest, SRT_SOCKET_NAME);
}

/** Whether srt's socket path passes macOS's 104-byte limit here: exactly when srt itself would fail. */
export function tmpdirTooLong(tmpdir?: string): boolean {
  return Buffer.byteLength(srtSocketProbe(tmpdir)) > SOCKET_PATH_LIMIT;
}

function preflightFix(platform: string): string {
  if (platform === 'darwin') return tmpdirTooLong() ? FIX_TMPDIR : '';
  if (platform === 'linux') return inContainer() ? FIX_CONTAINER : (namespaceRestriction()?.fix ?? '');
  return '';
}

function platformRefusal(platform: string): { reason: string; fix: string } {
  if (platform === 'win32') return { reason: 'there is no sandbox on native Windows', fix: FIX_WINDOWS };
  return { reason: `there is no sandbox on ${platform}; srt supports macOS and Linux`, fix: FIX_PLATFORM };
}

/** srt's own pins for the Linux helpers, so it never takes them from PATH. */
function linuxPins(deps: Record<string, string>): Record<string, unknown> {
  const pins: Record<string, unknown> = {};
  if (deps.bwrap) pins.bwrapPath = deps.bwrap;
  if (deps.socat) pins.socatPath = deps.socat;
  if (deps.rg) pins.ripgrep = { command: deps.rg };
  return pins;
}

// ---------------------------------------------------------------------------
// The status, once per process and per explicit-install setting
// ---------------------------------------------------------------------------

const STATUS = new Map<string, BackendStatus>();

export function resetBackendStatus(): void {
  STATUS.clear();
}

/**
 * What backend is in use, and why not, so `doctor` can say something true.
 * `available` is only true after a real `-c true` has run. `fix` is what to
 * do on THIS machine (the sandbox-engine lane, 2026-09-27): the install line
 * for the missing Linux programs, what a container must allow, WSL 2 on
 * Windows; empty when there is nothing better than `webagents sandbox setup`.
 */
export function backendStatus(): BackendStatus {
  const key = `${process.env[ENV_CLI] ?? ''}\n${process.env[ENV_NODE] ?? ''}`;
  const cached = STATUS.get(key);
  if (cached) return { ...cached };

  const status: BackendStatus = {
    platform: process.platform,
    backend: null,
    path: null,
    node: null,
    version: null,
    available: false,
    reason: '',
    fix: '',
    found: '',
  };
  const finish = (): BackendStatus => {
    STATUS.set(key, status);
    return { ...status };
  };
  if (process.platform !== 'darwin' && process.platform !== 'linux') {
    Object.assign(status, platformRefusal(process.platform));
    return finish();
  }
  let deps: Record<string, string> = {};
  if (process.platform === 'linux') {
    const programs = linuxPrograms();
    if (programs.missing.length) {
      status.reason = programsReason(programs.missing);
      status.fix = linuxInstallFix(osRelease(), programs.missing);
      return finish();
    }
    deps = programs.found;
  }
  const located = locateEngine();
  if (!located.location) {
    status.reason = located.reason;
    status.fix = located.fix;
    return finish();
  }
  const { location } = located;
  status.path = location.cli;
  status.node = location.node;
  status.version = location.version;
  status.found = `cli.js from ${location.cliFrom}, node from ${location.nodeFrom}`;
  const failure = preflight(location, deps);
  if (failure) {
    status.reason = failure;
    status.fix = preflightFix(process.platform);
  } else {
    status.backend = 'srt';
    status.available = true;
  }
  return finish();
}

/**
 * The refusal when the engine cannot run (fixture `sandbox_engine.json`
 * `unavailable`): what this machine lacks, its fix when it has one, then
 * `UNAVAILABLE_TAIL`, which names the check and the opt-out.
 */
export function unavailableMessage(status: { reason?: string; fix?: string }): string {
  const reason = status.reason || 'no sandbox backend here';
  return status.fix ? `${reason}: ${status.fix}. ${UNAVAILABLE_TAIL}` : `${reason}. ${UNAVAILABLE_TAIL}`;
}

/**
 * doctor's fix line for an engine that cannot run, the chat's `sandboxFix`
 * in the same words (fixture `sandbox_engine.json` `doctor`): this
 * machine's fix, then the opt-out.
 */
export function unavailableFix(status: { fix?: string }): string {
  return `${status.fix || FIX_SETUP_POINTER}, or pass --no-sandbox to run commands with your permissions for this run`;
}

/** One `webagents sandbox setup` line, in doctor's shape. */
export interface SetupCheck {
  name: string;
  status: 'ok' | 'warn' | 'fail';
  detail: string;
  fix?: string;
}

function setupCheck(name: string, status: SetupCheck['status'], detail: string, fix?: string): SetupCheck {
  return fix ? { name, status, detail, fix } : { name, status, detail };
}

/**
 * `webagents sandbox setup`: whether the engine runs on this machine, as
 * doctor-style checks in the order of the fixture's `setup.checks`. A CHECK,
 * NOT AN INSTALLER: it installs and changes nothing and never downloads. The
 * last check runs a real confined `true` (a fresh preflight, not the cached
 * status), so an `ok` here means commands are sandboxed here.
 */
export function setupChecks(env: Env = process.env): SetupCheck[] {
  const platform = process.platform;
  const checks: SetupCheck[] = [];
  const notRun = 'not run: fix the checks above first';
  if (platform !== 'darwin' && platform !== 'linux') {
    const refusal = platformRefusal(platform);
    return [setupCheck('platform', 'fail', refusal.reason, refusal.fix), setupCheck('confined', 'fail', notRun)];
  }
  if (platform === 'darwin') {
    const version = spawnSync('/usr/bin/sw_vers', ['-productVersion'], { encoding: 'utf8', env: {} as NodeJS.ProcessEnv }).stdout?.trim();
    checks.push(setupCheck('platform', 'ok', `macOS ${version || os.release()}`));
  } else {
    const pretty = /^PRETTY_NAME=(.*)$/m.exec(osRelease())?.[1]?.trim().replace(/^["']|["']$/g, '');
    checks.push(setupCheck('platform', 'ok', pretty ? `Linux (${pretty})` : 'Linux'));
  }
  let blocked = false;
  const cli = locateCli(env);
  if (cli.ok) checks.push(setupCheck('engine', 'ok', `srt ${SRT_VERSION}, ${cli.cliFrom}`));
  else {
    checks.push(setupCheck('engine', 'fail', cli.reason, cli.fix));
    blocked = true;
  }
  const node = chooseNode(env);
  if (node.ok) checks.push(setupCheck('node', 'ok', `v${(nodeVersion(node.node) ?? []).join('.')} from ${node.nodeFrom} (${node.node})`));
  else {
    checks.push(setupCheck('node', 'fail', node.reason, node.fix));
    blocked = true;
  }
  let deps: Record<string, string> = {};
  if (platform === 'linux') {
    const programs = linuxPrograms();
    deps = programs.found;
    if (programs.missing.length) {
      checks.push(setupCheck('programs', 'fail', programsReason(programs.missing), linuxInstallFix(osRelease(), programs.missing)));
      blocked = true;
    } else checks.push(setupCheck('programs', 'ok', 'bwrap, socat and rg'));
  }
  let confined: SetupCheck;
  if (blocked || !cli.ok || !node.ok) confined = setupCheck('confined', 'fail', notRun);
  else {
    const location: SrtLocation = { node: node.node, cli: cli.cli, version: SRT_VERSION, packageDir: path.dirname(path.dirname(cli.cli)), nodeFrom: node.nodeFrom, cliFrom: cli.cliFrom };
    const failure = preflight(location, deps);
    confined = failure
      ? setupCheck('confined', 'fail', `${failure}: shell commands are refused`, preflightFix(platform) || undefined)
      : setupCheck('confined', 'ok', 'a confined `true` ran: shell commands are sandboxed here');
  }
  const ran = confined.status === 'ok';
  if (platform === 'linux') {
    // Said only when the confined `true` failed: a restriction bubblewrap
    // is exempt from (an AppArmor profile for bwrap) is not a problem.
    const restriction = namespaceRestriction();
    if (restriction && !ran) {
      checks.push(setupCheck('namespaces', 'fail', `unprivileged user namespaces are restricted here (${restriction.file} is ${restriction.value})`, restriction.fix));
    }
    if (inContainer('/', env)) {
      checks.push(
        ran
          ? setupCheck('container', 'ok', 'inside a container, and bubblewrap works here')
          : setupCheck('container', 'fail', 'inside a container, where bubblewrap needs namespaces the container may not allow', FIX_CONTAINER),
      );
    }
  }
  if (platform === 'darwin' && tmpdirTooLong()) {
    const probe = srtSocketProbe();
    checks.push(setupCheck('tmpdir', 'fail', `srt's socket path here is ${Buffer.byteLength(probe)} bytes (${probe}), over macOS's 104`, FIX_TMPDIR));
  }
  // The install inside the folder the commands run in (S-316): said, since
  // it is why a confined `npm install` into the CLI's own node_modules fails.
  const install = installInsideCheck([process.cwd()], env);
  if (install) checks.push(install);
  checks.push(confined);
  return checks;
}

/** `-c true` under a minimal policy. Empty when it ran; else why not. */
function preflight(location: SrtLocation, deps: Record<string, string>): string {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'webagents-srt-preflight-'));
  try {
    const settings = {
      network: { allowedDomains: [], deniedDomains: [], strictAllowlist: true },
      filesystem: { denyRead: [], allowWrite: [fs.realpathSync(scratch)], denyWrite: [] },
      ...linuxPins(deps),
    };
    const dir = fs.mkdtempSync(path.join(scratch, 'webagents-srt-'));
    fs.chmodSync(dir, 0o700);
    const file = writeJson0600(path.join(dir, 'settings.json'), settings);
    const env = srtEnvironment(scrubEnvironment(process.env).kept, scratch);
    const result = spawnSync(location.node, [location.cli, '--settings', file, '-c', 'true'], {
      cwd: scratch,
      env: env as NodeJS.ProcessEnv,
      encoding: 'utf8',
      timeout: PREFLIGHT_TIMEOUT_MS,
    });
    if (result.error) return `srt cannot start a sandbox here: ${result.error.message}`;
    if (result.status !== 0) {
      const lines = `${result.stderr || result.stdout || ''}`.split('\n').filter((line) => line.trim());
      return `srt cannot start a sandbox here: ${lines[0]?.trim() || `exit ${result.status}`}`;
    }
    return '';
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// The settings file
// ---------------------------------------------------------------------------

/**
 * The srt settings for a policy. Pinned by the fixture's `settings_cases`.
 * `network.local` is srt's `allowLocalBinding` (bind and listen on any local
 * port, connect to loopback directly; on Linux the command has its own
 * network namespace, so a port it opens is reachable only inside it), and
 * `network.sockets` is `allowUnixSockets`, macOS only. Neither key appears
 * unless asked for, and `allowAllUnixSockets` never does.
 */
export function buildSettings(policy: SandboxPolicy, deps?: Record<string, string>): Record<string, unknown> {
  const linux = process.platform === 'linux';
  const pins = deps ?? (linux ? linuxDeps().found : {});
  let denied = denyWrites(policy);
  // The SDK's own install, when it lies inside a write root (S-316).
  for (const entry of installWriteDenies(policy.writeRoots, sdkInstallPaths())) if (!denied.includes(entry)) denied.push(entry);
  // A `denyWrite` for a missing path makes bubblewrap create a placeholder on
  // the host while the command runs; srt's own mandatory set covers
  // `.git/hooks` and `.git/config` when they exist.
  if (linux) denied = denied.filter((p) => fs.existsSync(p));
  const filesystem: Record<string, unknown> = {
    denyRead: denyReads(policy),
    allowWrite: [...policy.writeRoots],
    denyWrite: denied,
  };
  if (policy.scopedReads) filesystem.allowRead = allowReads(policy);
  const network: Record<string, unknown> = {
    allowedDomains: [...policy.networkDomains],
    deniedDomains: [],
    // Never consult an ask callback: the CLI has none, and saying so keeps
    // the file honest if it is ever read by the library.
    strictAllowlist: true,
  };
  if (policy.localNetwork) network.allowLocalBinding = true;
  if (policy.unixSockets.length && process.platform === 'darwin') network.allowUnixSockets = [...policy.unixSockets];
  return {
    network,
    filesystem,
    ...linuxPins(pins),
  };
}

function writeJson0600(file: string, data: unknown): string {
  const fd = fs.openSync(file, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL, 0o600);
  try {
    fs.writeSync(fd, JSON.stringify(data, null, 2));
  } finally {
    fs.closeSync(fd);
  }
  return file;
}

function inside(p: string, root: string): boolean {
  const trimmed = root.replace(/\/+$/, '') || '/';
  return trimmed === '/' || p === trimmed || p.startsWith(trimmed + '/');
}

/** A folder for the settings file that no write root contains, or a refusal. */
export function settingsBase(policy: SandboxPolicy): string {
  const candidates = [realOrSelf(os.tmpdir()), realOrSelf(path.join(os.homedir(), '.cache', 'webagents-srt'))];
  for (const base of candidates) {
    if (!policy.writeRoots.some((root) => inside(base, root))) {
      fs.mkdirSync(base, { recursive: true, mode: 0o700 });
      return base;
    }
  }
  throw new SandboxUnavailable(`no place for the settings file outside the writable folders (${policy.writeRoots.join(', ')}). ${UNAVAILABLE_TAIL}`);
}

function realOrSelf(p: string): string {
  try {
    return fs.realpathSync(p);
  } catch {
    return path.resolve(p);
  }
}

/** The settings file for one command: 0600, in a fresh 0700 folder outside every write root. */
export function writeSettings(policy: SandboxPolicy, settings?: Record<string, unknown>): string {
  const dir = fs.mkdtempSync(path.join(settingsBase(policy), 'webagents-srt-'));
  fs.chmodSync(dir, 0o700);
  return writeJson0600(path.join(dir, 'settings.json'), settings ?? buildSettings(policy));
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

function killGroup(pid: number): void {
  try {
    process.kill(-pid, 'SIGTERM');
  } catch {
    return;
  }
  setTimeout(() => {
    try {
      process.kill(-pid, 'SIGKILL');
    } catch {
      // gone already
    }
  }, KILL_GRACE_MS).unref();
}

function collect(
  child: ReturnType<typeof spawn>,
  timeoutMs: number | undefined,
  maxBuffer: number,
  signal?: AbortSignal,
): Promise<SandboxResult> {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let interrupted = false;
    let settled = false;
    child.stdout?.on('data', (chunk: Buffer) => {
      if (stdout.length < maxBuffer) stdout += chunk.toString();
    });
    child.stderr?.on('data', (chunk: Buffer) => {
      if (stderr.length < maxBuffer) stderr += chunk.toString();
    });
    const timer = timeoutMs
      ? setTimeout(() => {
          timedOut = true;
          if (child.pid) killGroup(child.pid);
        }, timeoutMs)
      : undefined;
    // The interrupt kills the group as the timeout does (file comment, 5).
    const onAbort = () => {
      interrupted = true;
      if (child.pid) killGroup(child.pid);
    };
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }
    const done = (exitCode: number) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      resolve({ stdout: stdout.slice(0, maxBuffer), stderr: stderr.slice(0, maxBuffer), exitCode, timedOut, ...(interrupted ? { interrupted } : {}) });
    };
    child.on('error', (err) => {
      stderr += `\n${err.message}`;
      done(1);
    });
    child.on('close', (code, signal) => done(code ?? (signal ? 1 : 0)));
  });
}

export interface SandboxRunOptions {
  /** Seconds; the whole process group is killed when it passes. */
  timeout?: number;
  /** The environment to scrub and hand on; `process.env` when unset. */
  env?: Env;
  /** Output cap per stream (default 1 MiB). */
  maxBuffer?: number;
  /**
   * Learn which hosts the proxy refused, from srt's own log: srt runs with
   * `--debug`, the command's stderr is merged into its stdout (so the
   * result's `stdout` is the command's whole output and `stderr` is empty),
   * and `refusedHosts` is filled. The interactive chat sets it, to ask the
   * owner about a host by name; nothing else needs it.
   */
  captureRefusals?: boolean;
  /**
   * The interrupt: when it aborts (the chat's Esc or Ctrl+C reach a tool as
   * `context.signal`), the command's whole process group is killed and the
   * result says `interrupted`. An already aborted signal runs nothing.
   */
  signal?: AbortSignal;
}

/**
 * Why a command from a caller other than the owner cannot run: undefined
 * when `policy` confines and the engine is here, else the reason (fixture
 * `refusals.not_owner_reasons`). The owner may run unconfined; nobody else
 * may, which is defense in depth behind the owner-only shell tool (S-248).
 */
export function sandboxRequiredReason(policy: SandboxPolicy | null | undefined): string | undefined {
  if (!policy) return 'this agent declares none';
  if (!policy.confined) return "this agent's sandbox is unrestricted, which is no sandbox";
  const status = backendStatus();
  if (!status.available) return `the sandbox is unavailable: ${status.reason}`;
  return undefined;
}

/**
 * Run `command` under the policy, or refuse.
 *
 * A confined policy runs under srt. An UNCONFINED one (`preset: unrestricted`)
 * runs with the agent's own permissions, because that is what the
 * declaration says; callers that must not do that check `policy.confined`
 * or `sandboxRequiredReason` first, as `ShellSkill` does for callers other
 * than the owner. Throws `SandboxUnavailable` when srt cannot enforce the
 * policy here; a timeout is reported in the result (`timedOut`), never as
 * exit 0.
 */
export async function runSandboxed(command: string, policy: SandboxPolicy, options: SandboxRunOptions = {}): Promise<SandboxResult> {
  const maxBuffer = options.maxBuffer ?? 1024 * 1024;
  const timeoutMs = options.timeout ? options.timeout * 1000 : undefined;
  if (options.signal?.aborted) return { stdout: '', stderr: '', exitCode: 1, timedOut: false, interrupted: true };
  const given = options.env ?? process.env;
  // A listed `env` name the process lacks is read from the `.env` files the
  // CLI loads; the files themselves stay unreadable to the command.
  const { kept } = scrubEnvironment({ ...envFromDotenv(policy.envPassthrough, policy.cwd, given), ...given }, policy.envPassthrough);

  if (!policy.confined) {
    const env = { ...kept, ...(policy.scratch ? { TMPDIR: policy.scratch } : {}) };
    const child = spawn('/bin/sh', ['-c', command], { cwd: policy.cwd, env: env as NodeJS.ProcessEnv, stdio: ['ignore', 'pipe', 'pipe'], detached: true });
    return collect(child, timeoutMs, maxBuffer, options.signal);
  }

  const status = backendStatus();
  if (!status.available || !status.node || !status.path) {
    throw new SandboxUnavailable(unavailableMessage(status));
  }
  const userPath = kept.PATH || '/usr/bin:/bin';
  const env = srtEnvironment(kept, policy.scratch);
  const deps = process.platform === 'linux' ? linuxDeps().found : {};
  const settingsFile = writeSettings(policy, buildSettings(policy, deps));
  const capture = Boolean(options.captureRefusals);
  try {
    const argv = [status.path, ...(capture ? ['--debug'] : []), '--settings', settingsFile, '-c', wrappedCommand(command, userPath, policy.networkDomains, capture)];
    const child = spawn(status.node, argv, {
      cwd: policy.cwd,
      env: env as NodeJS.ProcessEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
      // Its own process group, so a timeout can kill everything the command
      // started, not just srt.
      detached: true,
    });
    const result = await collect(child, timeoutMs, maxBuffer, options.signal);
    if (!capture) return result;
    // With the command's stderr merged into stdout, srt's stderr is srt's
    // own: the refused hosts are read from it, and none of it reaches the
    // caller (the debug log names the settings and the wrapped command).
    return { ...result, stderr: '', refusedHosts: refusedHostsFromSrtLog(result.stderr) };
  } finally {
    fs.rmSync(path.dirname(settingsFile), { recursive: true, force: true });
  }
}
