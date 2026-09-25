/**
 * The CLI's configuration and credential paths.
 *
 * THE SAME ON-DISK CONTRACT AS THE PYTHON SDK (`python/webagents/cli/
 * config_store.py` and `credentials.py`). Before this (2026-09-23) the two
 * CLIs were wrong in opposite directions:
 *
 *   - They COLLIDED on config. Both wrote `~/.webagents/config.json` with
 *     incompatible schemas: Python wrote nested objects, this one wrote flat
 *     strings, so `webagents config set daemon 8080` here replaced Python's
 *     `{"daemon": {"port": 8765}}` with the string `"8080"`.
 *   - They DIVERGED on credentials. Python wrote `credentials.json`, this one
 *     wrote `auth.json`, so logging in with one SDK left the other logged out.
 *
 * PRECEDENCE, highest first, matching Python exactly:
 *
 *   1. an explicit flag            (--token, --profile)
 *   2. the process environment     (WEBAGENTS_TOKEN, ${VAR} in config)
 *   3. ./.env
 *   4. ~/.webagents/.env
 *   5. ./.webagents/config.json    (project)
 *   6. ~/.webagents/config.json    (global)
 *   7. the defaults below
 *
 * THE TOKEN IS NOT IN ANY OF THESE FILES. It goes to the OS keystore via the
 * `SecretStore` this SDK already ships (`src/skills/secrets/store.ts`), with an
 * owner-only 0600 file as the documented fallback. Writing it at 0644 next to
 * the config is what S-211 was about.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

/** Built-in defaults, the lowest layer, and the set of keys this CLI knows. */
export const DEFAULTS: Record<string, unknown> = {
  'platform.url': 'https://robutler.ai',
  'daemon.port': 8765,
  'daemon.host': '127.0.0.1',
  model: null,
  profile: null,
  'telemetry.enabled': false,
  // The platform agent this directory publishes to. Same keys and file as the
  // Python CLI's `deploy`/`link`, so a directory linked by either CLI updates
  // the same agent from both (2026-09-24).
  'link.agentId': null,
  'link.agentName': null,
};

/** `${VAR}` or `${VAR:-fallback}`. */
const VAR_RE = /\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}/g;

export function profileName(explicit?: string): string | undefined {
  return explicit || process.env.WEBAGENTS_PROFILE || undefined;
}

/**
 * A `webagents` command as the user should type it here: with
 * `--profile <name>` while a profile is active (2026-09-25). Every hint that
 * names a command goes through this. They said `webagents login` whatever the
 * profile, so under `webagents --profile local` following the hint signed the
 * DEFAULT profile in and left the local one signed out. The twin of Python's
 * `cli_command`; a name that is not one shell word is quoted as `shlex.quote`
 * quotes it.
 */
export function cliCommand(rest = '', profile?: string): string {
  const active = profileName(profile);
  let base = 'webagents';
  if (active) {
    const word = /^[A-Za-z0-9._-]+$/.test(active) ? active : `'${active.replace(/'/g, `'"'"'`)}'`;
    base = `webagents --profile ${word}`;
  }
  return rest ? `${base} ${rest}` : base;
}

/**
 * A keystore namespace that carries the profile (S-219, 2026-09-23).
 *
 * The OS keystore is keyed by namespace ALONE (`webagents:<namespace>`), so
 * scoping only the fallback file's directory isolated two profiles on a machine
 * with no keystore and let them share one entry on a machine with one: a
 * `--profile test` login overwrote the default profile's token. The Python
 * side was fixed first; this side had the same defect and is kept in step by
 * `python/webagents/cli/config_store.py:scoped_namespace`. The default profile
 * keeps the bare name so an existing token stays readable.
 */
export function scopedNamespace(base: string, profile?: string): string {
  const resolved = profileName(profile);
  return resolved ? `${base}-${resolved}` : base;
}

/** `~/.webagents`, or `~/.webagents-<profile>`. Resolves the profile itself. */
export function globalDir(profile?: string): string {
  const resolved = profileName(profile);
  return path.join(os.homedir(), resolved ? `.webagents-${resolved}` : '.webagents');
}

/**
 * `./.webagents`. NOT created as a side effect of reading it: the Python CLI
 * used to scaffold eight directories in whatever folder you were standing in,
 * reached from `is_authenticated()`, so `auth whoami` littered any repo.
 */
export function projectDir(cwd: string = process.cwd()): string {
  return path.join(cwd, '.webagents');
}

function readJson(file: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf-8'));
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    // A corrupt config must not make every command unrunnable. `config
    // validate` is where a user asks about correctness.
    return {};
  }
}

/** A deliberately small `.env` reader: `KEY=value`, `#` comments, quotes. */
function parseDotenv(file: string): Record<string, string> {
  const out: Record<string, string> = {};
  let text: string;
  try {
    text = fs.readFileSync(file, 'utf-8');
  } catch {
    return out;
  }
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
    const eq = trimmed.indexOf('=');
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (value.length >= 2 && value[0] === value[value.length - 1] && (value[0] === '"' || value[0] === "'")) {
      value = value.slice(1, -1);
    }
    if (key) out[key] = value;
  }
  return out;
}

/**
 * The resolved environment: process env wins, then `./.env`, then
 * `~/.webagents/.env`. Files never override something already exported; a
 * developer who exports a variable in their shell means it.
 */
export function envChain(profile?: string, cwd: string = process.cwd()): Record<string, string> {
  return {
    ...parseDotenv(path.join(globalDir(profile), '.env')),
    ...parseDotenv(path.join(cwd, '.env')),
    ...(process.env as Record<string, string>),
  };
}

/**
 * Resolve `${VAR}` and `${VAR:-fallback}` in a string.
 *
 * An unresolvable `${VAR}` with no fallback is left AS WRITTEN rather than
 * becoming an empty string, so the failure is visible in the value.
 */
export function expand(value: unknown, env: Record<string, string>): unknown {
  if (typeof value !== 'string') return value;
  return value.replace(VAR_RE, (whole, name: string, fallback?: string) => {
    if (name in env) return env[name];
    if (fallback !== undefined) return fallback;
    return whole;
  });
}

export type LayerName = 'flag' | 'project' | 'global' | 'default';

export class ConfigStore {
  readonly profile?: string;
  readonly globalPath: string;
  readonly projectPath: string;
  private readonly cwd: string;
  private readonly overrides: Record<string, unknown>;

  constructor(opts: { profile?: string; cwd?: string; overrides?: Record<string, unknown> } = {}) {
    this.profile = profileName(opts.profile);
    this.cwd = opts.cwd ?? process.cwd();
    this.overrides = { ...(opts.overrides ?? {}) };
    this.globalPath = path.join(globalDir(this.profile), 'config.json');
    this.projectPath = path.join(projectDir(this.cwd), 'config.json');
  }

  get env(): Record<string, string> {
    return envChain(this.profile, this.cwd);
  }

  /** Every layer, highest precedence first. */
  layers(): [LayerName, Record<string, unknown>][] {
    return [
      ['flag', { ...this.overrides }],
      ['project', readJson(this.projectPath)],
      ['global', readJson(this.globalPath)],
      ['default', { ...DEFAULTS }],
    ];
  }

  get(key: string, fallback?: unknown): unknown {
    const env = this.env;
    for (const [, layer] of this.layers()) {
      if (key in layer) {
        const value = layer[key];
        return value === null || value === undefined ? fallback : expand(value, env);
      }
    }
    return fallback;
  }

  /** Which layer supplied `key`. What `config get --why` prints. */
  sourceOf(key: string): LayerName | undefined {
    for (const [name, layer] of this.layers()) {
      if (key in layer) return name;
    }
    return undefined;
  }

  effective(): Record<string, unknown> {
    const keys = new Set(Object.keys(DEFAULTS));
    for (const [, layer] of this.layers()) Object.keys(layer).forEach((k) => keys.add(k));
    const out: Record<string, unknown> = {};
    for (const key of [...keys].sort()) out[key] = this.get(key);
    return out;
  }

  set(key: string, value: unknown, scope: 'global' | 'project' = 'global'): string {
    const file = scope === 'project' ? this.projectPath : this.globalPath;
    const data = readJson(file);
    data[key] = value;
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, `${JSON.stringify(data, Object.keys(data).sort(), 2)}\n`);
    return file;
  }

  unset(key: string, scope: 'global' | 'project' = 'global'): boolean {
    const file = scope === 'project' ? this.projectPath : this.globalPath;
    const data = readJson(file);
    if (!(key in data)) return false;
    delete data[key];
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, `${JSON.stringify(data, Object.keys(data).sort(), 2)}\n`);
    return true;
  }

  /**
   * Problems with the config as written. Empty means clean.
   *
   * Reports UNKNOWN KEYS, because accepting a typo silently and then ignoring
   * it is the worst outcome: nothing ever says so.
   */
  validate(): string[] {
    const problems: string[] = [];
    const env = this.env;
    for (const file of [this.projectPath, this.globalPath]) {
      if (!fs.existsSync(file)) continue;
      let raw: string;
      try {
        raw = fs.readFileSync(file, 'utf-8');
      } catch (e) {
        problems.push(`${file}: cannot read (${(e as Error).message})`);
        continue;
      }
      let data: unknown;
      try {
        data = JSON.parse(raw);
      } catch (e) {
        problems.push(`${file}: not valid JSON (${(e as Error).message})`);
        continue;
      }
      if (!data || typeof data !== 'object' || Array.isArray(data)) {
        problems.push(`${file}: top level must be an object`);
        continue;
      }
      for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
        if (!(key in DEFAULTS)) {
          problems.push(`${file}: unknown key "${key}"`);
        }
        if (typeof value === 'string') {
          for (const match of value.matchAll(VAR_RE)) {
            if (!(match[1] in env) && match[2] === undefined) {
              problems.push(`${file}: ${key} references \${${match[1]}}, which is not set`);
            }
          }
        }
      }
    }
    return problems;
  }
}

/**
 * Points every platform command at another portal. The same variable the
 * Python CLI's `login` has always read, so one export means the same thing to
 * both SDKs and to every command in each.
 */
export const PLATFORM_URL_ENV_VAR = 'ROBUTLER_API_URL';

/**
 * The portal the platform commands talk to, and which layer said so.
 *
 * ONE ANSWER FOR LOGIN AND EVERYTHING AFTER IT (2026-09-24). `login` took
 * `--url` with a hardcoded `https://robutler.ai` default and never read
 * `platform.url`, while `platformAuth()` read `platform.url` and could never
 * reach the URL `login` saved: `ConfigStore.get` returns the DEFAULT when the
 * key is unset, so the `auth.json` fallback behind it was dead code. A login
 * to a local cluster therefore stored that cluster's token and sent it to
 * production on the next `sync`. Mirrors
 * `python/webagents/cli/config_store.py:resolve_platform_url`, precedence
 * environment over config.
 */
export function resolvePlatformUrl(profile?: string, cwd?: string): [string, string] {
  const fromEnv = (process.env[PLATFORM_URL_ENV_VAR] ?? '').trim();
  if (fromEnv) return [fromEnv.replace(/\/+$/, ''), PLATFORM_URL_ENV_VAR];
  const store = new ConfigStore({ profile, cwd });
  const value = store.get('platform.url') ?? DEFAULTS['platform.url'];
  return [String(value).replace(/\/+$/, ''), store.sourceOf('platform.url') ?? 'default'];
}
