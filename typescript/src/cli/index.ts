#!/usr/bin/env node
/**
 * WebAgents CLI
 *
 * Full command-line interface for WebAgents (17 command groups).
 */

import { Command } from 'commander';
import {
  ConfigStore,
  DEFAULTS,
  PLATFORM_URL_ENV_VAR,
  cliCommand,
  resolvePlatformUrl,
} from './config-store.js';
import { InteractiveREPL } from './app';
import { promptSecret } from './prompt';
import { streamJsonEvent } from './render';
import { firstOperandIndex, suggestSimilar } from './suggest';
import { setRequestErrorDetail } from '../skills/llm/request';
import { WebAgentsDaemon } from '../daemon/server';
import { setAgentTrace } from '../core/trace';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';

/**
 * The real package version, read from package.json at startup.
 *
 * This was a hardcoded `'0.1.0'` literal, and it was wrong in three places at
 * once because three things interpolate it (2026-09-23):
 *
 *   1. `--version` reported 0.1.0 while the package was 0.3.6.
 *   2. `webagents update` compares this against the npm registry, so every user
 *      on the newest build was told, permanently, to upgrade.
 *   3. `init` writes `webagents: ^${version}` into the scaffolded package.json.
 *      No 0.1.x was ever published, so `npm install` in a fresh project failed
 *      with ETARGET. The generated project's own `version` field is a separate
 *      literal and 0.1.0 is correct there; do not "fix" that one.
 *
 * Resolved relative to this module rather than cwd: `dist/cli/index.js` and
 * `src/cli/index.ts` are both two levels below the package root, so the same
 * path works built and under tsx. Matches `src/agents/index.ts`, which resolves
 * its embedded markdown the same way.
 */
const version: string = (() => {
  try {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const pkg = JSON.parse(fs.readFileSync(path.join(here, '..', '..', 'package.json'), 'utf-8'));
    return typeof pkg.version === 'string' ? pkg.version : 'unknown';
  } catch {
    // A missing or unreadable package.json means someone is running the CLI
    // from a tree we do not understand. Say so rather than inventing a number
    // that other commands will compare against the registry.
    return 'unknown';
  }
})();
// Written by older builds only (a token before Phase 3, a URL after it that
// nothing read). `logout` still sweeps it; nothing writes it any more.
const LEGACY_AUTH_FILE = path.join(os.homedir(), '.webagents', 'auth.json');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/**
 * The agent loop's trace stays out of the CLI's output (2026-09-24).
 *
 * The loop writes a trace line per iteration and per streamed chunk, which the
 * portal wants in its pod logs and a person at a terminal does not. It landed
 * on STDOUT, so `-p --output-format json` could not be piped into anything.
 * `WEBAGENTS_DEBUG` brings it back, on stderr. See `core/trace.ts`.
 */
const DEBUG = Boolean(process.env.WEBAGENTS_DEBUG);
setAgentTrace(DEBUG ? { enabled: true, sink: (line) => console.error(line) } : { enabled: false });

const program = new Command();

program
  .name('webagents')
  .description('Build and run AI agents')
  .version(version)
  // A GLOBAL `--json`, matching the Python CLI (2026-09-23). For a CLI whose
  // users are themselves building agents, the machine-readable contract is the
  // product, not a convenience, and it has to be the same shape on every
  // command or a script cannot rely on it. See `output.ts`.
  .option('--json', 'Machine-readable output: one JSON document on stdout, diagnostics on stderr')
  // GLOBAL FLAGS GO BEFORE THE COMMAND, as in the Python CLI (click). By
  // default commander reads a program option anywhere on the line, so the
  // global `--token` below took `login --token <key>`'s key, and `login` sat
  // waiting for a pasted one (2026-09-24).
  .enablePositionalOptions()
  // The Python CLI's global flags, so a script works with either CLI.
  .option('--profile <name>', 'Use a separate set of settings, keys and sign-in')
  .option('--token <token>', 'Use this platform token for this run, instead of the stored sign-in')
  .hook('preAction', async (command) => {
    const opts = command.opts() as { profile?: string; token?: string };
    // The profile is a name, not a secret, so the environment is where every
    // later lookup already reads it. The token is not put there (credentials.ts).
    if (opts.profile) process.env.WEBAGENTS_PROFILE = opts.profile;
    if (opts.token) (await import('./credentials.js')).setFlagToken(opts.token);
  });

// ============================================================================
// 1. chat (default)
// ============================================================================

const OUTPUT_FORMATS = ['text', 'json', 'stream-json'] as const;
type OutputFormat = (typeof OUTPUT_FORMATS)[number];

/**
 * `chat` and its alias `connect`.
 *
 * Three things here were flags that lied, which `serve` already has a comment
 * about rejecting (see `--multi` below): a flag that lies is worse than a
 * missing one. Fixed 2026-09-23.
 *
 *   - `--output-format stream-json` was accepted and silently behaved as
 *     `text`, because only `'json'` was ever tested for. It now streams real
 *     JSONL, one object per delta then a terminal `done`.
 *   - `--no-streaming` was never read at all. It is now honoured.
 *   - `--model` defaulted to the literal string `'default'`, which is truthy,
 *     so it was passed through as the API `model` field. Omitted when unset so
 *     the REPL's own default applies.
 */
async function chatAction(options: {
  model?: string;
  prompt?: string;
  outputFormat?: string;
  agent?: string;
  streaming?: boolean;
}) {
  const format = (options.outputFormat ?? 'text') as OutputFormat;
  if (!OUTPUT_FORMATS.includes(format)) {
    console.error(
      `Unknown --output-format '${options.outputFormat}'. Expected one of: ${OUTPUT_FORMATS.join(', ')}.`,
    );
    process.exit(2);
  }

  // The person at the terminal reads a failed model request's error, so it
  // names the server and the reason (request.ts; a server never does, S-228).
  setRequestErrorDetail(true);

  // Keys are omitted rather than set to undefined: REPLConfig is applied with a
  // spread over its defaults, and an explicit `undefined` overrides a default.
  const config: { model?: string; agentFile?: string | null; streaming: boolean; version: string } = {
    streaming: options.streaming !== false,
    version,
  };
  if (options.model) config.model = options.model;
  if (options.agent) {
    // By name, as `/agent` does, or refused (`agent-files.ts`).
    const { agentFileFor, AgentNotFound } = await import('./agent-files.js');
    try {
      config.agentFile = agentFileFor(process.cwd(), options.agent);
    } catch (error) {
      if (!(error instanceof AgentNotFound)) throw error;
      console.error(error.message);
      process.exit(1);
    }
  }

  if (options.prompt) {
    const repl = new InteractiveREPL(config);
    await orAgentFileError(() => repl.initialize());

    // Refused up front, like the Python CLI's `run` (2026-09-24). This used to
    // start, call the model, and die with a Node stack trace. Without a key
    // but signed in, `initialize()` has already chosen Robutler's models, so
    // this fires only when there is nothing at all to run the agent on.
    if (repl.modelProblem) {
      console.error(repl.modelProblem);
      process.exit(1);
    }

    try {
      if (format === 'stream-json') {
        // Every event of the turn, one JSON object per line (see
        // streamJsonEvent); a turn that ends in an `error` line exits 1.
        let failed = false;
        for await (const chunk of repl.streamTurn(options.prompt)) {
          const event = streamJsonEvent(chunk);
          if (event && chunk.type === 'error') {
            // The headline and hint the chat shows (`failures.ts`), not the
            // platform's protocol text.
            // Robutler's own refusals get their stable code; anything else
            // keeps the error's own.
            const { headline, hint, code } = repl.explainFailure(chunk.error?.message ?? '');
            event.error = {
              ...(event.error as Record<string, unknown>),
              message: headline,
              ...(code ? { code } : {}),
              ...(hint ? { hint } : {}),
            };
          }
          if (event) process.stdout.write(`${JSON.stringify(event)}\n`);
          if (chunk.type === 'error') failed = true;
        }
        if (failed) process.exit(1);
      } else {
        const response = await repl.sendMessage(options.prompt);
        console.log(format === 'json' ? JSON.stringify(response, null, 2) : response.content);
      }
    } catch (error) {
      // The chat's headline and hint (`failures.ts`), and a non-zero exit a
      // script can see. The stack only helps someone debugging the SDK
      // itself, so it waits for WEBAGENTS_DEBUG.
      const { headline, hint } = repl.explainFailure((error as Error).message);
      console.error(`Error: ${headline}`);
      if (hint) console.error(hint);
      if (DEBUG) console.error((error as Error).stack);
      process.exit(1);
    }
    process.exit(0);
  }

  if (format !== 'text') {
    // Interactive output is a terminal transcript; there is no sane JSON of it.
    // Refuse rather than accept the flag and ignore it.
    console.error(`--output-format ${format} applies to -p/--prompt only.`);
    process.exit(2);
  }
  const repl = new InteractiveREPL(config);
  await orAgentFileError(() => repl.run());
}

/**
 * An agent file that cannot be used as written (`AgentFileError`): its
 * sentence alone and exit 1, as the Python CLI answers `AgentFormatError`.
 * Anything else still throws.
 */
async function orAgentFileError<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (error) {
    if ((error as Error | undefined)?.name === 'AgentFileError') {
      console.error((error as Error).message);
      process.exit(1);
    }
    throw error;
  }
}

program
  .command('chat', { isDefault: true })
  .description('Start interactive chat session')
  // No default for --model: the literal string 'default' used to be sent as
  // the API model field, because it is truthy.
  .option('-m, --model <model>', 'Model to use, as provider/model')
  .option('-a, --agent <agent>', 'Agent name')
  .option('-p, --prompt <prompt>', 'Non-interactive prompt, then exit')
  .option('--output-format <format>', 'With -p: text, json, stream-json', 'text')
  .option('--no-streaming', 'Disable streaming')
  .action(chatAction);

// ============================================================================
// 2. connect
// ============================================================================

program
  .command('connect')
  .description('Start interactive session (alias for chat)')
  // Kept in step with `chat` deliberately: they share one action, so an option
  // declared on only one of them is a flag that silently does nothing there.
  .option('-m, --model <model>', 'Model to use, as provider/model')
  .option('-a, --agent <agent>', 'Agent name')
  .option('-p, --prompt <prompt>', 'Non-interactive prompt, then exit')
  .option('--output-format <format>', 'With -p: text, json, stream-json', 'text')
  .option('--no-streaming', 'Disable streaming')
  .action(chatAction);

// ============================================================================
// 3. serve
// ============================================================================

program
  .command('serve')
  .description('Serve an agent on HTTP')
  .argument('[path]', 'Path to agent config file', '.')
  .option('-p, --port <port>', 'Port', '3000')
  // No default and no `-h` (2026-09-24, S-226). It was `-h, --host` with
  // `0.0.0.0`: `-h` could not show help, and every `webagents serve` listened
  // on every interface. Unset, `serve()` binds loopback for an agent with no
  // public URL and no AuthSkill, and says so.
  .option('--host <host>', 'Interface to listen on (0.0.0.0 for every interface)')
  // NO `--multi`. It was declared here and read nowhere: `serveAction` serves
  // exactly one agent, so `webagents serve --multi` silently did the
  // single-agent thing while promising to "load all agents in directory".
  // Multi-agent hosting is `WebAgentsServer`, which is a different entry point
  // with a different shape (named mounts, per-agent scopes and rate limits);
  // reviving the flag means building that directory loader, not passing a
  // boolean through. A flag that lies is worse than a missing one.
  // The body lives in ./serve-action so it can be unit-tested: this module
  // calls `program.parse()` at the bottom, so importing IT to drive a command
  // runs the CLI against the test runner's argv.
  .action(async (agentPath, options) => {
    const { serveAction } = await import('./serve-action.js');
    await orAgentFileError(() => serveAction(agentPath, options));
  });

// ============================================================================
// 4. daemon
// ============================================================================

program
  .command('daemon')
  .description('Start the WebAgents daemon')
  // THE CONFIGURED ADDRESS (2026-09-24). This defaulted to port 8080 on every
  // interface, while `list`, `status` and `logs` look for the daemon at
  // `daemon.host`/`daemon.port` (127.0.0.1:8765), so the daemon this command
  // started was one they could not find, and one the LAN could reach (the
  // S-214 leftover). Both now come from the same config keys.
  .option('-p, --port <port>', 'Port (default: daemon.port, 8765)')
  .option('--host <host>', 'Interface to listen on (default: daemon.host, 127.0.0.1)')
  .option('-w, --watch <dir>', 'Watch directory')
  .option('--no-cron', 'Disable cron')
  .action(async (options) => {
    const store = new ConfigStore();
    const daemon = new WebAgentsDaemon({
      port: parseInt(String(options.port ?? store.get('daemon.port', 8765)), 10),
      hostname: String(options.host ?? store.get('daemon.host', '127.0.0.1')),
      watchDir: options.watch,
      cron: options.cron,
    });
    await daemon.start();
    process.on('SIGINT', () => { daemon.stop(); process.exit(0); });
  });

// ============================================================================
// 5. login / logout
// ============================================================================

program
  .command('login')
  .description('Authenticate with the portal')
  // NO DEFAULT HERE (2026-09-24). It was a hardcoded `https://robutler.ai`,
  // so `login` ignored `platform.url` while every other command read it: with
  // `platform.url` pointed at a local cluster, `login` still validated against
  // production. The default is now the shared resolver's answer.
  .option('-u, --url <url>', 'Portal URL (default: platform.url, or ROBUTLER_API_URL)')
  .option('-t, --token <token>', 'API key to sign in with, instead of the browser')
  .action(async (options) => {
    const explicitUrl: string | undefined = options.url
      ? String(options.url).trim().replace(/\/+$/, '')
      : undefined;
    const [resolvedUrl, resolvedFrom] = resolvePlatformUrl();
    const portalUrl = explicitUrl || resolvedUrl;
    let token: string | undefined = options.token;
    let result: import('./auth-check.js').ValidationResult;

    if (!token && process.stdin.isTTY) {
      // THE BROWSER, as the Python CLI does (2026-09-24, `browser-login.ts`):
      // the portal asks the person to approve and redirects the token back
      // to this terminal. A pasted key stays available as `--token`.
      const { browserLogin } = await import('./browser-login.js');
      try {
        const signedIn = await browserLogin(portalUrl);
        result = { ok: true, accessToken: signedIn.token, username: signedIn.username };
        token = signedIn.token;
      } catch (err) {
        console.error(`Could not sign in: ${(err as Error).message}`);
        process.exit(1);
      }
    } else {
      if (!token) {
        // `/settings/api-keys` is not a page in the portal and never was; keys
        // are issued on the Developer tab of Settings.
        console.log(`\nCreate an API key at: ${portalUrl}/settings?tab=developer\n`);
        token = await promptSecret('Paste your API key: ');
      }
      token = token?.trim();
      if (!token) {
        console.error('No key entered.');
        process.exit(1);
      }

      // VALIDATE IT (2026-09-23). This used to store whatever was typed and print
      // "Authenticated successfully" without making a single network call, so a
      // typo'd key was reported as a successful login and failed later, somewhere
      // else, with an unrelated-looking error.
      const { validateToken } = await import('./auth-check.js');
      result = await validateToken(portalUrl, token);
      if (!result.ok) {
        console.error(`Could not authenticate: ${result.reason}`);
        process.exit(1);
      }
    }
    if (!result.ok) process.exit(1);

    // Store the EXCHANGED token, not the key the user pasted.
    //
    // `/api/auth/cli/token` answers with a platform JWT scoped `agents:own`
    // that expires in 7 days. Keeping that instead of the long-lived `rok_*`
    // key means a stolen credential file is bounded in both time and scope.
    // Falls back to the pasted key only if the portal returned no token, which
    // a 2xx without a parseable body can do.
    const { setToken } = await import('./credentials.js');
    const backend = await setToken(result.accessToken ?? token);

    // THE TOKEN AND THE PORTAL HAVE TO STAY TOGETHER. An explicit `--url` used
    // to be written to `auth.json`, which nothing read (see `platformAuth`),
    // so the next `sync` sent this portal's token to `platform.url` instead.
    // Recording it where every command looks is what makes `--url` mean "log
    // in to THAT portal" rather than "check the key against it once".
    // Profile-scoped: this is the active profile's global file.
    if (explicitUrl && explicitUrl !== resolvedUrl) {
      if (resolvedFrom === PLATFORM_URL_ENV_VAR) {
        // Config cannot outrank the environment, so writing it would change
        // nothing; say what will happen instead of pretending.
        console.log(
          `Note: ${PLATFORM_URL_ENV_VAR}=${resolvedUrl} is set and later commands will use it, ` +
            `not ${explicitUrl}. Unset it, or set it to ${explicitUrl}.`,
        );
      } else {
        const written = new ConfigStore().set('platform.url', explicitUrl);
        console.log(`platform.url set to ${explicitUrl} (${written}).`);
      }
    }

    console.log(
      `Authenticated${result.username ? ` as @${result.username}` : ''} on ${portalUrl}.`,
    );
    console.log(
      backend === 'keystore'
        ? 'Token stored in your OS keystore.'
        : 'No OS keystore here; token stored in an owner-only file (0600).',
    );
  });

program
  .command('logout')
  .description('Sign out of Robutler')
  .action(async () => {
    // Clear BOTH: the keystore entry and the metadata file. Removing only the
    // file would leave the token behind in the keystore.
    const { clearToken } = await import('./credentials.js');
    await clearToken();
    try { fs.unlinkSync(LEGACY_AUTH_FILE); } catch { /* already gone */ }
    const [portalUrl] = resolvePlatformUrl();
    console.log(`Signed out of ${portalUrl.replace(/^https?:\/\//, '')}.`);
  });

program
  .command('whoami')
  .description('Show who you are signed in as')
  .action(async () => {
    const { whoAmI } = await import('./account.js');
    const { jsonEnabled, emit, fail } = await import('./output.js');
    const result = await whoAmI();
    if (jsonEnabled(program)) {
      if (!result.ok) {
        fail(result.code, result.message, result.fix);
        return;
      }
      emit({ username: result.username, platform: result.platform });
      return;
    }
    if (!result.ok) {
      console.error(result.message);
      if (result.fix) console.error(result.fix);
      process.exit(1);
    }
    console.log(result.message);
  });

program
  .command('link')
  .description('Link this folder to one of your agents on Robutler')
  .argument('[name]', "The agent's name; defaults to the name in this folder's agent file")
  .option('--show', 'Say what this folder is linked to')
  .action(async (name: string | undefined, options: { show?: boolean }) => {
    const { linkFolder, showLink } = await import('./account.js');
    const ok = options.show ? showLink(process.cwd(), console.log) : await linkFolder(process.cwd(), name, console.log, console.error);
    if (!ok) process.exit(1);
  });

program
  .command('unlink')
  .description('Forget the agent this folder is linked to')
  .action(async () => {
    const { unlinkFolder } = await import('./account.js');
    unlinkFolder(process.cwd(), console.log);
  });

program
  .command('doctor')
  .description('Check this setup and say what to fix')
  .action(async () => {
    const { runChecks, reportLines } = await import('./doctor.js');
    const { jsonEnabled, emit } = await import('./output.js');
    const checks = await runChecks();
    if (jsonEnabled(program)) emit({ checks });
    else for (const line of reportLines(checks)) console.log(line);
    if (checks.some((c) => c.status === 'fail')) process.exit(1);
  });

// ============================================================================
// 9. models
// ============================================================================

program
  .command('models')
  .description('List LLM providers and which are configured here')
  .action(async () => {
    // Providers, not model ids. The previous version of this command printed a
    // hardcoded list of ids (gpt-4o, claude-3-5-sonnet, gemini-1.5-pro,
    // grok-2), every one of which was superseded while it sat in the source
    // with nothing in the build able to notice. Provider names are stable;
    // model ids are not, and the provider's own docs are the only current list.
    const { LLM_PROVIDERS, configuredProviders } = await import('../skills/llm/providers.js');
    // Stored keys count: the chat runs on them. Only the shell was looked at,
    // so a key kept with `secrets set` showed its provider as not ready.
    const { readStoredProviderKeys } = await import('./provider-keys.js');
    const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
    const { available } = configuredProviders({ ...stored, ...process.env });
    const ready = new Set(available.map((p) => p.id));

    console.log('\nLLM providers:\n');
    // The in-browser runtimes (webllm, transformers) run in the browser
    // build, not here: this CLI cannot load them, so it does not list them
    // (2026-09-25), and the Python CLI lists the same providers.
    for (const p of LLM_PROVIDERS.filter((provider) => provider.credential !== 'local')) {
      const mark = ready.has(p.id) ? 'ready' : '-';
      const needs =
        p.credential === 'local' ? 'no credential needed'
          : p.credential === 'platform' ? `${p.envVar} (or ${cliCommand('login')})`
            : `${p.envVar}`;
      console.log(`  ${mark.padEnd(6)} ${p.id.padEnd(14)} ${p.modelFormat.padEnd(24)} ${needs}`);
    }
    console.log(`\n  "ready" means this machine has its key: set in this shell, or stored with \`${cliCommand('secrets set')}\`.`);
    // No example model id here on purpose: naming one is how the previous
    // version of this command ended up advertising four superseded models.
    console.log('  Pass --model as provider/model, using an id from the provider.\n');
  });

// ============================================================================
// 10. skills
// ============================================================================

const skillsCmd = program.command('skills').description('Skills an agent file can name');

skillsCmd
  .command('list')
  .description('Skills an agent file can name')
  .action(async () => {
    // What `skills:` can actually name here: the names the resolver builds
    // (`skills/resolve.ts`). This printed 37 hard-coded names, most of which
    // no agent file could load.
    const { resolvableSkillNames } = await import('../skills/resolve.js');
    const names = resolvableSkillNames();
    console.log('\nSkills an agent file can name:\n');
    for (const name of names) console.log(`  ${name}`);
    console.log();
  });

// ============================================================================
// 11. templates
// ============================================================================

/**
 * The templates `init` can actually make (2026-09-24).
 *
 * `templates list` advertised six, among them rag-agent, multi-agent,
 * browser-agent and mcp-agent, and `init --template` accepted any name at all:
 * everything that was not `chatbot` got the same openai + filesystem + shell
 * scaffold with the template's name pasted into its description. One table
 * now feeds both commands, and `init` refuses a name it cannot make.
 */
const INIT_TEMPLATES: Record<string, { description: string; skills: string[] }> = {
  chatbot: { description: 'A chat agent: one model, no tools', skills: ['openai'] },
  'tool-agent': {
    description: 'Can read and write files and run shell commands',
    skills: ['openai', 'filesystem', 'shell'],
  },
};

const templatesCmd = program.command('templates').description('Agent templates');

templatesCmd
  .command('list')
  .description('List available templates')
  .action(() => {
    console.log('\nAvailable Templates:\n');
    for (const [name, t] of Object.entries(INIT_TEMPLATES)) {
      console.log(`  ${name.padEnd(20)} ${t.description}`);
    }
    console.log('\nUse: webagents init <name> --template <template>\n');
  });

// ============================================================================
// 12. config
// ============================================================================

const configCmd = program.command('config').description('Manage configuration');

/**
 * `config` reads and writes through `ConfigStore`, the SAME store every other
 * command reads (2026-09-23).
 *
 * These four commands used to go through a private `loadConfig()` /
 * `saveConfig()` pair hard-wired to `~/.webagents/config.json`, while
 * `ConfigStore` (which the daemon client, `status` and `list` read) resolves
 * `~/.webagents-<profile>/config.json` under `--profile`. The two coincide only
 * without a profile, which is why nothing caught it: under a profile,
 * `config set daemon.port 8821` succeeded, `config get` echoed 8821 back, and
 * `status` went on dialling 8765. Found by the end-to-end run.
 *
 * They also wrote any key as a raw string, so a typo was accepted silently and
 * `config set daemon.port 8821` stored the STRING "8821". The Python CLI has
 * refused unknown keys since Phase 3; the two SDKs share this file, so they
 * must also share its rules.
 */
function coerceConfigValue(key: string, raw: string): unknown {
  // Typed by the default the key already has. `null`-defaulted keys stay text.
  const current = DEFAULTS[key];
  if (typeof current === 'number') {
    const n = Number(raw);
    if (!Number.isFinite(n)) throw new Error(`${key} expects a number, got "${raw}"`);
    return n;
  }
  if (typeof current === 'boolean') {
    if (['true', '1', 'yes', 'on'].includes(raw.toLowerCase())) return true;
    if (['false', '0', 'no', 'off'].includes(raw.toLowerCase())) return false;
    throw new Error(`${key} expects true or false, got "${raw}"`);
  }
  return raw;
}

function refuseUnknownKey(key: string): void {
  if (!(key in DEFAULTS)) {
    console.error(`Unknown config key: ${key}`);
    console.error(`Known keys: ${Object.keys(DEFAULTS).sort().join(', ')}`);
    process.exit(1);
  }
}

configCmd
  .command('get [key]')
  .description('Get a configuration value (the effective one, after every layer)')
  .action((key) => {
    const store = new ConfigStore();
    if (key) {
      refuseUnknownKey(key);
      const value = store.get(key);
      console.log(value === undefined || value === null ? '(not set)' : String(value));
    } else {
      const merged: Record<string, unknown> = {};
      for (const k of Object.keys(DEFAULTS).sort()) merged[k] = store.get(k);
      console.log(JSON.stringify(merged, null, 2));
    }
  });

configCmd
  .command('set <key> <value>')
  .description('Set a configuration value')
  .option('--project', 'Write to ./.webagents/config.json instead of the global file')
  .action((key, value, options) => {
    refuseUnknownKey(key);
    let typed: unknown;
    try {
      typed = coerceConfigValue(key, value);
    } catch (error) {
      console.error((error as Error).message);
      process.exit(1);
    }
    const file = new ConfigStore().set(key, typed, options.project ? 'project' : 'global');
    console.log(`${key} = ${JSON.stringify(typed)} in ${file}`);
  });

configCmd
  .command('unset <key>')
  .description('Remove a configuration value')
  .option('--project', 'Remove from ./.webagents/config.json instead of the global file')
  .action((key, options) => {
    refuseUnknownKey(key);
    const removed = new ConfigStore().unset(key, options.project ? 'project' : 'global');
    console.log(removed ? `Removed ${key}` : `${key} was not set there`);
  });

configCmd
  .command('validate')
  .description('Check the config files for unknown keys and bad values')
  .action(() => {
    const problems = new ConfigStore().validate();
    if (problems.length === 0) {
      console.log('Config is valid.');
      return;
    }
    for (const problem of problems) console.error(problem);
    process.exit(1);
  });

configCmd
  .command('path')
  .description('Show config file paths')
  .action(() => {
    const store = new ConfigStore();
    console.log(`global:  ${store.globalPath}`);
    console.log(`project: ${store.projectPath}`);
  });

// ============================================================================
// 13. init
// ============================================================================

program
  .command('init')
  .description('Initialize a new agent project')
  .argument('[name]', 'Project name', 'my-agent')
  .option('-t, --template <template>', 'Template to use', 'chatbot')
  .action(async (name, options) => {
    // Checked before anything is created, so a typo leaves no directory behind.
    const template = INIT_TEMPLATES[options.template];
    if (!template) {
      console.error(
        `Unknown template '${options.template}'. Available: ${Object.keys(INIT_TEMPLATES).join(', ')}.`,
      );
      process.exit(1);
    }

    const dir = path.resolve(name);
    if (fs.existsSync(dir)) {
      console.error(`Directory ${name} already exists.`);
      process.exit(1);
    }

    fs.mkdirSync(dir, { recursive: true });

    // The model comes from the provider registry rather than a literal here.
    // This used to be a bare `'gpt-4o'`: stale, and missing the `provider/model`
    // prefix every other part of the SDK expects.
    const { findProvider } = await import('../skills/llm/providers.js');
    const openai = findProvider('openai');
    const skills = template.skills;

    // AGENT.md, THE FORMAT THE DOCS DESCRIBE (2026-09-24). This wrote
    // `agent.json` plus `instructions.md`, which nothing read (see
    // `agent-project.ts`), while the CLI quickstart shows `webagents init`
    // producing `AGENT.md`. It is the one file both SDKs parse, so the same
    // project also runs under the Python CLI. Only keys the Python loader's
    // strict schema accepts: the old `template` key would be REJECTED there.
    const agentMd = [
      '---',
      `name: ${name}`,
      `description: A ${options.template} agent`,
      `model: openai/${openai?.defaultModel ?? 'gpt-4o-mini'}`,
      'skills:',
      ...skills.map((s) => `  - ${s}`),
      '---',
      '',
      `# ${name}`,
      '',
      'You are a helpful assistant.',
      '',
    ].join('\n');

    fs.writeFileSync(path.join(dir, 'AGENT.md'), agentMd);

    // AGENT.MD IS THE WHOLE PROJECT (2026-09-25), as in the Python CLI. This
    // also wrote a `package.json` pinning `webagents` for `npm install` and
    // `npm start`, which the CLI that just ran `init` does not need, and which
    // made the first steps differ between the two CLIs. A project that wants
    // its own Node package adds one when it adds code.
    // The same report as the Python CLI's `init`.
    const model = `openai/${openai?.defaultModel ?? 'gpt-4o-mini'}`;
    console.log(`\nCreated agent project: ${name}/`);
    console.log(`  AGENT.md       the agent: its model, skills and instructions`);
    console.log(`\nNext steps:`);
    console.log(`  cd ${name}`);
    // The commands as they must be typed here: with `--profile` under one.
    const chat = cliCommand();
    const serve = cliCommand('serve');
    const width = Math.max(21, serve.length + 2);
    console.log(`  ${chat.padEnd(width)}chat with it`);
    console.log(`  ${serve.padEnd(width)}serve it over HTTP`);
    // The model is the step every first run trips on, so both ways to it are
    // named here. It said `export OPENAI_API_KEY=...`, which puts the key in
    // shell history, and left out signing in.
    console.log(
      `\nIt runs on ${model}: add your key with \`${cliCommand('secrets set OPENAI_API_KEY')}\`, ` +
        `or sign in with \`${cliCommand('login')}\` to run it through Robutler.\n`,
    );
  });

// ============================================================================
// 14. publish
// ============================================================================

program
  .command('publish')
  .description('Publish the agent to Robutler, or update the one this folder is linked to')
  .argument('[path]', 'Path to agent config', '.')
  .option('-y, --yes', 'Do not ask before creating a new agent')
  .option('--dry-run', 'Show what would be sent, and send nothing')
  .action(async (agentPath, options: { yes?: boolean; dryRun?: boolean }) => {
    // `publish.ts` is the one implementation, shared with the chat's `/publish`.
    const { publishAgent } = await import('./publish.js');
    const result = await publishAgent(agentPath, {
      ok: (line) => console.log(line),
      print: (line) => console.log(line),
      error: (line) => console.error(line),
      confirm: async (question) => {
        if (!process.stdin.isTTY) {
          console.error('Pass --yes to create it without a prompt.');
          return false;
        }
        const { promptLine } = await import('./prompt.js');
        const answer = await promptLine(`${question} [y/N] `);
        return /^y(es)?$/i.test((answer ?? '').trim());
      },
    }, { yes: options.yes, dryRun: options.dryRun });
    if (!result.ok) process.exit(1);
  });

// ============================================================================
// 14b. secrets get
// ============================================================================

/**
 * Keys this CLI keeps on this machine: provider keys and agent API keys.
 *
 * The store is the Python CLI's (`provider-keys.ts`), so `secrets set` in
 * either CLI is seen by both. The chat loads the provider keys it knows
 * (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY) wherever
 * the environment has none, and offers to store one when it has no model to
 * run on.
 *
 * `get` exists because `publish` stores the agent's API key, which the
 * platform returns exactly once, and this CLI had no way to read it back.
 * Same contract as the Python command: redacted by default, the bare value on
 * stdout under `--show` (for `$(...)` in a script), exit 1 when absent.
 */
const secretsCmd = program.command('secrets').description('Keys this CLI stored on this machine');

/**
 * `secrets list`, in the words the Python CLI prints: each stored key, and
 * each provider key the shell sets, with where it comes from. A keychain
 * cannot be listed, so the names are those this CLI recorded, and it says so.
 */
async function secretsListing(
  open: () => Promise<{ list(): Promise<{ names: string[]; complete: boolean }>; status(): { backend: string } }>,
  providerVars: string[],
): Promise<string[]> {
  const store = await open();
  const { names, complete } = await store.list();
  const stored = new Set(names);
  const keychain = store.status().backend === 'keystore';
  const shown = [...new Set([...names, ...providerVars.filter((v) => process.env[v])])].sort();
  if (!shown.length) {
    const out = ['No keys stored, and none set in this shell.', `Add one with \`${cliCommand('secrets set OPENAI_API_KEY')}\`.`];
    if (!complete) out.push('An OS keychain cannot be listed, so keys other tools wrote there do not appear.');
    return out;
  }
  const width = Math.max(...shown.map((n) => n.length)) + 3;
  const where = (name: string) =>
    process.env[name]
      ? stored.has(name)
        ? 'set in this shell, which wins over the stored one'
        : 'set in this shell'
      : keychain
        ? 'stored in your keychain'
        : 'stored in an owner-only file';
  const out = shown.map((name) => `  ${name.padEnd(width)}${where(name)}`);
  if (!complete) out.push('', 'An OS keychain cannot be listed, so keys other tools wrote there do not appear.');
  return out;
}

secretsCmd
  .command('list')
  .description('Keys stored on this machine, and which this shell sets')
  .action(async () => {
    const { providerKeyStore } = await import('./provider-keys.js');
    const { keyProviders } = await import('./model-access.js');
    const { providerEnvVars } = await import('../skills/llm/providers.js');
    const lines = await secretsListing(providerKeyStore, keyProviders().flatMap((p) => [...providerEnvVars(p)]));
    for (const line of lines) console.log(line);
  });

secretsCmd
  .command('set <name>')
  .description('Store a key, for example OPENAI_API_KEY (asked for with echo off)')
  .action(async (name: string) => {
    // Never from an argument: that lands in shell history and the process list.
    if (!/^[A-Z][A-Z0-9_]*$/.test(name)) {
      console.error(`${name} does not look like an environment variable name.`);
      process.exit(1);
    }
    const value = (await promptSecret(`Value for ${name}: `)).trim();
    if (!value) {
      console.error('Nothing entered; nothing stored.');
      process.exit(1);
    }
    const { storeProviderKey } = await import('./provider-keys.js');
    const backend = await storeProviderKey(name, value);
    console.log(`Stored ${name} (${backend === 'keystore' ? 'your keychain' : 'an owner-only file'}).`);
    if (process.env[name]) {
      console.log(`${name} is also set in this environment, and the environment wins.`);
    }
  });

secretsCmd
  .command('unset <name>')
  .description('Remove a stored key')
  .action(async (name: string) => {
    const { providerKeyStore } = await import('./provider-keys.js');
    if (await (await providerKeyStore()).delete(name)) {
      console.log(`Removed ${name}.`);
    } else {
      console.error(`${name} was not stored.`);
      process.exit(1);
    }
  });

secretsCmd
  .command('get <name>')
  .description('Say whether a key is stored, or print it with --show')
  .option('--show', 'Print the value itself, for a script')
  .action(async (name: string, options: { show?: boolean }) => {
    const { providerKeyStore } = await import('./provider-keys.js');
    const store = await providerKeyStore();
    let value: string | null;
    try {
      value = await store.get(name);
    } catch (err) {
      console.error((err as Error).message);
      process.exit(1);
    }
    if (value === null) {
      console.error(`${name} is not stored.`);
      if (process.env[name]) console.error('It IS set in this environment, which takes precedence.');
      process.exit(1);
    }
    if (options.show) {
      process.stdout.write(`${value}\n`);
      return;
    }
    console.log(`${name} is stored. Use --show to print it.`);
  });

/**
 * A FIRST WORD THAT NAMES NO COMMAND IS A MISTYPED COMMAND (2026-09-24).
 *
 * With `chat` as the default command, commander hands `webagents publsh` to
 * `chat`, which answered "too many arguments for 'chat'. Expected 0 arguments
 * but got 1." It is now what commander says for a group, with its suggestion,
 * and what the Python CLI says: "error: unknown command 'publsh'" and
 * "(Did you mean publish?)". Options and their values are skipped, so
 * `webagents -p hi` still reaches the chat.
 */
{
  const chat = program.commands.find((c) => c.name() === 'chat');
  const knownFlags = new Set<string>(['-h', '--help', '-V', '--version']);
  const valueFlags = new Set<string>();
  for (const cmd of [program, ...(chat ? [chat] : [])]) {
    for (const option of cmd.options) {
      for (const flag of [option.short, option.long]) {
        if (!flag) continue;
        knownFlags.add(flag);
        if (option.required || option.optional) valueFlags.add(flag);
      }
    }
  }
  const argv = process.argv.slice(2);
  const names = [...program.commands.flatMap((c) => [c.name(), ...c.aliases()]), 'help'];
  const index = firstOperandIndex(argv, knownFlags, valueFlags);
  // `help <name>` names a command too: commander printed the whole help for a typo there.
  const word = index === -1 ? undefined : argv[index] === 'help' && argv[index + 1] && !argv[index + 1].startsWith('-') ? argv[index + 1] : argv[index];
  if (word !== undefined && !names.includes(word)) {
    program.error(`error: unknown command '${word}'${suggestSimilar(word, names)}`, {
      code: 'commander.unknownCommand',
    });
  }
}

program.parse();
