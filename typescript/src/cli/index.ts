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
import { forPrompt } from './failures';
import { MAX_TOOL_ROUNDS_ENV, TOOL_LOOP, isAgentFinish, parseMaxToolRounds, toolLoopSentence, toolRoundLimitSentence } from '../core/tool-budget';
import { INIT_TEMPLATES, ROBUTLER_CHOICE_MODEL, agentMarkdown, initLine, initModel } from './init-templates';
import { promptSecret, promptSecretOrPipe } from './prompt';
import { REFERENCE_NAME } from '../skills/secrets/references';
import { streamJsonEvent } from './render';
import { firstOperandIndex, suggestSimilar } from './suggest';
import { hoistGlobalOptions } from './sandbox-default-argv';
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
  .option('--max-tool-rounds <n>', 'Tool rounds one turn may run before its last answer (default 50)')
  .option('--token <token>', 'Use this platform token for this run, instead of the stored sign-in')
  // The sandbox is on by default (2026-09-27): this is the one-run opt-out,
  // for the owner's shell commands only (`sandbox/policy.ts`, ENV_NO_SANDBOX).
  .option('--no-sandbox', 'Run shell commands with your permissions for this run, outside the operating-system sandbox')
  .hook('preAction', async (command) => {
    const opts = command.opts() as { profile?: string; token?: string; sandbox?: boolean; maxToolRounds?: string };
    // The profile is a name, not a secret, so the environment is where every
    // later lookup already reads it. The token is not put there (credentials.ts).
    if (opts.profile) process.env.WEBAGENTS_PROFILE = opts.profile;
    // `--max-tool-rounds` (2026-09-28, `core/tool-budget.ts`): checked here,
    // once, then read by every agent this run builds (the chat, `-p`,
    // `serve`, the daemon) from the environment, as `--profile` is.
    if (opts.maxToolRounds !== undefined) {
      try {
        process.env[MAX_TOOL_ROUNDS_ENV] = String(parseMaxToolRounds(opts.maxToolRounds, '--max-tool-rounds'));
      } catch (error) {
        console.error((error as Error).message);
        process.exit(1);
      }
    }
    if (opts.token) (await import('./credentials.js')).setFlagToken(opts.token);
    // Commander reads `--no-sandbox` as `sandbox: false`. The shell reads the
    // environment, so a child `webagents` inherits the choice for the run.
    if (opts.sandbox === false) process.env.WEBAGENTS_NO_SANDBOX = '1';
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
  let format = (options.outputFormat ?? 'text') as OutputFormat;
  if (!OUTPUT_FORMATS.includes(format)) {
    console.error(
      `Unknown --output-format '${options.outputFormat}'. Expected one of: ${OUTPUT_FORMATS.join(', ')}.`,
    );
    process.exit(2);
  }
  // `--json` with `-p` is `--output-format json` (2026-09-27, the final e2e
  // re-run: the global flag was ignored here). An explicit format wins, and
  // the flag alone still opens the chat.
  if (options.prompt && format === 'text') {
    const { jsonEnabled } = await import('./output.js');
    if (jsonEnabled(program)) format = 'json';
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
    // By name, as `/agent` does, or refused (`agent-files.ts`). Under
    // `--json` the refusal is the error envelope (`output.ts` `fail`, code
    // `agent_not_found`), which was ignored here (2026-09-26, the e2e run).
    const { agentFileFor, AgentNotFound } = await import('./agent-files.js');
    try {
      config.agentFile = agentFileFor(process.cwd(), options.agent);
    } catch (error) {
      if (!(error instanceof AgentNotFound)) throw error;
      const { jsonEnabled, fail } = await import('./output.js');
      if (jsonEnabled(program)) fail('agent_not_found', error.message);
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
      // In the format asked for (2026-09-28, the e2e pass: `json` and
      // `stream-json` printed text).
      promptFailure(format, repl.modelProblem, 'no_model', undefined);
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
            const { headline, hint, code } = repl.explainTurnError(chunk.error);
            const shown = forPrompt(hint);
            event.error = {
              ...(event.error as Record<string, unknown>),
              message: headline,
              ...(code ? { code } : {}),
              ...(shown ? { hint: shown } : {}),
            };
          }
          if (event) process.stdout.write(`${JSON.stringify(event)}\n`);
          if (chunk.type === 'error') failed = true;
          // The agent ended the turn by its tool budget (2026-09-28,
          // `core/tool-budget.ts`): the `done` line carries the finish, and
          // the exit is 1, as the Python `-p` answers.
          if (chunk.type === 'done' && isAgentFinish(chunk.response?.finish?.reason)) failed = true;
        }
        if (failed) process.exit(1);
      } else {
        const response = await repl.sendMessage(options.prompt);
        console.log(format === 'json' ? JSON.stringify(response, null, 2) : response.content);
        // THE AGENT ENDED THE TURN (2026-09-28, `core/tool-budget.ts`): its
        // tool rounds ran out, or it repeated one call, and its last,
        // tool-less call gave the answer printed above. `json` carries the
        // finish, text says it on stderr, and the exit is 1, so a script can
        // tell it from a turn that finished. The Python `-p` answers the same.
        const ended = response.finish && isAgentFinish(response.finish.reason) ? response.finish : undefined;
        if (ended && response.content.trim() && format === 'text') {
          console.error(`Error: ${ended.reason === TOOL_LOOP ? toolLoopSentence(ended.tool) : toolRoundLimitSentence(ended.rounds, true)}`);
        }
        if (ended) process.exit(1);
        if (!response.content.trim()) {
          // NOTHING SAID IS SAID, in `-p` too (the ptypass-fixes lane,
          // 2026-09-27; the chat's `presentEmptyReply`, the Python `-p`'s
          // twin): an empty reply printed an empty line and exited 0, so a
          // script could not tell it from an answer. The truthful line goes
          // to stderr; stdout keeps the answer channel's shape.
          const { presentEmptyReply } = await import('./failures.js');
          const { headline, hint } = presentEmptyReply(response.finish ?? {});
          console.error(headline);
          if (hint) console.error(hint);
        }
      }
    } catch (error) {
      // The chat's headline and hint (`failures.ts`), and a non-zero exit a
      // script can see. The stack only helps someone debugging the SDK
      // itself, so it waits for WEBAGENTS_DEBUG.
      const { headline, hint, code } = repl.explainTurnError(error);
      // `json` answers in JSON (2026-09-28), and the hint is `-p`'s: no chat command.
      promptFailure(format, headline, code ?? ((error as { code?: unknown }).code as string | undefined), forPrompt(hint));
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
  // The chat is over and its agent cleaned up (`run()`), so the event loop
  // should drain by itself. When something still holds it (a stdio MCP
  // server's pipes, a socket a skill left open), `/exit` used to hang for
  // good (2026-09-26, the e2e run's HIGH bug). A watchdog that does not
  // itself keep the loop alive ends the process a moment later instead.
  setTimeout(() => process.exit(0), 1500).unref();
}

/**
 * A failed `-p` in the format asked for (2026-09-28, the e2e pass):
 * `stream-json` an `error` line, `json` one `{"error": {...}}` document on
 * stdout, `text` the headline and hint on stderr. The Python `-p` answers the
 * same (`one_shot._fail`; fixture `cli/final_sdk_low_items.json`).
 */
function promptFailure(format: OutputFormat, message: string, code: string | undefined, hint: string | undefined): void {
  const body = { message, ...(typeof code === 'string' && code ? { code } : {}), ...(hint ? { hint } : {}) };
  if (format === 'stream-json') {
    process.stdout.write(`${JSON.stringify({ type: 'error', error: body })}\n`);
  } else if (format === 'json') {
    process.stdout.write(`${JSON.stringify({ error: body }, null, 2)}\n`);
  } else {
    console.error(code === 'no_model' ? message : `Error: ${message}`);
    if (hint) console.error(hint);
  }
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

/**
 * A port already in use (`server/listen-error.ts`, 2026-09-26): its one
 * sentence on stderr and exit 1, as the Python CLI answers, instead of an
 * unhandled 'error' event with a stack trace. Anything else still throws.
 */
async function orListenError<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (error) {
    if ((error as Error | undefined)?.name === 'PortInUseError') {
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
    await orListenError(() => orAgentFileError(() => serveAction(agentPath, options)));
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
      // Without -w, the working directory's agents, as the Python daemon.
      watchDir: options.watch ?? process.cwd(),
      cron: options.cron,
    });
    await orListenError(() => daemon.start());
    process.on('SIGINT', () => { daemon.stop(); process.exit(0); });
  });

// ============================================================================
// 4b. mcp
// ============================================================================

// `webagents mcp serve` (plan item 1.8, 2026-09-26): the agent's tools to an
// MCP client, over stdio by default (how Claude Code, Codex and OpenCode start
// a local server) or over Streamable HTTP with `--http`. The caller over stdio
// is the person at this terminal, the owner; over HTTP the rules are `serve`'s:
// a credential is required, the agent's auth skills and access block say who
// it is, and the tools are the ones that caller may use. The body lives in
// ./mcp-serve-action for the reason serve's does (this module parses argv).
// The Python CLI has the same group and words (`tests/cli/test_cli_parity.py`).
const mcpCmd = program.command('mcp').description('Serve an agent over the Model Context Protocol');
mcpCmd
  .command('serve')
  .description("Serve an agent's tools to an MCP client: over stdio, or over Streamable HTTP with --http")
  .argument('[path]', 'Path to agent config file', '.')
  .option('--http <port>', 'Serve over Streamable HTTP on this port instead of stdio')
  .option('--host <host>', 'Interface to listen on with --http (0.0.0.0 for every interface)')
  .action(async (agentPath, options) => {
    const { mcpServeAction } = await import('./mcp-serve-action.js');
    await orListenError(() => orAgentFileError(() => mcpServeAction(agentPath, options)));
  });

// ============================================================================
// 4c. cron
// ============================================================================

// `webagents cron` (plan item 1.7, 2026-09-26): the `cron:` schedules the
// agent files in a folder declare, as the daemon runs them. `list` reads the
// files and the runner's state and writes nothing; `run` builds the one agent
// and runs the schedule now, delivering as configured. The bodies live in
// ./cron-action for the reason serve's does (this module parses argv). The
// Python CLI has the same group and words (`tests/cli/test_cli_parity.py`);
// the lines are held by `python/tests/fixtures/cli/cron.json`.
const cronCmd = program.command('cron').description('Schedules the agents in a folder declare');
cronCmd
  .command('list')
  .description('List the schedules of the agents in a folder')
  .option('-w, --watch <dir>', 'Folder whose agents to read (default: this folder)')
  .action(async (options) => {
    const { cronListAction } = await import('./cron-action.js');
    const { jsonEnabled } = await import('./output.js');
    // Non-zero when a file was refused (2026-09-26): the listing says so and exits 1.
    const code = await orAgentFileError(() => cronListAction(options, { json: jsonEnabled(program) }));
    if (code) process.exit(code);
  });
cronCmd
  .command('run')
  .description('Run a schedule now and deliver as configured')
  .argument('<agent>', 'The agent, by name')
  .argument('<name>', 'The schedule, by name')
  .option('-w, --watch <dir>', 'Folder whose agents to read (default: this folder)')
  .action(async (agent, name, options) => {
    const { cronRunAction } = await import('./cron-action.js');
    const { jsonEnabled } = await import('./output.js');
    const code = await orAgentFileError(() => cronRunAction(agent, name, options, { json: jsonEnabled(program) }));
    if (code) process.exit(code);
  });

// ============================================================================
// 4d. acp
// ============================================================================

// `webagents acp` (plan item 1.6, 2026-09-26): the agent to a code editor over
// the Agent Client Protocol, on stdin and stdout (how Zed and the JetBrains
// IDEs start an agent). The caller is the person whose editor spawned this
// process, the owner. The body lives in ./acp-action for the reason serve's
// does (this module parses argv). The Python CLI has the same command and
// words (`tests/cli/test_cli_parity.py`); the words are also held by
// `python/tests/fixtures/acp/acp_protocol.json`.
program
  .command('acp')
  .description('Serve an agent to a code editor over the Agent Client Protocol (stdio)')
  .argument('[path]', 'Path to agent config file', '.')
  .action(async (agentPath) => {
    const { acpAction } = await import('./acp-action.js');
    await orAgentFileError(() => acpAction(agentPath));
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
    // file would leave the token behind in the keystore. The steps and words
    // are `logoutCommand`'s (keychain-ux, 2026-09-27): an old `webagents:cli`
    // item only a macOS dialog could remove is named, and a token that needs
    // one with nobody to answer is the one sentence and exit 1.
    const { logoutCommand } = await import('./account');
    const code = await logoutCommand(console.log, console.error, () => {
      try { fs.unlinkSync(LEGACY_AUTH_FILE); } catch { /* already gone */ }
    });
    if (code) process.exit(code);
  });

program
  .command('whoami')
  .description('Show who you are signed in as')
  .action(async () => {
    const { settleKeychain, whoAmI } = await import('./account.js');
    const { jsonEnabled, emit, fail } = await import('./output.js');
    // In a terminal, what a run with nobody to answer macOS could not read is
    // read now, so macOS asks once, here (keychain-ux, 2026-09-27): this is
    // the command that run's one sentence names.
    await settleKeychain();
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
  .command('budget')
  .description('Show the budget tree of a run: a payment token and every child a hop derived from it')
  .argument('<token_id>', 'A payment token id, from your token list on the platform')
  .action(async (tokenId: string) => {
    const { budgetTree } = await import('./budget-tree.js');
    const { jsonEnabled, emit, fail } = await import('./output.js');
    const result = await budgetTree(tokenId);
    if (jsonEnabled(program)) {
      if (!result.ok) {
        fail(result.code, result.message, result.fix);
        return;
      }
      emit({ tree: result.tree });
      return;
    }
    if (!result.ok) {
      console.error(result.message);
      if (result.fix) console.error(result.fix);
      process.exit(1);
    }
    for (const line of result.lines) console.log(line);
    console.log(result.totals);
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
  // `-a`, as the chat takes it (2026-09-26): `doctor -a helper` was "unknown option".
  .option('-a, --agent <agent>', 'Agent name')
  .action(async (options: { agent?: string }) => {
    const { runChecks, reportLines } = await import('./doctor.js');
    const { jsonEnabled, emit, fail } = await import('./output.js');
    const { AgentNotFound } = await import('./agent-files.js');
    let checks: Awaited<ReturnType<typeof runChecks>>;
    try {
      checks = await runChecks({ agent: options.agent });
    } catch (error) {
      // An agent the folder does not have: the chat's sentence, or the error envelope.
      if (!(error instanceof AgentNotFound)) throw error;
      if (jsonEnabled(program)) fail('agent_not_found', error.message);
      console.error(error.message);
      process.exit(1);
    }
    if (jsonEnabled(program)) emit({ checks });
    else for (const line of reportLines(checks)) console.log(line);
    // An explicit exit either way (2026-09-26): `runChecks` closes the
    // agent it built, and this makes sure nothing else the agent started
    // can hold the process open once the report is out.
    process.exit(checks.some((c) => c.status === 'fail') ? 1 : 0);
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
    const { LLM_PROVIDERS, MODELS_READY_FOOTNOTE, configuredProviders, providerBaseUrl, providerNeeds } = await import('../skills/llm/providers.js');
    // Stored keys count: the chat runs on them. Only the shell was looked at,
    // so a key kept with `secrets set` showed its provider as not ready.
    const { readStoredProviderKeys } = await import('./provider-keys.js');
    const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
    const { available } = configuredProviders({ ...stored, ...process.env });
    const ready = new Set(available.map((p) => p.id));
    // A local server (Ollama, plan item 2.8) is ready when it answers.
    const { probeOllama } = await import('../skills/llm/ollama/probe.js');
    for (const p of LLM_PROVIDERS) {
      const base = providerBaseUrl(p);
      if (p.credential === 'none' && base && (await probeOllama(base)).ok) ready.add(p.id);
    }
    // Robutler's models are ready when this profile is signed in
    // (2026-09-27): the row said `-` for a signed-in person, since only the
    // proxy URL variable was looked at. The Python CLI decides the same way.
    const { getToken } = await import('./credentials.js');
    if (await getToken().catch(() => null)) {
      for (const p of LLM_PROVIDERS) if (p.credential === 'platform') ready.add(p.id);
    }

    // The in-browser runtimes (webllm, transformers) run in the browser
    // build, not here: this CLI cannot load them, so it does not list them
    // (2026-09-25), and the Python CLI lists the same providers.
    const rows = LLM_PROVIDERS.filter((provider) => provider.credential !== 'local').map((p) => ({
      id: p.id,
      ready: ready.has(p.id),
      model_format: p.modelFormat,
      needs: providerNeeds(p, cliCommand('login')),
    }));
    // `--json` (2026-09-27): the table's rows as one document (fixture `cli/json_documents.json`, `models`).
    const { jsonEnabled, emit } = await import('./output.js');
    if (jsonEnabled(program)) {
      emit({ providers: rows });
      return;
    }
    console.log('\nLLM providers:\n');
    for (const row of rows) {
      const mark = row.ready ? 'ready' : '-';
      console.log(`  ${mark.padEnd(6)} ${row.id.padEnd(14)} ${row.model_format.padEnd(24)} ${row.needs}`);
    }
    console.log(`\n${MODELS_READY_FOOTNOTE.replace('{command}', cliCommand('secrets set'))}`);
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
    // The folder's SKILL.md skills, apart from the coded names (plan item 1.4).
    const { skillmdListLines } = await import('./skills-edit.js');
    console.log();
    for (const line of skillmdListLines(process.cwd())) console.log(line);
    console.log();
  });

// `skills add` and `skills remove` change the `skills:` list of this folder's
// agent file and nothing else (`skills-edit.ts`, the Python `skills_edit.py`),
// except for a SOURCE (`owner/repo`, a git URL, a folder), which installs a
// SKILL.md skill into `.agents/skills` instead (plan item 1.4).
skillsCmd
  .command('add')
  .description('Add skills to an agent file')
  .argument('<names...>', 'Skills to add, by the names `skills list` shows, or a SKILL.md source: owner/repo, a git URL or a folder')
  .option('-a, --agent <agent>', 'Agent name')
  .option('--skill <name>', 'Install only this skill from the source')
  .option('-y, --yes', 'Install without asking')
  .action(async (names: string[], options: { agent?: string; skill?: string; yes?: boolean }) => {
    const { skillsCommand } = await import('./skills-edit.js');
    const { jsonEnabled, emit, fail } = await import('./output.js');
    if (!jsonEnabled(program)) {
      const code = await skillsCommand('add', names, { agent: options.agent, skill: options.skill, yes: options.yes });
      if (code) process.exit(code);
      return;
    }
    // `--json` (2026-09-27): one document, the editor's facts and the lines
    // it would have printed (fixture `cli/json_documents.json`, `skills_add`);
    // a refusal is the error envelope with the lines it would have printed.
    const messages: string[] = [];
    const errors: string[] = [];
    let edited: import('./skills-edit.js').SkillsEdited | undefined;
    const code = await skillsCommand(
      'add',
      names,
      { agent: options.agent, skill: options.skill, yes: options.yes },
      { out: (line) => messages.push(line), err: (line) => errors.push(line), edited: (facts) => { edited = facts; } },
    );
    if (code) fail('skills_add_failed', [...errors, ...messages].join('\n'), '', code);
    emit({ file: edited?.file ?? null, added: edited?.added ?? [], already: edited?.already ?? [], messages });
  });

skillsCmd
  .command('remove')
  .description('Remove skills from an agent file')
  .argument('<names...>', 'Skills to remove')
  .option('-a, --agent <agent>', 'Agent name')
  .action(async (names: string[], options: { agent?: string }) => {
    const { skillsCommand } = await import('./skills-edit.js');
    const code = await skillsCommand('remove', names, { agent: options.agent });
    if (code) process.exit(code);
  });

// ============================================================================
// 11. templates
// ============================================================================

// The templates `init` can make, and the AGENT.md each writes, live in
// `./init-templates` (2026-09-26): the chat's `/agent new` writes the same
// file, so one table and one renderer feed both.
const templatesCmd = program.command('templates').description('Agent templates');

templatesCmd
  .command('list')
  .description('List available templates')
  .action(async () => {
    // `--json` (2026-09-27): the same table as one document (fixture `cli/json_documents.json`, `templates_list`).
    const { jsonEnabled, emit } = await import('./output.js');
    if (jsonEnabled(program)) {
      emit({ templates: Object.entries(INIT_TEMPLATES).map(([name, t]) => ({ name, description: t.description })) });
      return;
    }
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
    // `--json` (2026-09-27): one document either way (fixture
    // `cli/json_documents.json`, `init`; the refusals in `cli/json_errors.json`).
    const { jsonEnabled, emit, fail } = await import('./output.js');
    const json = jsonEnabled(program);
    // Checked before anything is created, so a typo leaves no directory behind.
    const template = INIT_TEMPLATES[options.template];
    if (!template) {
      const message = `Unknown template '${options.template}'. Available: ${Object.keys(INIT_TEMPLATES).join(', ')}.`;
      if (json) fail('unknown_template', message);
      console.error(message);
      process.exit(1);
    }

    const dir = path.resolve(name);
    if (fs.existsSync(dir)) {
      if (json) fail('directory_exists', `Directory ${name} already exists.`);
      console.error(`Directory ${name} already exists.`);
      process.exit(1);
    }

    fs.mkdirSync(dir, { recursive: true });

    // AGENT.md, THE FORMAT THE DOCS DESCRIBE (2026-09-24). This wrote
    // `agent.json` plus `instructions.md`, which nothing read (see
    // `agent-project.ts`), while the CLI quickstart shows `webagents init`
    // producing `AGENT.md`. It is the one file both SDKs parse, so the same
    // project also runs under the Python CLI. Only keys the Python loader's
    // strict schema accepts: the old `template` key would be REJECTED there.
    // The bytes are `agentMarkdown`'s (`init-templates.ts`), which the chat's
    // `/agent new` writes too. The model is a provider's when this machine
    // holds its key, else Robutler's choice (B3, 2026-09-28; `initModel`).
    const keyed = await initModel();
    fs.writeFileSync(path.join(dir, 'AGENT.md'), agentMarkdown(name, options.template, keyed));

    // AGENT.MD IS THE WHOLE PROJECT (2026-09-25), as in the Python CLI. This
    // also wrote a `package.json` pinning `webagents` for `npm install` and
    // `npm start`, which the CLI that just ran `init` does not need, and which
    // made the first steps differ between the two CLIs. A project that wants
    // its own Node package adds one when it adds code.
    // The same report as the Python CLI's `init`.
    const model = keyed ?? ROBUTLER_CHOICE_MODEL;
    if (json) {
      // The model the file names; null when it names none (B3).
      emit({ name, template: options.template, path: dir, files: ['AGENT.md'], model: keyed ?? null });
      return;
    }
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
    // What runs it, and the way in only when one is needed: no `login` hint
    // for a person already signed in (2026-09-28, fixture `init_line`).
    const { getToken } = await import('./credentials.js');
    console.log(`\n${initLine(model, keyed !== undefined, Boolean(await getToken().catch(() => undefined)))}\n`);
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
    // `--json` (2026-09-27): the lines go to stderr and the outcome is one
    // document, for `--dry-run` the request that would have been sent
    // (fixture `cli/json_documents.json`, `publish_dry_run`).
    const { jsonEnabled, emit, fail, note } = await import('./output.js');
    const json = jsonEnabled(program);
    const errors: string[] = [];
    const result = await publishAgent(agentPath, {
      ok: (line) => (json ? note(line) : console.log(line)),
      print: (line) => (json ? note(line) : console.log(line)),
      error: (line) => {
        if (json) errors.push(line);
        else console.error(line);
      },
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
    if (json) {
      if (!result.ok) fail('publish_failed', errors.join('\n'));
      emit(
        result.request
          ? { method: result.request.method, url: result.request.url, body: result.request.body, created: result.created ?? false }
          : { username: result.username ?? null, agent_id: result.agentId ?? null, created: result.created ?? false },
      );
      return;
    }
    if (!result.ok) process.exit(1);
  });

// ============================================================================
// 14b. secrets
// ============================================================================

/**
 * Keys and secrets this CLI keeps on this machine: provider keys, agent API
 * keys, and since S-292 (2026-09-26) the secrets an agent file's MCP servers
 * name as `${secret:NAME}` in their `env`, `headers` or `url`.
 *
 * The store is the Python CLI's (`provider-keys.ts`), so `secrets set` in
 * either CLI is seen by both. The chat loads the provider keys it knows
 * (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY) wherever
 * the environment has none, and offers to store one when it has no model to
 * run on. The MCP skill reads any other name on demand, at connect time.
 *
 * `set` takes the value with echo off at a terminal, or from a pipe in a
 * script (`promptSecretOrPipe`), never from an argument. `get` exists because
 * `publish` stores the agent's API key, which the platform returns exactly
 * once, and this CLI had no way to read it back. Same contract as the Python
 * command: redacted by default, the bare value on stdout under `--show` (for
 * `$(...)` in a script), exit 1 when absent. `remove` was `unset` until
 * 2026-09-26; the old name still works, hidden. The words are pinned by
 * `python/tests/fixtures/cli/secrets.json`.
 */
const secretsCmd = program.command('secrets').description('Keys and secrets this CLI keeps on this machine');

/** What `secrets set NAME VALUE` answers (the Python CLI's `VALUE_AS_ARGUMENT`, fixture `cli/final_sdk_low_items.json`). */
const SECRETS_VALUE_AS_ARGUMENT =
  '`secrets set` takes the name alone: it asks for the value with echo off, or reads it from a pipe, ' +
  'so the value never lands in your shell history. Nothing was stored.';

/**
 * `secrets list`, in the words the Python CLI prints: each stored key, and
 * each provider key the shell sets, with where it comes from. A keychain
 * cannot be listed, so the names are those this CLI recorded, and it says so.
 */
/** One key as `secrets list` knows it (the `--json` document's rows, the Python CLI's `listing_rows`). */
interface SecretRow {
  name: string;
  stored: boolean;
  where: 'keychain' | 'file' | null;
  set_in_shell: boolean;
}

/**
 * What `secrets list` knows, as data (pinned by `cli/secrets.json` `list_json`,
 * 2026-09-26): each stored key and each provider key this shell sets, sorted
 * by name, whether it is stored and where, whether the shell sets it (the
 * shell wins), and whether the listing is complete (a keychain cannot be listed).
 */
async function secretsRows(
  open: () => Promise<{ list(): Promise<{ names: string[]; complete: boolean }>; status(): { backend: string } }>,
  providerVars: string[],
): Promise<{ keys: SecretRow[]; complete: boolean }> {
  const store = await open();
  const { names, complete } = await store.list();
  const stored = new Set(names);
  const where = store.status().backend === 'keystore' ? 'keychain' : 'file';
  const shown = [...new Set([...names, ...providerVars.filter((v) => process.env[v])])].sort();
  return {
    keys: shown.map((name) => ({ name, stored: stored.has(name), where: stored.has(name) ? where : null, set_in_shell: Boolean(process.env[name]) })),
    complete,
  };
}

async function secretsListing(
  open: () => Promise<{ list(): Promise<{ names: string[]; complete: boolean }>; status(): { backend: string } }>,
  providerVars: string[],
): Promise<string[]> {
  const { keys, complete } = await secretsRows(open, providerVars);
  const caveat = 'An OS keychain cannot be listed, so keys other tools wrote there do not appear.';
  if (!keys.length) {
    const out = ['No keys stored, and none set in this shell.', `Add one with \`${cliCommand('secrets set OPENAI_API_KEY')}\`.`];
    const { otherRuntimeKeysLine } = await import('./account');
    const other = await otherRuntimeKeysLine(open);
    if (other) out.push(other);
    if (!complete) out.push(caveat);
    return out;
  }
  const width = Math.max(...keys.map((k) => k.name.length)) + 3;
  const where = (key: SecretRow) =>
    key.set_in_shell
      ? key.stored
        ? 'set in this shell, which wins over the stored one'
        : 'set in this shell'
      : key.where === 'keychain'
        ? 'stored in your keychain'
        : 'stored in an owner-only file';
  const out = keys.map((key) => `  ${key.name.padEnd(width)}${where(key)}`);
  if (!complete) out.push('', caveat);
  return out;
}

secretsCmd
  .command('list')
  .description('Keys stored on this machine, and which this shell sets')
  .action(async () => {
    const { providerKeyStore } = await import('./provider-keys.js');
    const { keyProviders } = await import('./model-access.js');
    const { providerEnvVars } = await import('../skills/llm/providers.js');
    const providerVars = keyProviders().flatMap((p) => [...providerEnvVars(p)]);
    // `--json` was ignored here (2026-09-26, the e2e run): one document now.
    const { jsonEnabled, emit } = await import('./output.js');
    if (jsonEnabled(program)) {
      emit(await secretsRows(providerKeyStore, providerVars));
      return;
    }
    for (const line of await secretsListing(providerKeyStore, providerVars)) console.log(line);
  });

secretsCmd
  .command('set <name>')
  .description('Store a key, for example OPENAI_API_KEY (asked for with echo off, or read from a pipe)')
  // A value typed after the name is answered with where values come from
  // (2026-09-28, the e2e pass), not "too many arguments for 'set'".
  .allowExcessArguments()
  .action(async (name: string, _options: unknown, command: { args: string[] }) => {
    // Never from an argument: that lands in shell history and the process
    // list. The name grammar is the `${secret:NAME}` grammar, so every
    // reference an agent file can write is one this command can store.
    if (command.args.length > 1) {
      console.error(SECRETS_VALUE_AS_ARGUMENT);
      process.exit(1);
    }
    if (!REFERENCE_NAME.test(name)) {
      console.error(`${name} does not look like an environment variable name.`);
      process.exit(1);
    }
    const value = (await promptSecretOrPipe(`Value for ${name}: `)).trim();
    if (!value) {
      console.error('Nothing entered; nothing stored.');
      process.exit(1);
    }
    const { storeProviderKey } = await import('./provider-keys.js');
    const { KeychainDialogBlocked } = await import('../skills/secrets/keychain-ux');
    let backend: 'keystore' | 'file';
    try {
      backend = await storeProviderKey(name, value);
    } catch (error) {
      // Replacing it needs macOS to ask, and nobody can answer here.
      if (!(error instanceof KeychainDialogBlocked)) throw error;
      console.error(error.message);
      process.exit(1);
    }
    console.log(`Stored ${name} (${backend === 'keystore' ? 'your keychain' : 'an owner-only file'}).`);
    if (process.env[name]) {
      console.log(`${name} is also set in this environment, and the environment wins.`);
    }
  });

async function removeSecret(name: string): Promise<void> {
  if (!REFERENCE_NAME.test(name)) {
    console.error(`${name} does not look like an environment variable name.`);
    process.exit(1);
  }
  const { providerKeyStore } = await import('./provider-keys.js');
  const { KeychainDialogBlocked } = await import('../skills/secrets/keychain-ux');
  const { leftBehindLines } = await import('./account');
  const store = await providerKeyStore();
  let removed: boolean;
  try {
    removed = await store.delete(name);
  } catch (error) {
    if (!(error instanceof KeychainDialogBlocked)) throw error;
    console.error(error.message);
    process.exit(1);
  }
  if (removed) {
    console.log(`Removed ${name}.`);
    // An old `webagents:` item only a macOS dialog could remove is named.
    for (const line of await leftBehindLines(store.leftBehind)) console.log(line);
  } else {
    console.error(`${name} was not stored.`);
    process.exit(1);
  }
}

secretsCmd.command('remove <name>').description('Remove a stored key').action(removeSecret);

secretsCmd
  .command('unset <name>', { hidden: true })
  .description('Remove a stored key (the old name of remove)')
  .action(removeSecret);

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

// ============================================================================
// sandbox
// ============================================================================

/**
 * `webagents sandbox setup` (the sandbox-engine lane, 2026-09-27): A CHECK,
 * NOT AN INSTALLER. The sandbox is on by default and fails closed, and the
 * engine ships with the package, so what can still be missing is the
 * machine's: the Linux programs (named with this distribution's install
 * line), what a container must allow, WSL 2 on Windows. It installs and
 * downloads nothing, runs a real confined `true`, and exits 1 when shell
 * commands would be refused. The words are the Python CLI's, pinned by
 * `python/tests/fixtures/sandbox/sandbox_engine.json`; `--json` is doctor's
 * document shape, `{checks: [...]}`.
 */
const sandboxCmd = program.command('sandbox').description('The sandbox that confines shell commands');

sandboxCmd
  .command('setup')
  .description('Check that the sandbox engine runs on this machine')
  .action(async () => {
    const { setupChecks } = await import('../sandbox/index.js');
    const { reportLines } = await import('./doctor.js');
    const { jsonEnabled, emit } = await import('./output.js');
    const checks = setupChecks();
    if (jsonEnabled(program)) emit({ checks });
    else for (const line of reportLines(checks)) console.log(line);
    process.exit(checks.some((c) => c.status === 'fail') ? 1 : 0);
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
// `--profile <name>` and `--no-sandbox` typed after the subcommand mean the
// same as before it (2026-09-27, `sandbox-default-argv.ts`).
process.argv = [...process.argv.slice(0, 2), ...hoistGlobalOptions(process.argv.slice(2))];

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
