/**
 * Interactive REPL
 * 
 * Interactive chat session with slash commands.
 * Based on Gemini CLI architecture.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as readline from 'readline';
import { BaseAgent } from '../core/agent';
import { LOCAL_OWNER as LOCAL_OWNER_CALLER } from '../access/caller';
import type { RunResponse, StreamChunk } from '../core/types';
import { providerBaseUrl, type LLMProvider } from '../skills/llm/providers';
import type { Message } from '../uamp/types';
import { TurnRecorder, historyForModel, spokenCount } from './turn-history';
import { withCliPreamble } from './preamble';
import { spawnSync } from 'node:child_process';
import { folderAgents, type FolderAgent } from './agent-files';
import { parseAgentMarkdown, readAgentFile, type ParsedAgent } from '../agents/index';
import { presentEmptyReply, presentFailure, type FailureText } from './failures';
import {
  CONTINUE_MESSAGE,
  DEFAULT_MAX_TOOL_ITERATIONS,
  TOOL_ROUND_LIMIT,
  agentFinishOf,
  continueQuestion,
  effectiveMaxToolRounds,
  parseMaxToolRounds,
  roundsSourceWords,
  roundsWords,
  toolRoundLimitSentence,
} from '../core/tool-budget';
import {
  CHAT_COMMANDS,
  CHAT_GROUPS,
  CHAT_KEYS,
  MOVED_COMMANDS,
  OUTSIDE_THE_CHAT,
  chatCommand,
  takesNoArguments,
} from './chat-commands';
import { ANSWER_WORDS, CHAT_WORDS, fill, plural } from './chat-words';
import { appendChatHistory, chatHistoryFile, readChatHistory } from './chat-history';
import { loadedAgentOf, reloadDiff, sameVersion, toolChangeLines, type LoadedAgent } from './agent-reload';
import { AGENT_NAME_RE, INIT_TEMPLATES, agentMarkdown } from './init-templates';
import { applySkills, editFrontMatterScalar, planSkills, spokenList, unsafeTargetReason, writeByRename } from './skills-edit';
import { suggestSimilar } from './suggest';
import { cliCommand, profileName, resolvePlatformUrl } from './config-store';
import { describeAccess, describeLocalRoute, keyProviders, type ModelAccess } from './model-access';
import { NO_COST, addTurnCost, costWords, type RunningCost } from '../skills/llm/pricing';
import type { AccessPolicy, GroupRule } from '../access/policy';
import type { MemoryOwnerSummary } from '../skills/memory/skill';
import { promptLine, promptSecret } from './prompt';
import { HOST_WORDS, addNetworkHost, fillWords, hostAnswer } from './sandbox-default-hosts';
import type { HostAsker } from '../skills/shell/skill';
import { TurnPrinter } from './render';
import { CONTROL_HEADER, CONTROL_QUESTION } from '../skills/filesystem/agent-secrets-guard';
import {
  listSessions,
  loadSession,
  markRecorded,
  newSessionId,
  saveSession,
  sessionsDir,
  whenLabel,
  type SessionMessage,
} from './sessions';
import {
  UNDO_WORDS,
  checkpointsDir,
  listCheckpoints,
  planChangesAnything,
  planLines,
  planRestore,
  restoreSnapshot,
  restoredSentence,
  rewindHeader,
  scanFolder,
  snapshotsOffReason,
  takeSnapshot,
  takeSnapshotYielding,
  turnLabel,
  type Manifest,
  type RestorePlan,
} from './checkpoints';
import {
  linkedPlatformAgent,
  listPlatformConversations,
  mergeConversations,
  readPlatformConversation,
  recordWords,
  unavailableReason,
  wordsSince,
  type ConversationsTarget,
  type PlatformConversation,
  type Unavailable,
} from './robutler-sessions';
import { compactNumber, duration, shortPath, terminalColumns, truncate, truncateStart, wrapStyled } from './ui/ansi';
import { playWordmark, welcomeCard, type WelcomeInfo } from './ui/banner';
import { promptBox, sentMessage } from './ui/input';
import { recordScreen, type ScreenRecord } from './ui/screen';
import { queryBackground, queryCursorRow } from './ui/terminal';
import { themeFor, type Theme } from './ui/theme';

/** How long an interrupted turn gets to wind down before the prompt returns. */
const INTERRUPT_GRACE_MS = 2000;

/**
 * Who the person at this terminal is to the agent: its OWNER (2026-09-25,
 * ADR-0045). Every turn ran anonymous, so an owner-scoped tool or prompt (the
 * REST tool is one) never reached the one person the local chat serves, while
 * an HTTP caller of the same agent file is whoever it proves to be. The Python
 * chat runs its turns the same way (`repl/session.py`, `one_shot.py`). One
 * definition, shared with the daemon's schedule runner (`access/caller.ts`).
 */
const LOCAL_OWNER: Record<string, unknown> = { ...LOCAL_OWNER_CALLER };

/** The embedded agent's name (`agents/ROBUTLER.md`), offered by /agent in every folder. */
const BUILT_IN_AGENT = 'robutler';

/**
 * The content tools `BaseAgent` registers for the length of one turn
 * (`core/agent.ts`, `present`, `read_content`, `save_content`) and deletes
 * when the turn ends. A turn stopped with Ctrl+C never reaches that deletion,
 * so they lingered in the registry and `/tools` listed them, which the Python
 * chat never does (2026-09-26): they are not the agent's tools, and are left
 * out of every listing and reload diff here.
 */
const TRANSIENT_TOOLS: ReadonlySet<string> = new Set(['present', 'read_content', 'save_content']);

/** How long the agent's cleanup (an MCP server that will not close) may hold the exit. */
const SHUTDOWN_GRACE_MS = 3000;

/**
 * REPL configuration
 */
export interface REPLConfig {
  /** Model to use */
  model?: string;
  /** System instructions */
  instructions?: string;
  /** Enable streaming */
  streaming?: boolean;
  /** Agent name to connect to (defaults to 'robutler') */
  agentName?: string;
  /** The agent file `-a` chose (`agent-files.ts`); null for the built-in agent. */
  agentFile?: string | null;
  /** The SDK's version, shown on the welcome card. */
  version?: string;
  /**
   * Whether the chat is at a terminal (spec W2): a test sets it to drive the
   * write commands. Defaults to `process.stdin.isTTY && process.stdout.isTTY`.
   */
  interactive?: boolean;
}

/**
 * Slash command handler
 */
interface SlashCommand {
  name: string;
  description: string;
  handler: (args: string) => Promise<void>;
}

/**
 * Everything `buildState` computes and `applyState` swaps in (spec 3.1). It
 * sets none of it, so a build that throws leaves the running agent whole.
 */
interface PreparedState {
  agent: BaseAgent;
  agentName: string;
  agentFile: string | undefined;
  agentDescription: string;
  configModel: string | undefined;
  modelAccess: ModelAccess | undefined;
  providerEnvVar: string | undefined;
  declaredLLM: LLMProvider | undefined;
  modelProblem: string | undefined;
  sessionBackend: 'local' | 'robutler';
  canChangeFiles: boolean;
  proxyUrl: string;
  toolNames: string[];
  /** The loaded version's fingerprint (`agent-reload.ts`); undefined for the built-in agent. */
  loaded: LoadedAgent | undefined;
  /** The skill names the agent file declares (the built-in one's too), for `/skills` (2026-09-27). */
  declaredSkills: string[];
  /** The file's `access:` block, parsed (ADR-0045), for `/access`; undefined without one. */
  accessPolicy: AccessPolicy | undefined;
}

/** An agent's tool names, sorted, so the reload diff reads the same however they were registered. */
/** The conversation's saved cost (`saveConversation`), for /resume; nothing when the file has none. */
function savedCost(metadata: Record<string, unknown> | undefined): RunningCost {
  const credits = metadata?.cost_credits;
  if (typeof credits !== 'number' || !Number.isFinite(credits)) return NO_COST;
  return { credits, estimated: metadata?.cost_estimated === true, known: true };
}

function toolNamesOf(agent: BaseAgent): string[] {
  const registry = (agent as unknown as { toolRegistry?: Map<string, { name: string }> }).toolRegistry;
  return registry ? [...registry.values()].map((t) => t.name).filter((name) => !TRANSIENT_TOOLS.has(name)).sort() : [];
}

/**
 * Who may use a tool, from its scopes (`/tools`, interactive-mode spec 3.7,
 * the same words as the Python chat): `only you` for a tool scoped to the
 * owner or admin, `you and {groups}` for one an `access.tools` grant named,
 * `every caller` for one open to all (or with no scope at all).
 */
export function scopeLabel(scopes: readonly string[] | undefined): string {
  const list = (scopes ?? []).map(String);
  const groups = list.filter((s) => s.startsWith('group:')).map((s) => s.slice('group:'.length));
  if (groups.length) return fill('scopeYouAnd', { groups: groups.join(', ') });
  if (!list.length || list.includes('all') || list.includes('none')) return CHAT_WORDS.scopeEveryCaller;
  if (list.every((s) => s === 'owner' || s === 'admin')) return CHAT_WORDS.scopeOnlyYou;
  return list.join(', ');
}

/** A pattern as the file wrote it, for `/access`. */
function patternText(pattern: { kind: string; value: string }): string {
  return pattern.kind === 'domain' ? `agent:*.${pattern.value}` : `${pattern.kind}:${pattern.value}`;
}

/** A group's members and trust, in one phrase (`/access`). */
function groupText(rule: GroupRule): string {
  const trust = rule.trust
    ? rule.trust.topic
      ? fill('accessTrust', { min: rule.trust.min, topic: rule.trust.topic })
      : fill('accessTrustOverall', { min: rule.trust.min })
    : '';
  if (rule.members === null) return trust;
  const members = rule.members.length ? rule.members.map(patternText).join(', ') : CHAT_WORDS.accessNobody;
  return trust ? `${members}, ${trust}` : members;
}

/** How many files an installed SKILL.md skill has, for `remove {skill} from .agents/skills ({n} files)`. */
function installedFileCount(folder: string, name: string): number {
  try {
    const lock = JSON.parse(fs.readFileSync(path.join(folder, '.webagents', 'skills.lock'), 'utf8')) as {
      skills?: Record<string, { files?: string[] }>;
    };
    return lock.skills?.[name]?.files?.length ?? 0;
  } catch {
    return 0;
  }
}

/**
 * Interactive REPL for agent conversations
 */
/**
 * A control file's path as the person reads it in the chat's question
 * (the ptypass-fixes lane, 2026-09-27): relative to this folder when it is
 * inside it (`AGENT.md`), else under `~`, else as given. The file tools name
 * a file as the model did, often absolutely, and the question wrapped such a
 * path mid-name across three lines. The Python chat's `_control_path` is
 * the twin.
 */
export function controlPath(file: string, folder: string = process.cwd()): string {
  const absolute = path.resolve(folder, file);
  const relative = path.relative(folder, absolute);
  if (relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)) return relative;
  return shortPath(absolute);
}

export class InteractiveREPL {
  private config: REPLConfig;
  private agent: BaseAgent | null = null;
  private messages: Message[] = [];
  private commands: Map<string, SlashCommand> = new Map();
  private running = false;
  /**
   * Lines typed at the prompt, newest first, carried from one prompt to the
   * next, and kept on disk per profile, owner-only (`chat-history.ts`, S-291).
   */
  private inputHistory: string[] = [];
  private readonly historyFile = chatHistoryFile();
  /** Colours and what the terminal can do; `run()` refines it with the terminal's real background. */
  private theme: Theme = themeFor(process.stdout);
  /**
   * What the terminal shows, kept while the chat runs at a terminal, so the
   * `/` menu can open over the conversation instead of scrolling it
   * (`ui/screen.ts`).
   */
  private screen: ScreenRecord | undefined;
  /** The agent file's one-line description, for the welcome card. */
  private agentDescription = '';
  /** The key the agent's model provider reads, for error hints. */
  private providerEnvVar: string | undefined;
  private sessionTokens = 0;
  private inputTokens = 0;
  private outputTokens = 0;
  /**
   * The conversation's cost in credits (plan item 2.4, `skills/llm/pricing.ts`):
   * what Robutler's models reported, or an estimate from the list prices for
   * a provider key. `sessionCost` is what THIS chat spent, for the goodbye
   * line, as `sessionTokens` is for tokens.
   */
  private cost: RunningCost = NO_COST;
  private sessionCost: RunningCost = NO_COST;
  private turns = 0;
  private readonly sessionStarted = Date.now();

  /** The conversation's id and start, for the file `sessions.ts` keeps it in. */
  private sessionId = newSessionId();
  private sessionCreatedAt = new Date().toISOString();

  /**
   * Where conversations are kept: always on this machine, and on Robutler too
   * when the agent file names `session: {backend: robutler}`
   * (`robutler-sessions.ts`). The chat keeps its conversations itself, so the
   * session skill is never loaded into the agent here.
   */
  private sessionBackend: 'local' | 'robutler' = 'local';
  /** The platform chat this conversation is recorded into, once it is. */
  private platformChatId: string | undefined;
  /** How many of `messages` are on Robutler. */
  private recordedCount = 0;
  /** Recording runs behind the conversation, one turn after another. */
  private recording: Promise<void> = Promise.resolve();
  /** Said once per conversation, before the next prompt: why it is not on Robutler. */
  private recordingProblem: string | undefined;
  private recordingNoticed = false;

  /**
   * The snapshot taken before each message of this conversation, oldest first
   * (`checkpoints.ts`), for `/undo`. Taken only when the agent can change
   * files: it has `filesystem` or `shell`.
   */
  private turnSnapshots: string[] = [];
  private canChangeFiles = false;
  private snapshotsNoticed = false;

  /** The agent file in use; undefined for the built-in agent. */
  private agentFile: string | undefined;

  /** The file's `access:` block, parsed, for `/access`; undefined without one. */
  private accessPolicy: AccessPolicy | undefined;

  /** The owner's note keys, read before each prompt for `/memory forget` completion. */
  private memoryKeys: string[] = [];

  /**
   * The version of the agent file the chat is running, for `/reload` and the
   * before-prompt notice (`agent-reload.ts`, spec 3.1); undefined for the
   * built-in agent, which has no file to read again.
   */
  private loaded: LoadedAgent | undefined;
  /** The skill names the agent file declares, the built-in one's too (`/skills`). */
  private declaredSkills: string[] = [];

  /** The running agent's tool names, sorted, so a reload can say which came and which went. */
  private toolNames: string[] = [];

  /** The version whose "changed since the chat loaded it" notice has already been shown, said once. */
  private versionNoticed: string | undefined;

  /** A file change seen during the last reply: `/reload shows what changed`, said once. */
  private changedDuringReply = false;

  /**
   * Whether this chat may ask before it changes a file: stdin and stdout are
   * terminals (spec W2). A test sets it through the `interactive` config; a
   * pipe leaves it false, and a command that writes refuses with the "not at
   * a terminal" sentence.
   */
  private interactive: boolean;

  /**
   * The agent file /agent chose: a path, null for the built-in agent, or
   * undefined to find it the usual way (`-a`, then AGENT.md).
   */
  private selectedFile: string | null | undefined;

  /**
   * `run()` has started the interactive loop (S-314, 2026-09-27). Only then
   * may a file-tool write to a control file ask the person at the terminal;
   * `-p` builds the same object and never sets it, so there the write is
   * refused, as it is under `serve` and the daemon.
   */
  private chatting = false;

  /**
   * How to take the live turn display down and put it back while a
   * question is asked mid-turn (the control-file prompt): set by the turn
   * that is running, cleared when it ends.
   */
  private turnPause?: { pause(): void; resume(): void };

  /**
   * A model the person chose, by `--model` or by `/model`, kept apart from the
   * REPL default (see load). Not readonly: `/model` sets it, because
   * `initialize()` re-applies the agent file's `model:` and would otherwise
   * undo the switch (2026-09-24).
   */
  private explicitModel: string | undefined;
  /**
   * `/rounds <n>`: this chat's tool-round budget, above the flag and the
   * file (2026-09-28, `core/tool-budget.ts`); kept across rebuilds.
   */
  private sessionRounds: number | undefined;

  /**
   * How the agent reaches its model, decided by `initialize()`
   * (`model-access.ts`): this machine's key, Robutler's models, or nothing.
   */
  private modelAccess: ModelAccess | undefined;

  /** The LLM skill the agent file names, when it names one it could build. */
  private declaredLLM: LLMProvider | undefined;

  /** Robutler's `/llm` socket, for error hints. */
  private proxyUrl = '';

  /**
   * Provider keys from the CLI's store (`provider-keys.ts`) or typed into the
   * offer, by variable name. Read once, and handed to the model client only:
   * never put into `process.env`, which the `shell` skill passes to every
   * command it runs.
   */
  private storedKeys: Record<string, string> | undefined;

  /**
   * Why the agent has no model to run on, as one sentence naming the ways
   * out; undefined when it has one. Set by `initialize()`. The CLI refuses
   * `-p` on it, and a chat at a terminal offers to fix it
   * (`offerModelAccess`).
   */
  modelProblem: string | undefined;

  constructor(config: REPLConfig = {}) {
    this.explicitModel = config.model;
    // No default model here: `initialize()` decides it from the keys and the
    // sign-in this machine has. It was a fixed 'gpt-4o', so OpenAI whatever
    // else was set up (2026-09-24).
    this.config = {
      streaming: true,
      ...config,
    };
    if (config.agentFile !== undefined) this.selectedFile = config.agentFile;
    this.interactive = config.interactive ?? Boolean(process.stdin.isTTY && process.stdout.isTTY);
    // Earlier chats' lines, oldest first on disk, newest first here.
    this.inputHistory = readChatHistory(this.historyFile).reverse();

    this.setupCommands();
  }

  /** Whether the chat may ask before it changes a file (spec W2): stdin and stdout are terminals. */
  private atTerminal(): boolean {
    return this.interactive;
  }
  
  /**
   * The commands, from `chat-commands.ts` (the list both SDKs share), each
   * bound to its handler below. A listed command without a handler is a bug
   * reported at construction, not a command that silently does nothing.
   */
  private setupCommands(): void {
    const handlers: Record<string, (args: string) => Promise<void>> = {
      help: async (args) => this.commandHelp(args),
      new: async () => this.startNewConversation(),
      clear: async () => this.clearScreen(),
      resume: async (args) => this.commandResume(args),
      undo: () => this.commandUndo(),
      rewind: (args) => this.commandRewind(args),
      reload: () => this.commandReload(),
      model: (args) => this.commandModel(args),
      rounds: (args) => this.commandRounds(args),
      agent: (args) => this.commandAgent(args),
      skills: (args) => this.commandSkills(args),
      tools: async () => this.commandTools(),
      mcp: async () => this.commandMcp(),
      access: async () => this.commandAccess(),
      cron: (args) => this.commandCron(args),
      memory: (args) => this.commandMemory(args),
      status: () => this.commandStatus(),
      login: () => this.signIn(),
      logout: () => this.commandLogout(),
      keys: (args) => this.commandKeys(args),
      secrets: (args) => this.commandSecrets(args),
      sandbox: () => this.commandSandbox(),
      publish: (args) => this.commandPublish(args),
      exit: async () => {
        this.running = false;
      },
    };
    for (const spec of CHAT_COMMANDS) {
      const handler = handlers[spec.name];
      if (!handler) throw new Error(`No handler for /${spec.name}`);
      this.registerCommand({ name: spec.name, description: spec.description, handler });
    }
  }

  /** `/help`: the commands by group and the keys; `/help <command>`: its usage, one line, and its forms. */
  private commandHelp(args: string): void {
    const { paint, palette } = this.theme;
    const asked = args.trim();
    if (asked) {
      const spec = chatCommand(asked);
      if (!spec) {
        this.sayUnknownCommand(asked);
        return;
      }
      this.notice('info', spec.usage, [spec.description, ...spec.details].join('\n'));
      return;
    }
    const width = Math.max(...CHAT_COMMANDS.map((c) => c.usage.length)) + 2;
    const lines: string[] = [];
    for (const [group, heading] of CHAT_GROUPS) {
      lines.push(paint.bold(paint.fg(palette.text, heading)));
      for (const c of CHAT_COMMANDS) {
        if (c.group !== group) continue;
        lines.push(`  ${paint.fg(palette.accent, c.usage.padEnd(width))}${paint.fg(palette.muted, c.description)}`);
      }
      lines.push('');
    }
    lines.push(paint.bold(paint.fg(palette.text, 'Keys')));
    for (const [key, what] of CHAT_KEYS) {
      lines.push(`  ${paint.fg(palette.text, key.padEnd(width))}${paint.fg(palette.muted, what)}`);
    }
    // Wrapped at words: the terminal broke this line wherever its width fell
    // ("web / agents", 2026-09-26), where the Python chat wraps at a space.
    lines.push('', ...wrapStyled(paint.fg(palette.faint, OUTSIDE_THE_CHAT), terminalColumns() - 1));
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** `✗ Unknown command /x.`, with a did-you-mean for a moved name (`/edit`) or a near one. */
  private sayUnknownCommand(name: string): void {
    const bare = name.replace(/^\//, '').toLowerCase();
    const moved = MOVED_COMMANDS[bare];
    if (moved) {
      this.notice('error', fill('unknownCommand', { name: bare }), `Did you mean /${moved}? Type / to see the commands.`);
      return;
    }
    const near = suggestSimilar(bare, CHAT_COMMANDS.map((c) => c.name)).match(/Did you mean (\w+)\?/);
    if (near) {
      this.notice('error', fill('unknownCommand', { name: bare }), fill('didYouMean', { name: near[1] }));
      return;
    }
    this.notice('error', fill('unknownCommand', { name: bare }), CHAT_WORDS.unknownCommandHint);
  }

  /** `/new`: an empty conversation with a new id; the old one stays saved for /resume. */
  private startNewConversation(say = true): void {
    this.messages = [];
    this.sessionId = newSessionId();
    this.sessionCreatedAt = new Date().toISOString();
    this.platformChatId = undefined;
    this.recordedCount = 0;
    this.recordingNoticed = false;
    this.turnSnapshots = [];
    // The conversation's tokens start over; `sessionTokens`, what THIS chat
    // has spent, does not (the goodbye line, 2026-09-26).
    this.inputTokens = 0;
    this.outputTokens = 0;
    this.cost = NO_COST;
    if (say) this.notice('ok', 'Started a new conversation.');
  }

  /** `/clear`: a new conversation on a clean screen, under the agent's card. */
  private clearScreen(): void {
    this.startNewConversation(false);
    if (process.stdout.isTTY) {
      // Screen and scrollback, then the card: what a fresh start looks like.
      process.stdout.write('\x1b[2J\x1b[3J\x1b[H');
      console.log(`${welcomeCard(this.theme, terminalColumns(), this.welcomeInfo()).join('\n')}\n`);
    } else {
      console.clear();
    }
  }

  /** The folder the conversation belongs to: the agent file's, or where the chat started. */
  private agentFolder(): string {
    return this.agentFile ? path.dirname(this.agentFile) : process.cwd();
  }

  /** Where this folder's conversations with this agent are kept (`sessions.ts`). */
  private sessionDir(): string {
    return sessionsDir(this.agentFolder(), this.agent?.name ?? 'agent');
  }

  /** Saved after every turn, so /resume finds it after a crash as well as after /exit. */
  private saveConversation(): void {
    if (!this.messages.length) return;
    try {
      saveSession(this.sessionDir(), {
        session_id: this.sessionId,
        agent_name: this.agent?.name ?? 'agent',
        created_at: this.sessionCreatedAt,
        updated_at: '',
        messages: this.messages as unknown as SessionMessage[],
        metadata: {
          model: this.modelLabel(),
          sdk: 'typescript',
          ...(this.platformChatId ? { robutler_chat_id: this.platformChatId, robutler_recorded: this.recordedCount } : {}),
          // The conversation's cost, so /resume shows it again (plan item 2.4).
          ...(this.cost.known ? { cost_credits: this.cost.credits, cost_estimated: this.cost.estimated } : {}),
        },
        input_tokens: this.inputTokens,
        output_tokens: this.outputTokens,
      });
    } catch {
      // A conversation that cannot be saved is still a conversation.
    }
  }

  /**
   * Where this agent's conversations are kept on Robutler, as whom, or why
   * they cannot be (`robutler-sessions.ts`): the person's sign-in, and the
   * folder's link to the agent `webagents publish` made.
   */
  private async conversationsTarget(): Promise<ConversationsTarget | Unavailable> {
    const { getToken } = await import('./credentials.js');
    const token = await getToken();
    if (!token) return 'signed_out';
    const agentId = linkedPlatformAgent(this.agentFolder(), this.agent?.name ?? '');
    if (!agentId) return 'not_published';
    return { base: resolvePlatformUrl()[0], token, agentId };
  }

  /**
   * After a turn: record what the conversation added into the person's chat
   * with the agent on Robutler, behind the conversation (the next prompt does
   * not wait). A problem is said once, before the next prompt.
   */
  private recordOnRobutler(): void {
    if (this.sessionBackend !== 'robutler') return;
    const words = wordsSince(this.messages, this.recordedCount);
    if (!words.length) return;
    const sessionId = this.sessionId;
    const dir = this.sessionDir();
    const upTo = this.messages.length;
    const chatId = this.platformChatId;
    this.recording = this.recording.then(async () => {
      const target = await this.conversationsTarget();
      if (typeof target === 'string') {
        this.recordingProblem = unavailableReason(target, cliCommand);
        return;
      }
      try {
        const recordedInto = await recordWords(target, sessionId, chatId, words, cliCommand);
        if (!recordedInto) return;
        if (sessionId === this.sessionId) {
          this.platformChatId = recordedInto;
          this.recordedCount = upTo;
          this.saveConversation();
        } else {
          // The person moved on (/new, /resume): note it on the one recorded.
          markRecorded(dir, sessionId, recordedInto, upTo);
        }
      } catch (error) {
        this.recordingProblem = `This conversation is not being kept on Robutler: ${(error as Error).message}.`;
      }
    });
  }

  /** Before a prompt: the recording problem, once per conversation. */
  private sayRecordingProblem(): void {
    if (!this.recordingProblem) return;
    if (!this.recordingNoticed) this.notice('warn', this.recordingProblem);
    this.recordingNoticed = true;
    this.recordingProblem = undefined;
  }

  /**
   * Before a message: a snapshot of the folder, for `/undo`, when the agent can
   * change files. Never in the home folder or above (`snapshotsOffReason`),
   * and quiet about it until `/undo` is asked for. A snapshot that cannot be
   * taken is said once, and the message goes ahead.
   */
  private async snapshotBeforeTurn(message: string): Promise<void> {
    if (!this.canChangeFiles) return;
    const folder = this.agentFolder();
    if (snapshotsOffReason(folder)) return;
    try {
      // Taken once the spinner is up, and yielding while it walks: run
      // straight through before the spinner, a large folder froze the chat
      // with no sign of life after Enter (2026-09-28). It still finishes
      // before the model is asked anything, so no tool can change a file first.
      this.turnSnapshots.push((await takeSnapshotYielding(folder, turnLabel(message))).id);
    } catch (error) {
      if (!this.snapshotsNoticed) this.notice('warn', UNDO_WORDS.snapshotFailed((error as Error).message));
      this.snapshotsNoticed = true;
    }
  }

  /** A yes to `question`, asked only at a terminal (as `/publish` asks). */
  private async confirm(question: string): Promise<boolean> {
    if (!this.atTerminal()) return false;
    const { paint, palette } = this.theme;
    const answer = await promptLine(`  ${paint.fg(palette.text, question)}`);
    return /^y(es)?$/i.test((answer ?? '').trim());
  }

  /**
   * The file tools' question before they change one of the agent's control
   * files (S-314, 2026-09-27; `skills/filesystem/agent-secrets-guard.ts`):
   * the diff, then a yes or no from the person at the terminal, with the
   * live turn display paused so the question is not drawn over. Anything
   * but the interactive chat at a terminal answers no, and the tool says
   * the owner declined.
   */
  private async confirmControlWrite(file: string, diff: string): Promise<boolean> {
    if (!this.chatting || !this.atTerminal()) return false;
    const pause = this.turnPause;
    pause?.pause();
    try {
      const { paint, palette } = this.theme;
      // The path as this folder knows it (`controlPath`), in the header, the
      // diff's own two header lines and the answer: an absolute path wrapped
      // mid-name across three lines (the ptypass-fixes lane, 2026-09-27).
      const shown = controlPath(file);
      const width = terminalColumns() - 1;
      console.log(`\n${wrapStyled(paint.fg(palette.warning, CONTROL_HEADER.replace('{path}', shown)), width, '  ', '  ').join('\n')}`);
      for (const line of diff.split('\n')) {
        const header = /^(---|\+\+\+) /.exec(line);
        const text = header ? `${header[1]} ${shown}` : line;
        const colour = header ? palette.muted : line.startsWith('+') ? palette.success : line.startsWith('-') ? palette.error : palette.muted;
        console.log(`  ${paint.fg(colour, text)}`);
      }
      console.log();
      const allowed = await this.confirm(CONTROL_QUESTION);
      this.answerLine(allowed ? ANSWER_WORDS.control.yes : ANSWER_WORDS.control.no, { path: shown });
      return allowed;
    } finally {
      pause?.resume();
    }
  }

  /** The one line that says what was decided, under a mid-turn question and its answer (`ANSWER_WORDS`, the Python chat's too). */
  private answerLine(template: string, values: Record<string, string>): void {
    const { paint, palette } = this.theme;
    let text: string = template;
    for (const [key, value] of Object.entries(values)) text = text.split(`{${key}}`).join(value);
    const mark = text.slice(0, 1);
    console.log(`  ${paint.fg(mark === '✓' ? palette.success : palette.warning, mark)}${paint.fg(palette.muted, text.slice(1))}`);
  }

  /**
   * A notice printed during a turn, above its live region (the ptypass-fixes
   * lane, 2026-09-27): after `always`, "Added ... to network.hosts" printed
   * under the resumed live frame and the next redraw erased it (the PTY
   * pass, `08`). The region is taken down for the notice and put back.
   */
  private aboveTurn(show: () => void): void {
    const pause = this.turnPause;
    pause?.pause();
    try {
      show();
    } finally {
      pause?.resume();
    }
  }

  /** Show what a restore would do, ask, and do it. */
  private async confirmAndRestore(
    folder: string,
    target: Manifest,
    plan: RestorePlan,
    header: string,
    beforeLabel: string,
    done?: () => void,
  ): Promise<void> {
    const { paint, palette } = this.theme;
    console.log(`\n  ${paint.fg(palette.text, header)}`);
    for (const line of planLines(plan)) console.log(paint.fg(palette.muted, line));
    if (target.partial) console.log(paint.fg(palette.muted, `  ${UNDO_WORDS.partialNote}`));
    console.log();
    if (!(await this.confirm(UNDO_WORDS.confirm))) {
      this.notice('info', UNDO_WORDS.leftAsIs);
      return;
    }
    const result = restoreSnapshot(folder, target.id, beforeLabel);
    done?.();
    if (result.written.length || result.removed.length) {
      this.notice('ok', restoredSentence(result.written.length, result.removed.length));
    }
    for (const failure of result.failed) this.notice('warn', UNDO_WORDS.failed(failure.path, failure.reason));
  }

  /** `/undo`: put back what the last message changed (`checkpoints.ts`). */
  private async commandUndo(): Promise<void> {
    const folder = this.agentFolder();
    const off = snapshotsOffReason(folder);
    if (off && this.canChangeFiles) {
      this.notice('info', off);
      return;
    }
    const id = this.turnSnapshots[this.turnSnapshots.length - 1];
    const store = checkpointsDir(folder);
    const all = id ? listCheckpoints(store) : [];
    const target = all.find((m) => m.id === id);
    if (!target) {
      if (id) this.turnSnapshots.pop();
      this.notice('info', UNDO_WORDS.nothingToUndo);
      return;
    }
    const plan = planRestore(target, scanFolder(folder, store, all[0]));
    if (!planChangesAnything(plan)) {
      this.turnSnapshots.pop();
      this.notice('ok', UNDO_WORDS.nothingChanged);
      return;
    }
    await this.confirmAndRestore(folder, target, plan, UNDO_WORDS.undoHeader, 'before /undo', () => this.turnSnapshots.pop());
  }

  /** `/rewind`: this folder's snapshots; `/rewind <number>`: put the folder back as that one has it. */
  private async commandRewind(args: string): Promise<void> {
    const { paint, palette } = this.theme;
    const folder = this.agentFolder();
    const store = checkpointsDir(folder);
    const all = listCheckpoints(store);
    if (!all.length) {
      this.notice('info', UNDO_WORDS.noSnapshots);
      return;
    }
    const pick = args.trim();
    if (!pick) {
      const lines = [paint.bold(paint.fg(palette.text, UNDO_WORDS.rewindTitle))];
      all.slice(0, 9).forEach((m, i) => {
        lines.push(
          `  ${paint.fg(palette.accent, String(i + 1))}  ${paint.fg(palette.muted, whenLabel(m.created_at).padEnd(12))}${paint.fg(palette.text, truncate(m.label, terminalColumns() - 20))}`,
        );
      });
      lines.push('', paint.fg(palette.faint, `  ${UNDO_WORDS.rewindHint}`));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    const target = /^\d+$/.test(pick) ? all[Number(pick) - 1] : undefined;
    if (!target) {
      this.notice('error', UNDO_WORDS.rewindMissing(pick), UNDO_WORDS.rewindMissingHint);
      return;
    }
    const plan = planRestore(target, scanFolder(folder, store, all[0]));
    if (!planChangesAnything(plan)) {
      this.notice('ok', UNDO_WORDS.rewindSame);
      return;
    }
    await this.confirmAndRestore(folder, target, plan, rewindHeader(whenLabel(target.created_at), target.label), 'before /rewind');
  }

  /** `/resume`: the earlier conversations, here and on Robutler; `/resume <number>`: continue one. */
  private async commandResume(args: string): Promise<void> {
    const { paint, palette } = this.theme;
    const dir = this.sessionDir();
    const name = this.agent?.name ?? 'the agent';
    let target: ConversationsTarget | undefined;
    let platform: PlatformConversation[] = [];
    if (this.sessionBackend === 'robutler') {
      const found = await this.conversationsTarget();
      if (typeof found === 'string') {
        this.notice('info', unavailableReason(found, cliCommand));
      } else {
        target = found;
        try {
          platform = await listPlatformConversations(found, cliCommand);
        } catch (error) {
          this.notice('warn', `Could not list the conversations on Robutler: ${(error as Error).message}.`);
        }
      }
    }
    const current = (e: { id?: string; chatId?: string }) =>
      this.messages.length > 0 && ((e.id !== undefined && e.id === this.sessionId) || (e.chatId !== undefined && e.chatId === this.platformChatId));
    const entries = mergeConversations(listSessions(dir), platform).filter((e) => !current(e));
    if (!entries.length) {
      this.notice(
        'info',
        this.sessionBackend === 'robutler' && target
          ? `No earlier conversations with ${name}, in this folder or on Robutler.`
          : `No earlier conversations with ${name} in this folder.`,
      );
      return;
    }
    const pick = args.trim();
    if (!pick) {
      const shown = entries.slice(0, 9);
      const width = (terminalColumns()) - 1;
      const lines = [paint.bold(paint.fg(palette.text, 'Earlier conversations'))];
      shown.forEach((e, i) => {
        const when = whenLabel(e.updatedAt).padEnd(12);
        const count = `${Math.max(e.localCount, e.platformCount)} messages`.padEnd(14);
        const room = Math.max(10, width - 32);
        const where = e.onlyOnRobutler ? paint.fg(palette.muted, 'Robutler: ') : '';
        const preview = truncate(e.preview || '(no text)', room - (e.onlyOnRobutler ? 'Robutler: '.length : 0));
        lines.push(
          `  ${paint.fg(palette.accent, String(i + 1))}  ${paint.fg(palette.muted, when)}${paint.fg(palette.faint, count)}${where}${paint.fg(palette.text, preview)}`,
        );
      });
      lines.push('', paint.fg(palette.faint, '  Continue one with /resume <number>.'));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    const index = /^\d+$/.test(pick)
      ? Number(pick) - 1
      : entries.findIndex((e) => (e.id ?? '').startsWith(pick) || (e.chatId ?? '').startsWith(pick));
    const chosen = entries[index];
    if (!chosen) {
      this.notice('error', `There is no conversation ${pick}.`, 'Type /resume to see the list.');
      return;
    }
    const local = chosen.id ? loadSession(dir, chosen.id) : null;
    // Robutler has more of it (continued on the web, or only there): read it from there.
    if (target && chosen.chatId && (!local || chosen.platformCount > chosen.localCount)) {
      let words: { role: 'user' | 'assistant'; content: string }[];
      try {
        words = await readPlatformConversation(target, chosen.chatId, cliCommand);
      } catch (error) {
        this.notice('error', `Could not read that conversation from Robutler: ${(error as Error).message}.`);
        return;
      }
      this.messages = words as unknown as Message[];
      this.sessionId = chosen.id ?? chosen.sessionId ?? newSessionId();
      this.sessionCreatedAt = local?.created_at || new Date().toISOString();
      this.inputTokens = local?.input_tokens ?? 0;
      this.outputTokens = local?.output_tokens ?? 0;
      this.cost = savedCost(local?.metadata);
      this.platformChatId = chosen.chatId;
      this.recordedCount = words.length;
      // Kept on this machine as well from now on.
      this.saveConversation();
    } else if (local) {
      this.messages = local.messages as unknown as Message[];
      this.sessionId = local.session_id;
      this.sessionCreatedAt = local.created_at;
      this.inputTokens = local.input_tokens;
      this.outputTokens = local.output_tokens;
      this.cost = savedCost(local.metadata);
      const chatId = local.metadata.robutler_chat_id;
      const recorded = local.metadata.robutler_recorded;
      this.platformChatId = typeof chatId === 'string' && chatId ? chatId : undefined;
      this.recordedCount = this.platformChatId && typeof recorded === 'number' ? recorded : 0;
    } else {
      this.notice('error', `There is no conversation ${pick}.`, 'Type /resume to see the list.');
      return;
    }
    this.recordingNoticed = false;
    this.turnSnapshots = [];
    // The resumed conversation's tokens are its own (/status, the footer);
    // they are not what this chat spent (the goodbye line said they were,
    // 2026-09-26).
    this.printRecap();
    this.notice('ok', `Continuing the conversation from ${whenLabel(chosen.updatedAt)} (${spokenCount(this.messages)} messages).`);
  }

  /** The last few exchanges of a resumed conversation, so it reads as a continuation. */
  private printRecap(): void {
    const { paint, palette } = this.theme;
    const columns = terminalColumns();
    const said = this.messages.filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim());
    const shown = said.slice(-6);
    const lines: string[] = ['', paint.fg(palette.faint, '── Earlier in this conversation ──')];
    if (said.length > shown.length) lines.push(paint.fg(palette.faint, `   … ${said.length - shown.length} earlier messages`));
    console.log(lines.join('\n'));
    for (const m of shown) {
      const text = String(m.content).trim();
      console.log();
      if (m.role === 'user') {
        console.log(sentMessage(this.theme, columns, text.length > 240 ? `${text.slice(0, 240)}…` : text).join('\n'));
      } else {
        const preview = text.split('\n').filter((l) => l.trim()).slice(0, 3);
        preview.forEach((line, i) => {
          const marker = i === 0 ? paint.fg(palette.agent, '✦ ') : '  ';
          console.log(`${marker}${paint.fg(palette.muted, truncate(line.trim(), columns - 3))}`);
        });
        if (text.split('\n').filter((l) => l.trim()).length > 3) console.log(paint.fg(palette.faint, '  …'));
      }
    }
    console.log(`\n${paint.fg(palette.faint, '── Continue below ──')}`);
  }

  /**
   * `/model`: which model and how it is reached; `/model <provider/model>`:
   * switch, for this chat; `--save` also keeps it in the agent file (spec 3.5).
   *
   * THE FILE'S PROVIDER DECIDES WHAT IT RUNS (D3, 2026-09-26). An agent
   * naming `openai` asked for `anthropic/claude-sonnet-4` used to print
   * "Model set to anthropic/claude-sonnet-4" and send the turn to OpenAI's
   * default on the OpenAI key (`modelForProvider` drops another provider's
   * model, and the label repeated what was typed); the Python chat quietly
   * kept its own model. Now such a switch is refused with the way out, and a
   * switch that is made is the one the label says.
   */
  private async commandModel(args: string): Promise<void> {
    const parts = args.trim().split(/\s+/).filter(Boolean);
    const save = parts.includes('--save');
    const rest = parts.filter((p) => p !== '--save');
    if (rest.length > 1 || rest.some((p) => p.startsWith('-'))) {
      this.notice('error', fill('usage', { usage: '/model [provider/model] [--save]' }));
      return;
    }
    if (!rest.length && !save) {
      this.notice('info', `Model: ${this.modelLabel() || '(none)'}`, 'Switch with /model <provider/model>.');
      return;
    }
    const { findProvider } = await import('../skills/llm/providers.js');
    const current = this.modelAccess?.kind !== 'none' ? this.modelAccess?.model : undefined;
    const wanted = rest[0] ?? current;
    if (!wanted) {
      this.notice('error', fill('usage', { usage: '/model [provider/model] [--save]' }));
      return;
    }
    if (save && !this.agentFile) {
      this.notice('error', CHAT_WORDS.builtInNoFileToKeep);
      return;
    }
    // The file names a provider's skill: it runs that provider's models, and
    // asking for another's changes nothing (D3). Robutler's proxy serves them all.
    const declared = this.declaredLLM;
    const slash = wanted.indexOf('/');
    const asked = slash > 0 ? findProvider(wanted.slice(0, slash)) : undefined;
    if (declared && declared.id !== 'proxy' && asked && asked.id !== declared.id) {
      this.notice(
        'error',
        fill('modelRefusal', { name: this.agent?.name ?? 'agent', provider: declared.id }),
        fill('modelRefusalHint', { provider: declared.id, other: asked.id }),
      );
      return;
    }
    const file = this.agentFile ? path.basename(this.agentFile) : '';
    if (wanted !== current || !save) {
      // REBUILD THE AGENT, do not just set the field: the already-built LLM
      // skill keeps its own model. Through `explicitModel`, because
      // `initialize()` re-applies the agent file's `model:` otherwise.
      const previous = this.explicitModel;
      this.explicitModel = wanted;
      try {
        await this.initialize();
        // A model with no way to run here (no key, not signed in) is not a
        // switch; keep the one that works.
        if (this.modelProblem) throw new Error(this.modelProblem);
      } catch (error) {
        this.explicitModel = previous;
        this.notice('error', `Could not switch model: ${(error as Error).message}`);
        await this.initialize().catch(() => {});
        return;
      }
      const model = this.modelAccess?.model ?? wanted;
      this.notice('ok', `Model set to ${this.modelLabel()}`, save || !this.agentFile ? undefined : fill('modelSwitchHint', { model, file }));
    }
    if (!save) return;
    await this.saveModel(this.modelAccess?.model ?? wanted);
  }

  /** `/model --save`: the model into the file's front matter (W2 to W6), then reload. */
  /** The running agent's tool-round budget, and where it came from. */
  private roundsNow(): { rounds: number; source: string } {
    const agent = this.agent as unknown as { maxToolIterations?: number; maxToolRoundsSource?: string } | null;
    return { rounds: agent?.maxToolIterations ?? DEFAULT_MAX_TOOL_ITERATIONS, source: agent?.maxToolRoundsSource ?? 'default' };
  }

  /** `/status`'s line: the budget and its source. */
  private roundsStatus(): string {
    const { rounds, source } = this.roundsNow();
    const file = this.agentFile ? path.basename(this.agentFile) : undefined;
    return roundsWords('status', { rounds, source: roundsSourceWords(source, file) });
  }

  /**
   * `/rounds`: the tool rounds one turn may run, and where that came from;
   * `/rounds <n>`: set it for this chat; `--save` also keeps it in the agent
   * file as `max_tool_rounds`, as `/model --save` keeps a model (2026-09-28,
   * `core/tool-budget.ts`). The Python chat's `cmd_rounds` says the same.
   */
  private async commandRounds(args: string): Promise<void> {
    const parts = args.split(/\s+/).filter(Boolean);
    const save = parts.includes('--save');
    const rest = parts.filter((p) => p !== '--save');
    const file = this.agentFile ? path.basename(this.agentFile) : undefined;
    if (rest.length > 1 || (save && rest.length === 0)) {
      this.notice('error', fill('usage', { usage: '/rounds [n] [--save]' }));
      return;
    }
    if (rest.length === 0) {
      const { rounds, source } = this.roundsNow();
      this.notice('info', roundsWords('show', { rounds, source: roundsSourceWords(source, file) }), roundsWords('showHint'));
      return;
    }
    let rounds: number;
    try {
      rounds = parseMaxToolRounds(rest[0], '/rounds');
    } catch (error) {
      this.notice('error', (error as Error).message);
      return;
    }
    if (save && !this.agentFile) {
      this.notice('error', CHAT_WORDS.builtInNoFileToKeep);
      return;
    }
    this.sessionRounds = rounds;
    const agent = this.agent as unknown as { maxToolIterations?: number; maxToolRoundsSource?: string } | null;
    if (agent) {
      agent.maxToolIterations = rounds;
      agent.maxToolRoundsSource = 'session';
    }
    if (!save) {
      this.notice('ok', roundsWords('set', { rounds }), file ? roundsWords('setHint', { rounds, file }) : undefined);
      return;
    }
    await this.saveRounds(rounds);
  }

  /** `/rounds <n> --save`: `max_tool_rounds` into the file's front matter, then reload. */
  private async saveRounds(rounds: number): Promise<void> {
    if (!this.agentFile) return;
    const file = path.basename(this.agentFile);
    if (!this.atTerminal()) {
      this.notice('error', fill('notAtTerminal', { command: 'rounds --save' }));
      return;
    }
    const unsafe = unsafeTargetReason(this.agentFile, this.agentFolder(), 'the chat');
    if (unsafe) {
      this.notice('error', unsafe);
      return;
    }
    let edit: ReturnType<typeof editFrontMatterScalar>;
    try {
      edit = editFrontMatterScalar(readAgentFile(this.agentFile), 'max_tool_rounds', String(rounds), this.agentFile);
    } catch (error) {
      this.notice('error', (error as Error).message);
      return;
    }
    if (!edit.changed) {
      this.notice('info', roundsWords('already', { file, rounds }));
      return;
    }
    const { paint, palette } = this.theme;
    console.log(`  ${paint.fg(palette.muted, `max_tool_rounds: ${rounds}`)}`);
    if (!(await this.confirm(CHAT_WORDS.makeThisChange))) {
      this.notice('info', CHAT_WORDS.leftAsIs);
      return;
    }
    this.snapshotForCommand('/rounds');
    writeByRename(this.agentFile, edit.text);
    // The file carries the budget now: it is no longer this chat's alone.
    this.sessionRounds = undefined;
    this.notice('ok', roundsWords('kept', { rounds, file }));
    await this.reloadNow();
  }

  private async saveModel(model: string): Promise<void> {
    if (!this.agentFile) return;
    const file = path.basename(this.agentFile);
    if (!this.atTerminal()) {
      this.notice('error', fill('notAtTerminal', { command: 'model --save' }));
      return;
    }
    const folder = this.agentFolder();
    const unsafe = unsafeTargetReason(this.agentFile, folder, 'the chat');
    if (unsafe) {
      this.notice('error', unsafe);
      return;
    }
    let edit: ReturnType<typeof editFrontMatterScalar>;
    try {
      edit = editFrontMatterScalar(readAgentFile(this.agentFile), 'model', model, this.agentFile);
    } catch (error) {
      this.notice('error', (error as Error).message);
      return;
    }
    if (!edit.changed) {
      this.notice('info', `${file} already keeps ${fill('planModel', { model })}.`);
      return;
    }
    const { paint, palette } = this.theme;
    console.log(`  ${paint.fg(palette.muted, fill('planModel', { model }))}`);
    if (!(await this.confirm(CHAT_WORDS.makeThisChange))) {
      this.notice('info', CHAT_WORDS.leftAsIs);
      return;
    }
    this.snapshotForCommand('/model');
    writeByRename(this.agentFile, edit.text);
    // The file carries the model now: the switch is no longer this chat's alone.
    this.explicitModel = undefined;
    this.notice('ok', fill('modelKept', { model, file }));
    await this.reloadNow();
  }

  /** The agent files in this folder: AGENT.md and AGENT-<name>.md, with their names. */
  private folderAgents(): FolderAgent[] {
    return folderAgents(process.cwd());
  }

  /**
   * `/agent`: this folder's agents and the built-in one; `/agent <name>`:
   * switch; `/agent new <name>` and `/agent edit [name]` (spec 3.2, 3.3). A
   * bare `new` or `edit` switches to an agent of that name, as any other does.
   */
  private async commandAgent(args: string): Promise<void> {
    const trimmed = args.trim();
    const [verb, ...rest] = trimmed.split(/\s+/).filter(Boolean);
    if (verb === 'new' && rest.length) return this.commandAgentNew(rest);
    if (verb === 'edit') return this.commandAgentEdit(rest);

    const { paint, palette } = this.theme;
    const agents = this.folderAgents();
    const current = this.agent?.name;
    const wanted = trimmed;
    if (!wanted) {
      const width = Math.max(10, ...agents.map((a) => a.name.length), BUILT_IN_AGENT.length) + 2;
      const lines = [paint.bold(paint.fg(palette.text, 'Agents'))];
      const row = (name: string, where: string, description: string) => {
        const mark = name === current ? paint.fg(palette.accent, '●') : paint.fg(palette.faint, '○');
        return `  ${mark} ${paint.bold(paint.fg(palette.text, name.padEnd(width)))}${paint.fg(palette.faint, where.padEnd(18))}${paint.fg(palette.muted, truncate(description, Math.max(10, (terminalColumns()) - width - 24)))}`;
      };
      for (const a of agents) {
        // A file that does not load is listed with its sentence, never hidden (D2).
        if (a.problem) lines.push(`  ${paint.bold(paint.fg(palette.error, '✗'))} ${paint.fg(palette.text, a.name)}  ${paint.fg(palette.faint, path.basename(a.file))}  ${paint.fg(palette.muted, a.problem)}`);
        else lines.push(row(a.name, path.basename(a.file), a.description));
      }
      lines.push(row(BUILT_IN_AGENT, 'built in', 'The general assistant'));
      lines.push('', paint.fg(palette.faint, '  Switch with /agent <name>.'));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    const target = agents.find((a) => a.name === wanted);
    if (!target && wanted !== BUILT_IN_AGENT) {
      this.notice('error', fill('noAgentCalled', { name: wanted }), CHAT_WORDS.seeTheList);
      return;
    }
    // A broken agent is not switched to: the chat keeps the one it has (D2).
    if (target?.problem) {
      this.notice('error', target.problem, fill('keepsTalking', { name: current ?? BUILT_IN_AGENT }));
      return;
    }
    if (wanted === current) {
      this.notice('info', `Already talking to ${current}.`);
      return;
    }
    await this.switchTo(target ? target.file : null, wanted);
  }

  /** Switch to `file` (null: the built-in agent), rebuild, and greet, keeping the running agent if the build fails (D2). */
  private async switchTo(file: string | null, wanted: string): Promise<void> {
    const previousFile = this.selectedFile;
    const previousModel = this.explicitModel;
    this.selectedFile = file;
    this.explicitModel = undefined;
    try {
      await this.initialize();
    } catch (error) {
      this.selectedFile = previousFile;
      this.explicitModel = previousModel;
      await this.initialize().catch(() => {});
      this.notice('error', (error as Error).message, fill('keepsTalking', { name: this.agent?.name ?? BUILT_IN_AGENT }));
      return;
    }
    this.startNewConversation(false);
    if (process.stdout.isTTY) {
      console.log(`\n${welcomeCard(this.theme, terminalColumns(), this.welcomeInfo()).join('\n')}`);
    }
    this.notice('ok', fill('nowTalking', { name: this.agent?.name ?? wanted }), this.modelProblem);
  }

  /** After the card, on the built-in agent in a folder with no agent file: `/agent new <name> makes one`. */
  private sayNewAgentTip(): void {
    if (this.agentFile) return;
    if (this.folderAgents().length) return;
    const { paint, palette } = this.theme;
    console.log(`${paint.fg(palette.faint, CHAT_WORDS.tip)}\n`);
  }

  /** Whether the agent's file differs from the loaded version now; false for the built-in agent or an unreadable file. */
  private fileChangedNow(): LoadedAgent | false {
    if (!this.agentFile || !this.loaded) return false;
    try {
      const parsed = parseAgentMarkdown(readAgentFile(this.agentFile), this.agentFile);
      const after = loadedAgentOf(this.agentFile, this.agentFolder(), parsed);
      return sameVersion(this.loaded, after) ? false : after;
    } catch {
      // A file that no longer parses: /reload says why; nothing here.
      return false;
    }
  }

  /** Before a prompt: the agent file changed since the chat loaded it (S-283); said once per version, never reloaded unasked. */
  private sayFileChanged(): void {
    const after = this.fileChangedNow();
    if (!after || !this.agentFile) return;
    if (this.versionNoticed === after.sha) return;
    this.versionNoticed = after.sha;
    const file = path.basename(this.agentFile);
    if (this.changedDuringReply) this.notice('warn', fill('changedDuringReply', { file }));
    else this.notice('info', fill('changedSinceLoaded', { file }));
    this.changedDuringReply = false;
  }

  /** `/reload`: read the agent file again and use it (spec 3.1), asking when a policy part changed. */
  private async commandReload(): Promise<void> {
    if (!this.agentFile || !this.loaded) {
      this.notice('info', CHAT_WORDS.builtInNoFileToRead);
      return;
    }
    const file = path.basename(this.agentFile);
    let after: LoadedAgent;
    try {
      const parsed = parseAgentMarkdown(readAgentFile(this.agentFile), this.agentFile);
      after = loadedAgentOf(this.agentFile, this.agentFolder(), parsed);
    } catch (error) {
      this.notice('error', (error as Error).message, CHAT_WORDS.keepsAgent);
      return;
    }
    if (sameVersion(this.loaded, after)) {
      this.notice('info', fill('upToDate', { name: this.agent?.name ?? 'agent', file }));
      return;
    }
    const diff = reloadDiff(this.loaded, after);
    const { paint, palette } = this.theme;
    console.log(`\n  ${paint.fg(palette.text, fill('changedHeader', { file }))}`);
    for (const part of diff.parts) console.log(`  ${paint.fg(palette.muted, part)}`);
    console.log();
    // A version the chat did not write, whose policy-bearing parts changed,
    // is taken only on a yes (S-283).
    if (diff.policyChanged && !(await this.confirm(CHAT_WORDS.useThisVersion))) {
      this.notice('info', CHAT_WORDS.keepsAgent);
      return;
    }
    await this.reloadNow();
  }

  /** Build the agent from its file again and swap it in, saying what changed; a failed build keeps the running agent. */
  private async reloadNow(): Promise<void> {
    const before = this.toolNames;
    const wasName = this.agent?.name;
    const file = this.agentFile ? path.basename(this.agentFile) : '';
    let state: PreparedState;
    try {
      state = await this.buildState();
    } catch (error) {
      this.notice('error', (error as Error).message, CHAT_WORDS.keepsAgent);
      return;
    }
    await this.applyState(state);
    const name = this.agent?.name ?? 'agent';
    const detail = this.modelProblem ?? (toolChangeLines(before, this.toolNames).join(' ') || undefined);
    this.notice('ok', fill('reloaded', { name, file }), detail);
    // A new `name:` starts a new conversation.
    if (wasName && name !== wasName) this.startNewConversation(false);
  }

  /** `/agent edit [name]`: open the agent's file in $VISUAL/$EDITOR, then use it (spec 3.2). */
  private async commandAgentEdit(rest: string[]): Promise<void> {
    if (rest.length > 1) {
      this.notice('error', fill('usage', { usage: '/agent edit [name]' }));
      return;
    }
    const name = rest[0];
    let file: string;
    if (name) {
      const target = this.folderAgents().find((a) => a.name === name);
      if (!target) {
        this.notice('error', fill('noAgentCalled', { name }), CHAT_WORDS.seeTheList);
        return;
      }
      file = target.file;
    } else {
      if (!this.agentFile) {
        this.notice('info', CHAT_WORDS.builtInNoFileToEdit, CHAT_WORDS.makeOneHint);
        return;
      }
      file = this.agentFile;
    }
    if (!this.atTerminal()) {
      this.notice('error', fill('notAtTerminal', { command: 'agent edit' }), this.scriptHint('webagents skills add'));
      return;
    }
    const folder = path.dirname(file);
    const unsafe = unsafeTargetReason(file, folder, 'the chat');
    if (unsafe) {
      this.notice('error', unsafe);
      return;
    }
    const editor = process.env.VISUAL || process.env.EDITOR;
    if (!editor) {
      this.notice('info', fill('fileIsAt', { file: path.basename(file), path: file }), CHAT_WORDS.setEditor);
      return;
    }
    const code = this.runEditor(editor, file);
    // Re-anchor the screen record: the editor scrolled the terminal.
    if (this.screen) {
      const row = await queryCursorRow();
      if (row !== null) this.screen.anchor(row);
    }
    if (code !== 0) {
      this.notice('error', fill('editorExited', { code }), CHAT_WORDS.keepsAgent);
      return;
    }
    // The current agent's file: reload as approved (the person saw the whole
    // file). Another agent's: saved, and how to switch to it.
    if (file === this.agentFile) {
      await this.reloadNow();
    } else {
      this.notice('ok', fill('saved', { file: path.basename(file) }), fill('switchHint', { name: name ?? '' }));
    }
  }

  /** Run the person's editor on `file`, stdio inherited; the exit code, or 1 when it could not start. */
  private runEditor(editor: string, file: string): number {
    const result = process.platform === 'win32'
      ? spawnSync('cmd.exe', ['/d', '/s', '/c', `${editor} "${file}"`], { stdio: 'inherit' })
      : spawnSync('/bin/sh', ['-c', `${editor} "$1"`, 'sh', file], { stdio: 'inherit' });
    return result.status ?? 1;
  }

  /** `/agent new <name> [chatbot|tool-agent]`: make an agent file here and switch to it (spec 3.3). */
  private async commandAgentNew(rest: string[]): Promise<void> {
    const [name, template = 'chatbot', ...extra] = rest;
    if (!name || extra.length) {
      this.notice('error', fill('usage', { usage: '/agent new <name> [chatbot|tool-agent]' }));
      return;
    }
    if (!AGENT_NAME_RE.test(name)) {
      this.notice('error', CHAT_WORDS.badName);
      return;
    }
    if (!INIT_TEMPLATES[template]) {
      this.notice('error', fill('unknownTemplate', { template }), CHAT_WORDS.templates);
      return;
    }
    const folder = process.cwd();
    if (this.folderAgents().some((a) => a.name === name)) {
      this.notice('error', fill('taken', { name }), fill('switchHint', { name }));
      return;
    }
    if (snapshotsOffReason(folder)) {
      this.notice('error', CHAT_WORDS.homeFolder, CHAT_WORDS.homeFolderHint);
      return;
    }
    if (!this.atTerminal()) {
      this.notice('error', fill('notAtTerminal', { command: 'agent new' }));
      return;
    }
    const hasAgent = this.folderAgents().length > 0 || fs.existsSync(path.join(folder, 'AGENT.md'));
    const file = path.join(folder, hasAgent ? `AGENT-${name}.md` : 'AGENT.md');
    const shown = path.basename(file);
    const unsafe = unsafeTargetReason(file, folder, 'the chat');
    if (unsafe) {
      this.notice('error', unsafe);
      return;
    }
    if (!(await this.confirm(fill('makeFile', { file: shown, template })))) {
      this.notice('info', CHAT_WORDS.leftAsIs);
      return;
    }
    this.snapshotForCommand('/agent new');
    // The provider the chat runs on with a key here, so the agent runs at once.
    const model = this.modelAccess?.kind === 'direct' && this.config.model ? this.config.model : undefined;
    fs.writeFileSync(file, agentMarkdown(name, template, model));
    this.notice('ok', fill('made', { file: shown }), CHAT_WORDS.madeHint);
    await this.switchTo(file, name);
  }

  /** Take a W5 snapshot before a command writes, so `/undo` can put it back; quiet where snapshots are off. */
  private snapshotForCommand(typed: string): void {
    const folder = this.agentFolder();
    if (snapshotsOffReason(folder)) return;
    try {
      this.turnSnapshots.push(takeSnapshot(folder, fill('beforeCommand', { command: typed })).id);
    } catch (error) {
      if (!this.snapshotsNoticed) this.notice('warn', UNDO_WORDS.snapshotFailed((error as Error).message));
      this.snapshotsNoticed = true;
    }
  }

  /** The "in a script, use ..." detail, omitted when there is no CLI twin for the action. */
  private scriptHint(command: string): string | undefined {
    return command ? fill('inAScript', { command }) : undefined;
  }

  /** `/skills`: this agent's skills; `/skills add|remove` change them; `/skills list` shows every name (spec 3.4). */
  private async commandSkills(args: string): Promise<void> {
    const parts = args.trim().split(/\s+/).filter(Boolean);
    const action = parts[0];
    const rest = parts.slice(1);
    if (!action) return this.commandSkillsShow();
    if (action === 'list') {
      const { skillmdListLines } = await import('./skills-edit.js');
      const { resolvableSkillNames } = await import('../skills/resolve.js');
      const { paint, palette } = this.theme;
      const lines = [paint.bold(paint.fg(palette.text, 'Skills an agent file can name:')), ''];
      for (const name of resolvableSkillNames()) lines.push(`  ${paint.fg(palette.text, name)}`);
      lines.push('', ...skillmdListLines(process.cwd()).map((l) => paint.fg(palette.muted, l)));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    if (action !== 'add' && action !== 'remove') {
      this.notice('error', fill('usage', { usage: '/skills [list|add|remove]' }));
      return;
    }
    if (!rest.length) {
      const usage = action === 'add'
        ? '/skills add <name>... | <owner/repo | git URL | folder> [--skill <name>]'
        : '/skills remove <name>...';
      this.notice('error', fill('usage', { usage }));
      return;
    }
    await this.commandSkillsEdit(action, rest);
  }

  /** `/skills`: this agent's coded skills and the folder's SKILL.md skills. */
  private commandSkillsShow(): void {
    const { paint, palette } = this.theme;
    if (!this.agentFile) {
      // The built-in agent has skills too (2026-09-27): it used to refuse
      // with "no agent file to change", which is /skills add's answer.
      const lines = [paint.bold(paint.fg(palette.text, fill('skillsOfBuiltIn', { name: this.agent?.name ?? BUILT_IN_AGENT })))];
      for (const name of this.declaredSkills) lines.push(`  ${paint.fg(palette.text, name)}`);
      if (!this.declaredSkills.length) lines.push(`  ${paint.fg(palette.faint, CHAT_WORDS.none)}`);
      lines.push('', paint.fg(palette.faint, `  ${CHAT_WORDS.builtInSkillsHint}`));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    const file = path.basename(this.agentFile);
    const lines = [paint.bold(paint.fg(palette.text, fill('skillsOf', { name: this.agent?.name ?? 'agent', file })))];
    const skills = this.loaded?.skills ?? [];
    for (const name of skills) lines.push(`  ${paint.fg(palette.text, name)}`);
    if (!skills.length) lines.push(`  ${paint.fg(palette.faint, CHAT_WORDS.none)}`);
    const md = this.loaded?.skillmd ?? [];
    if (md.length) {
      lines.push('', paint.fg(palette.text, CHAT_WORDS.skillmdHeading));
      for (const name of md) lines.push(`  ${paint.fg(palette.muted, fill('skillmdIn', { skill: name, folder: `.agents/skills/${name}` }))}`);
    }
    lines.push('', paint.fg(palette.faint, `  ${CHAT_WORDS.skillsHint}`));
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** `/skills add|remove`: plan, show, ask, snapshot, apply, reload (spec 3.4). */
  private async commandSkillsEdit(action: 'add' | 'remove', rest: string[]): Promise<void> {
    if (!this.agentFile) {
      this.notice('info', CHAT_WORDS.builtInNoFileToChange, CHAT_WORDS.makeOneHint);
      return;
    }
    const folder = this.agentFolder();
    // A `--skill` value and `-y`/`--yes` belong to a source; the chat never passes yes.
    const askedYes = rest.includes('--yes') || rest.includes('-y');
    const skillAt = rest.indexOf('--skill');
    const wantedSkill = skillAt !== -1 && rest[skillAt + 1] && !rest[skillAt + 1].startsWith('-') ? rest[skillAt + 1] : undefined;
    const names = rest.filter((token, i) => {
      if (token === '--yes' || token === '-y' || token === '--skill') return false;
      if (skillAt !== -1 && i === skillAt + 1) return false;
      return true;
    });

    const plan = planSkills(action, names, { file: this.agentFile, folder, who: 'the chat' });
    if (plan.errors.length) {
      for (const line of plan.errors) this.notice('error', line);
      return;
    }
    // A source install: W2, never --yes, then the CLI installer with the chat's confirm.
    if (plan.sources.length) {
      await this.installSources(plan.sources, wantedSkill, askedYes);
      return;
    }
    if (!plan.changed) {
      // Nothing to change: the file already names it, or does not.
      const file = path.basename(this.agentFile);
      if (plan.already.length) this.notice('info', `${file} already names ${spokenList(plan.already)}.`, `Skills: ${plan.skillsAfter.length ? plan.skillsAfter.join(', ') : 'none'}`);
      else this.notice('info', `${file} does not name ${spokenList(plan.absent)}.`, `Skills: ${plan.skillsAfter.length ? plan.skillsAfter.join(', ') : 'none'}`);
      return;
    }
    if (!this.atTerminal()) {
      this.notice('error', fill('notAtTerminal', { command: `skills ${action}` }), this.scriptHint(`webagents skills ${action} ${names.join(' ')}`));
      return;
    }
    // Show the change (W4).
    const { paint, palette } = this.theme;
    const changeLines: string[] = [];
    if (plan.add.length) changeLines.push(fill('planAdd', { names: plan.add.join(', ') }));
    if (plan.remove.length) changeLines.push(fill('planRemove', { names: plan.remove.join(', ') }));
    for (const name of plan.installedRemovals) {
      changeLines.push(fill('planRemoveInstalled', { skill: name, files: plural(installedFileCount(folder, name), 'file') }));
    }
    if (plan.add.length || plan.remove.length) changeLines.push(fill('planAfter', { list: plan.skillsAfter.length ? plan.skillsAfter.join(', ') : 'none' }));
    for (const line of changeLines) console.log(`  ${paint.fg(palette.muted, line)}`);
    if (!(await this.confirm(CHAT_WORDS.makeThisChange))) {
      this.notice('info', CHAT_WORDS.leftAsIs);
      return;
    }
    // W5, apply, W6.
    this.snapshotForCommand(`/skills ${action}`);
    const shown = path.basename(this.agentFile);
    const applied = applySkills(action, plan, folder);
    for (const name of applied.installedRemoved) this.notice('ok', `Removed ${name} from .agents/skills.`);
    if (applied.added.length) this.notice('ok', `Added ${spokenList(applied.added)} to ${shown}.`);
    if (applied.removed.length) this.notice('ok', `Removed ${spokenList(applied.removed)} from ${shown}.`);
    if (applied.added.length || applied.removed.length) {
      console.log(`  ${paint.fg(palette.muted, `Skills: ${applied.skillsAfter.length ? applied.skillsAfter.join(', ') : 'none'}`)}`);
    }
    await this.reloadNow();
  }

  /** A SKILL.md source: refuse --yes (W2), then the CLI installer with the chat's confirm; reload after. */
  private async installSources(sources: string[], skill: string | undefined, askedYes: boolean): Promise<void> {
    if (askedYes) {
      this.notice('error', CHAT_WORDS.alwaysAsks, CHAT_WORDS.alwaysAsksHint);
      return;
    }
    if (!this.atTerminal()) {
      this.notice('error', fill('notAtTerminal', { command: `skills add` }), this.scriptHint(`webagents skills add ${sources.join(' ')}`));
      return;
    }
    const { installFromSource, parseSource } = await import('../skills/skillmd/skillmd-install.js');
    const folder = this.agentFolder();
    let installedAny = false;
    let snapped = false;
    for (const source of sources) {
      const code = await installFromSource(parseSource(source), folder, {
        skill,
        yes: false,
        tty: true,
        confirm: async (question) => {
          const yes = await this.confirm(question);
          if (yes && !snapped) {
            this.snapshotForCommand('/skills add');
            snapped = true;
          }
          return yes;
        },
      }, { out: (line) => console.log(`  ${this.theme.paint.fg(this.theme.palette.muted, line)}`), err: (line) => this.notice('error', line) });
      if (code === 0) installedAny = true;
    }
    if (installedAny) await this.reloadNow();
  }

  /** `/tools`: what the agent can use, by name, and who else may (the scopes column, spec 3.7). */
  private commandTools(): void {
    const tools = this.toolList();
    if (!tools.length) {
      this.notice('info', 'This agent has no tools.', 'Add skills to its AGENT.md, for example `filesystem`.');
      return;
    }
    const { paint, palette } = this.theme;
    const width = Math.min(28, Math.max(...tools.map((t) => t.name.length))) + 2;
    const labels = tools.map((t) => scopeLabel(t.scopes));
    const scopeWidth = Math.max(...labels.map((l) => l.length)) + 2;
    const room = (terminalColumns()) - width - scopeWidth - 6;
    const lines = [paint.bold(paint.fg(palette.text, `Tools (${tools.length})`))];
    tools.forEach((tool, i) => {
      const description = (tool.description ?? '').split('\n')[0];
      lines.push(
        `  ${paint.fg(palette.success, '●')} ${paint.bold(paint.fg(palette.text, truncate(tool.name, width - 2).padEnd(width)))}${paint.fg(palette.faint, labels[i].padEnd(scopeWidth))}${paint.fg(palette.muted, truncate(description, Math.max(10, room)))}`,
      );
    });
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** `/access`: who may call this agent, and what each caller gets, from the parsed block (spec 3.7). */
  private commandAccess(): void {
    const { paint, palette } = this.theme;
    const policy = this.accessPolicy;
    if (!policy) {
      const yours = this.toolList()
        .filter((t) => scopeLabel(t.scopes) === CHAT_WORDS.scopeOnlyYou)
        .map((t) => t.name);
      this.notice('info', fill('noAccessBlock', { tools: yours.length ? yours.join(', ') : 'no tools' }), CHAT_WORDS.noAccessBlockHint);
      return;
    }
    const rows: Array<[string, string]> = [];
    if (policy.deny.length) rows.push([CHAT_WORDS.accessRefusedLabel, policy.deny.map(patternText).join(', ')]);
    for (const [group, rule] of policy.groups) rows.push([CHAT_WORDS.accessGroupsLabel, `${group}: ${groupText(rule)}`]);
    rows.push([CHAT_WORDS.accessOthersLabel, policy.default ? fill('accessOthersGroup', { group: policy.default }) : CHAT_WORDS.accessRefused]);
    // Tools by the set of groups that name them, so `shell, todo: you and staff` reads as one line.
    const byGroups = new Map<string, string[]>();
    for (const [group, names] of policy.tools) {
      for (const name of names) {
        const groups = [...policy.tools].filter(([, list]) => list.includes(name)).map(([g]) => g);
        const key = groups.join(', ');
        const list = byGroups.get(key) ?? [];
        if (!list.includes(name) && groups[0] === group) list.push(name);
        byGroups.set(key, list);
      }
    }
    for (const [groups, names] of byGroups) rows.push([CHAT_WORDS.accessToolsLabel, fill('accessToolsRow', { names: names.join(', '), groups })]);
    rows.push([CHAT_WORDS.accessToolsLabel, CHAT_WORDS.accessEveryOther]);
    const width = Math.max(...rows.map(([label]) => label.length)) + 3;
    const columns = terminalColumns() - 1;
    const lines = [paint.bold(paint.fg(palette.text, fill('accessTitle', { file: this.agentFile ? path.basename(this.agentFile) : 'AGENT.md' })))];
    let last = '';
    for (const [label, value] of rows) {
      const shown = label === last ? '' : label;
      last = label;
      lines.push(...wrapStyled(paint.fg(palette.text, value), columns, `  ${paint.fg(palette.muted, shown.padEnd(width))}`, ' '.repeat(width + 2)));
    }
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** `/mcp`: the servers the agent uses, from the skill's own report, values masked (spec 3.7, S-292). */
  private commandMcp(): void {
    const { paint, palette } = this.theme;
    const name = this.agent?.name ?? 'agent';
    const file = this.agentFile ? path.basename(this.agentFile) : 'AGENT.md';
    const skill = this.mcpSkill();
    const rows = skill?.serverReport() ?? [];
    if (!rows.length) {
      this.notice('info', fill('mcpNone', { name }), fill('mcpNoneHint', { file }));
      return;
    }
    const heading = skill?.configSource === 'mcp.json' ? CHAT_WORDS.mcpFromJson : fill('mcpFromFile', { file });
    const lines = [paint.bold(paint.fg(palette.text, heading))];
    for (const row of rows) {
      let text: string;
      if (row.rejected) text = fill('mcpRejected', { server: row.name, reason: row.rejected });
      else if (!row.connected) text = fill('mcpNotConnected', { server: row.name, transport: row.transport, error: row.error ?? 'not connected' });
      else {
        const shown = row.tools.slice(0, 3).join(', ');
        const more = row.tools.length > 3 ? fill('mcpMore', { count: row.tools.length - 3 }) : '';
        const tools = row.tools.length ? `${plural(row.tools.length, 'tool')}: ${shown}${more}` : plural(0, 'tool');
        text = fill('mcpConnected', { server: row.name, transport: row.transport, tools });
      }
      lines.push(...wrapStyled(paint.fg(row.connected ? palette.text : palette.muted, text), terminalColumns() - 1, '  ', '    '));
    }
    lines.push('', paint.fg(palette.faint, `  ${CHAT_WORDS.mcpServe}`));
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** `/cron`: this agent's schedules as the daemon runs them; `/cron run <name>`: one now, after asking (spec 3.7). */
  private async commandCron(args: string): Promise<void> {
    const parts = args.trim().split(/\s+/).filter(Boolean);
    const name = this.agent?.name ?? 'agent';
    const file = this.agentFile ? path.basename(this.agentFile) : 'AGENT.md';
    const folder = this.agentFolder();
    const { cronListAction, cronRunAction, folderSchedules } = await import('./cron-action.js');
    if (parts.length && (parts[0] !== 'run' || parts.length !== 2)) {
      this.notice('error', fill('usage', { usage: '/cron [run <name>]' }));
      return;
    }
    const { paint, palette } = this.theme;
    if (!parts.length) {
      if (!this.agentFile) {
        this.notice('info', fill('cronNone', { name }), fill('cronNoneHint', { file }));
        return;
      }
      const lines: string[] = [];
      await cronListAction({ watch: folder, agent: name }, { log: (line) => lines.push(line), error: (line) => this.notice('error', line) });
      if (!lines.length || lines[0].startsWith('No schedules')) {
        this.notice('info', fill('cronNone', { name }), fill('cronNoneHint', { file }));
        return;
      }
      console.log(`\n${lines.map((l) => `  ${paint.fg(palette.text, l)}`).join('\n')}\n\n  ${paint.fg(palette.faint, CHAT_WORDS.cronDaemon)}\n`);
      return;
    }
    const schedule = parts[1];
    const found = this.agentFile ? folderSchedules(folder, () => {}).find(({ definition }) => definition.name === name) : undefined;
    const entry = found?.schedules.find((s) => s.name === schedule);
    if (!entry) {
      this.notice('error', fill('cronNoSchedule', { schedule }), CHAT_WORDS.seeTheSchedules);
      return;
    }
    const target =
      entry.deliver.kind === 'file' ? entry.deliver.path : entry.deliver.kind === 'webhook' ? new URL(entry.deliver.url).host : CHAT_WORDS.cronChatTarget;
    if (!(await this.confirm(fill('cronRun', { schedule, target })))) {
      this.notice('info', CHAT_WORDS.notRun);
      return;
    }
    let outcome = '';
    const code = await cronRunAction(name, schedule, { watch: folder, agent: name }, { log: (line) => (outcome = line), error: (line) => this.notice('error', line) });
    if (outcome) this.notice(code === 0 ? 'ok' : 'error', outcome);
  }

  /** The agent's memory skill, when it has one. */
  private memorySkill(): { ownerSummary(): Promise<MemoryOwnerSummary>; hasOwnNote(key: string): Promise<boolean>; forgetOwn(key: string): Promise<boolean> } | undefined {
    return (this.agent as unknown as { skills?: Array<{ name?: string }> } | null)?.skills?.find((s) => s.name === 'memory') as
      | { ownerSummary(): Promise<MemoryOwnerSummary>; hasOwnNote(key: string): Promise<boolean>; forgetOwn(key: string): Promise<boolean> }
      | undefined;
  }

  /** `/memory`: what the agent remembers, and where; `/memory forget <key>`: one of your notes, after asking (spec 3.7). */
  private async commandMemory(args: string): Promise<void> {
    const parts = args.trim().split(/\s+/).filter(Boolean);
    const name = this.agent?.name ?? 'agent';
    if (parts.length && (parts[0] !== 'forget' || parts.length !== 2)) {
      this.notice('error', fill('usage', { usage: '/memory [forget <key>]' }));
      return;
    }
    const skill = this.memorySkill();
    if (!skill) {
      this.notice('info', fill('memoryNone', { name }), CHAT_WORDS.memoryNoneHint);
      return;
    }
    if (parts.length) {
      const key = parts[1];
      if (!(await skill.hasOwnNote(key))) {
        this.notice('error', fill('memoryNoNote', { key }));
        return;
      }
      if (!(await this.confirm(fill('memoryForget', { key })))) {
        this.notice('info', CHAT_WORDS.leftAsIs);
        return;
      }
      if (await skill.forgetOwn(key)) this.notice('ok', fill('memoryForgot', { key }));
      else this.notice('error', fill('memoryNoNote', { key }));
      return;
    }
    const summary = await skill.ownerSummary();
    const { paint, palette } = this.theme;
    // "on Robutler" only when the tier has the agent's key to get there (B5).
    const robutler = summary.portal && summary.portalKey !== false;
    const keptIn = summary.local
      ? `${CHAT_WORDS.memoryLocal}${robutler ? CHAT_WORDS.memoryAndRobutler : summary.portal ? CHAT_WORDS.memoryRobutlerNoKey : ''}`
      : robutler ? CHAT_WORDS.memoryRobutlerOnly : CHAT_WORDS.memoryRobutlerOnlyNoKey;
    const rows: Array<[string, string]> = [
      [CHAT_WORDS.memoryKeptInLabel, keptIn],
      [CHAT_WORDS.memoryYoursLabel, plural(summary.owner, 'note')],
      [CHAT_WORDS.memorySharedLabel, plural(summary.shared, 'note')],
      [CHAT_WORDS.memoryCallersLabel, fill('memoryCallers', { callers: plural(summary.callers, 'caller'), notes: plural(summary.callerNotes, 'note') })],
    ];
    const width = Math.max(...rows.map(([label]) => label.length)) + 3;
    const lines = [paint.bold(paint.fg(palette.text, fill('memoryTitle', { name })))];
    for (const [label, value] of rows) lines.push(`  ${paint.fg(palette.muted, label.padEnd(width))}${paint.fg(palette.text, value)}`);
    if (summary.recent.length) {
      lines.push('');
      const keyWidth = Math.min(32, Math.max(...summary.recent.map((n) => n.key.length))) + 2;
      for (const note of summary.recent) {
        lines.push(`  ${paint.fg(palette.text, truncate(note.key, keyWidth - 2).padEnd(keyWidth))}${paint.fg(palette.muted, truncate(note.firstLine, Math.max(10, terminalColumns() - keyWidth - 4)))}`);
      }
    }
    lines.push('', paint.fg(palette.faint, `  ${CHAT_WORDS.memoryHint}`));
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** Who the stored sign-in belongs to: the username, 'expired', or null when unknown. */
  private async whoAmI(portalUrl: string, token: string): Promise<string | 'expired' | null> {
    try {
      const res = await fetch(`${portalUrl}/api/users/me`, {
        headers: { Authorization: `Bearer ${token}` },
        signal: AbortSignal.timeout(4000),
      });
      if (res.status === 401) return 'expired';
      if (!res.ok) return null;
      const data = (await res.json()) as { user?: { username?: string }; username?: string };
      return data.user?.username ?? data.username ?? null;
    } catch {
      return null;
    }
  }

  /**
   * The agent and its model, for `webagents doctor`: the same decision the
   * chat just made, so the two never disagree.
   */
  doctorFacts(): {
    agent: string;
    agentFile?: string;
    modelOk: boolean;
    model: string;
    hasShell: boolean;
    /** The shell's resolved policy (the file's, the defaults or the opt-out), a declaration that could not be resolved, and where the policy came from. */
    sandbox?: import('../sandbox/policy.js').SandboxPolicy | null;
    sandboxError?: string | null;
    sandboxOrigin?: import('../sandbox/policy.js').SandboxOrigin;
    /** The state SKILL.md scripts run under, when the agent has such skills: they are confined commands too. */
    skillScripts?: string;
    /** Every MCP server the file named, as the skill reports it (S-292): never a value. */
    mcp?: import('../skills/mcp/skill.js').McpServerReportRow[];
    /** A local model (Ollama, plan item 2.8): the model and where it is reached, for doctor to probe. */
    localModel?: { model: string; baseUrl: string };
  } {
    const access = this.modelAccess;
    const shell = this.shellSkill();
    const localBase = access?.kind === 'direct' && access.provider?.credential === 'none' && access.model ? providerBaseUrl(access.provider) : undefined;
    const hasShell = Boolean(shell);
    const mcp = this.mcpSkill()?.serverReport();
    let model: string;
    if (this.modelProblem || !access || access.kind === 'none') {
      model = `none (${access?.kind === 'proxy' ? 'not signed in' : (access?.reason ?? 'no model')})`;
    } else if (access.kind === 'proxy') {
      model = `${this.modelLabel()}, paid from your Robutler credits`;
    } else {
      model = describeLocalRoute(access) ?? (this.providerEnvVar ? `${this.modelLabel()}, with your ${this.providerEnvVar}` : this.modelLabel());
    }
    return {
      agent: this.agent?.name ?? 'agent',
      ...(this.agentFile ? { agentFile: path.basename(this.agentFile) } : {}),
      modelOk: !this.modelProblem && Boolean(access) && access?.kind !== 'none',
      model,
      hasShell,
      ...(shell ? { sandbox: shell.policy, sandboxError: shell.sandboxError, sandboxOrigin: shell.sandboxOrigin } : {}),
      ...(this.skillScriptState() ? { skillScripts: this.skillScriptState() } : {}),
      ...(mcp ? { mcp } : {}),
      ...(localBase && access?.model ? { localModel: { model: access.model, baseUrl: localBase } } : {}),
    };
  }

  /** The agent's MCP skill, when it has one. */
  private mcpSkill(): { serverReport(): import('../skills/mcp/skill.js').McpServerReportRow[]; configSource?: 'config' | 'mcp.json' } | undefined {
    return (this.agent as unknown as { skills?: Array<{ name?: string }> } | null)?.skills?.find((s) => s.name === 'MCPSkill') as
      | { serverReport(): import('../skills/mcp/skill.js').McpServerReportRow[]; configSource?: 'config' | 'mcp.json' }
      | undefined;
  }

  /** How the agent reaches its model, in words: for /status. */
  private modelRoute(): string {
    const access = this.modelAccess;
    if (this.modelProblem || !access || access.kind === 'none') {
      const reason = access?.kind === 'proxy' ? 'not signed in' : (access?.reason ?? 'no model');
      return `none (${reason}). /login, or /keys set <NAME>.`;
    }
    if (access.kind === 'proxy') return `${this.modelLabel()}, paid from your Robutler credits`;
    // A local model says where it is reached (plan item 2.8).
    return describeLocalRoute(access) ?? (this.providerEnvVar ? `${this.modelLabel()}, with your ${this.providerEnvVar}` : this.modelLabel());
  }

  /** The `Robutler` row of /status (spec 3.6): published as what, or what stands in the way. */
  private async robutlerRow(signedIn: boolean): Promise<string> {
    if (!this.agentFile) return CHAT_WORDS.statusBuiltIn;
    const { projectLink } = await import('./publish.js');
    const link = projectLink(this.agentFolder());
    const linked = linkedPlatformAgent(this.agentFolder(), this.agent?.name ?? '');
    if (linked) return fill('statusPublished', { agentName: link.agentName ?? this.agent?.name ?? 'agent' });
    return signedIn ? CHAT_WORDS.statusNotPublished : CHAT_WORDS.statusSignedOut;
  }

  /** `/status`: the account, the agent, its model, the sandbox, the folder, the conversation, Robutler. */
  private async commandStatus(): Promise<void> {
    const { paint, palette } = this.theme;
    const { getToken } = await import('./credentials.js');
    const [portalUrl] = resolvePlatformUrl();
    const host = portalUrl.replace(/^https?:\/\//, '');
    const token = await getToken();
    let account = 'Not signed in. /login signs in.';
    if (token) {
      const who = await this.whoAmI(portalUrl, token);
      account = who === 'expired' ? `Sign-in expired on ${host}. /login signs in again.` : who ? `@${who} on ${host}` : `Signed in on ${host}`;
    }
    const name = this.agent?.name ?? 'agent';
    const tokens = this.inputTokens + this.outputTokens;
    const profile = profileName();
    // Where the sign-in and keys live and whether the next use makes macOS
    // ask: doctor's `keychain` line (keychain-ux, 2026-09-27).
    const keychain = await this.keychainRow();
    const rows: Array<[string, string]> = [
      ['Account', account],
      ...(profile ? [['Profile', profile] as [string, string]] : []),
      ...(keychain ? [['Keychain', keychain] as [string, string]] : []),
      ['Agent', this.agentFile ? `${name} (${path.basename(this.agentFile)})` : `${name} (built in)`],
      ['Model', this.modelRoute()],
      ['Tool rounds', this.roundsStatus()],
      ['Sandbox', (await this.sandboxSummary()).headline],
      ['Folder', shortPath(this.agentFolder())],
      [
        'Conversation',
        `${spokenCount(this.messages)} messages${tokens ? `, ${compactNumber(tokens)} tokens` : ''}` +
          (this.cost.known ? `, ${costWords(this.cost.credits, this.cost.estimated)}` : '') +
          (this.platformChatId ? ', also on Robutler' : ''),
      ],
      ['Robutler', await this.robutlerRow(Boolean(token))],
    ];
    const width = Math.max(...rows.map(([label]) => label.length)) + 3;
    const columns = (terminalColumns()) - 1;
    const lines = [paint.bold(paint.fg(palette.text, 'Status'))];
    for (const [label, value] of rows) {
      lines.push(...wrapStyled(paint.fg(palette.text, value), columns, `  ${paint.fg(palette.muted, label.padEnd(width))}`, ' '.repeat(width + 2)));
    }
    console.log(`\n${lines.join('\n')}\n`);
  }

  /** `/logout`: forget the stored sign-in, and say what the agent runs on now. */
  private async commandLogout(): Promise<void> {
    const { clearToken, getToken, leftBehind, TOKEN_ENV_VAR } = await import('./credentials.js');
    const { KeychainDialogBlocked } = await import('../skills/secrets/keychain-ux');
    const [portalUrl] = resolvePlatformUrl();
    if (!(await getToken())) {
      this.notice('info', 'Not signed in.');
      return;
    }
    try {
      await clearToken();
    } catch (error) {
      if (!(error instanceof KeychainDialogBlocked)) throw error;
      this.notice('error', error.message);
      return;
    }
    await this.initialize();
    if (process.env[TOKEN_ENV_VAR]) {
      this.notice('warn', `${TOKEN_ENV_VAR} is set in this shell, and it keeps you signed in.`, 'Unset it to sign out completely.');
      return;
    }
    this.notice('ok', `Signed out of ${portalUrl.replace(/^https?:\/\//, '')}.`, this.modelProblem ?? `${this.agent?.name ?? 'The agent'} runs on ${this.modelLabel()}.`);
    await this.leftBehindNotices(leftBehind());
  }

  /** The `Keychain` row of /status: the detail of doctor's `keychain` line, found without reading a value. */
  private async keychainRow(): Promise<string> {
    try {
      const { keychainFacts } = await import('./doctor');
      const { doctorLine } = await import('../skills/secrets/keychain-ux');
      return doctorLine(await keychainFacts()).detail;
    } catch {
      // A status row, never a failure.
      return '';
    }
  }

  /** An old `webagents:` item only a macOS dialog could remove, named with where to remove it (keychain-ux). */
  private async leftBehindNotices(items: Array<{ item: string; account: string }>): Promise<void> {
    const { leftBehindLines } = await import('./account');
    for (const line of await leftBehindLines(items)) this.notice('info', line);
  }

  /** `/keys`: each provider key and where it comes from; `set`/`unset` change the stored ones. */
  private async commandKeys(args: string): Promise<void> {
    const { paint, palette } = this.theme;
    const [verb = '', rawName = ''] = args.trim().split(/\s+/);
    const known = keyProviders().map((p) => p.envVar!);
    const name = rawName.toUpperCase();
    if (!verb) {
      const { providerKeyStore } = await import('./provider-keys.js');
      let backend = 'stored';
      try {
        backend = (await providerKeyStore()).status().backend === 'keystore' ? 'stored in your keychain' : 'stored in an owner-only file';
      } catch {
        // The label is a nicety.
      }
      const width = Math.max(...known.map((k) => k.length)) + 3;
      const lines = [paint.bold(paint.fg(palette.text, 'Model provider keys'))];
      for (const key of known) {
        const where = process.env[key] ? 'set in this shell' : this.storedKeys?.[key] ? backend : 'not set';
        const mark = where === 'not set' ? paint.fg(palette.faint, '○') : paint.fg(palette.success, '●');
        lines.push(`  ${mark} ${paint.fg(palette.text, key.padEnd(width))}${paint.fg(where === 'not set' ? palette.faint : palette.muted, where)}`);
      }
      lines.push('', paint.fg(palette.faint, '  /keys set <NAME> stores one; /keys unset <NAME> removes a stored one.'));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    if ((verb !== 'set' && verb !== 'unset') || !name) {
      this.notice('error', 'Usage: /keys [set|unset NAME]', `NAME is one of ${known.join(', ')}.`);
      return;
    }
    if (!known.includes(name)) {
      this.notice('error', `${name} is not a model provider key.`, `One of ${known.join(', ')}.`);
      return;
    }
    const { providerKeyStore, storeProviderKey } = await import('./provider-keys.js');
    if (verb === 'set') {
      // Ctrl+C cancels the entry, not the chat (D5): the TS chat used to end.
      const entered = await promptSecret(`  ${paint.fg(palette.muted, `${name} (hidden):`)} `, { onInterrupt: 'cancel' });
      if (entered === null) {
        this.notice('info', 'Nothing entered; nothing stored.');
        return;
      }
      const value = entered.trim();
      if (!value) {
        this.notice('info', 'Nothing entered; nothing stored.');
        return;
      }
      let backend: string;
      try {
        backend = await storeProviderKey(name, value);
      } catch (error) {
        this.notice('error', `Could not store ${name}: ${(error as Error).message}`);
        return;
      }
      this.storedKeys = { ...(this.storedKeys ?? {}), [name]: value };
      await this.initialize();
      this.notice(
        'ok',
        `Stored ${name} (${backend === 'keystore' ? 'your keychain' : 'an owner-only file'}).`,
        process.env[name] ? `${name} is also set in this shell, which wins.` : this.modelProblem ?? `${this.agent?.name ?? 'The agent'} runs on ${this.modelLabel()}.`,
      );
      return;
    }
    let removed = false;
    let left: Array<{ item: string; account: string }> = [];
    try {
      const store = await providerKeyStore();
      removed = await store.delete(name);
      left = store.leftBehind;
    } catch (error) {
      this.notice('error', `Could not remove ${name}: ${(error as Error).message}`);
      return;
    }
    if (this.storedKeys) delete this.storedKeys[name];
    await this.initialize();
    if (!removed) {
      this.notice('info', `${name} was not stored.`, process.env[name] ? 'It is set in this shell; unset it there.' : undefined);
      return;
    }
    this.notice('ok', `Removed ${name}.`, process.env[name] ? 'It is still set in this shell.' : this.modelProblem);
    await this.leftBehindNotices(left);
  }

  /**
   * `/secrets`: the names in the store an MCP server's `${secret:NAME}`
   * reads (S-292), and `set`/`remove` to change them. Handled like `/keys
   * set`: the value is taken at a hidden local prompt, goes to the store and
   * nowhere else, never to the model. The words are pinned by
   * `python/tests/fixtures/cli/secrets.json` (`chat`).
   */
  private async commandSecrets(args: string): Promise<void> {
    const { paint, palette } = this.theme;
    const [verb = '', name = ''] = args.trim().split(/\s+/);
    const { providerKeyStore } = await import('./provider-keys.js');
    const { REFERENCE_NAME } = await import('../skills/secrets/references.js');
    if (!verb) {
      let names: string[] = [];
      let complete = true;
      let where = 'stored';
      try {
        const store = await providerKeyStore();
        ({ names, complete } = await store.list());
        where = store.status().backend === 'keystore' ? 'stored in your keychain' : 'stored in an owner-only file';
      } catch {
        // The listing is a nicety; the hint below still says how to add one.
      }
      const lines = [paint.bold(paint.fg(palette.text, 'Secrets stored on this machine'))];
      if (!names.length) lines.push(paint.fg(palette.faint, '  (none)'));
      const width = Math.max(0, ...names.map((n) => n.length)) + 3;
      for (const stored of names) {
        lines.push(`  ${paint.fg(palette.success, '●')} ${paint.fg(palette.text, stored.padEnd(width))}${paint.fg(palette.muted, where)}`);
      }
      if (!complete) lines.push(paint.fg(palette.faint, '  An OS keychain cannot be listed, so secrets other tools wrote there do not appear.'));
      lines.push('', paint.fg(palette.faint, '  /secrets set <NAME> stores one; /secrets remove <NAME> removes it. An MCP server uses one as ${secret:NAME} in its env, headers or url.'));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    if ((verb !== 'set' && verb !== 'remove') || !name) {
      this.notice('error', 'Usage: /secrets [set|remove NAME]');
      return;
    }
    if (!REFERENCE_NAME.test(name)) {
      this.notice('error', `${name} does not look like an environment variable name.`);
      return;
    }
    if (verb === 'set') {
      // Ctrl+C cancels the entry, not the chat (D5), as at `/keys set`.
      const entered = await promptSecret(`  ${paint.fg(palette.muted, `${name} (hidden):`)} `, { onInterrupt: 'cancel' });
      const value = (entered ?? '').trim();
      if (!value) {
        this.notice('info', 'Nothing entered; nothing stored.');
        return;
      }
      let backend: string;
      try {
        backend = await (await providerKeyStore()).set(name, value);
      } catch (error) {
        this.notice('error', `Could not store ${name}: ${(error as Error).message}`);
        return;
      }
      // Rebuilt, so an MCP server that named the secret connects now, and a
      // provider key stored this way is picked up as `/keys set` would.
      this.storedKeys = undefined;
      await this.initialize();
      this.notice('ok', `Stored ${name} (${backend === 'keystore' ? 'your keychain' : 'an owner-only file'}).`, `An MCP server uses it as \${secret:${name}} in its env, headers or url.`);
      return;
    }
    let removed = false;
    let left: Array<{ item: string; account: string }> = [];
    try {
      const store = await providerKeyStore();
      removed = await store.delete(name);
      left = store.leftBehind;
    } catch (error) {
      this.notice('error', `Could not remove ${name}: ${(error as Error).message}`);
      return;
    }
    this.storedKeys = undefined;
    await this.initialize();
    if (!removed) {
      this.notice('info', `${name} was not stored.`);
      return;
    }
    this.notice('ok', `Removed ${name}.`);
    await this.leftBehindNotices(left);
  }

  /**
   * Ask on first use (the sandbox-default lane, 2026-09-27): the shell asks
   * the person at the terminal about a host a confined command was refused,
   * by name, read from srt's own proxy log (`skills/shell/skill.ts`,
   * `HostAsker`). `once` re-runs the command with the host for that run;
   * `always` writes it into the agent file's `network.hosts`
   * (`sandbox-default-hosts.ts`), keeps it for this session and re-runs;
   * anything else returns the output with the hint. Only the interactive
   * chat attaches this: `-p`, `serve()` and the daemon refuse with the hint,
   * and the shell asks for the owner's commands only.
   */
  private attachHostAsker(): void {
    const shell = (this.agent as unknown as { skills?: Array<{ name?: string }> } | null)?.skills?.find((s) => s.name === 'ShellSkill') as
      | { asker?: HostAsker; policy: { network: boolean; networkDomains: string[] } | null; declaredInAgentFile?: () => void }
      | undefined;
    if (!shell) return;
    if (!this.interactive) {
      shell.asker = undefined;
      return;
    }
    shell.asker = {
      askHost: async ({ host, command }) => {
        if (!this.chatting || !this.atTerminal()) return 'no';
        const pause = this.turnPause;
        pause?.pause();
        try {
          const { paint, palette } = this.theme;
          console.log(`\n  ${paint.fg(palette.warning, fillWords(HOST_WORDS.hostRefused, { host, command }))}`);
          const answer = hostAnswer((await promptLine(`  ${paint.fg(palette.text, HOST_WORDS.hostQuestion)}`)) ?? '');
          this.answerLine(ANSWER_WORDS.host[answer], { host });
          return answer;
        } finally {
          pause?.resume();
        }
      },
      allowHostAlways: async (host) => {
        const file = this.agentFile;
        const shown = file ? path.basename(file) : 'the built-in agent';
        if (!file) {
          this.aboveTurn(() => this.notice('warn', fillWords(HOST_WORDS.hostNotWritten, { host, file: shown, problem: HOST_WORDS.hostNoFile })));
          return;
        }
        const edit = addNetworkHost(fs.readFileSync(file, 'utf8'), host);
        if ('problem' in edit) {
          this.aboveTurn(() => this.notice('warn', fillWords(HOST_WORDS.hostNotWritten, { host, file: shown, problem: edit.problem })));
          return;
        }
        // Whether the file was still the version the chat loaded, before its own write.
        const untouched = !this.fileChangedNow();
        fs.writeFileSync(file, edit.text);
        // This session keeps the host too; the file is read again on the next start or /reload.
        if (shell.policy && !shell.policy.networkDomains.includes(host)) {
          shell.policy.networkDomains.push(host);
          shell.policy.network = true;
        }
        // THE CHAT'S OWN WRITE IS NOT A CHANGE (the ptypass-fixes lane,
        // 2026-09-27): the edit above made the next prompt say "AGENT.md
        // changed during the last reply", and `/sandbox` said "(default)"
        // until /reload, though this session already runs what the file now
        // says. When nothing else had changed the file, the written version
        // is the loaded one; the policy is now the file's own block.
        if (untouched) {
          const written = this.fileChangedNow();
          if (written) {
            this.loaded = written;
            this.versionNoticed = written.sha;
          }
        }
        shell.declaredInAgentFile?.();
        this.aboveTurn(() => this.notice('ok', fillWords(HOST_WORDS.hostWritten, { host, file: shown })));
      },
    };
  }

  /** The agent's shell skill, when it has one. */
  private shellSkill():
    | { policy: import('../sandbox/policy.js').SandboxPolicy | null; sandboxError: string | null; sandboxOrigin: import('../sandbox/policy.js').SandboxOrigin }
    | undefined {
    return (this.agent as unknown as { skills?: Array<{ name?: string }> } | null)?.skills?.find((s) => s.name === 'ShellSkill') as
      | { policy: import('../sandbox/policy.js').SandboxPolicy | null; sandboxError: string | null; sandboxOrigin: import('../sandbox/policy.js').SandboxOrigin }
      | undefined;
  }

  /** The state the agent's SKILL.md scripts run under (`SkillMdSkill.scriptState`), when it has such skills. */
  private skillScriptState(): string | undefined {
    const skill = (this.agent as unknown as { skills?: Array<{ name?: string; scriptState?: () => string }> } | null)?.skills?.find((s) => s.name === 'SkillMdSkill');
    return skill?.scriptState?.();
  }

  /** The effective folders, hosts and switches of a confined policy, one line (`sandboxDetail`). */
  private sandboxDetail(policy: import('../sandbox/policy.js').SandboxPolicy): string {
    const folders = (roots: string[]) => roots.filter((root) => root !== policy.scratch).map((root) => shortPath(root)).join(', ');
    return fill('sandboxDetail', {
      writes: `${folders(policy.writeRoots) || 'none'} and a scratch folder`,
      reads: policy.scopedReads ? `${folders(policy.readRoots) || 'none'} and the system folders` : 'all but credential folders and .env',
      hosts: policy.networkDomains.join(', ') || 'none',
      local: policy.localNetwork ? 'on' : 'off',
      sockets: policy.unixSockets.join(', ') || 'none',
      env: policy.envPassthrough.join(', ') || 'none',
    });
  }

  /**
   * What the agent's commands may do, in one line and a detail: for /sandbox
   * and /status. The same words as the Python chat (`repl/session.py`,
   * `sandbox_summary`), from the shell's own policy (plan item 1.2). The
   * headline is the STATE, preset and origin (`development (default)`, `off
   * (agent file)`, `off (--no-sandbox)`), since the sandbox is on by default
   * (2026-09-27); the detail lists the effective folders, hosts and switches.
   */
  private async sandboxSummary(): Promise<{ kind: 'ok' | 'warn' | 'info'; headline: string; detail?: string }> {
    const shell = this.shellSkill();
    const { backendStatus, FIX_SETUP_POINTER, sandboxState } = await import('../sandbox/index.js');
    if (!shell) {
      // SKILL.md scripts are confined commands too.
      const scripts = this.skillScriptState();
      if (!scripts) return { kind: 'info', headline: 'Not needed: this agent cannot run commands.' };
      const status = backendStatus();
      if (!status.available) {
        return { kind: 'warn', headline: fill('sandboxUnavailable', { state: scripts, reason: status.reason || 'no sandbox backend here' }), detail: fill('sandboxFix', { fix: status.fix || FIX_SETUP_POINTER }) };
      }
      return { kind: 'ok', headline: fill('sandboxScripts', { state: scripts }) };
    }
    if (shell.sandboxError) {
      return { kind: 'warn', headline: `Invalid: ${shell.sandboxError}`, detail: 'Every command is refused until the declaration is fixed.' };
    }
    const policy = shell.policy;
    const state = sandboxState(policy, shell.sandboxOrigin);
    if (!policy || !policy.confined) {
      return {
        kind: 'warn',
        headline: fill('sandboxOff', { state }),
        detail: CHAT_WORDS[shell.sandboxOrigin === '--no-sandbox' ? 'sandboxFlagFix' : 'sandboxOffFix'],
      };
    }
    // The engine, as `doctor` checks it (G9, 2026-09-26): a policy srt
    // cannot enforce here refuses every command, and the state alone would be a lie.
    const status = backendStatus();
    if (!status.available) {
      return {
        kind: 'warn',
        headline: fill('sandboxUnavailable', { state, reason: status.reason || 'no sandbox backend here' }),
        detail: fill('sandboxFix', { fix: status.fix || FIX_SETUP_POINTER }),
      };
    }
    return { kind: 'ok', headline: state, detail: this.sandboxDetail(policy) };
  }

  /** `/sandbox`: what the agent's commands are allowed to do. */
  private async commandSandbox(): Promise<void> {
    const summary = await this.sandboxSummary();
    this.notice(summary.kind, `Sandbox: ${summary.headline}`, summary.detail);
  }

  /**
   * `/publish`: send this folder's agent to Robutler, or update the linked
   * one after asking (owner decision 6); `/publish --dry-run`: show what would
   * be sent and send nothing (spec 3.6).
   */
  private async commandPublish(args: string): Promise<void> {
    const { paint, palette } = this.theme;
    const parts = args.trim().split(/\s+/).filter(Boolean);
    if (parts.some((p) => p !== '--dry-run')) {
      this.notice('error', fill('usage', { usage: '/publish [--dry-run]' }));
      return;
    }
    const dryRun = parts.includes('--dry-run');
    if (!this.agentFile) {
      this.notice('warn', 'Publishing needs an AGENT.md in this folder.', 'Create one with `webagents init`, then /publish.');
      return;
    }
    const { publishAgent } = await import('./publish.js');
    console.log();
    const result = await publishAgent(
      this.agentFile,
      {
        ok: (line) => this.notice('ok', line),
        print: (line) => console.log(`  ${paint.fg(palette.muted, line)}`),
        error: (line) => this.notice('error', line),
        confirm: (question) => this.confirm(`${question} [y/N] `),
        confirmUpdate: (question) => this.confirm(question),
      },
      { dryRun },
    );
    if (result.ok) console.log();
  }

  /**
   * Register a slash command
   */
  registerCommand(command: SlashCommand): void {
    this.commands.set(command.name, command);
  }
  
  /**
   * Initialize the agent: build the state and apply it (spec 3.1). The split
   * lets `/reload`, `/model`, `/login` and the rest build a new agent and keep
   * the running one when the build fails.
   */
  async initialize(): Promise<void> {
    // The opt-out, in the chat, is one of its notices, the words `/sandbox`
    // prints (`setOptOutAnnouncer`; the ptypass-fixes lane, 2026-09-27): the
    // shell's raw stderr line wrapped mid-word above the welcome card and
    // said something else than `/sandbox`. `-p` and `doctor` keep theirs.
    const optedOut: string[] = [];
    const restore = this.chatting
      ? (await import('../skills/shell/skill.js')).setOptOutAnnouncer((_message, origin) => void optedOut.push(origin))
      : undefined;
    let state: PreparedState;
    try {
      state = await this.buildState();
    } finally {
      restore?.();
    }
    await this.applyState(state);
    if (optedOut.length) await this.commandSandbox();
  }

  /**
   * Everything the running chat derives from the agent file, computed and
   * set NOWHERE (spec 3.1). `applyState` swaps it in and cleans the old agent
   * up. A build that throws (`AgentFileError`, a model client that will not
   * load) leaves the running agent untouched.
   */
  private async buildState(): Promise<PreparedState> {
    const { getRobutlerContent, parseAgentMarkdown, findAgentFile } = await import(
      '../agents/index.js'
    );
    const { resolveSkillsByName, skillEntryParts } = await import('../skills/resolve.js');

    let agentName = this.config.agentName || 'robutler';
    let instructions = this.config.instructions;
    let agentDescription = '';
    let agentFile: string | undefined;
    let parsedAgent: ParsedAgent | undefined;
    let declaredSkills: string[] = [];
    // The entries as written, config included (`- rest: {sign: always}`).
    let declaredEntries: Array<string | Record<string, unknown>> = [];
    // The file's `access:` block (ADR-0045), when it has one.
    let declaredAccess: unknown;
    // The file's `sandbox:` block, checked, for the shell to enforce (plan item 1.2).
    let declaredSandbox: import('../sandbox/policy.js').SandboxDeclaration | undefined;
    // The file's `agent_skills:`, folders of SKILL.md skills (plan item 1.4).
    let declaredAgentSkills: string[] | undefined;
    let declaredModel: string | undefined;

    // LOOK FOR A LOCAL AGENT FILE FIRST (2026-09-23).
    //
    // This used to load the embedded ROBUTLER.md and nothing else. The guard
    // was `if (agentName === 'robutler' || !instructions)`, and nothing in the
    // CLI ever set `instructions`, so the branch was ALWAYS taken: `-a myagent`
    // changed the display name and not one thing more. An agent sitting in the
    // working directory was never read, and the skills the file declared were
    // parsed and then discarded.
    const localFile = this.selectedFile !== undefined ? this.selectedFile : findAgentFile(process.cwd(), this.config.agentName);
    agentFile = localFile ?? undefined;
    if (localFile) {
      // A file that cannot be used as written (a `sandbox:` with a mistyped
      // key, a link, S-270/S-290) stops the chat with its sentence, as the
      // Python CLI does, rather than running a different agent. `readAgentFile`
      // refuses a symbolic link; `parseAgentMarkdown` refuses bad YAML, an
      // unknown key and the string cron form. Both throw `AgentFileError`,
      // which the caller keeps as-is.
      const parsed = parseAgentMarkdown(readAgentFile(localFile), localFile);
      parsedAgent = parsed;
      // With its path, so a file with no `name:` is named for its file
      // (`AGENT-helper.md` -> helper), as /agent lists it.
      agentName = parsed.name !== 'unknown' ? parsed.name : agentName;
      instructions = instructions ?? parsed.instructions;
      declaredSkills = parsed.skills;
      declaredEntries = parsed.skillEntries;
      declaredAccess = parsed.access;
      declaredSandbox = parsed.sandbox;
      declaredAgentSkills = parsed.agentSkills;
      declaredModel = parsed.model;
      agentDescription = parsed.description ?? '';
    } else if (!instructions) {
      // No local agent: fall back to the embedded one.
      try {
        const parsed = parseAgentMarkdown(getRobutlerContent());
        instructions = parsed.instructions;
        declaredSkills = parsed.skills;
        declaredEntries = parsed.skillEntries;
        if (agentName === 'robutler') agentName = parsed.name;
        // The card's description line was blank for the built-in agent, which
        // does have one (the Python chat shows it).
        agentDescription = parsed.description ?? '';
      } catch (error) {
        console.warn('Could not load embedded robutler agent:', (error as Error).message);
      }
    }

    // The model an explicit --model gives wins over the file's.
    //
    // It read `this.config.model ?? declaredModel`, and the constructor had
    // already defaulted `config.model` to 'gpt-4o', so the left side was never
    // nullish and the file's `model:` never applied (2026-09-24, found by the
    // onboarding docs audit). The explicit flag is now kept apart.
    const chosenModel = this.explicitModel ?? declaredModel;

    // THE MODEL, AND HOW IT IS REACHED (2026-09-24, `model-access.ts`).
    //
    // An agent that named no LLM skill got `OpenAISkill`, whatever this
    // machine had, so with no OPENAI_API_KEY every reply was "OpenAI API key
    // not configured": for a person signed in to Robutler, which serves
    // models, and for one with an Anthropic key exported. Now any provider
    // with a key is used, else Robutler's models when signed in, else
    // nothing, said before the first message (at a terminal `run()` offers
    // to sign in or take a key). An agent that names its LLM skill keeps it
    // while it can run, and without its key falls back the same way, as the
    // Python daemon does (`model_access.choose_model`).
    const { findProvider, missingProviderKey, modelForProvider } = await import('../skills/llm/providers.js');
    const { resolveModelAccess, unavailableMessage, platformLlmUrl } = await import('./model-access.js');
    const { getToken } = await import('./credentials.js');
    const { resolvePlatformUrl } = await import('./config-store.js');
    if (!this.storedKeys) {
      const { readStoredProviderKeys } = await import('./provider-keys.js');
      this.storedKeys = await readStoredProviderKeys();
    }
    // What the decision sees: the environment, plus stored keys where it has
    // none. The skills get the stored ones through `apiKeys`.
    const keyEnv: Record<string, string | undefined> = { ...process.env, ...this.storedKeys };
    const apiKeys: Record<string, string> = {};
    const { storedKeyFor } = await import('./provider-keys.js');
    for (const provider of keyProviders()) {
      const stored = storedKeyFor(provider, this.storedKeys);
      if (stored) apiKeys[provider.id] = stored;
    }
    const signedIn = async () => Boolean(await getToken());
    const proxyUrl = platformLlmUrl(resolvePlatformUrl()[0]);
    const proxy = {
      proxyUrl,
      // Read on every request, so a `/login` reaches the agent already built.
      platformToken: async () => (await getToken()) ?? undefined,
    };

    // Instantiate the skills the agent DECLARES, rather than hardcoding
    // OpenAI. `parseAgentMarkdown` has always returned this list and `app.ts`
    // has always thrown it away.
    const agentDir = localFile ? path.dirname(path.resolve(localFile)) : process.cwd();
    // `discovery` searches as the person at this terminal when the agent has
    // no platform credential of its own; only the chat passes this
    // (`skills/discovery/skill.ts`, rule 3). Read per search, like the proxy's.
    const personToken = async () => (await getToken()) ?? undefined;
    // The chat keeps its conversations itself (`sessions.ts`), so the session
    // skill only says WHERE, and is never loaded into the agent here: it
    // would be a second writer of the same conversation.
    const sessionEntry = declaredEntries.map(skillEntryParts).find((entry) => entry.name?.toLowerCase() === 'session');
    const sessionBackend: 'local' | 'robutler' = sessionEntry?.config.backend === 'robutler' ? 'robutler' : 'local';
    const agentEntries = declaredEntries.filter((entry) => skillEntryParts(entry).name?.toLowerCase() !== 'session');
    const { skills, byName, unknown, failed, skillmd } = await resolveSkillsByName(agentEntries, {
      model: chosenModel,
      proxy,
      apiKeys,
      agentDir,
      personToken,
      // The file tools ask here, and nowhere else, before they change one
      // of the agent's control files (S-314).
      ...(localFile ? { agentFile: path.resolve(localFile) } : {}),
      confirmControlWrite: (file, diff) => this.confirmControlWrite(file, diff),
      ...(declaredSandbox ? { sandbox: declaredSandbox } : {}),
      ...(declaredAgentSkills ? { agentSkills: declaredAgentSkills } : {}),
    });
    // SKILL.md skills that could not load are said, never fatal (plan item 1.4).
    const { skillmdReportLines } = await import('../skills/resolve.js');
    for (const line of skillmdReportLines(skillmd)) console.warn(line);
    // An agent with these can change the folder, so each message to it is
    // preceded by a snapshot `/undo` can put back (`checkpoints.ts`).
    const canChangeFiles = [...byName.keys()].some((name) => ['filesystem', 'shell'].includes(name.toLowerCase()));
    for (const name of unknown) {
      // The embedded agent names only skills both SDKs have (`filesystem`,
      // `rest`, 2026-09-25); an unknown one there is ours to fix, not
      // something the person can act on, so it is said only when debugging.
      if (localFile || process.env.WEBAGENTS_DEBUG) {
        console.warn(`Unknown skill "${name}" in ${localFile ?? 'the embedded agent'}; skipping.`);
      }
    }
    for (const f of failed) {
      console.warn(`Skill "${f.name}" failed to load: ${f.reason}`);
    }

    // Decided from the DECLARED names against the provider registry, not by
    // inspecting instances: the registry is the one place that knows which
    // names are LLM providers. A declared LLM skill that could not be built
    // (`fireworks` has no client in this SDK) does not count.
    const notBuilt = new Set([...unknown, ...failed.map((f) => f.name)]);
    const declaredLLM = declaredSkills
      .filter((name) => !notBuilt.has(name))
      .map((name) => findProvider(name))
      .find((provider): provider is LLMProvider => provider !== undefined);
    let modelProblem: string | undefined;
    let modelAccess: ModelAccess | undefined;
    let providerEnvVar: string | undefined;
    let configModel: string | undefined;

    const declared = declaredLLM;
    if (declared && !missingProviderKey([declared.id], keyEnv)) {
      // The agent's own provider, with its key here: its choice stands. The
      // model it RUNS is the one its skill was built with (`resolve.ts`,
      // `modelForProvider`): another provider's model is dropped there, so
      // the label says the provider's own rather than what was asked for
      // (D3, 2026-09-26; the proxy serves every provider's models as given).
      const viaRobutler = declared.credential === 'platform';
      const own = declared.id === 'proxy' ? chosenModel : modelForProvider(declared.id, chosenModel);
      const built = [...byName.entries()].find(([entryName]) => findProvider(entryName)?.id === declared.id)?.[1] as
        | { modelConfig?: { model?: string }; model?: string }
        | undefined;
      const skillModel = built?.modelConfig?.model ?? built?.model;
      const running = own ?? (skillModel ? (skillModel.includes('/') ? skillModel : `${declared.id}/${skillModel}`) : undefined) ?? (declared.defaultModel ? `${declared.id}/${declared.defaultModel}` : chosenModel);
      const effective = running && !running.includes('/') ? `${declared.id}/${running}` : running;
      modelAccess = { kind: viaRobutler ? 'proxy' : 'direct', model: effective, provider: declared };
      providerEnvVar = declared.credential === 'api-key' ? declared.envVar : undefined;
      if (viaRobutler && !(await signedIn())) {
        modelProblem = `This agent runs on Robutler's models. Sign in with \`${cliCommand('login')}\`.`;
      }
      configModel = effective;
    } else {
      // No LLM skill named, or the named provider's key is missing. For a
      // named provider the decision is asked about THAT provider's model, so a
      // signed-in person runs the same model through Robutler: the `init`
      // scaffold lists `openai`, and a first run without the key must still
      // be offered the sign-in.
      let wanted = chosenModel;
      if (declared) {
        const own = modelForProvider(declared.id, chosenModel);
        wanted = own
          ? (own.includes('/') ? own : `${declared.id}/${own}`)
          : (declared.defaultModel ? `${declared.id}/${declared.defaultModel}` : undefined);
        // It cannot run without its key; whatever runs instead takes its place.
        const index = skills.findIndex((skill) => (skill as { name?: string }).name === declared.id);
        if (index !== -1) skills.splice(index, 1);
      }
      const access = await resolveModelAccess(wanted, { signedIn, env: keyEnv });
      modelAccess = access;
      providerEnvVar = access.kind === 'direct' ? access.provider?.envVar : undefined;
      if (access.kind === 'direct') {
        const built = await resolveSkillsByName([access.provider!.id], { model: access.model, apiKeys });
        skills.push(...built.skills);
        if (built.failed.length) {
          modelProblem = `Could not load the ${access.provider!.id} client: ${built.failed[0].reason}`;
        }
      } else if (access.kind === 'proxy') {
        const { LLMProxySkill } = await import('../skills/llm/proxy/skill.js');
        skills.push(new LLMProxySkill({ model: access.model, ...proxy }) as unknown as (typeof skills)[number]);
      } else {
        modelProblem = unavailableMessage(access);
      }
      configModel = access.model ?? chosenModel;
    }

    // `fallback_models:` (plan item 2.8): the model's skill becomes a chain
    // that moves to the next model on a provider error, with a note in the
    // transcript. A fallback that cannot be built here is said and left out.
    if (parsedAgent?.fallbackModels?.length && !modelProblem) {
      const { withFallbackModels } = await import('../skills/resolve.js');
      const chained = await withFallbackModels(skills, parsedAgent.fallbackModels, {
        primaryModel: modelAccess?.model,
        ...((await signedIn()) ? { proxy } : {}),
        apiKeys,
        env: keyEnv,
      });
      skills.splice(0, skills.length, ...chained.skills);
      for (const f of chained.failed) console.warn(`Skill "${f.name}" failed to load: ${f.reason}`);
    }

    // Who may call it, and what each group gets (ADR-0045). In the chat the
    // person is the owner, so the block decides nothing here; it is still
    // installed, and a malformed one refused, so the file means one thing
    // wherever it runs.
    const { AccessConfigError } = await import('../access/policy.js');
    const { AgentFileError } = await import('../agents/index.js');
    const { accessSkillFor, applyAccessTools } = await import('../access/install.js');
    let access: ReturnType<typeof accessSkillFor> | undefined;
    try {
      access = declaredAccess !== undefined ? accessSkillFor(declaredAccess, localFile ?? undefined) : undefined;
    } catch (err) {
      if (err instanceof AccessConfigError) throw new AgentFileError(`${localFile}: ${err.message}`);
      throw err;
    }

    const agent = new BaseAgent({
      name: agentName,
      // Where the instructions come from, the folder and the small-talk rule,
      // for an agent that can explore its folder (preamble.ts).
      instructions: withCliPreamble(instructions, agentFile, declaredEntries.length ? declaredEntries : declaredSkills),
      model: configModel,
      skills: (access ? [...skills, access.skill] : skills) as never,
      // `observability: {otel: true}` in the file records the run as
      // OpenTelemetry spans (plan item 2.4).
      ...(parsedAgent?.observability !== undefined ? { observability: parsedAgent.observability } : {}),
    });
    // The tool rounds a turn may run (2026-09-28, `core/tool-budget.ts`):
    // this chat's `/rounds`, then `--max-tool-rounds`, then the file's
    // `max_tool_rounds`, then 50.
    if (this.sessionRounds !== undefined) {
      agent.maxToolIterations = this.sessionRounds;
      agent.maxToolRoundsSource = 'session';
    } else {
      const effective = effectiveMaxToolRounds(parsedAgent?.maxToolRounds);
      agent.maxToolIterations = effective.rounds;
      agent.maxToolRoundsSource = effective.source;
    }
    if (access) {
      try {
        applyAccessTools(access.policy, byName);
      } catch (err) {
        if (err instanceof AccessConfigError) throw new AgentFileError(`${localFile}: ${err.message}`);
        throw err;
      }
    }

    await agent.initialize();

    // The loaded version and the tool set, so `/reload` can tell what changed
    // and which tools came or went (spec 3.1). The built-in agent has no file.
    const loaded = localFile && parsedAgent ? loadedAgentOf(localFile, agentDir, parsedAgent) : undefined;
    return {
      agent,
      agentName,
      agentFile,
      agentDescription,
      configModel,
      modelAccess,
      providerEnvVar,
      declaredLLM,
      modelProblem,
      sessionBackend,
      canChangeFiles,
      proxyUrl,
      toolNames: toolNamesOf(agent),
      loaded,
      declaredSkills,
      accessPolicy: access?.policy,
    };
  }

  /**
   * Swap in a built state and clean up the agent it replaces (spec 3.1):
   * `BaseAgent.cleanup()` closes the previous agent's MCP connections, which
   * every `/model`, `/login` and `/keys` used to leave open.
   */
  private async applyState(state: PreparedState): Promise<void> {
    const previous = this.agent;
    this.agent = state.agent;
    this.agentFile = state.agentFile;
    this.agentDescription = state.agentDescription;
    this.config.model = state.configModel;
    this.modelAccess = state.modelAccess;
    this.providerEnvVar = state.providerEnvVar;
    this.declaredLLM = state.declaredLLM;
    this.modelProblem = state.modelProblem;
    this.sessionBackend = state.sessionBackend;
    this.canChangeFiles = state.canChangeFiles;
    this.proxyUrl = state.proxyUrl;
    this.toolNames = state.toolNames;
    this.loaded = state.loaded;
    this.declaredSkills = state.declaredSkills;
    this.accessPolicy = state.accessPolicy;
    // The version in use has been read; forget any pending "changed" notice
    // for an older one.
    this.versionNoticed = state.loaded?.sha;
    this.changedDuringReply = false;
    // Only the interactive loop asks (`run()` attaches for the first agent);
    // a `/reload` while chatting attaches to the new one here. `-p` never has it.
    if (this.chatting) this.attachHostAsker();
    if (previous && previous !== state.agent) {
      try {
        await previous.cleanup();
      } catch {
        // A cleanup that fails must not fail the reload.
      }
    }
  }

  /**
   * Close the running agent's connections without ending the process, for a
   * one-shot caller such as `webagents doctor` (2026-09-26, the e2e run's
   * HIGH bug): `runChecks` built a chat, initialised it and never cleaned it
   * up, so a connected stdio MCP server's child process kept the event loop
   * alive and doctor never exited once a server had connected. The agent
   * stays in place (its report is still readable); only what it holds open
   * is closed. A cleanup that fails must not fail the caller.
   */
  async closeAgent(): Promise<void> {
    const agent = this.agent;
    if (!agent) return;
    try {
      await agent.cleanup();
    } catch {
      // Said nowhere: the caller has its report, and the process ends.
    }
  }

  /**
   * Send a message and get response
   */
  async sendMessage(content: string): Promise<RunResponse> {
    // The same stream as the other paths, so the turn's tool rounds are kept
    // with it (turn-history.ts); it used to call `agent.run` and keep text only.
    const steps = this.sendMessageStreaming(content);
    for (;;) {
      const step = await steps.next();
      if (step.done) return step.value;
    }
  }
  
  /**
   * The chunks of one turn, straight from the agent: text, tool calls and
   * their results, thinking, `done` or `error`. Records nothing; the callers
   * below decide what the conversation keeps.
   */
  private async *turnChunks(content: string, signal?: AbortSignal): AsyncGenerator<StreamChunk, void, unknown> {
    if (!this.agent) {
      throw new Error('Agent not initialized');
    }
    // A copy: the conversation records the turn once it is over (recordTurn).
    // Older tool results past the budget go as a one-line note (turn-history.ts).
    const conversation: Message[] = [...historyForModel(this.messages), { role: 'user', content }];
    yield* this.agent.runStreaming(conversation, { ...(signal ? { signal } : {}), auth: LOCAL_OWNER });
  }

  /**
   * Adds a turn to the conversation: the message, its tool rounds (`tools`,
   * from a TurnRecorder: kept so the next message does not make the model
   * list and read everything again), and the answer's text if there was any.
   * A turn that failed before it said anything leaves no trace, so the next
   * message is not sent after an unanswered one.
   */
  private recordTurn(content: string, answer: string, failed: boolean, tools: Message[] = []): void {
    // A turn that said nothing, failed or not, leaves no trace, so the next
    // message is not sent after an unanswered one (the Python chat pops the
    // message the same way; 2026-09-27, when empty completions were common).
    void failed;
    if (!answer) return;
    this.messages.push({ role: 'user', content });
    this.messages.push(...tools);
    this.messages.push({ role: 'assistant', content: answer });
  }

  /**
   * Every event of one turn, for `-p --output-format stream-json`: text
   * deltas, tool calls and their results, then `done` or `error`.
   */
  async *streamTurn(content: string): AsyncGenerator<StreamChunk, void, unknown> {
    let answer = '';
    let failed = false;
    const recorder = new TurnRecorder();
    for await (const chunk of this.turnChunks(content)) {
      recorder.observe(chunk);
      if (chunk.type === 'delta' && chunk.delta) answer += chunk.delta;
      if (chunk.type === 'error') failed = true;
      yield chunk;
    }
    this.recordTurn(content, answer, failed, recorder.messages());
  }

  /**
   * Send message with streaming: the answer's text only.
   *
   * Throws when the model reports an error. It used to drop the error chunk,
   * so a rejected key or an unknown model produced an empty answer and no
   * message at all (2026-09-24).
   */
  async *sendMessageStreaming(content: string): AsyncGenerator<string, RunResponse, unknown> {
    let answer = '';
    let response: RunResponse | undefined;
    const recorder = new TurnRecorder();
    for await (const chunk of this.turnChunks(content)) {
      recorder.observe(chunk);
      if (chunk.type === 'delta' && chunk.delta) {
        answer += chunk.delta;
        yield chunk.delta;
      } else if (chunk.type === 'done' && chunk.response) {
        response = chunk.response;
      } else if (chunk.type === 'error') {
        this.recordTurn(content, answer, true, recorder.messages());
        throw chunk.error ?? new Error('The model returned an error.');
      }
    }
    this.recordTurn(content, answer, false, recorder.messages());
    return response || { content: answer };
  }

  /**
   * Handle input line
   */
  private async handleInput(line: string): Promise<void> {
    const trimmed = line.trim();
    
    if (!trimmed) {
      return;
    }
    
    // Check for slash command. Only a typed line reaches here (spec W1): the
    // model, a tool, an `@file` and a resumed transcript never do.
    if (trimmed.startsWith('/')) {
      const parts = trimmed.slice(1).split(/\s+/);
      const cmdName = parts[0].toLowerCase();
      const args = parts.slice(1).join(' ');

      const spec = chatCommand(cmdName);
      const command = spec ? this.commands.get(spec.name) : undefined;
      if (!spec || !command) {
        this.sayUnknownCommand(cmdName);
        return;
      }
      // W8: a command whose usage names no argument refuses one, nothing done.
      if (args.trim() && takesNoArguments(spec)) {
        this.notice('error', fill('usage', { usage: spec.usage }));
        return;
      }
      // W7: a handler's exception becomes `✗ {message}`, and the chat goes on.
      // A rebuild that failed keeps the running agent (the handler restores it).
      try {
        await command.handler(args);
      } catch (error) {
        this.notice('error', (error as Error).message);
      }
      return;
    }
    
    // No model: say so and how to get one, instead of a model error per message.
    if (this.modelProblem) {
      this.notice('warn', this.modelProblem, 'Type /login to sign in without leaving the chat.');
      return;
    }
    const message = this.expandFileReferences(trimmed);
    // The file as it is when the turn starts, so a change made while the
    // chat sat idle at the prompt is not called one made "during the last
    // reply" (2026-09-26): only a version that appears between here and the
    // end of the turn is the agent's own doing.
    const beforeTurn = this.fileChangedNow();

    // Send message to agent
    try {
      if (this.config.streaming) {
        await this.streamToTerminal(message, () => this.snapshotBeforeTurn(message));
      } else {
        const printer = new TurnPrinter({ theme: this.theme, explainError: (text) => this.explainFailure(text) });
        printer.start();
        await this.snapshotBeforeTurn(message);
        this.turnPause = { pause: () => printer.suspend(), resume: () => printer.resume() };
        let response;
        try {
          response = await this.sendMessage(message);
        } finally {
          this.turnPause = undefined;
        }
        printer.feed({ type: 'delta', delta: `${response.content}\n` });
        printer.finish();
        console.log();
      }
    } catch (error) {
      const { headline, hint } = this.explainTurnError(error);
      this.notice('error', headline, hint);
    } finally {
      this.saveConversation();
      this.recordOnRobutler();
      // If the agent rewrote its own file during the reply, the next
      // before-prompt notice says so with `▲` rather than `✦` (S-283).
      const afterTurn = this.fileChangedNow();
      if (afterTurn && (!beforeTurn || beforeTurn.sha !== afterTurn.sha)) this.changedDuringReply = true;
    }
  }

  /**
   * `@path/to/file` includes that file when it exists; anything else stays as
   * typed (an `@name` that is not a file is left alone). The Python chat does
   * the same, in the same words.
   */
  private expandFileReferences(text: string): string {
    return text.replace(/(?<![\w.])@([A-Za-z0-9_\-./~]+)/g, (match, ref: string) => {
      const expanded = ref.startsWith('~') ? path.join(os.homedir(), ref.slice(1)) : ref;
      const file = path.isAbsolute(expanded) ? expanded : path.join(process.cwd(), expanded);
      let isFile = false;
      try {
        isFile = fs.statSync(file).isFile();
      } catch {
        return match;
      }
      if (!isFile) return match;
      try {
        return `\n\n<file path="${ref}">\n${fs.readFileSync(file, 'utf-8')}\n</file>\n\n`;
      } catch (error) {
        return `@${ref} (could not read it: ${(error as Error).message})`;
      }
    });
  }

  /** A one-line result of a command: ✓ done, ✦ information, ▲ warning, ✗ failure. */
  private notice(kind: 'ok' | 'info' | 'warn' | 'error', text: string, detail?: string): void {
    const { paint, palette } = this.theme;
    const marker = {
      ok: paint.fg(palette.success, '✓'),
      info: paint.fg(palette.agent, '✦'),
      warn: paint.fg(palette.warning, '▲'),
      error: paint.bold(paint.fg(palette.error, '✗')),
    }[kind];
    const colour = kind === 'error' ? palette.error : kind === 'warn' ? palette.warning : palette.text;
    const width = (terminalColumns()) - 1;
    const lines = wrapStyled(paint.fg(colour, text), width, `${marker} `, '  ');
    if (detail) lines.push(...wrapStyled(paint.fg(palette.faint, detail), width, '  ', '  '));
    console.log(`\n${lines.join('\n')}\n`);
  }

  /**
   * The headline and hint for a failed turn (`failures.ts`, the same rules
   * and cases as the Python chat): Robutler's own refusals when the turn ran
   * on its models, else the provider's words and `genericErrorHint`.
   */
  explainFailure(message: string): FailureText {
    return presentFailure(message, {
      proxyUrl: this.modelAccess?.kind === 'proxy' ? this.proxyUrl : undefined,
      genericHint: (text) => this.genericErrorHint(text),
      model: this.costModel(),
    });
  }

  /**
   * `explainFailure` for a thrown or streamed error. The agent's tool-round
   * cap (its `max_iterations` error, 2026-09-28, `core/tool-budget.ts`) is not
   * the model's failure: it gets the empty-reply line's sentence and its own
   * code, as the Python chat and `-p` say it.
   */
  explainTurnError(error: unknown): FailureText {
    const spent = agentFinishOf(error);
    if (spent) return { ...presentEmptyReply(spent), code: spent.reason };
    return this.explainFailure((error as Error | undefined)?.message ?? '');
  }

  /**
   * What to do about a provider's own error, for the errors a first run
   * meets: a key the provider refused, a server that is not there, a model
   * that does not exist, a rate limit. Names environment variables, never
   * their values.
   */
  private genericErrorHint(message: string): string | undefined {
    const key = this.providerEnvVar;
    if (/\b401\b|unauthori[sz]ed|invalid.{0,10}api.?key|incorrect api key|api key not valid|authentication/i.test(message)) {
      return key ? `The provider refused the key. Check ${key}, then start the chat again.` : 'The provider refused the key.';
    }
    if (/\b429\b|rate.?limit|quota/i.test(message)) return 'The provider is limiting requests. Wait a moment, then try again.';
    if (/could not reach|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|fetch failed|bad port/i.test(message)) {
      // This provider's own override (OPENAI_API_KEY -> OPENAI_BASE_URL), not
      // whichever *_BASE_URL happens to be in the environment.
      const base = key?.replace(/_API_KEY$/, '_BASE_URL');
      return base && base !== key && process.env[base]
        ? `${base} is set; check that it points at a running server.`
        : 'Check the network connection.';
    }
    if (/\b404\b|model.{0,40}(not found|does not exist)|unknown model/i.test(message)) {
      return 'Check the model name; /model switches it.';
    }
    return undefined;
  }

  /** The model as the card, the footer and /model show it: `auto/balanced via Robutler`. */
  private modelLabel(): string {
    const access = this.modelAccess;
    if (!access || access.kind === 'none') return this.config.model ?? '';
    return describeAccess(access, access.kind === 'proxy' ? 'auto/balanced' : (access.provider?.id ?? ''));
  }

  /** The providers whose key would give this agent a model, for the offer. */
  private keyCandidates(): LLMProvider[] {
    const takesKey = keyProviders();
    const named = this.declaredLLM ?? this.modelAccess?.provider;
    if (named) return takesKey.filter((p) => p.id === named.id);
    // `auto/...` runs only on Robutler; a key would not help.
    if (this.modelAccess?.reason?.endsWith('is served by Robutler')) return [];
    return takesKey;
  }

  /**
   * No model to run on: ask once, before the chat opens (2026-09-24).
   *
   * It said "OPENAI_API_KEY is not set ... Exit, export it, then start
   * again" and then failed every message. The ways out are offered there and
   * then: sign in to Robutler (first, since it needs nothing the person has
   * to go and find), type a provider key (kept for next time, in the store
   * both CLIs read), or carry on without a model.
   */
  private async offerModelAccess(): Promise<void> {
    const { paint, palette } = this.theme;
    const choices: Array<{ label: string; act: () => Promise<void> }> = [
      { label: "Sign in to Robutler and use its models, paid from your credits", act: () => this.signIn() },
    ];
    const candidates = this.keyCandidates();
    if (candidates.length) {
      const which = candidates.length === 1 ? candidates[0].envVar! : 'a provider key';
      choices.push({ label: `Enter ${which}, kept for next time`, act: () => this.enterKey(candidates) });
    }
    choices.push({ label: 'Continue without a model', act: async () => {} });

    this.notice('warn', this.modelProblem!);
    for (const [i, choice] of choices.entries()) {
      console.log(`  ${paint.fg(palette.accent, String(i + 1))}  ${paint.fg(palette.text, choice.label)}`);
    }
    const answer = await promptLine(`\n  ${paint.fg(palette.muted, `Choose 1-${choices.length} [1]:`)} `);
    if (answer === null) {
      // Ctrl+C (or Ctrl+D) at the offer cancels the offer, not the chat, and
      // says so; the Python chat does the same (2026-09-26, the e2e run).
      this.notice('info', CHAT_WORDS.continuingWithoutModel);
      return;
    }
    const picked = choices[answer.trim() === '' ? 0 : Number.parseInt(answer.trim(), 10) - 1];
    if (!picked) {
      this.notice('info', 'Continuing without a model.');
      return;
    }
    await picked.act();
  }

  /**
   * Sign in through the browser (`browser-login.ts`), store the token, and
   * rebuild the agent: with no key of its own it now runs on Robutler's
   * models. `/login` and the offer both come here. Ctrl+C cancels the wait.
   */
  private async signIn(): Promise<void> {
    const { browserLogin } = await import('./browser-login.js');
    const { setToken } = await import('./credentials.js');
    const { resolvePlatformUrl } = await import('./config-store.js');
    const [portalUrl] = resolvePlatformUrl();
    const { paint, palette } = this.theme;
    const controller = new AbortController();
    const cancel = () => controller.abort();
    process.on('SIGINT', cancel);
    try {
      console.log();
      const signedIn = await browserLogin(portalUrl, {
        signal: controller.signal,
        print: (line) => console.log(`  ${paint.fg(palette.muted, line)}`),
      });
      await setToken(signedIn.token);
      await this.initialize();
      const who = signedIn.username ? ` as @${signedIn.username}` : '';
      if (this.modelProblem) {
        this.notice('warn', `Signed in${who} on ${portalUrl}.`, this.modelProblem);
      } else {
        this.notice('ok', `Signed in${who} on ${portalUrl}.`, `${this.agent?.name ?? 'The agent'} runs on ${this.modelLabel()}.`);
      }
    } catch (error) {
      this.notice('error', `Could not sign in: ${(error as Error).message}`);
    } finally {
      process.removeListener('SIGINT', cancel);
    }
  }

  /** Take a provider key with echo off, keep it (`provider-keys.ts`), and rebuild the agent. */
  private async enterKey(candidates: LLMProvider[]): Promise<void> {
    const { paint, palette } = this.theme;
    let provider: LLMProvider | undefined = candidates[0];
    if (candidates.length > 1) {
      console.log();
      for (const [i, p] of candidates.entries()) {
        console.log(`  ${paint.fg(palette.accent, String(i + 1))}  ${paint.fg(palette.text, p.id)} ${paint.fg(palette.faint, p.envVar!)}`);
      }
      const answer = await promptLine(`\n  ${paint.fg(palette.muted, `Which provider? 1-${candidates.length} [1]:`)} `);
      if (answer === null) return;
      provider = candidates[answer.trim() === '' ? 0 : Number.parseInt(answer.trim(), 10) - 1];
      if (!provider) {
        this.notice('info', 'No provider chosen; continuing without a model.');
        return;
      }
    }
    const envVar = provider.envVar!;
    const value = (await promptSecret(`  ${paint.fg(palette.muted, `${envVar} (hidden):`)} `)).trim();
    if (!value) {
      this.notice('info', 'Nothing entered; continuing without a model.');
      return;
    }
    // For this session: to the model client, not the environment.
    this.storedKeys = { ...(this.storedKeys ?? {}), [envVar]: value };
    let kept: string;
    try {
      const { storeProviderKey } = await import('./provider-keys.js');
      const backend = await storeProviderKey(envVar, value);
      kept = backend === 'keystore' ? 'Kept in your OS keystore' : 'Kept in an owner-only file';
      kept += `; \`webagents secrets remove ${envVar}\` removes it.`;
    } catch (error) {
      kept = `Set for this session only; it could not be kept: ${(error as Error).message}`;
    }
    await this.initialize();
    if (this.modelProblem) {
      this.notice('warn', this.modelProblem, kept);
    } else {
      this.notice('ok', `${this.agent?.name ?? 'The agent'} runs on ${this.modelLabel()}.`, kept);
    }
  }

  /** The agent's tools, by name, less the turn-scoped content tools (`TRANSIENT_TOOLS`). */
  private toolList(): Array<{ name: string; description?: string; scopes?: string[] }> {
    const registry = (this.agent as unknown as { toolRegistry?: Map<string, { name: string; description?: string; scopes?: string[] }> } | null)
      ?.toolRegistry;
    // By name, as the Python chat lists them: registration order is each
    // SDK's own business, and the list reads the same in both.
    return registry
      ? [...registry.values()].filter((t) => !TRANSIENT_TOOLS.has(t.name)).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
      : [];
  }

  /** What the welcome card shows. */
  private welcomeInfo(): WelcomeInfo {
    return {
      agent: this.agent?.name || 'agent',
      description: this.agentDescription,
      model: this.modelLabel(),
      tools: this.toolList().map((tool) => tool.name),
      folder: shortPath(process.cwd()),
      warnings: this.modelProblem ? [this.modelProblem] : [],
      version: this.config.version,
    };
  }

  /** The left side of the box's footer: who, which model, what the conversation has cost so far, where. */
  private footerParts(): string[] {
    const parts = [this.agent?.name ?? 'agent'];
    const model = this.modelLabel();
    if (model) parts.push(model);
    const tokens = this.inputTokens + this.outputTokens;
    if (tokens) parts.push(`${compactNumber(tokens)} tokens`);
    // What it has cost, in credits, next to the tokens (plan item 2.4): the
    // platform's number for Robutler's models, an estimate (tilde) for a key.
    if (this.cost.known) parts.push(costWords(this.cost.credits, this.cost.estimated));
    // The folder last and short: its tail is the part that says where you are.
    parts.push(truncateStart(shortPath(process.cwd()), 28));
    return parts;
  }

  /** The model a turn's cost is estimated for: the one that answered (a failover may have moved it), else the agent's. */
  private costModel(): string | undefined {
    const answered = this.failoverSkill()?.answeredModel;
    return answered ?? this.modelAccess?.model;
  }

  /** The agent's failover skill (`fallback_models:`, plan item 2.8), when it has one. */
  private failoverSkill(): { answeredModel?: string; notes: string[] } | undefined {
    return (this.agent as unknown as { skills?: Array<{ name?: string; answeredModel?: string; notes?: string[] }> } | null)?.skills?.find(
      (s) => s.name === 'failover',
    ) as { answeredModel?: string; notes: string[] } | undefined;
  }

  /**
   * One turn, drawn as it streams (see render.ts), and stoppable.
   *
   * Ctrl+C or Esc stops the turn, not the program: the model request is
   * cancelled, what already arrived stays on screen, and the prompt comes
   * back. Before, the chat printed text deltas only, so tool calls, their
   * results and model errors never appeared, and Ctrl+C mid-answer killed the
   * process (2026-09-24).
   */
  private async streamToTerminal(content: string, beforeTurn?: () => Promise<void>): Promise<void> {
    const controller = new AbortController();
    const printer = new TurnPrinter({ theme: this.theme, explainError: (message) => this.explainFailure(message) });
    const recorder = new TurnRecorder();
    const stop = () => controller.abort();
    const onKey = (_text: string, key?: { ctrl?: boolean; name?: string }) => {
      if (key && ((key.ctrl && key.name === 'c') || key.name === 'escape')) stop();
    };
    // Between prompts no readline owns the terminal: raw mode makes Ctrl+C and
    // Esc keypresses (and keeps typing from echoing into the answer), and the
    // SIGINT handler covers input that is not a terminal.
    const tty = Boolean(process.stdin.isTTY);
    process.on('SIGINT', stop);
    if (tty) {
      readline.emitKeypressEvents(process.stdin);
      process.stdin.setRawMode(true);
      process.stdin.on('keypress', onKey);
      process.stdin.resume();
    }

    const chunks = this.turnChunks(content, controller.signal);
    const aborted = new Promise<'aborted'>((resolve) => {
      controller.signal.addEventListener('abort', () => resolve('aborted'), { once: true });
    });
    // A blank line between what was typed and the answer.
    console.log();
    printer.start();
    // A question mid-turn (the control-file prompt, S-314) needs the live
    // region down and the terminal cooked; both come back after the answer.
    this.turnPause = {
      pause: () => {
        printer.suspend();
        if (tty) {
          process.stdin.removeListener('keypress', onKey);
          process.stdin.setRawMode(false);
        }
      },
      resume: () => {
        if (tty) {
          process.stdin.setRawMode(true);
          process.stdin.on('keypress', onKey);
          process.stdin.resume();
        }
        printer.resume();
      },
    };
    try {
      // The /undo snapshot, with the spinner already drawing (snapshotBeforeTurn).
      if (beforeTurn) await beforeTurn();
      for (;;) {
        // Raced, so an interrupt shows at once even while a tool is running.
        const step = await Promise.race([chunks.next(), aborted]);
        if (step === 'aborted' || step.done) break;
        // The model request fails when it is cancelled; that is not an error
        // to show.
        if (step.value.type === 'error' && controller.signal.aborted) break;
        recorder.observe(step.value);
        printer.feed(step.value);
      }
    } finally {
      this.turnPause = undefined;
      process.removeListener('SIGINT', stop);
      if (tty) {
        process.stdin.removeListener('keypress', onKey);
        process.stdin.setRawMode(false);
        process.stdin.pause();
      }
    }

    if (controller.signal.aborted) {
      printer.finish({ interrupted: true });
      // The request is already cancelled and the tool loop stops at its next
      // check; a tool that ignores the signal is not waited on for long.
      const drain = (async () => {
        try {
          for (;;) if ((await chunks.next()).done) return;
        } catch {
          // An aborted request may end in a rejection; the turn is over either way.
        }
      })();
      await Promise.race([drain, new Promise((resolve) => setTimeout(resolve, INTERRUPT_GRACE_MS).unref())]);
    } else {
      printer.finish();
    }
    this.recordTurn(content, printer.plainText, printer.failed, recorder.messages());
    const turnFinish = controller.signal.aborted || printer.failed ? undefined : printer.turnFinish;
    // A REPLY IS SOMETHING SAID (2026-09-25, narrowed 2026-09-27): the
    // session's last line counted every turn, so a chat whose only message was
    // refused ended "1 reply", and then every turn that ended quietly, so a
    // chat of seven empty completions said "7 replies". A turn counts only
    // when it said something. The Python chat counts the same way.
    if (printer.plainText.trim()) this.turns += 1;
    this.sessionTokens += printer.usageTokens ?? 0;
    this.inputTokens += printer.usageSplit.input;
    this.outputTokens += printer.usageSplit.output;
    // The turn's cost: reported by the platform, else estimated for the
    // model that answered (plan item 2.4).
    const turnUsage = {
      input_tokens: printer.usageSplit.input,
      output_tokens: printer.usageSplit.output,
      ...(printer.usageCostCredits !== null ? { cost: { total_cost: printer.usageCostCredits, currency: 'credits' } } : {}),
    };
    if (printer.usageTokens || printer.usageCostCredits !== null) {
      const model = this.costModel();
      this.cost = addTurnCost(this.cost, model, turnUsage);
      this.sessionCost = addTurnCost(this.sessionCost, model, turnUsage);
    }
    console.log();
    // THE CAP ASKS (2026-09-28, the owner): a turn that spent its tool rounds
    // ends with the answer its last, tool-less call gave, and the interactive
    // chat offers another budget. Yes continues from the conversation with a
    // fresh budget (a turn that brought no answer is sent again); no, or
    // anything else, ends the turn. Only the interactive chat asks: `-p`,
    // `serve`, the daemon and ACP end with the answer and the finish reason.
    // The Python chat asks the same (`session.py`, `_keep_going`).
    if (turnFinish?.reason === TOOL_ROUND_LIMIT) {
      if (await this.keepGoing(turnFinish.rounds)) {
        await this.streamToTerminal(printer.plainText.trim() ? CONTINUE_MESSAGE : content);
      } else if (!this.atTerminal() && printer.plainText.trim()) {
        this.notice('warn', toolRoundLimitSentence(turnFinish.rounds, true));
      }
    }
  }

  /**
   * The chat's question after a turn that spent its tool rounds
   * (`core/tool-budget.ts`, `continueQuestion`): yes is the default, and
   * Ctrl+C, Ctrl+D or Esc answer no. Asked only at a terminal.
   */
  private async keepGoing(rounds: number | undefined): Promise<boolean> {
    if (!this.atTerminal()) return false;
    const { paint, palette } = this.theme;
    const used = rounds ?? (this.agent as unknown as { maxToolIterations?: number } | null)?.maxToolIterations ?? 0;
    const answer = await promptLine(`  ${paint.fg(palette.text, continueQuestion(used))} `);
    if (answer === null) return false;
    return /^(y(es)?)?$/i.test(answer.trim());
  }

  /**
   * Run the interactive REPL.
   *
   * At a terminal: the wordmark, the agent's card, then the input box
   * (ui/input.ts) for every message. Anything else (a pipe, a script) gets a
   * plain prompt and plain output, which is what a script can read.
   */
  async run(): Promise<void> {
    // The interactive loop, the one place a control-file write may ask (S-314),
    // and the one place a refused host is asked about (`attachHostAsker`).
    this.chatting = true;
    await this.initialize();
    const out = process.stdout;
    const tty = Boolean(process.stdin.isTTY && out.isTTY);

    // Everything the chat writes from here on, recorded (`ui/screen.ts`), and
    // tied to the screen by where the cursor starts.
    const recording = tty
      ? recordScreen([out, process.stderr], () => terminalColumns(out), () => out.rows ?? 24)
      : null;
    this.screen = recording?.screen;
    if (recording) {
      const startRow = await queryCursorRow();
      if (startRow !== null) recording.screen.anchor(startRow);
    }

    if (tty) {
      this.theme = themeFor(out, process.env, { background: await queryBackground() });
      await playWordmark(out, this.theme);
      // Before the card, so the card shows what the agent will run on.
      if (this.modelProblem) await this.offerModelAccess();
      console.log(`\n${welcomeCard(this.theme, terminalColumns(out), this.welcomeInfo()).join('\n')}\n`);
      this.sayNewAgentTip();
    } else {
      console.log(`\nWebAgents CLI - Connected to ${this.agent?.name || 'cli-agent'}`);
      // Before the first message, not after it: the alternative is a stack
      // trace in reply to "hello".
      if (this.modelProblem) console.log(this.modelProblem);
      console.log('Type /help for available commands, or start chatting.\n');
      this.sayNewAgentTip();
    }

    this.running = true;
    // Not at a terminal (a pipe, a script): ONE line reader for the whole
    // session. A reader per prompt read the whole pipe into the first one's
    // buffer and closed it, so everything after the first line was lost.
    const piped = tty ? null : readline.createInterface({ input: process.stdin, terminal: false });
    const lines = piped ? piped[Symbol.asyncIterator]() : null;
    while (this.running) {
      this.sayRecordingProblem();
      // The agent's file may have changed since the chat loaded it (S-283):
      // say so, once per version, and never reload unasked.
      this.sayFileChanged();
      let line: string | null;
      if (lines) {
        process.stdout.write('> ');
        const next = await lines.next();
        line = next.done ? null : next.value;
      } else {
        line = await this.readBox();
      }
      if (line === null) break;
      await this.handleInput(line);
    }
    piped?.close();
    // The last turn's recording, before the process ends with it (bounded:
    // an unreachable platform must not hold the terminal).
    await Promise.race([this.recording, new Promise((resolve) => setTimeout(resolve, 10_000).unref())]);
    this.sayRecordingProblem();
    this.goodbye();
    recording?.stop();
    await this.shutdown();
  }

  /**
   * The agent's skills closed, MCP servers included, so the process can end
   * (2026-09-26): `/exit` with a stdio server left its pipes open and the
   * chat hung for good. Bounded, so a server that will not close cannot hold
   * the exit either (`cli/index.ts` has the last word).
   */
  private async shutdown(): Promise<void> {
    const agent = this.agent;
    if (!agent) return;
    await Promise.race([
      agent.cleanup().catch(() => undefined),
      new Promise<void>((resolve) => setTimeout(resolve, SHUTDOWN_GRACE_MS).unref()),
    ]);
  }

  /**
   * What the box offers after `/<command> ` (spec 3.8): the next word's
   * values, by command. Read-only lookups, computed as the menu opens.
   */
  private completions(): Record<string, (args: string) => Array<{ value: string; description: string }>> {
    const agents = () => [
      ...this.folderAgents().filter((a) => !a.problem).map((a) => ({ value: a.name, description: a.description })),
      { value: BUILT_IN_AGENT, description: 'The general assistant' },
    ];
    const words = (args: string) => args.split(/\s+/).filter(Boolean);
    const keys = () => keyProviders().map((p) => ({ value: p.envVar!, description: p.id }));
    // A word after the last one the command takes closes the menu, so enter
    // then sends the line; a list command (`skills add a b`) keeps offering
    // the names not yet typed, and esc closes its menu.
    return {
      agent: (args) => {
        const [verb, ...rest] = words(args);
        if (verb === 'edit') return rest.length ? [] : agents().filter((a) => a.value !== BUILT_IN_AGENT);
        if (verb) return [];
        return [...agents(), { value: 'new', description: 'make one here' }, { value: 'edit', description: 'open its file in your editor' }];
      },
      skills: (args) => {
        const [verb, ...rest] = words(args);
        if (verb === 'add') return this.skillNamesToOffer.filter((name) => !rest.includes(name)).map((name) => ({ value: name, description: '' }));
        if (verb === 'remove') {
          return [...(this.loaded?.skills ?? []), ...(this.loaded?.skillmd ?? [])].filter((name) => !rest.includes(name)).map((name) => ({ value: name, description: '' }));
        }
        if (verb) return [];
        return [
          { value: 'list', description: 'every name an agent file can name' },
          { value: 'add', description: 'give the agent a skill' },
          { value: 'remove', description: 'take one away' },
        ];
      },
      help: (args) => (words(args).length ? [] : CHAT_COMMANDS.map((c) => ({ value: c.name, description: c.description }))),
      keys: (args) => {
        const [verb, ...rest] = words(args);
        if (verb === 'set' || verb === 'unset') return rest.length ? [] : keys();
        if (verb) return [];
        return [{ value: 'set', description: 'store a key' }, { value: 'unset', description: 'remove a stored key' }];
      },
      cron: (args) => {
        const [verb, ...rest] = words(args);
        if (verb === 'run') return rest.length ? [] : this.scheduleNames().map((name) => ({ value: name, description: '' }));
        if (verb) return [];
        return [{ value: 'run', description: 'run a schedule now' }];
      },
      memory: (args) => {
        const [verb, ...rest] = words(args);
        if (verb === 'forget') return rest.length ? [] : this.memoryKeys.map((key) => ({ value: key, description: '' }));
        if (verb) return [];
        return [{ value: 'forget', description: 'remove one of your notes' }];
      },
    };
  }

  /** The names `/skills add` completes, read once (`resolvableSkillNames`). */
  private skillNamesToOffer: string[] = [];

  /** This agent's schedule names, for `/cron run` completion, read before the box opens. */
  private scheduleNamesCache: string[] = [];

  private scheduleNames(): string[] {
    return this.scheduleNamesCache;
  }

  /** What the completers need that is async: read before the box opens, quietly. */
  private async refreshCompletionData(): Promise<void> {
    try {
      if (!this.skillNamesToOffer.length) {
        const { resolvableSkillNames } = await import('../skills/resolve.js');
        this.skillNamesToOffer = resolvableSkillNames();
      }
      const memory = this.memorySkill();
      this.memoryKeys = memory ? (await memory.ownerSummary()).recent.map((n) => n.key) : [];
      this.scheduleNamesCache = [];
      if (this.agentFile && this.loaded?.cron) {
        const { folderSchedules } = await import('./cron-action.js');
        const found = folderSchedules(this.agentFolder(), () => {}).find(({ definition }) => definition.name === this.agent?.name);
        this.scheduleNamesCache = found ? found.schedules.map((s) => s.name) : [];
      }
    } catch {
      // Completion is a convenience.
    }
  }

  /** One message from the input box, or null when the person leaves. */
  private async readBox(): Promise<string | null> {
    await this.refreshCompletionData();
    const completions = this.completions();
    const result = await promptBox({
      theme: this.theme,
      commands: [...this.commands.values()].map((c) => ({ name: c.name, description: c.description, complete: completions[c.name] })),
      // readline keeps history newest first; the box walks it oldest first.
      history: [...this.inputHistory].reverse(),
      placeholder: `Message ${this.agent?.name ?? 'the agent'}, or type / for commands`,
      footer: () => ({ left: this.footerParts() }),
      screen: this.screen,
    });
    if (result.kind === 'exit') return null;
    if (result.text.trim() && this.inputHistory[0] !== result.text) {
      this.inputHistory.unshift(result.text);
      appendChatHistory(this.historyFile, result.text);
    }
    return result.text;
  }

  /** The session's last line: how many replies, what they cost, how long. */
  private goodbye(): void {
    const { paint, palette } = this.theme;
    if (!this.turns) {
      console.log();
      return;
    }
    const parts = [
      `${this.turns} ${this.turns === 1 ? 'reply' : 'replies'}`,
      ...(this.sessionTokens ? [`${this.sessionTokens.toLocaleString('en-US')} tokens`] : []),
      ...(this.sessionCost.known ? [costWords(this.sessionCost.credits, this.sessionCost.estimated)] : []),
      duration((Date.now() - this.sessionStarted) / 1000),
    ];
    console.log(`${paint.fg(palette.faint, `✦ ${parts.join(' · ')}`)}\n`);
  }
}
