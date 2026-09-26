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
import type { RunResponse, StreamChunk } from '../core/types';
import type { LLMProvider } from '../skills/llm/providers';
import type { Message } from '../uamp/types';
import { folderAgents, type FolderAgent } from './agent-files';
import { presentFailure, type FailureText } from './failures';
import { CHAT_COMMANDS, CHAT_KEYS, chatCommand } from './chat-commands';
import { cliCommand, resolvePlatformUrl } from './config-store';
import { describeAccess, keyProviders, type ModelAccess } from './model-access';
import { promptLine, promptSecret } from './prompt';
import { TurnPrinter } from './render';
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
 * chat runs its turns the same way (`repl/session.py`, `one_shot.py`).
 */
const LOCAL_OWNER: Record<string, unknown> = { authenticated: true, scope: 'owner', provider: 'local' };

/** The embedded agent's name (`agents/ROBUTLER.md`), offered by /agent in every folder. */
const BUILT_IN_AGENT = 'robutler';

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
 * Interactive REPL for agent conversations
 */
export class InteractiveREPL {
  private config: REPLConfig;
  private agent: BaseAgent | null = null;
  private messages: Message[] = [];
  private commands: Map<string, SlashCommand> = new Map();
  private running = false;
  /** Lines typed at the prompt, newest first, carried from one prompt to the next. */
  private inputHistory: string[] = [];
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

  /**
   * The agent file /agent chose: a path, null for the built-in agent, or
   * undefined to find it the usual way (`-a`, then AGENT.md).
   */
  private selectedFile: string | null | undefined;
  
  /**
   * A model the person chose, by `--model` or by `/model`, kept apart from the
   * REPL default (see load). Not readonly: `/model` sets it, because
   * `initialize()` re-applies the agent file's `model:` and would otherwise
   * undo the switch (2026-09-24).
   */
  private explicitModel: string | undefined;

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

    this.setupCommands();
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
      model: (args) => this.commandModel(args),
      agent: (args) => this.commandAgent(args),
      tools: async () => this.commandTools(),
      status: () => this.commandStatus(),
      login: () => this.signIn(),
      logout: () => this.commandLogout(),
      keys: (args) => this.commandKeys(args),
      sandbox: async () => this.commandSandbox(),
      publish: () => this.commandPublish(),
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

  /** `/help`: every command and key; `/help <command>`: that command's usage. */
  private commandHelp(args: string): void {
    const { paint, palette } = this.theme;
    const asked = args.trim();
    if (asked) {
      const spec = chatCommand(asked);
      if (!spec) {
        this.notice('error', `Unknown command /${asked.replace(/^\//, '')}.`, 'Type /help for the list.');
        return;
      }
      this.notice('info', spec.usage, spec.description);
      return;
    }
    const width = Math.max(...CHAT_COMMANDS.map((c) => c.usage.length)) + 2;
    const lines = [paint.bold(paint.fg(palette.text, 'Commands'))];
    for (const c of CHAT_COMMANDS) {
      lines.push(`  ${paint.fg(palette.accent, c.usage.padEnd(width))}${paint.fg(palette.muted, c.description)}`);
    }
    lines.push('', paint.bold(paint.fg(palette.text, 'Keys')));
    for (const [key, what] of CHAT_KEYS) {
      lines.push(`  ${paint.fg(palette.text, key.padEnd(width))}${paint.fg(palette.muted, what)}`);
    }
    console.log(`\n${lines.join('\n')}\n`);
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
    this.sessionTokens = 0;
    this.inputTokens = 0;
    this.outputTokens = 0;
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
  private snapshotBeforeTurn(message: string): void {
    if (!this.canChangeFiles) return;
    const folder = this.agentFolder();
    if (snapshotsOffReason(folder)) return;
    try {
      this.turnSnapshots.push(takeSnapshot(folder, turnLabel(message)).id);
    } catch (error) {
      if (!this.snapshotsNoticed) this.notice('warn', UNDO_WORDS.snapshotFailed((error as Error).message));
      this.snapshotsNoticed = true;
    }
  }

  /** A yes to `question`, asked only at a terminal (as `/publish` asks). */
  private async confirm(question: string): Promise<boolean> {
    if (!process.stdin.isTTY) return false;
    const { paint, palette } = this.theme;
    const answer = await promptLine(`  ${paint.fg(palette.text, question)}`);
    return /^y(es)?$/i.test((answer ?? '').trim());
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
    this.sessionTokens = this.inputTokens + this.outputTokens;
    this.printRecap();
    this.notice('ok', `Continuing the conversation from ${whenLabel(chosen.updatedAt)} (${this.messages.length} messages).`);
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

  /** `/model`: which model and how it is reached; `/model <provider/model>`: switch. */
  private async commandModel(args: string): Promise<void> {
    if (!args.trim()) {
      this.notice('info', `Model: ${this.modelLabel() || '(none)'}`, 'Switch with /model <provider/model>.');
      return;
    }
    // REBUILD THE AGENT, do not just set the field: the already-built LLM
    // skill keeps its own model. Through `explicitModel`, because
    // `initialize()` re-applies the agent file's `model:` otherwise.
    const previous = this.explicitModel;
    this.explicitModel = args.trim();
    try {
      await this.initialize();
      // A model with no way to run here (no key, not signed in) is not a
      // switch; keep the one that works.
      if (this.modelProblem) throw new Error(this.modelProblem);
      this.notice('ok', `Model set to ${this.modelLabel()}`);
    } catch (error) {
      this.explicitModel = previous;
      this.notice('error', `Could not switch model: ${(error as Error).message}`);
      await this.initialize().catch(() => {});
    }
  }

  /** The agent files in this folder: AGENT.md and AGENT-<name>.md, with their names. */
  private folderAgents(): FolderAgent[] {
    return folderAgents(process.cwd());
  }

  /** `/agent`: this folder's agents and the built-in one; `/agent <name>`: switch to it. */
  private async commandAgent(args: string): Promise<void> {
    const { paint, palette } = this.theme;
    const agents = this.folderAgents();
    const current = this.agent?.name;
    const wanted = args.trim();
    if (!wanted) {
      const width = Math.max(10, ...agents.map((a) => a.name.length), BUILT_IN_AGENT.length) + 2;
      const lines = [paint.bold(paint.fg(palette.text, 'Agents'))];
      const row = (name: string, where: string, description: string) => {
        const mark = name === current ? paint.fg(palette.accent, '●') : paint.fg(palette.faint, '○');
        return `  ${mark} ${paint.bold(paint.fg(palette.text, name.padEnd(width)))}${paint.fg(palette.faint, where.padEnd(18))}${paint.fg(palette.muted, truncate(description, Math.max(10, (terminalColumns()) - width - 24)))}`;
      };
      for (const a of agents) lines.push(row(a.name, path.basename(a.file), a.description));
      lines.push(row(BUILT_IN_AGENT, 'built in', 'The general assistant'));
      lines.push('', paint.fg(palette.faint, '  Switch with /agent <name>.'));
      console.log(`\n${lines.join('\n')}\n`);
      return;
    }
    const target = agents.find((a) => a.name === wanted);
    if (!target && wanted !== BUILT_IN_AGENT) {
      this.notice('error', `There is no agent called ${wanted} in this folder.`, 'Type /agent to see the list.');
      return;
    }
    if (wanted === current) {
      this.notice('info', `Already talking to ${current}.`);
      return;
    }
    this.selectedFile = target ? target.file : null;
    this.explicitModel = undefined;
    await this.initialize();
    this.startNewConversation(false);
    if (process.stdout.isTTY) {
      console.log(`\n${welcomeCard(this.theme, terminalColumns(), this.welcomeInfo()).join('\n')}`);
    }
    this.notice('ok', `Now talking to ${this.agent?.name ?? wanted}.`, this.modelProblem);
  }

  /** `/tools`: what the agent can use, by name. */
  private commandTools(): void {
    const tools = this.toolList();
    if (!tools.length) {
      this.notice('info', 'This agent has no tools.', 'Add skills to its AGENT.md, for example `filesystem`.');
      return;
    }
    const { paint, palette } = this.theme;
    const width = Math.min(28, Math.max(...tools.map((t) => t.name.length))) + 2;
    const room = (terminalColumns()) - width - 6;
    const lines = [paint.bold(paint.fg(palette.text, `Tools (${tools.length})`))];
    for (const tool of tools) {
      const description = (tool.description ?? '').split('\n')[0];
      lines.push(
        `  ${paint.fg(palette.success, '●')} ${paint.bold(paint.fg(palette.text, truncate(tool.name, width - 2).padEnd(width)))}${paint.fg(palette.muted, truncate(description, Math.max(10, room)))}`,
      );
    }
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
  doctorFacts(): { agent: string; agentFile?: string; modelOk: boolean; model: string; hasShell: boolean } {
    const access = this.modelAccess;
    const hasShell = Boolean(
      (this.agent as unknown as { skills?: Array<{ name?: string }> } | null)?.skills?.some((s) => s.name === 'ShellSkill'),
    );
    let model: string;
    if (this.modelProblem || !access || access.kind === 'none') {
      model = `none (${access?.kind === 'proxy' ? 'not signed in' : (access?.reason ?? 'no model')})`;
    } else if (access.kind === 'proxy') {
      model = `${this.modelLabel()}, paid from your Robutler credits`;
    } else {
      model = this.providerEnvVar ? `${this.modelLabel()}, with your ${this.providerEnvVar}` : this.modelLabel();
    }
    return {
      agent: this.agent?.name ?? 'agent',
      ...(this.agentFile ? { agentFile: path.basename(this.agentFile) } : {}),
      modelOk: !this.modelProblem && Boolean(access) && access?.kind !== 'none',
      model,
      hasShell,
    };
  }

  /** How the agent reaches its model, in words: for /status. */
  private modelRoute(): string {
    const access = this.modelAccess;
    if (this.modelProblem || !access || access.kind === 'none') {
      const reason = access?.kind === 'proxy' ? 'not signed in' : (access?.reason ?? 'no model');
      return `none (${reason}). /login, or /keys set <NAME>.`;
    }
    if (access.kind === 'proxy') return `${this.modelLabel()}, paid from your Robutler credits`;
    return this.providerEnvVar ? `${this.modelLabel()}, with your ${this.providerEnvVar}` : this.modelLabel();
  }

  /** `/status`: the account, the agent, its model, the sandbox, the folder, the conversation. */
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
    const rows: Array<[string, string]> = [
      ['Account', account],
      ['Agent', this.agentFile ? `${name} (${path.basename(this.agentFile)})` : `${name} (built in)`],
      ['Model', this.modelRoute()],
      ['Sandbox', this.sandboxSummary().headline],
      ['Folder', shortPath(this.agentFolder())],
      [
        'Conversation',
        `${this.messages.length} messages${this.sessionTokens ? `, ${compactNumber(this.sessionTokens)} tokens` : ''}` +
          (this.platformChatId ? ', also on Robutler' : ''),
      ],
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
    const { clearToken, getToken, TOKEN_ENV_VAR } = await import('./credentials.js');
    const [portalUrl] = resolvePlatformUrl();
    if (!(await getToken())) {
      this.notice('info', 'Not signed in.');
      return;
    }
    await clearToken();
    await this.initialize();
    if (process.env[TOKEN_ENV_VAR]) {
      this.notice('warn', `${TOKEN_ENV_VAR} is set in this shell, and it keeps you signed in.`, 'Unset it to sign out completely.');
      return;
    }
    this.notice('ok', `Signed out of ${portalUrl.replace(/^https?:\/\//, '')}.`, this.modelProblem ?? `${this.agent?.name ?? 'The agent'} runs on ${this.modelLabel()}.`);
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
      const value = (await promptSecret(`  ${paint.fg(palette.muted, `${name} (hidden):`)} `)).trim();
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
    try {
      removed = await (await providerKeyStore()).delete(name);
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
  }

  /** What the agent's commands may do, in one line and a detail: for /sandbox and /status. */
  private sandboxSummary(): { kind: 'ok' | 'warn' | 'info'; headline: string; detail?: string } {
    const runsCommands = (this.agent as unknown as { skills?: Array<{ name?: string }> } | null)?.skills?.some((s) => s.name === 'ShellSkill');
    if (!runsCommands) {
      return { kind: 'info', headline: 'Not needed: this agent cannot run commands.' };
    }
    return {
      kind: 'warn',
      headline: 'Off: commands run with your permissions.',
      detail: 'This SDK does not confine the shell. Keep agents with `shell` to folders you trust.',
    };
  }

  /** `/sandbox`: what the agent's commands are allowed to do. */
  private commandSandbox(): void {
    const summary = this.sandboxSummary();
    this.notice(summary.kind, `Sandbox: ${summary.headline}`, summary.detail);
  }

  /** `/publish`: send this folder's agent to Robutler, or update the linked one. */
  private async commandPublish(): Promise<void> {
    const { paint, palette } = this.theme;
    if (!this.agentFile) {
      this.notice('warn', 'Publishing needs an AGENT.md in this folder.', 'Create one with `webagents init`, then /publish.');
      return;
    }
    const { publishAgent } = await import('./publish.js');
    console.log();
    const result = await publishAgent(this.agentFile, {
      ok: (line) => this.notice('ok', line),
      print: (line) => console.log(`  ${paint.fg(palette.muted, line)}`),
      error: (line) => this.notice('error', line),
      confirm: async (question) => {
        if (!process.stdin.isTTY) return false;
        const answer = await promptLine(`  ${paint.fg(palette.text, question)} ${paint.fg(palette.faint, '[y/N]')} `);
        return /^y(es)?$/i.test((answer ?? '').trim());
      },
    });
    if (result.ok) console.log();
  }

  /**
   * Register a slash command
   */
  registerCommand(command: SlashCommand): void {
    this.commands.set(command.name, command);
  }
  
  /**
   * Initialize the agent
   */
  async initialize(): Promise<void> {
    const { getRobutlerContent, parseAgentMarkdown, findAgentFile } = await import(
      '../agents/index.js'
    );
    const { resolveSkillsByName, skillEntryParts } = await import('../skills/resolve.js');
    const { readFileSync } = await import('node:fs');

    let agentName = this.config.agentName || 'robutler';
    let instructions = this.config.instructions;
    let declaredSkills: string[] = [];
    // The entries as written, config included (`- rest: {sign: always}`).
    let declaredEntries: Array<string | Record<string, unknown>> = [];
    // The file's `access:` block (ADR-0045), when it has one.
    let declaredAccess: unknown;
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
    this.agentFile = localFile ?? undefined;
    if (localFile) {
      try {
        // With its path, so a file with no `name:` is named for its file
        // (`AGENT-helper.md` -> helper), as /agent lists it.
        const parsed = parseAgentMarkdown(readFileSync(localFile, 'utf-8'), localFile);
        agentName = parsed.name !== 'unknown' ? parsed.name : agentName;
        instructions = instructions ?? parsed.instructions;
        declaredSkills = parsed.skills;
        declaredEntries = parsed.skillEntries;
        declaredAccess = parsed.access;
        declaredModel = parsed.model;
        this.agentDescription = parsed.description ?? '';
      } catch (error) {
        console.warn(`Could not read ${localFile}: ${(error as Error).message}`);
      }
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
        this.agentDescription = parsed.description ?? '';
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
    this.proxyUrl = platformLlmUrl(resolvePlatformUrl()[0]);
    const proxy = {
      proxyUrl: this.proxyUrl,
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
    this.sessionBackend = sessionEntry?.config.backend === 'robutler' ? 'robutler' : 'local';
    const agentEntries = declaredEntries.filter((entry) => skillEntryParts(entry).name?.toLowerCase() !== 'session');
    const { skills, byName, unknown, failed } = await resolveSkillsByName(agentEntries, {
      model: chosenModel,
      proxy,
      apiKeys,
      agentDir,
      personToken,
    });
    // An agent with these can change the folder, so each message to it is
    // preceded by a snapshot `/undo` can put back (`checkpoints.ts`).
    this.canChangeFiles = [...byName.keys()].some((name) => ['filesystem', 'shell'].includes(name.toLowerCase()));
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
    this.declaredLLM = declaredSkills
      .filter((name) => !notBuilt.has(name))
      .map((name) => findProvider(name))
      .find((provider): provider is LLMProvider => provider !== undefined);
    this.modelProblem = undefined;

    const declared = this.declaredLLM;
    if (declared && !missingProviderKey([declared.id], keyEnv)) {
      // The agent's own provider, with its key here: its choice stands.
      const viaRobutler = declared.credential === 'platform';
      this.modelAccess = { kind: viaRobutler ? 'proxy' : 'direct', model: chosenModel, provider: declared };
      this.providerEnvVar = declared.credential === 'api-key' ? declared.envVar : undefined;
      if (viaRobutler && !(await signedIn())) {
        this.modelProblem = `This agent runs on Robutler's models. Sign in with \`${cliCommand('login')}\`.`;
      }
      this.config.model = chosenModel;
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
      this.modelAccess = access;
      this.providerEnvVar = access.kind === 'direct' ? access.provider?.envVar : undefined;
      if (access.kind === 'direct') {
        const built = await resolveSkillsByName([access.provider!.id], { model: access.model, apiKeys });
        skills.push(...built.skills);
        if (built.failed.length) {
          this.modelProblem = `Could not load the ${access.provider!.id} client: ${built.failed[0].reason}`;
        }
      } else if (access.kind === 'proxy') {
        const { LLMProxySkill } = await import('../skills/llm/proxy/skill.js');
        skills.push(new LLMProxySkill({ model: access.model, ...proxy }) as unknown as (typeof skills)[number]);
      } else {
        this.modelProblem = unavailableMessage(access);
      }
      this.config.model = access.model ?? chosenModel;
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

    this.agent = new BaseAgent({
      name: agentName,
      instructions,
      model: this.config.model,
      skills: (access ? [...skills, access.skill] : skills) as never,
    });
    if (access) {
      try {
        applyAccessTools(access.policy, byName);
      } catch (err) {
        if (err instanceof AccessConfigError) throw new AgentFileError(`${localFile}: ${err.message}`);
        throw err;
      }
    }

    await this.agent.initialize();
  }
  
  /**
   * Send a message and get response
   */
  async sendMessage(content: string): Promise<RunResponse> {
    if (!this.agent) {
      throw new Error('Agent not initialized');
    }
    
    // Add user message
    this.messages.push({ role: 'user', content });
    
    // Get response
    const response = await this.agent.run(this.messages, { auth: LOCAL_OWNER });
    
    // Add assistant message
    this.messages.push({ role: 'assistant', content: response.content });
    
    return response;
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
    const conversation: Message[] = [...this.messages, { role: 'user', content }];
    yield* this.agent.runStreaming(conversation, { ...(signal ? { signal } : {}), auth: LOCAL_OWNER });
  }

  /**
   * Adds a turn to the conversation: the message, and the answer's text if
   * there was any. A turn that failed before it said anything leaves no trace,
   * so the next message is not sent after an unanswered one.
   */
  private recordTurn(content: string, answer: string, failed: boolean): void {
    if (failed && !answer) return;
    this.messages.push({ role: 'user', content });
    if (answer) this.messages.push({ role: 'assistant', content: answer });
  }

  /**
   * Every event of one turn, for `-p --output-format stream-json`: text
   * deltas, tool calls and their results, then `done` or `error`.
   */
  async *streamTurn(content: string): AsyncGenerator<StreamChunk, void, unknown> {
    let answer = '';
    let failed = false;
    for await (const chunk of this.turnChunks(content)) {
      if (chunk.type === 'delta' && chunk.delta) answer += chunk.delta;
      if (chunk.type === 'error') failed = true;
      yield chunk;
    }
    this.recordTurn(content, answer, failed);
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
    for await (const chunk of this.turnChunks(content)) {
      if (chunk.type === 'delta' && chunk.delta) {
        answer += chunk.delta;
        yield chunk.delta;
      } else if (chunk.type === 'done' && chunk.response) {
        response = chunk.response;
      } else if (chunk.type === 'error') {
        this.recordTurn(content, answer, true);
        throw chunk.error ?? new Error('The model returned an error.');
      }
    }
    this.recordTurn(content, answer, false);
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
    
    // Check for slash command
    if (trimmed.startsWith('/')) {
      const parts = trimmed.slice(1).split(/\s+/);
      const cmdName = parts[0].toLowerCase();
      const args = parts.slice(1).join(' ');
      
      const command = this.commands.get(cmdName);
      if (command) {
        await command.handler(args);
      } else {
        this.notice('error', `Unknown command /${cmdName}.`, 'Type / to see the commands, or /help.');
      }
      return;
    }
    
    // No model: say so and how to get one, instead of a model error per message.
    if (this.modelProblem) {
      this.notice('warn', this.modelProblem, 'Type /login to sign in without leaving the chat.');
      return;
    }
    const message = this.expandFileReferences(trimmed);
    this.snapshotBeforeTurn(message);

    // Send message to agent
    try {
      if (this.config.streaming) {
        await this.streamToTerminal(message);
      } else {
        const printer = new TurnPrinter({ theme: this.theme, explainError: (text) => this.explainFailure(text) });
        printer.start();
        const response = await this.sendMessage(message);
        printer.feed({ type: 'delta', delta: `${response.content}\n` });
        printer.finish();
        console.log();
      }
    } catch (error) {
      const { headline, hint } = this.explainFailure((error as Error).message);
      this.notice('error', headline, hint);
    } finally {
      this.saveConversation();
      this.recordOnRobutler();
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
    });
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
    if (answer === null) return;
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
      kept += `; \`webagents secrets unset ${envVar}\` removes it.`;
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

  /** The agent's tools, by name. */
  private toolList(): Array<{ name: string; description?: string }> {
    const registry = (this.agent as unknown as { toolRegistry?: Map<string, { name: string; description?: string }> } | null)
      ?.toolRegistry;
    // By name, as the Python chat lists them: registration order is each
    // SDK's own business, and the list reads the same in both.
    return registry ? [...registry.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) : [];
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

  /** The left side of the box's footer: who, which model, what it has cost so far, where. */
  private footerParts(): string[] {
    const parts = [this.agent?.name ?? 'agent'];
    const model = this.modelLabel();
    if (model) parts.push(model);
    if (this.sessionTokens) parts.push(`${compactNumber(this.sessionTokens)} tokens`);
    // The folder last and short: its tail is the part that says where you are.
    parts.push(truncateStart(shortPath(process.cwd()), 28));
    return parts;
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
  private async streamToTerminal(content: string): Promise<void> {
    const controller = new AbortController();
    const printer = new TurnPrinter({ theme: this.theme, explainError: (message) => this.explainFailure(message) });
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
    try {
      for (;;) {
        // Raced, so an interrupt shows at once even while a tool is running.
        const step = await Promise.race([chunks.next(), aborted]);
        if (step === 'aborted' || step.done) break;
        // The model request fails when it is cancelled; that is not an error
        // to show.
        if (step.value.type === 'error' && controller.signal.aborted) break;
        printer.feed(step.value);
      }
    } finally {
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
    this.recordTurn(content, printer.plainText, printer.failed);
    // A REPLY IS SOMETHING SAID (2026-09-25): the session's last line counted
    // every turn, so a chat whose only message was refused ended "1 reply".
    // A turn counts when it said something, or ended without failing or
    // being stopped. The Python chat counts the same way.
    if (printer.plainText.trim() || (!printer.failed && !controller.signal.aborted)) this.turns += 1;
    this.sessionTokens += printer.usageTokens ?? 0;
    this.inputTokens += printer.usageSplit.input;
    this.outputTokens += printer.usageSplit.output;
    console.log();
  }

  /**
   * Run the interactive REPL.
   *
   * At a terminal: the wordmark, the agent's card, then the input box
   * (ui/input.ts) for every message. Anything else (a pipe, a script) gets a
   * plain prompt and plain output, which is what a script can read.
   */
  async run(): Promise<void> {
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
    } else {
      console.log(`\nWebAgents CLI - Connected to ${this.agent?.name || 'cli-agent'}`);
      // Before the first message, not after it: the alternative is a stack
      // trace in reply to "hello".
      if (this.modelProblem) console.log(this.modelProblem);
      console.log('Type /help for available commands, or start chatting.\n');
    }

    this.running = true;
    // Not at a terminal (a pipe, a script): ONE line reader for the whole
    // session. A reader per prompt read the whole pipe into the first one's
    // buffer and closed it, so everything after the first line was lost.
    const piped = tty ? null : readline.createInterface({ input: process.stdin, terminal: false });
    const lines = piped ? piped[Symbol.asyncIterator]() : null;
    while (this.running) {
      this.sayRecordingProblem();
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
  }

  /** One message from the input box, or null when the person leaves. */
  private async readBox(): Promise<string | null> {
    const result = await promptBox({
      theme: this.theme,
      commands: [...this.commands.values()].map((c) => ({ name: c.name, description: c.description })),
      // readline keeps history newest first; the box walks it oldest first.
      history: [...this.inputHistory].reverse(),
      placeholder: `Message ${this.agent?.name ?? 'the agent'}, or type / for commands`,
      footer: () => ({ left: this.footerParts() }),
      screen: this.screen,
    });
    if (result.kind === 'exit') return null;
    if (result.text.trim() && this.inputHistory[0] !== result.text) this.inputHistory.unshift(result.text);
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
      duration((Date.now() - this.sessionStarted) / 1000),
    ];
    console.log(`${paint.fg(palette.faint, `✦ ${parts.join(' · ')}`)}\n`);
  }
}
