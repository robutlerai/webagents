/**
 * The chat's commands and keys (2026-09-24), the same in both SDKs.
 *
 * Names, usage, wording, groups and details here are the reference held by
 * the shared fixture `python/tests/fixtures/cli/chat_commands.json`: the
 * Python chat keeps the same list in `python/webagents/cli/repl/commands.py`,
 * and both suites compare their table with the fixture
 * (`tests/unit/cli/chat-commands-fixture-interactive.test.ts`,
 * `tests/cli/test_chat_commands_fixture_interactive.py`), so a command cannot
 * be added, renamed or reworded in one chat only.
 *
 * The order is the menu's, and the order within each /help group
 * (2026-09-26, interactive-mode spec 3.8).
 */

/** The /help headings, in order. */
export type ChatCommandGroup = 'agent' | 'conversation' | 'account' | 'chat';

export const CHAT_GROUPS: ReadonlyArray<readonly [ChatCommandGroup, string]> = [
  ['agent', 'This agent'],
  ['conversation', 'Conversation'],
  ['account', 'Account'],
  ['chat', 'Chat'],
];

export interface ChatCommandSpec {
  /** Typed after the slash. */
  name: string;
  /** How it is typed, for /help. */
  usage: string;
  /** One line, for the menu and /help. */
  description: string;
  /** Which /help heading it sits under. */
  group: ChatCommandGroup;
  /** Its forms, for `/help <command>`: one line each. */
  details: readonly string[];
}

export const CHAT_COMMANDS: readonly ChatCommandSpec[] = [
  { name: 'help', usage: '/help [command]', description: 'Show the commands and keys', group: 'chat', details: [] },
  {
    name: 'agent',
    usage: '/agent [name]',
    description: "List this folder's agents, switch to one, or make one",
    group: 'agent',
    details: [
      '/agent                                    the agents here',
      '/agent <name>                             switch to one',
      '/agent new <name> [chatbot|tool-agent]    make one here',
      '/agent edit [name]                        open its file in your editor, then use it',
    ],
  },
  {
    name: 'skills',
    usage: '/skills [list|add|remove]',
    description: "This agent's skills, and adding or removing one",
    group: 'agent',
    details: [
      '/skills list',
      '/skills add <name>... | <owner/repo | git URL | folder> [--skill <name>]',
      '/skills remove <name>...',
    ],
  },
  { name: 'reload', usage: '/reload', description: 'Read the agent file again and use it', group: 'agent', details: [] },
  {
    name: 'model',
    usage: '/model [provider/model] [--save]',
    description: 'Show or switch the model',
    group: 'agent',
    details: [
      '/model                            which model, and how it is reached',
      '/model <provider/model>           switch, for this chat',
      '/model <provider/model> --save    switch, and keep it in the agent file',
    ],
  },
  {
    name: 'rounds',
    usage: '/rounds [n] [--save]',
    description: 'Show or set the tool rounds one turn may run',
    group: 'agent',
    details: [
      '/rounds               how many, and where that comes from',
      '/rounds <n>           set it, for this chat',
      '/rounds <n> --save    set it, and keep it in the agent file',
    ],
  },
  { name: 'tools', usage: '/tools', description: 'List what the agent can use, and who else may', group: 'agent', details: [] },
  { name: 'mcp', usage: '/mcp', description: 'The MCP servers this agent uses', group: 'agent', details: [] },
  { name: 'access', usage: '/access', description: 'Who may call this agent, and what each caller gets', group: 'agent', details: [] },
  {
    name: 'cron',
    usage: '/cron [run <name>]',
    description: "This agent's schedules; run one now",
    group: 'agent',
    details: ['/cron                 the schedules, as the daemon runs them', '/cron run <name>      run one now, and deliver its result'],
  },
  {
    name: 'memory',
    usage: '/memory [forget <key>]',
    description: 'What this agent remembers',
    group: 'agent',
    details: ['/memory                  the notes it keeps, and where', '/memory forget <key>     remove one of your notes'],
  },
  { name: 'sandbox', usage: '/sandbox', description: "What the agent's commands are allowed to do", group: 'agent', details: [] },
  { name: 'status', usage: '/status', description: 'Account, agent, model, sandbox, folder and Robutler', group: 'agent', details: [] },
  { name: 'new', usage: '/new', description: 'Start a new conversation', group: 'conversation', details: [] },
  { name: 'clear', usage: '/clear', description: 'Start a new conversation and clear the screen', group: 'conversation', details: [] },
  { name: 'resume', usage: '/resume [number]', description: 'Continue an earlier conversation in this folder', group: 'conversation', details: [] },
  { name: 'undo', usage: '/undo', description: 'Put back the files your last message or command changed', group: 'conversation', details: [] },
  { name: 'rewind', usage: '/rewind [number]', description: 'Put the folder back as it was before an earlier message', group: 'conversation', details: [] },
  { name: 'login', usage: '/login', description: 'Sign in to Robutler', group: 'account', details: [] },
  { name: 'logout', usage: '/logout', description: 'Sign out of Robutler', group: 'account', details: [] },
  { name: 'keys', usage: '/keys [set|unset NAME]', description: 'Model provider keys, and where each comes from', group: 'account', details: [] },
  {
    name: 'secrets',
    usage: '/secrets [set|remove NAME]',
    description: 'Secrets for MCP servers, and adding or removing one',
    group: 'account',
    details: [
      '/secrets                  the names stored on this machine',
      '/secrets set <NAME>       store one, asked for with echo off; an MCP server uses it as ${secret:NAME}',
      '/secrets remove <NAME>    remove a stored one',
    ],
  },
  {
    name: 'publish',
    usage: '/publish [--dry-run]',
    description: 'Publish this agent to Robutler, or update it',
    group: 'account',
    details: ['/publish              create it, or update the linked one after asking', '/publish --dry-run    show what would be sent, and send nothing'],
  },
  { name: 'exit', usage: '/exit', description: 'Leave the chat', group: 'chat', details: [] },
];

/**
 * The commands whose arguments the box completes (interactive-mode spec 3.8,
 * 2026-09-26): after `/<command> ` the menu stays open with the values the
 * chat offers for the next word, and enter inserts one rather than running
 * the command. The same list in the Python box (`cli/repl/commands.py`),
 * pinned by the fixture's `completion`.
 */
export const COMPLETED_COMMANDS: readonly string[] = ['agent', 'skills', 'help', 'keys', 'cron', 'memory'];

/**
 * The typed-line history both chats keep (owner decision D1, S-291,
 * 2026-09-26): one file per profile, in the profile's own folder, readable
 * by its owner only. The Python box writes it through prompt_toolkit's
 * `FileHistory`, so the format is that one's: a `# <timestamp>` line, the
 * entry's lines each prefixed with `+`, and a blank line between entries.
 * The TypeScript chat reads and writes the same file (`cli/chat-history.ts`).
 */
export const CHAT_HISTORY = { file: 'history', dirMode: 0o700, fileMode: 0o600, keep: 1000 } as const;

export const CHAT_KEYS: ReadonlyArray<readonly [string, string]> = [
  ['enter', 'send'],
  ['alt+enter', 'new line (or end a line with \\)'],
  ['↑ ↓', 'earlier messages'],
  ['tab', 'complete a command'],
  ['esc', 'stop a reply; twice in the box, clear it'],
  ['ctrl+c', 'clear the box; twice, leave'],
];

/**
 * Names a person may type that are not commands, and the command each
 * became: the unknown-command path answers "Did you mean /agent edit?" for
 * `/edit`, where a nearest-name search would not find it.
 */
export const MOVED_COMMANDS: Readonly<Record<string, string>> = { edit: 'agent edit' };

/** The last line of /help: the ways to run an agent outside the chat. */
export const OUTSIDE_THE_CHAT =
  'Outside the chat: webagents serve (HTTP), webagents mcp serve (MCP clients), webagents acp (code editors), webagents daemon (every agent here, with schedules).';

/** The spec for `name`, with or without its slash. */
export function chatCommand(name: string): ChatCommandSpec | undefined {
  const bare = name.replace(/^\//, '').toLowerCase();
  return CHAT_COMMANDS.find((command) => command.name === bare);
}

/** Whether a command takes no arguments at all (its usage names none): an argument is then a usage error. */
export function takesNoArguments(spec: ChatCommandSpec): boolean {
  return !/[[<]/.test(spec.usage);
}

/** The width the /help columns are padded to: the longest usage, plus two. */
export function helpColumnWidth(): number {
  return Math.max(...CHAT_COMMANDS.map((c) => c.usage.length)) + 2;
}

/**
 * /help as plain rows, in the fixture's order: a heading per group, its
 * commands, the keys, then the "outside the chat" line. `kind` says how a
 * row is drawn; the text is exactly what the fixture pins.
 */
export type HelpRow =
  | { kind: 'heading'; text: string }
  | { kind: 'command'; text: string; usage: string; description: string }
  | { kind: 'key'; text: string; key: string; what: string }
  | { kind: 'blank'; text: '' }
  | { kind: 'outside'; text: string };

export function helpRows(): HelpRow[] {
  const width = helpColumnWidth();
  const rows: HelpRow[] = [];
  for (const [group, heading] of CHAT_GROUPS) {
    rows.push({ kind: 'heading', text: heading });
    for (const c of CHAT_COMMANDS) {
      if (c.group !== group) continue;
      rows.push({ kind: 'command', text: `  ${c.usage.padEnd(width)}${c.description}`, usage: c.usage.padEnd(width), description: c.description });
    }
    rows.push({ kind: 'blank', text: '' });
  }
  rows.push({ kind: 'heading', text: 'Keys' });
  for (const [key, what] of CHAT_KEYS) rows.push({ kind: 'key', text: `  ${key.padEnd(width)}${what}`, key: key.padEnd(width), what });
  rows.push({ kind: 'blank', text: '' });
  rows.push({ kind: 'outside', text: OUTSIDE_THE_CHAT });
  return rows;
}
