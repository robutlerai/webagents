/**
 * The chat's commands and keys (2026-09-24), the same in both SDKs.
 *
 * Names, usage and wording here are the reference. The Python chat keeps the
 * same list in `python/webagents/cli/repl/commands.py`, and
 * `python/tests/cli/test_chat_command_parity.py` reads THIS file and fails
 * when the two differ, so a command cannot be added, renamed or reworded in
 * one chat only.
 */

export interface ChatCommandSpec {
  /** Typed after the slash. */
  name: string;
  /** How it is typed, for /help. */
  usage: string;
  /** One line, for the menu and /help. */
  description: string;
}

export const CHAT_COMMANDS: readonly ChatCommandSpec[] = [
  { name: 'help', usage: '/help [command]', description: 'Show the commands and keys' },
  { name: 'new', usage: '/new', description: 'Start a new conversation' },
  { name: 'clear', usage: '/clear', description: 'Start a new conversation and clear the screen' },
  { name: 'resume', usage: '/resume [number]', description: 'Continue an earlier conversation in this folder' },
  { name: 'model', usage: '/model [provider/model]', description: 'Show or switch the model' },
  { name: 'agent', usage: '/agent [name]', description: "List this folder's agents, or switch to one" },
  { name: 'tools', usage: '/tools', description: 'List what the agent can use' },
  { name: 'status', usage: '/status', description: 'Account, agent, model, sandbox and folder' },
  { name: 'login', usage: '/login', description: 'Sign in to Robutler' },
  { name: 'logout', usage: '/logout', description: 'Sign out of Robutler' },
  { name: 'keys', usage: '/keys [set|unset NAME]', description: 'Model provider keys, and where each comes from' },
  { name: 'sandbox', usage: '/sandbox', description: "What the agent's commands are allowed to do" },
  { name: 'publish', usage: '/publish', description: 'Publish this agent to Robutler, or update it' },
  { name: 'exit', usage: '/exit', description: 'Leave the chat' },
];

export const CHAT_KEYS: ReadonlyArray<readonly [string, string]> = [
  ['enter', 'send'],
  ['alt+enter', 'new line (or end a line with \\)'],
  ['↑ ↓', 'earlier messages'],
  ['tab', 'complete a command'],
  ['esc', 'stop a reply; twice in the box, clear it'],
  ['ctrl+c', 'clear the box; twice, leave'],
];

/** The spec for `name`, with or without its slash. */
export function chatCommand(name: string): ChatCommandSpec | undefined {
  const bare = name.replace(/^\//, '').toLowerCase();
  return CHAT_COMMANDS.find((command) => command.name === bare);
}
