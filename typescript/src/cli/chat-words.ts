/**
 * What the chat says when a command reads or changes the agent's file
 * (2026-09-26, interactive-mode spec section 3): one table of sentences, as
 * templates, the same in both SDKs.
 *
 * The Python chat keeps the same table in `cli/repl/chat_words.py`, and the
 * shared fixture `python/tests/fixtures/cli/chat_edits.json` (`words`) holds
 * the reference both suites compare their table with, so a sentence cannot
 * change in one chat only. `{name}`-style holes are filled by `fill`.
 */

export const CHAT_WORDS = {
  usage: 'Usage: {usage}',
  unknownCommand: 'Unknown command /{name}.',
  unknownCommandHint: 'Type / to see the commands, or /help.',
  didYouMean: 'Did you mean /{name}? Type / to see the commands.',
  notAtTerminal: '/{command} asks before it changes a file, and this chat is not at a terminal.',
  inAScript: 'In a script, use `{command}`.',
  linkRefusal: '{file} is a link or lies outside this folder, so {who} will not change it.',
  makeThisChange: 'Make this change? [y/N] ',
  makeTheseChanges: 'Make these changes? [y/N] ',
  leftAsIs: 'Left as it is.',
  keepsTalking: 'The chat keeps talking to {name}.',
  keepsAgent: 'The chat keeps the agent it had.',
  nothingEntered: 'Nothing entered; nothing stored.',
  noAgentCalled: 'There is no agent called {name} in this folder.',
  seeTheList: 'Type /agent to see the list.',
  brokenAgentRow: '✗ {name}  {file}  {sentence}',
  nowTalking: 'Now talking to {name}.',
  builtInNoFileToRead: 'The built-in robutler has no file to read again.',
  upToDate: '{name} is up to date with {file}.',
  changedHeader: '{file} changed since the chat loaded it:',
  instructionsChanged: 'instructions ({lines})',
  skillmdChanged: 'SKILL.md skills: +{added} -{gone}',
  useThisVersion: 'Use this version? [y/N] ',
  reloaded: 'Reloaded {name} from {file}.',
  toolsAdded: 'Tools added: {list}.',
  toolsGone: 'Tools gone: {list}.',
  changedSinceLoaded: '{file} changed since the chat loaded it. /reload uses the new version.',
  changedDuringReply: '{file} changed during the last reply. /reload shows what changed.',
  builtInNoFileToEdit: 'The built-in robutler has no file to edit.',
  makeOneHint: '/agent new <name> makes one.',
  saved: 'Saved {file}.',
  switchHint: 'Switch to it with /agent {name}.',
  fileIsAt: '{file} is at {path}.',
  setEditor: 'Set EDITOR to open it from here, or edit it and type /reload.',
  editorExited: 'The editor exited with {code}.',
  badName: 'Agent names use lower-case letters, digits and hyphens, for example support-bot.',
  unknownTemplate: 'Unknown template {template}.',
  templates: 'Templates: chatbot, tool-agent.',
  taken: 'There is already an agent called {name} in this folder.',
  homeFolder: 'Agent files do not go in your home folder.',
  homeFolderHint: 'Make a folder for the agent, start webagents in it, then /agent new <name>.',
  makeFile: 'Make {file} here from the {template} template? [y/N] ',
  made: 'Made {file}.',
  madeHint: '/agent edit changes its instructions; /skills add gives it skills.',
  tip: 'No agent in this folder yet. /agent new <name> makes one.',
  skillsOf: 'Skills of {name} ({file})',
  none: '(none)',
  skillmdHeading: 'SKILL.md skills',
  // Where an installed SKILL.md skill came from (2026-09-29): the lock's
  // source and short commit, or its local folder; a folder the lock does
  // not know is the person's own. Both chats showed only the folder.
  skillmdFrom: '{skill}  from {source} at {commit}',
  skillmdFromLocal: '{skill}  from local folder {source}',
  skillmdIn: '{skill}  in {folder}, your own folder',
  notLoaded: 'Not loaded',
  skillsHint: '/skills add <name> adds one; /skills list shows every name.',
  skillsOfBuiltIn: 'Skills of {name} (built in)',
  builtInSkillsHint: '/agent new <name> makes an agent file; /skills add then gives it more.',
  builtInNoFileToChange: 'The built-in robutler has no agent file to change.',
  alwaysAsks: 'The chat always asks before it installs a skill.',
  alwaysAsksHint: 'In a script, use `webagents skills add <source> --yes`.',
  planAdd: 'add {names}',
  planRemove: 'remove {names}',
  planAfter: 'skills after: {list}',
  planRemoveInstalled: 'remove {skill} from .agents/skills ({files})',
  beforeCommand: 'before {command}',
  // /model (spec 3.5)
  modelRefusal: '{name} names the {provider} skill, so it runs {provider} models.',
  // The way out has to work (B3, 2026-09-28): "/skills add google first"
  // left the file naming `openai` first, so the next /model was refused the
  // same way, round and round; and it said "Pick a openai/ model".
  modelRefusalHint: "Name a model that starts with {provider}/, or /skills remove {provider} to run another provider's.",
  modelSwitchHint: 'For this chat. /model {model} --save keeps it in {file}.',
  planModel: 'model: {model}',
  modelKept: 'Kept model: {model} in {file}.',
  modelLayout: 'The model: line in {file} is laid out in a way this command cannot change safely. Edit it by hand.',
  builtInNoFileToKeep: 'The built-in robutler has no file to keep it in.',
  continuingWithoutModel: 'Continuing without a model.',
  // /status, /publish, /sandbox (spec 3.6)
  statusPublished: 'Published as {agentName}. /publish updates it.',
  statusNotPublished: 'Not published. /publish creates it.',
  statusSignedOut: 'Not published. /login, then /publish.',
  statusBuiltIn: 'The built-in assistant is not published.',
  publishUpdate: 'Update {agentName} on {host} from {file}? [y/N] ',
  notUpdated: 'Not updated.',
  sandboxUnavailable: '{state}, but {reason}: every command is refused.',
  // The engine ships with webagents (the sandbox-engine lane, 2026-09-27):
  // {fix} is what THIS machine lacks (srt.ts `unavailableFix`).
  sandboxFix: '{fix}, or pass --no-sandbox to run commands with your permissions for this run',
  // The sandbox is on by default (2026-09-27): the state names the preset and where it came from.
  sandboxOff: '{state}: commands are not confined and run with your permissions.',
  sandboxOffFix: 'Remove `sandbox: off` (or `preset: unrestricted`) from the agent file to confine them.',
  sandboxFlagFix: 'Run without --no-sandbox to confine them.',
  sandboxDetail: 'writes: {writes}; reads: {reads}; hosts: {hosts}; local: {local}; sockets: {sockets}; env: {env}',
  sandboxScripts: '{state} for SKILL.md scripts; this agent has no shell.',
  // /tools and /access (spec 3.7)
  scopeOnlyYou: 'only you',
  scopeEveryCaller: 'every caller',
  scopeYouAnd: 'you and {groups}',
  accessTitle: 'Access (from {file})',
  accessRefusedLabel: 'Refused',
  accessGroupsLabel: 'Groups',
  accessOthersLabel: 'Others',
  accessToolsLabel: 'Tools',
  accessNobody: 'nobody yet',
  accessTrust: 'agents with TrustFlow {min} on {topic}',
  accessTrustOverall: 'agents with TrustFlow {min}',
  accessOthersGroup: 'the {group} group',
  accessRefused: 'refused',
  accessToolsRow: '{names}: you and {groups}',
  accessEveryOther: 'every other tool: every caller',
  noAccessBlock: 'No access block: {tools} stay yours; every other tool answers every caller the agent lets in.',
  noAccessBlockHint: '/agent edit adds one; webagents init -t tool-agent shows the shape.',
  // /mcp (spec 3.7)
  mcpFromFile: 'MCP servers (from {file})',
  mcpFromJson: 'MCP servers (from mcp.json)',
  mcpConnected: '● {server}  {transport}  {tools}',
  mcpNotConnected: '○ {server}  {transport}  not connected: {error}',
  mcpRejected: '✗ {server}  {reason}',
  mcpMore: ', +{count} more',
  mcpNone: '{name} uses no MCP servers.',
  mcpNoneHint: 'List them under - mcp: in {file} with /agent edit, or in mcp.json after /skills add mcp.',
  mcpServe: 'To use this agent from an MCP client: webagents mcp serve (stdio) or webagents mcp serve --http <port>.',
  // /cron (spec 3.7)
  cronDaemon: 'Schedules run under webagents daemon, not in the chat.',
  cronNone: '{name} has no schedules.',
  cronNoneHint: 'Add a cron: list to {file} with /agent edit.',
  cronRun: 'Run {schedule} now? It delivers to {target}. [y/N] ',
  cronChatTarget: 'your chat on Robutler',
  cronNoSchedule: 'There is no schedule called {schedule}.',
  seeTheSchedules: 'Type /cron to see the list.',
  notRun: 'Not run.',
  // /memory (spec 3.7)
  memoryTitle: 'Memory of {name}',
  memoryKeptInLabel: 'Kept in',
  memoryYoursLabel: 'Yours',
  memorySharedLabel: 'Shared',
  memoryCallersLabel: 'Callers',
  memoryLocal: '.webagents/memory in this folder',
  memoryAndRobutler: ', and on Robutler',
  memoryRobutlerOnly: 'on Robutler',
  // The Robutler tier with no agent key reaches nothing (B5, 2026-09-28).
  memoryRobutlerNoKey: '; on Robutler once the agent has its own key (`webagents publish`)',
  memoryRobutlerOnlyNoKey: 'on Robutler once the agent has its own key (`webagents publish`)',
  memoryCallers: '{callers}, {notes}',
  memoryHint: '/memory forget <key> removes one of yours. The notes are Markdown files you can edit.',
  memoryNone: '{name} keeps no memory.',
  memoryNoneHint: '/skills add memory gives it notes that last between conversations.',
  memoryForget: 'Forget your note {key}? [y/N] ',
  memoryForgot: 'Forgot {key}.',
  memoryNoNote: 'You have no note called {key}.',
  resumeDeleteAsk: 'Delete the conversation last used {when} ({count} messages)? [y/N] ',
  resumeDeleted: 'Deleted the conversation last used {when}.',
  resumeDeletedRemote: 'Its copy on Robutler stays; delete it there.',
  resumeOnlyRemote: 'That conversation is only on Robutler; delete it there.',
  resumeNotDeleted: 'Nothing deleted.',
  resumeNoNumber: 'There is no conversation {pick}.',
  resumeNoNumberHint: 'Type /resume to see the list.',
  lastConversation: 'Last conversation here {when} ({count} messages): /resume 1 continues it.',
  compacting: 'Compacting the conversation…',
  compactFailed: 'Could not compact the conversation: {reason}',
  compactEmpty: 'Nothing to compact: the conversation has not started.',
  contextHeading: 'Context: {model}, {window} tokens',
  contextConversation: 'The conversation: about {tokens} tokens, {percent}% of it, in {messages} messages.',
  contextSummary: 'It carries a summary of its earlier part.',
  contextAuto: 'It compacts on its own at {at}% ({tokens} tokens). /compact does it now.',
  contextAutoOff: 'It does not compact on its own (compaction: auto: false). /compact does it now.',
  contextFooter: 'context {percent}%',
  resumeRow: '{when} · {count} messages · {preview}',
  resumeRowRobutler: '{when} · {count} messages · Robutler: {preview}',
  resumeNoText: '(no text)',
  resumeDeleteRow: 'delete an earlier conversation, after asking',
  rewindRow: '{when} · {label}',
  modelInUse: 'in use',
} as const;

export type ChatWord = keyof typeof CHAT_WORDS;

/**
 * The line left under a question the chat asked mid-turn, saying what was
 * decided (the ptypass-fixes lane, 2026-09-27): the Python chat's live
 * display redrew over the question and the answer, so the scrollback kept no
 * record of what was allowed; both chats now end the exchange with one of
 * these. The Python `ANSWER_WORDS` is the twin;
 * `python/tests/fixtures/cli/ptypass_fixes_answers.json` holds both.
 */
export const ANSWER_WORDS = {
  control: {
    yes: '✓ Allowed the change to {path}.',
    no: '✗ Declined: {path} was not changed.',
  },
  host: {
    once: '✓ Allowed {host} for this command.',
    always: '✓ Allowed {host} from now on.',
    no: '✗ Not allowed: {host}.',
  },
} as const;

/** The sentence `word`, with its `{holes}` filled from `values`. A hole with no value stays as written. */
export function fill(word: ChatWord, values: Record<string, string | number> = {}): string {
  return CHAT_WORDS[word].replace(/\{(\w+)\}/g, (match, key: string) => (key in values ? String(values[key]) : match));
}

/** `1 file`, `2 files`. */
export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`;
}
