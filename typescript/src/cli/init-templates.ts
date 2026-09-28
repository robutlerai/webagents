/**
 * The agent files `webagents init` and the chat's `/agent new` write
 * (2026-09-26, interactive-mode spec 3.3): one table of templates and one
 * function that renders AGENT.md, so the two commands cannot drift.
 *
 * Byte for byte what the Python CLI writes (`cli/init_templates.py`), pinned
 * by `python/tests/fixtures/cli/init_templates.json` (`agent_md`,
 * `with_model`, `with_key`). The chat passes the provider it runs on when this
 * machine holds that provider's key, so the new agent runs at once; `init`
 * passes the model `initModel` finds the same way.
 *
 * NO KEY, THE CHAT'S OWN RULE (B3, 2026-09-28). With no model the file
 * named `openai/gpt-4o-mini` and listed `openai`, whatever the machine had. A
 * new developer with no OpenAI key then ran OpenAI's model through Robutler,
 * and when that route failed, `/model` refused every other provider because
 * the file named `openai`. With no provider key the file now names no model
 * and no provider skill, so it runs as the zero-config chat does: Robutler's
 * choice, `auto/balanced`, until a provider key is set, then that key's
 * default model (the quickstart's order, `init` then `secrets set`, keeps
 * working). Two comment lines say so, and how to pin a model.
 */

import { cliCommand } from './config-store';
import { findProvider } from '../skills/llm/providers';

/**
 * The `access:` block the tool-agent template ships with (2026-09-26, S-248).
 * `shell` and `filesystem` are owner-only by default now; the block names the
 * group the owner would open them to, empty, so the file shows where a caller
 * goes rather than sending the reader to ADR-0045.
 */
export const TOOL_AGENT_ACCESS: readonly string[] = [
  'access:',
  '  # shell and filesystem run as you, so only you (and admins) can use them.',
  '  # To let other callers use them, name the callers in the group:',
  '  # user:@handle, agent:https://host/**, key:<thumbprint>, domain:host.',
  '  groups:',
  '    trusted: []',
  '  tools:',
  '    trusted: [filesystem, shell]',
];

/**
 * The `sandbox:` block the tool-agent template ships with (2026-09-26, the
 * e2e run): a template that runs shell commands declared no sandbox, so its
 * own `doctor` warned "off: shell commands run with your permissions" on the
 * first run. `development` confines writes to the folder and allows no
 * network until hosts are listed.
 */
export const TOOL_AGENT_SANDBOX: readonly string[] = [
  'sandbox:',
  '  # Shell commands run confined: writes stay in this folder, no network.',
  '  # List hosts under network: to let commands reach them.',
  '  preset: development',
];

export interface InitTemplate {
  description: string;
  skills: readonly string[];
  /** The `sandbox:` lines, before `access:`. */
  sandbox: readonly string[];
  access: readonly string[];
}

/**
 * The templates `init` can actually make (2026-09-24). `templates list`
 * advertised six, among them rag-agent, multi-agent, browser-agent and
 * mcp-agent, and `init --template` accepted any name at all: everything that
 * was not `chatbot` got the same openai + filesystem + shell scaffold with the
 * template's name pasted into its description. One table now feeds both
 * commands and the chat, and each refuses a name it cannot make.
 */
// The descriptions are the files' `description:` too (2026-09-26): the file
// said "A tool-agent agent", which described nothing. No colon in them: a
// plain YAML scalar cannot hold `: `.
export const INIT_TEMPLATES: Readonly<Record<string, InitTemplate>> = {
  chatbot: { description: 'A chat agent with one model and no tools', skills: [], sandbox: [], access: [] },
  'tool-agent': {
    description: 'Reads and writes files and runs shell commands',
    skills: ['filesystem', 'shell'],
    sandbox: TOOL_AGENT_SANDBOX,
    access: TOOL_AGENT_ACCESS,
  },
};

/** What a new agent with no provider key runs on while none is set: the zero-config chat's choice. */
export const ROBUTLER_CHOICE_MODEL = 'auto/balanced';

/** The lines a file with no `model:` carries in its place, so it says what runs. */
export const NO_MODEL_COMMENT: readonly string[] = [
  '# No model named: a provider key\'s default model when one is set, else',
  "# Robutler's choice (auto/balanced). Add a model: line to pin one.",
];

/** The names, in the order `templates list` shows them. */
export const TEMPLATE_NAMES: readonly string[] = Object.keys(INIT_TEMPLATES);

/** What `/agent new` accepts as a name (fixture `name_grammar`). */
export const AGENT_NAME_RE = /^[a-z0-9][a-z0-9-]{0,47}$/;

/**
 * The model `init` writes: the default model of the first provider whose key
 * this machine holds (the environment, or `secrets set`), as the zero-config
 * chat would run it; `undefined`, Robutler's choice, with no key. The Python
 * `init_model` answers the same.
 */
export async function initModel(env?: Record<string, string | undefined>): Promise<string | undefined> {
  let seen = env;
  if (!seen) {
    const { readStoredProviderKeys } = await import('./provider-keys');
    const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
    seen = { ...stored, ...process.env };
  }
  const { resolveModelAccess } = await import('./model-access');
  const access = await resolveModelAccess(undefined, { signedIn: () => false, env: seen });
  return access.kind === 'direct' ? access.model : undefined;
}

/**
 * The last line `init` prints: what the new agent runs on and, only when it
 * is needed, the way in (fixture `init_line`; the Python `init_line`). It said
 * "add your key ..., or sign in ..." to everyone, the signed in included
 * (2026-09-28).
 */
export function initLine(model: string, keyed: boolean, signedIn: boolean): string {
  if (keyed) {
    const provider = findProvider(model.split('/')[0]);
    return `It runs on ${model} with your ${provider?.envVar ?? 'its key'}.`;
  }
  if (signedIn) return `It runs on ${model} through Robutler, paid from your Robutler credits, until a provider key is set.`;
  return (
    `It runs on ${model} through Robutler once you sign in with \`${cliCommand('login')}\`, ` +
    `or on your own provider key: \`${cliCommand('secrets set OPENAI_API_KEY')}\`.`
  );
}

/**
 * AGENT.md for `name` from `template` (file comment). With `model`
 * (`provider/model`), that provider's skill leads the list and the model line
 * names it; a model only Robutler serves (`auto/...`) is named with no
 * provider skill; without one, the file names no model and says what runs.
 */
export function agentMarkdown(name: string, template: string, model?: string): string {
  const chosen = INIT_TEMPLATES[template];
  if (!chosen) throw new Error(`Unknown template '${template}'.`);
  let skills = [...chosen.skills];
  const prefix = model ? model.split('/')[0] : '';
  const head = ['---', `name: ${name}`, `description: ${chosen.description}`];
  if (!model) {
    head.push(...NO_MODEL_COMMENT);
  } else if (['auto', 'proxy', 'robutler'].includes(prefix)) {
    head.push(`model: ${model}`);
  } else {
    skills = [findProvider(prefix)?.id ?? 'openai', ...skills];
    head.push(`model: ${model}`);
  }
  return [
    ...head,
    ...(skills.length ? ['skills:', ...skills.map((s) => `  - ${s}`)] : []),
    ...chosen.sandbox,
    ...chosen.access,
    '---',
    '',
    `# ${name}`,
    '',
    'You are a helpful assistant.',
    '',
  ].join('\n');
}
