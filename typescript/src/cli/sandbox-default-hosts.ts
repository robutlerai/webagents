/**
 * Writing a host into an agent file's `sandbox: network: hosts:` (the
 * sandbox-default lane, 2026-09-27), for the chat's ask-on-first-use: when a
 * confined command was refused a host and the owner answers "always", the
 * host goes into the file, where the owner can see and review it, and the
 * command re-runs.
 *
 * A LINE EDIT, NOT A REWRITE, as the chat's other edits are (`skills-edit.ts`):
 * a YAML dump would drop the file's comments and reorder its keys. Four
 * shapes are handled, by indentation: no `sandbox:` key at all (a block is
 * added before the closing fence); `sandbox:` with `network:` holding a
 * `hosts:` block list (the host is appended to it); `sandbox:` with the old
 * bare `network:` list (appended there); an empty `hosts: []` or `network:
 * []` (replaced by a one-entry block list). Anything else (a flow list with
 * entries, `sandbox: off`, a `network:` mapping without `hosts:`) is left
 * alone with a sentence, and the owner adds the host by hand. The result is
 * read back through the loader by the caller. The Python twin is
 * `cli/sandbox_default_hosts.py`; the fixture
 * `python/tests/fixtures/cli/sandbox_default_hosts.json` pins the cases.
 */

export type HostEdit = { text: string } | { problem: string };

/** What the chat says and asks (fixture `words`). */
export const HOST_WORDS = {
  hostRefused: 'The sandbox refused {host} for `{command}`.',
  hostQuestion: 'Allow it? [o]nce, [a]lways (adds it to the agent file), [N]o: ',
  hostWritten: 'Added {host} to network.hosts in {file}.',
  hostNotWritten: 'Could not add {host} to {file}: {problem}. Add it under `sandbox: network: hosts:` by hand.',
  hostNoFile: 'the built-in agent has no file',
} as const;

/** `{name}` placeholders filled by plain replacement (a command may hold braces of its own). */
export function fillWords(template: string, values: Record<string, string>): string {
  let out = template;
  for (const [name, value] of Object.entries(values)) out = out.split(`{${name}}`).join(value);
  return out;
}

/** The owner's typed answer: `o`/`once`, `a`/`always`, anything else is no (fixture `answers`). */
export function hostAnswer(typed: string): 'once' | 'always' | 'no' {
  const word = typed.trim().toLowerCase();
  if (word === 'o' || word === 'once') return 'once';
  if (word === 'a' || word === 'always') return 'always';
  return 'no';
}

const FENCE = /^---\s*$/;

/** The indentation of `line`: its leading spaces. */
function indentOf(line: string): number {
  return line.length - line.trimStart().length;
}

/** `key:` and what follows on the line, at exactly `indent` spaces. */
function keyLine(line: string, key: string, indent: number): { rest: string } | null {
  if (indentOf(line) !== indent) return null;
  const trimmed = line.trim();
  if (trimmed === `${key}:`) return { rest: '' };
  if (trimmed.startsWith(`${key}: `)) return { rest: trimmed.slice(key.length + 2).trim() };
  return null;
}

/** The index of the line after `start` where a block at `indent` ends: the first line with a smaller indent that is not blank or a comment. */
function blockEnd(lines: readonly string[], start: number, indent: number): number {
  for (let index = start; index < lines.length; index++) {
    const line = lines[index];
    if (!line.trim() || line.trim().startsWith('#')) continue;
    if (indentOf(line) < indent) return index;
  }
  return lines.length;
}

/** The last line of a block list under `listLine` (items at any indent greater than the key's), or the key's line when the list is empty. */
function lastItem(lines: readonly string[], listLine: number, indent: number, end: number): { at: number; itemIndent: number } {
  let at = listLine;
  let itemIndent = indent + 2;
  for (let index = listLine + 1; index < end; index++) {
    const line = lines[index];
    if (!line.trim() || line.trim().startsWith('#')) continue;
    if (line.trim().startsWith('- ') || line.trim() === '-') {
      at = index;
      itemIndent = indentOf(line);
    }
  }
  return { at, itemIndent };
}

/** The file with `host` in `sandbox.network.hosts`, or why it cannot be added by a line edit. */
export function addNetworkHost(text: string, host: string): HostEdit {
  const newline = text.includes('\r\n') ? '\r\n' : '\n';
  const lines = text.split(/\r?\n/);
  if (!lines.length || !FENCE.test(lines[0])) return { problem: 'the file has no front matter' };
  const close = lines.findIndex((line, index) => index > 0 && FENCE.test(line));
  if (close < 0) return { problem: 'the front matter never closes' };
  const front = lines.slice(0, close);

  const sandboxAt = front.findIndex((line) => keyLine(line, 'sandbox', 0));
  if (sandboxAt < 0) {
    const block = ['sandbox:', '  network:', '    hosts:', `      - ${host}`];
    return { text: [...front, ...block, ...lines.slice(close)].join(newline) };
  }
  const sandboxRest = keyLine(front[sandboxAt], 'sandbox', 0)!.rest;
  if (sandboxRest) return { problem: `\`sandbox: ${sandboxRest}\` is not a block that can take a host` };
  const sandboxEnd = blockEnd(front, sandboxAt + 1, 1);
  const inner = front.slice(sandboxAt + 1, sandboxEnd).find((line) => line.trim() && !line.trim().startsWith('#'));
  const indent = inner ? indentOf(inner) : 2;

  const networkAt = front.findIndex((line, index) => index > sandboxAt && index < sandboxEnd && keyLine(line, 'network', indent));
  if (networkAt < 0) {
    const pad = ' '.repeat(indent);
    const block = [`${pad}network:`, `${pad}  hosts:`, `${pad}    - ${host}`];
    return { text: [...front.slice(0, sandboxEnd), ...block, ...front.slice(sandboxEnd), ...lines.slice(close)].join(newline) };
  }
  const networkRest = keyLine(front[networkAt], 'network', indent)!.rest;
  const networkEnd = blockEnd(front, networkAt + 1, indent + 1);
  if (networkRest === '[]') {
    // The old bare list, empty: one entry, as a block.
    const pad = ' '.repeat(indent);
    const replaced = [`${pad}network:`, `${pad}  - ${host}`];
    return { text: [...front.slice(0, networkAt), ...replaced, ...front.slice(networkAt + 1), ...lines.slice(close)].join(newline) };
  }
  if (networkRest) return { problem: `\`network: ${networkRest}\` is not a block that can take a host` };

  const body = front.slice(networkAt + 1, networkEnd).filter((line) => line.trim() && !line.trim().startsWith('#'));
  const isBareList = body.length > 0 && body.every((line) => line.trim().startsWith('- ') || line.trim() === '-');
  if (isBareList) {
    const { at, itemIndent } = lastItem(front, networkAt, indent, networkEnd);
    return { text: [...front.slice(0, at + 1), `${' '.repeat(itemIndent)}- ${host}`, ...front.slice(at + 1), ...lines.slice(close)].join(newline) };
  }
  const innerIndent = body.length ? indentOf(body[0]) : indent + 2;
  const hostsAt = front.findIndex((line, index) => index > networkAt && index < networkEnd && keyLine(line, 'hosts', innerIndent));
  if (hostsAt < 0) {
    if (body.length === 0) {
      const pad = ' '.repeat(indent + 2);
      return { text: [...front.slice(0, networkAt + 1), `${pad}hosts:`, `${pad}  - ${host}`, ...front.slice(networkAt + 1), ...lines.slice(close)].join(newline) };
    }
    const pad = ' '.repeat(innerIndent);
    return { text: [...front.slice(0, networkEnd), `${pad}hosts:`, `${pad}  - ${host}`, ...front.slice(networkEnd), ...lines.slice(close)].join(newline) };
  }
  const hostsRest = keyLine(front[hostsAt], 'hosts', innerIndent)!.rest;
  const hostsEnd = blockEnd(front, hostsAt + 1, innerIndent + 1);
  if (hostsRest === '[]') {
    const pad = ' '.repeat(innerIndent);
    return { text: [...front.slice(0, hostsAt), `${pad}hosts:`, `${pad}  - ${host}`, ...front.slice(hostsAt + 1), ...lines.slice(close)].join(newline) };
  }
  if (hostsRest) return { problem: `\`hosts: ${hostsRest}\` is not a block list that can take a host` };
  const { at, itemIndent } = lastItem(front, hostsAt, innerIndent, hostsEnd);
  return { text: [...front.slice(0, at + 1), `${' '.repeat(itemIndent)}- ${host}`, ...front.slice(at + 1), ...lines.slice(close)].join(newline) };
}
