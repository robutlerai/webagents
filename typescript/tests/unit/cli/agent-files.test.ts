/**
 * `-a <name>` by name, or refused (2026-09-24, `src/cli/agent-files.ts`).
 *
 * `findAgentFile(dir, name)` falls back to AGENT.md when AGENT-<name>.md is
 * missing, so `-a <typo>` opened this folder's other agent under the name
 * that was typed. The Python CLI's `agent_files.py` answers the same way, in
 * the same words.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { AgentNotFound, agentFileFor, folderAgents } from '../../../src/cli/agent-files';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

let folder = '';

beforeEach(() => {
  folder = tempDir('wa-agent-files-');
});

describe('agentFileFor', () => {
  it('finds an agent by its declared name or by its file name', () => {
    fs.writeFileSync(path.join(folder, 'AGENT-foo.md'), '---\nname: bar\n---\nBody\n');
    expect(agentFileFor(folder, 'bar')).toBe(path.join(folder, 'AGENT-foo.md'));
    expect(agentFileFor(folder, 'foo')).toBe(path.join(folder, 'AGENT-foo.md'));
  });

  it('means the built-in agent by its own name', () => {
    expect(agentFileFor(folder, 'robutler')).toBeNull();
  });

  it('refuses a name that matches nothing, naming the agents here', () => {
    fs.writeFileSync(path.join(folder, 'AGENT.md'), '---\nname: helper\n---\nBody\n');
    expect(() => agentFileFor(folder, 'helpr')).toThrow(AgentNotFound);
    expect(() => agentFileFor(folder, 'helpr')).toThrow(
      'There is no agent called helpr in this folder. Agents here: helper, and the built-in robutler.',
    );
  });

  it('says what runs instead in a folder with no agents', () => {
    expect(() => agentFileFor(folder, 'x')).toThrow('The built-in robutler runs without -a.');
  });
});

describe('folderAgents', () => {
  it('lists AGENT.md and AGENT-<name>.md under their declared names, skipping a broken one', () => {
    fs.writeFileSync(path.join(folder, 'AGENT.md'), '---\nname: main\ndescription: The main one.\n---\nBody\n');
    fs.writeFileSync(path.join(folder, 'AGENT-good.md'), '---\nname: good\n---\nBody\n');
    fs.writeFileSync(path.join(folder, 'notes.md'), 'not an agent');
    const agents = folderAgents(folder);
    expect(agents.map((a) => a.name).sort()).toEqual(['good', 'main']);
    expect(agents.find((a) => a.name === 'main')?.description).toBe('The main one.');
  });
});
