/**
 * Cross-language test harness.
 *
 * Starts a Python agent server (uvicorn) and a TypeScript agent server
 * (WebAgentsServer) so cross-language interop tests can run against both.
 *
 * Usage:
 *   import { startServers, stopServers, PYTHON_URL, TS_URL } from './harness.js';
 *   beforeAll(() => startServers());
 *   afterAll(() => stopServers());
 */

import { spawn, type ChildProcess } from 'node:child_process';
import * as path from 'node:path';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { tool, handoff } from '../../src/core/decorators.js';
import { WebAgentsServer } from '../../src/server/multi.js';
import type { Context } from '../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
} from '../../src/uamp/events.js';

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------

export const PYTHON_PORT = 9100;
export const TS_PORT = 9200;
export const PYTHON_URL = `http://localhost:${PYTHON_PORT}`;
export const TS_URL = `http://localhost:${TS_PORT}`;

// ---------------------------------------------------------------------------
// TS agent (echo + priced tool)
// ---------------------------------------------------------------------------

class EchoHandoff extends Skill {
  @handoff({ name: 'echo' })
  async *processUAMP(
    events: ClientEvent[],
    _ctx: Context,
  ): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text') {
        texts.push((e as { type: string; text: string }).text);
      }
    }
    const reply = texts.join(' | ');
    yield createResponseDeltaEvent('r1', { type: 'text', text: reply });
    yield createResponseDoneEvent(
      'r1',
      [{ type: 'text', text: reply }],
      'completed',
      { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    );
  }
}

class PricedTool extends Skill {
  @tool({ description: 'A tool that costs credits' })
  async pricedEcho(params: { text: string }, _ctx: Context) {
    return `[priced] ${params.text}`;
  }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let tsServer: WebAgentsServer | null = null;
let pythonProcess: ChildProcess | null = null;

// ---------------------------------------------------------------------------
// Start / Stop
// ---------------------------------------------------------------------------

async function waitForPort(port: number, timeoutMs = 15000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`http://localhost:${port}/`);
      if (res.ok || res.status < 500) return;
    } catch {
      // not ready yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Port ${port} did not become ready within ${timeoutMs}ms`);
}

export async function startTsServer(): Promise<void> {
  tsServer = new WebAgentsServer({ port: TS_PORT });

  const agent = new BaseAgent({
    name: 'ts-echo',
    description: 'TypeScript echo agent for interop tests',
    skills: [new EchoHandoff(), new PricedTool()],
  });

  await tsServer.addAgent('ts-echo', agent);
  await tsServer.start();
}

export async function startPythonServer(): Promise<void> {
  const scriptPath = path.resolve(
    __dirname,
    '..',
    '..',
    '..',
    'python',
    'tests',
    'interop',
    'serve_echo.py',
  );

  pythonProcess = spawn('python3', [scriptPath, String(PYTHON_PORT)], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
  });

  pythonProcess.stderr?.on('data', (chunk: Buffer) => {
    const msg = chunk.toString();
    if (!msg.includes('INFO')) {
      process.stderr.write(`[python] ${msg}`);
    }
  });

  await waitForPort(PYTHON_PORT);
}

export async function startServers(): Promise<void> {
  await Promise.all([startTsServer(), startPythonServer()]);
}

export async function stopServers(): Promise<void> {
  if (pythonProcess) {
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
  }
  if (tsServer) {
    await tsServer.stop?.();
    tsServer = null;
  }
}
