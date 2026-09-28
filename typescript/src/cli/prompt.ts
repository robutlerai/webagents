/**
 * Questions asked at the terminal outside the chat's input box.
 *
 * `promptSecret` came out of `index.ts` (2026-09-24) so the chat can take a
 * provider key the same way `login` takes an API key: with echo off.
 */

import * as readline from 'node:readline';

export interface SecretPromptOptions {
  /**
   * What Ctrl+C does: `exit` (the default) ends the process with 130, as a
   * command at a shell does; `cancel` answers null and the caller goes on.
   * The chat passes `cancel` (2026-09-26, D5 of the interactive-mode review:
   * Ctrl+C at `/keys set`'s hidden prompt ended the whole chat).
   */
  onInterrupt?: 'exit' | 'cancel';
  input?: NodeJS.ReadStream;
  output?: NodeJS.WriteStream;
}

/**
 * Read a secret from the terminal WITHOUT echoing it.
 *
 * S-212: the old prompt used `readline` with `output: process.stdout` and no
 * echo suppression, so a long-lived `rok_*` key was rendered on screen as it
 * was pasted and left in scrollback, screen shares and terminal recordings.
 *
 * Falls back to a visible prompt when stdin is not a TTY (a pipe, CI), where
 * there is nothing to hide from and no raw mode to enter.
 */
export async function promptSecret(prompt: string): Promise<string>;
export async function promptSecret(prompt: string, options: SecretPromptOptions & { onInterrupt: 'cancel' }): Promise<string | null>;
export async function promptSecret(prompt: string, options?: SecretPromptOptions): Promise<string | null>;
export async function promptSecret(prompt: string, options: SecretPromptOptions = {}): Promise<string | null> {
  const stdin = options.input ?? process.stdin;
  const out = options.output ?? process.stdout;
  const cancel = options.onInterrupt === 'cancel';
  if (!stdin.isTTY) {
    const rl = readline.createInterface({ input: stdin, output: out });
    return new Promise((resolve) => {
      let answered = false;
      rl.on('close', () => {
        if (!answered) resolve(cancel ? null : '');
      });
      rl.question(prompt, (answer) => {
        answered = true;
        rl.close();
        resolve(answer);
      });
    });
  }

  return new Promise((resolve) => {
    out.write(prompt);
    const wasRaw = stdin.isRaw;
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding('utf8');

    let value = '';
    const finish = (answer: string | null) => {
      stdin.setRawMode(Boolean(wasRaw));
      stdin.pause();
      stdin.removeListener('data', onData);
      out.write('\n');
      resolve(answer);
    };
    const onData = (chunk: string) => {
      for (const ch of chunk) {
        if (ch === '\n' || ch === '\r' || ch === '\u0004') {
          finish(value);
          return;
        }
        if (ch === '\u0003') {
          // Ctrl-C: restore the terminal rather than leaving it in raw mode,
          // then end the process, or (in the chat) answer nothing.
          if (cancel) {
            finish(null);
            return;
          }
          stdin.setRawMode(false);
          stdin.pause();
          out.write('\n');
          process.exit(130);
        }
        if (ch === '\u007f' || ch === '\b') {
          value = value.slice(0, -1);
          continue;
        }
        value += ch;
      }
    };
    stdin.on('data', onData);
  });
}

/**
 * A secret for a command (`webagents secrets set`, S-292): with echo off at a
 * terminal, else the first line of what is piped in, with no prompt printed,
 * so `printf '%s\n' "$TOKEN" | webagents secrets set NAME` works in a script
 * and its stdout holds nothing but the command's answer. Never an argument:
 * that lands in shell history and the process list.
 */
export async function promptSecretOrPipe(prompt: string): Promise<string> {
  if (process.stdin.isTTY) return promptSecret(prompt);
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  return Buffer.concat(chunks).toString('utf8').split(/\r?\n/)[0] ?? '';
}

/** One visible line, or null on Ctrl+C / Ctrl+D. */
export function promptLine(prompt: string): Promise<string | null> {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    let answered = false;
    rl.on('SIGINT', () => rl.close());
    rl.on('close', () => {
      if (!answered) {
        process.stdout.write('\n');
        resolve(null);
      }
    });
    rl.question(prompt, (answer) => {
      answered = true;
      rl.close();
      resolve(answer);
    });
  });
}
