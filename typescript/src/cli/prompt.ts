/**
 * Questions asked at the terminal outside the chat's input box.
 *
 * `promptSecret` came out of `index.ts` (2026-09-24) so the chat can take a
 * provider key the same way `login` takes an API key: with echo off.
 */

import * as readline from 'node:readline';

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
export async function promptSecret(prompt: string): Promise<string> {
  if (!process.stdin.isTTY) {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    return new Promise((resolve) => rl.question(prompt, (answer) => { rl.close(); resolve(answer); }));
  }

  return new Promise((resolve) => {
    process.stdout.write(prompt);
    const stdin = process.stdin;
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding('utf8');

    let value = '';
    const onData = (chunk: string) => {
      for (const ch of chunk) {
        if (ch === '\n' || ch === '\r' || ch === '\u0004') {
          stdin.setRawMode(false);
          stdin.pause();
          stdin.removeListener('data', onData);
          process.stdout.write('\n');
          resolve(value);
          return;
        }
        if (ch === '\u0003') {
          // Ctrl-C: restore the terminal rather than leaving it in raw mode.
          stdin.setRawMode(false);
          stdin.pause();
          process.stdout.write('\n');
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
