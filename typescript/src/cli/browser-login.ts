/**
 * Signing in through the browser, the way the Python CLI does (2026-09-24).
 *
 * `webagents login` here only took a pasted API key, so the chat could not
 * offer "sign in to use Robutler's models" without first sending the person to
 * Settings to create one. This is the Python flow's contract
 * (`python/webagents/cli/platform/auth.py`): open `{portal}/cli/auth?port=&state=`,
 * where the portal asks the person to approve, then take the token from the
 * redirect to `http://127.0.0.1:<port>/?token=&username=&state=`. The `state`
 * must come back unchanged, so a page elsewhere cannot hand this terminal a
 * token of its own choosing; Deny arrives as `error=access_denied`.
 */

import { createServer } from 'node:http';
import { createHash, randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { cliCommand } from './config-store';

/** The port the portal redirects to; the Python CLI uses the same one. */
export const CALLBACK_PORT = 8789;
const TIMEOUT_MS = 5 * 60 * 1000;

export interface BrowserLoginResult {
  token: string;
  username: string;
}

/**
 * The program and arguments that open `url` in the default browser on `platform`.
 * On Windows this is `rundll32 url.dll,FileProtocolHandler <url>`, not `cmd /c start`:
 * cmd treats `&` as a command separator, so a URL with a query string would be cut
 * at its first `&` and the rest run as a command. rundll32 takes the URL as one argument.
 */
export function browserOpenCommand(url: string, platform: NodeJS.Platform = process.platform): [string, string[]] {
  if (platform === 'darwin') return ['open', [url]];
  if (platform === 'win32') return ['rundll32', ['url.dll,FileProtocolHandler', url]];
  return ['xdg-open', [url]];
}

/** Open a URL in the person's browser; the caller also prints it, in case this cannot. */
export function openInBrowser(url: string): void {
  const [command, args] = browserOpenCommand(url);
  try {
    const child = spawn(command, args, { stdio: 'ignore', detached: true });
    child.on('error', () => {});
    child.unref();
  } catch {
    // Printing the URL is the fallback.
  }
}

/** Lucide icons, inlined: the page loads nothing from anywhere. */
const ICONS = {
  ok: '<path d="M20 6 9 17l-5-5"/>', // lucide: check
  cancel: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>', // lucide: x
  error: // lucide: triangle-alert
    '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
} as const;

/**
 * The page the browser lands on, the Python CLI's (`_callback_page`): the
 * consent screen's look, light or dark with the system.
 *
 * It also takes the token OUT OF THE ADDRESS BAR. The portal delivers it in
 * this URL's query string (S-213), so until this page loads the bearer is on
 * screen in full; `history.replaceState` swaps the visible URL for a bare
 * `/callback`.
 */
export function callbackPage(title: string, message: string, tone: keyof typeof ICONS): string {
  const escape = (s: string) =>
    s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
  const tile =
    tone === 'ok'
      ? 'linear-gradient(135deg,#6366f1,#a855f7)'
      : tone === 'cancel'
        ? 'linear-gradient(135deg,#71717a,#3f3f46)'
        : 'linear-gradient(135deg,#f59e0b,#ef4444)';
  return (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<meta name="referrer" content="no-referrer">' +
    `<title>${escape(title)}</title><style>` +
    ':root{color-scheme:light dark;--bg:#fafafa;--fg:#0a0a0a;--muted:#6b7280;--card:#ffffff;--border:#e5e7eb}' +
    '@media (prefers-color-scheme:dark){:root{--bg:#0a0a0a;--fg:#fafafa;--muted:#a1a1aa;--card:#171717;--border:#27272a}}' +
    '*{box-sizing:border-box}body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;' +
    'padding:16px;background:var(--bg);color:var(--fg);' +
    'font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}' +
    '.card{width:100%;max-width:380px;text-align:center;background:var(--card);border:1px solid var(--border);' +
    'border-radius:16px;padding:32px 24px}' +
    '.tile{width:56px;height:56px;border-radius:16px;display:inline-flex;align-items:center;justify-content:center;' +
    `margin-bottom:16px;background:${tile};color:#fff}` +
    'h1{font-size:20px;font-weight:600;margin:0 0 8px}p{margin:0;color:var(--muted);font-size:14px;text-wrap:balance}' +
    '</style></head><body><main class="card"><div class="tile">' +
    '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
    `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[tone]}</svg>` +
    `</div><h1>${escape(title)}</h1><p>${escape(message)}</p></main>` +
    '<script>history.replaceState(null,"","/callback")</script>' +
    '</body></html>'
  );
}

/**
 * Wait for the portal's redirect and return the token it carries.
 *
 * Rejects when the person denies, when the wait times out, or when the port is
 * taken (another sign-in running).
 */
export function browserLogin(
  portalUrl: string,
  options: {
    port?: number;
    timeoutMs?: number;
    open?: (url: string) => void;
    print?: (line: string) => void;
    /** Cancels the wait (the chat's Ctrl+C). */
    signal?: AbortSignal;
  } = {},
): Promise<BrowserLoginResult> {
  const port = options.port ?? CALLBACK_PORT;
  const state = randomBytes(16).toString('base64url');
  const authUrl = `${portalUrl.replace(/\/+$/, '')}/cli/auth?port=${port}&state=${encodeURIComponent(state)}`;
  const print = options.print ?? ((line: string) => console.log(line));

  print(`Browser on another machine? Run ${cliCommand('login --code')} instead.`);

  return new Promise((resolve, reject) => {
    let settled = false;
    const server = createServer((req, res) => {
      const url = new URL(req.url ?? '/', `http://127.0.0.1:${port}`);
      const reply = (status: number, title: string, message: string, tone: keyof typeof ICONS) => {
        // `Connection: close`, so no idle keep-alive socket holds the process
        // open once the server is closed.
        res.writeHead(status, {
          'Content-Type': 'text/html; charset=utf-8',
          'Cache-Control': 'no-store',
          Connection: 'close',
        });
        res.end(callbackPage(title, message, tone));
      };
      if (url.searchParams.get('state') !== state) {
        reply(400, 'Sign-in failed', `This does not match the sign-in started in your terminal. Run ${cliCommand('login')} again.`, 'error');
        return;
      }
      const error = url.searchParams.get('error');
      if (error) {
        reply(200, 'Sign-in cancelled', 'Nothing was shared. You can close this tab.', 'cancel');
        finish(new Error(error === 'access_denied' ? 'Sign-in was cancelled in the browser.' : `Sign-in failed: ${error}`));
        return;
      }
      const token = url.searchParams.get('token');
      const username = url.searchParams.get('username') ?? '';
      if (!token) {
        reply(400, 'Sign-in failed', `No sign-in details arrived. Run ${cliCommand('login')} again.`, 'error');
        return;
      }
      reply(200, username ? `Signed in as @${username}` : 'Signed in', 'You can close this tab and return to your terminal.', 'ok');
      finish(undefined, { token, username });
    });

    const timer = setTimeout(() => finish(new Error('No sign-in arrived within five minutes.')), options.timeoutMs ?? TIMEOUT_MS);
    const onAbort = () => finish(new Error('Sign-in cancelled.'));
    if (options.signal?.aborted) {
      queueMicrotask(onAbort);
    } else {
      options.signal?.addEventListener('abort', onAbort, { once: true });
    }

    function finish(error?: Error, result?: BrowserLoginResult) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal?.removeEventListener('abort', onAbort);
      server.close();
      if (error) reject(error);
      else resolve(result!);
    }

    server.on('error', (err: NodeJS.ErrnoException) => {
      finish(
        err.code === 'EADDRINUSE'
          ? new Error(`Port ${port} is in use; another sign-in may be running.`)
          : err,
      );
    });
    // Loopback only: the token must not be reachable from anywhere else.
    server.listen(port, '127.0.0.1', () => {
      if (settled) {
        // Cancelled before the port was bound: close it now it is.
        server.close();
        return;
      }
      print('Opening your browser to confirm the sign-in...');
      print(`If the browser does not open, visit: ${authUrl}`);
      (options.open ?? openInBrowser)(authUrl);
    });
  });
}

/**
 * Signing in from a machine whose browser is somewhere else (a server, a container, an SSH session): the portal shows a
 * one-time code and the person types it here (portal lib/auth/cli-code.ts, 2026-11-27). PKCE: this terminal keeps the
 * verifier and sends only its S256 challenge in the link, so the code is worthless anywhere else. The portal takes the
 * code once, right or wrong.
 */
export async function codeLogin(
  portalUrl: string,
  options: {
    print?: (line: string) => void;
    /** Reads the code the person types. */
    ask: (prompt: string) => Promise<string>;
    fetchImpl?: typeof fetch;
  },
): Promise<BrowserLoginResult> {
  const base = portalUrl.replace(/\/+$/, '');
  const verifier = randomBytes(32).toString('base64url');
  const challenge = createHash('sha256').update(verifier).digest('base64url');
  const print = options.print ?? ((line: string) => console.log(line));
  print('Open this link in a browser on any device, sign in and press Authorize:');
  print(`  ${base}/cli/auth?mode=code&challenge=${challenge}`);
  const code = (await options.ask('Then type the code it shows: ')).trim();
  if (!code) throw new Error('No code entered.');
  const res = await (options.fetchImpl ?? fetch)(`${base}/api/auth/cli/exchange`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, verifier }),
  });
  const body = (await res.json().catch(() => ({}))) as { token?: unknown; username?: unknown; message?: unknown };
  if (!res.ok || typeof body.token !== 'string') {
    throw new Error(typeof body.message === 'string' ? body.message : `The code was not accepted (${res.status}).`);
  }
  return { token: body.token, username: typeof body.username === 'string' ? body.username : '' };
}
