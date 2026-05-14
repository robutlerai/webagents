/**
 * Template: `minimal_html_page`.
 *
 * The smallest possible HTML page served from an agent function. No
 * auth, no kv, no fetch — pure string template. Use as a starting
 * point and layer on `visitor_session_personalized`,
 * `kv_visitor_state`, etc. once the page works.
 */

import type { WebappTemplate } from './types';

export const minimalHtmlPage: WebappTemplate = {
  name: 'minimal_html_page',
  description: 'Plain HTML page (no auth) — the simplest custom_http endpoint.',
  when_to_use:
    'You want to serve a static HTML page from your agent (landing page, marketing page, status page) and personalisation is not yet required.',
  security_notes:
    "- Default response headers are applied automatically (CSP, X-Frame-Options DENY, COOP, CORP, no-sniff). Don't override them unless you know exactly what you're loosening.\n- 2 MB response cap is enforced — keep the HTML lean and link to assets instead of inlining heavy resources.\n- Set `auth: 'public'` on the endpoint declaration in `agent_configs.skills.custom_http.endpoints[]`; this template does NOT use cookies or sessions.",
  required_permissions: {},
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: {}
export async function handler(ctx) {
  const html = \`<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Hello from \${ctx.metadata.agentSlug}</title>
</head>
<body>
  <h1>Hello \${ctx.metadata.agentSlug}!</h1>
  <p>Served by your custom_http endpoint at \${ctx.request.path}.</p>
</body>
</html>\`;
  return {
    status: 200,
    headers: { 'content-type': 'text/html; charset=utf-8' },
    body: html,
  };
}
`,
};
