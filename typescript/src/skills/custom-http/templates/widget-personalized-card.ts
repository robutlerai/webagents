/**
 * Template: `widget_personalized_card`.
 *
 * HTML endpoint flagged as a widget — the dispatcher applies
 * widget-friendly headers (`frame-ancestors 'self'`, `XFO SAMEORIGIN`)
 * so the platform can rasterise it on-device for the home-screen
 * widget surface (Phase 10).
 *
 * The HTML is sized for a small viewport. Keep it simple: a single
 * card-shaped layout that reads cleanly at 200×200.
 */

import type { WebappTemplate } from './types';

export const widgetPersonalizedCard: WebappTemplate = {
  name: 'widget_personalized_card',
  description: 'Widget-flagged HTML card sized for on-device rasterisation.',
  when_to_use:
    'You want a home-screen widget that shows a small personalised summary (latest message, unread count, current weather, etc.).',
  security_notes:
    "- Set `widget: { ... }` on the endpoint declaration in `agent_configs.skills.custom_http.endpoints[]` so the dispatcher relaxes XFO/frame-ancestors to allow same-origin embedding.\n- Keep total HTML + inline CSS under ~50 KB. The widget surface re-fetches frequently.\n- Use `auth: 'visitor_session'` if personalised; the platform widget runner replays the visitor's cookie for you.",
  required_permissions: {
    visitor_profile: ['name'] as const,
  },
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: { visitor_profile: ['name'] }
export async function handler(ctx) {
  const name = ctx.auth?.profile?.displayName ?? 'there';
  const html = \`<!doctype html>
<html><head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    html, body { margin: 0; height: 100%; font-family: -apple-system, system-ui, sans-serif; }
    body { display: grid; place-items: center; background: linear-gradient(135deg, #6e8efb, #a777e3); color: white; }
    .card { padding: 16px; border-radius: 16px; backdrop-filter: blur(8px); background: rgba(0,0,0,0.15); text-align: center; }
    h1 { font-size: 18px; margin: 0 0 4px; }
    p { font-size: 13px; margin: 0; opacity: 0.9; }
  </style>
</head><body>
  <div class="card">
    <h1>Hi \${escapeHtml(name)}</h1>
    <p>3 new updates</p>
  </div>
</body></html>\`;
  return { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' }, body: html };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
`,
};
