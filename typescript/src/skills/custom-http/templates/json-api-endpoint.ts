/**
 * Template: `json_api_endpoint`.
 *
 * REST-style JSON endpoint backing a single-page app. Returns
 * `application/json` with a small payload assembled from
 * `ctx.request.body` / `ctx.request.query`.
 */

import type { WebappTemplate } from './types';

export const jsonApiEndpoint: WebappTemplate = {
  name: 'json_api_endpoint',
  description: 'JSON API endpoint with method routing.',
  when_to_use:
    "You're building a SPA / mobile client backend and just need machine-readable responses (no HTML rendering on the server).",
  security_notes:
    '- Validate `ctx.request.body` shape before touching it — even with `auth: \'visitor_session\'`, the body is attacker-controlled.\n- Prefer JSON.stringify for serialisation; never concatenate user-controlled strings into a JSON literal.\n- The 2 MB response cap applies to JSON too — paginate large lists rather than dumping everything.',
  required_permissions: {},
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: {}
export async function handler(ctx) {
  switch (ctx.request.method) {
    case 'GET':    return json(200, { items: [] });
    case 'POST':   return json(201, { ok: true, received: ctx.request.body ?? null });
    case 'DELETE': return json(204, null);
    default:       return json(405, { error: 'method not allowed' });
  }
}

function json(status, payload) {
  return {
    status,
    headers: { 'content-type': 'application/json' },
    body: payload === null ? '' : JSON.stringify(payload),
  };
}
`,
};
