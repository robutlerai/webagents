/**
 * Template: `multi_route_dispatch`.
 *
 * One function backing several `custom_http` endpoints. The endpoint
 * declarations in `agent_configs.skills.custom_http.endpoints[]` all
 * point to the same `use:` function name; the function inspects
 * `ctx.request.path` (or `ctx.request.params`) and routes internally.
 *
 * Use this when you have a small webapp (e.g. login + dashboard +
 * logout) that shares state — keeping it in one function avoids the
 * "deploy three functions in lockstep" problem.
 */

import type { WebappTemplate } from './types';

export const multiRouteDispatch: WebappTemplate = {
  name: 'multi_route_dispatch',
  description: 'One function fronting several routes — internal dispatch on `ctx.request.path`.',
  when_to_use:
    'Your webapp has a few related routes (e.g. /login, /dashboard, /logout) that share helpers or state. Avoids the deploy-multiple-functions overhead.',
  security_notes:
    "- Register one endpoint per public path; all of them point to the same `use:` function name.\n- Match exact paths inside the dispatch — never trust a regex that could let the visitor hit private routes by accident.\n- Default to a 404 for unknown paths so the function fails closed.",
  required_permissions: {},
  code: `// @robutler-function
// runtime: js-v1
// entrypoint: handler
// permissions: {}
export async function handler(ctx) {
  const path = ctx.request.path;
  if (path.endsWith('/login')) return loginPage(ctx);
  if (path.endsWith('/dashboard')) return dashboardPage(ctx);
  if (path.endsWith('/logout')) return logout(ctx);
  return { status: 404, body: 'Not found' };
}

function htmlResponse(status, html) {
  return { status, headers: { 'content-type': 'text/html; charset=utf-8' }, body: html };
}

function loginPage(_ctx) {
  return htmlResponse(200, '<form method="post" action="./login"><button>Sign in</button></form>');
}

function dashboardPage(_ctx) {
  return htmlResponse(200, '<h1>Dashboard</h1>');
}

function logout(_ctx) {
  return { status: 302, headers: { location: './' }, body: '' };
}
`,
};
