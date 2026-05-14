/**
 * Webapp templates registry.
 *
 * Re-exported by the `CustomHttpSkill` so its `get_webapp_template` /
 * `list_webapp_templates` tools and the `webappPatterns` @prompt
 * have a single source of truth.
 *
 * Templates are organised into four sets that mirror the agent-author
 * mental model:
 *
 *   - Identity-first: lean on Robutler-as-IdP (visitor_session +
 *     visitor_profile) wherever possible. No third-party OAuth needed
 *     for "who is this visitor".
 *   - Google API access: real OAuth flow, but ONLY when you need to
 *     call Google APIs on the visitor's behalf — not for identity.
 *   - Agent-to-agent: portal-token-authenticated machine APIs.
 *   - Widget: HTML endpoints rasterised on-device for home-screen.
 */

import type { WebappTemplate } from './types';

import { minimalHtmlPage } from './minimal-html-page';
import { multiRouteDispatch } from './multi-route-dispatch';
import { visitorSessionPersonalized } from './visitor-session-personalized';
import { signinWithRobutler } from './signin-with-robutler';
import { sessionCheck } from './session-check';
import { kvVisitorState } from './kv-visitor-state';
import { jsonApiEndpoint } from './json-api-endpoint';
import { csrfProtectedForm } from './csrf-protected-form';
import { logout } from './logout';
import { oauthGoogleLogin } from './oauth-google-login';
import { oauthGoogleCallback } from './oauth-google-callback';
import { agentToAgentEndpoint } from './agent-to-agent-endpoint';
import { callOtherAgent } from './call-other-agent';
import { widgetPersonalizedCard } from './widget-personalized-card';

export type { WebappTemplate, RequiredPermissions } from './types';

/** Authoritative list — order matters for `list_webapp_templates`. */
export const WEBAPP_TEMPLATES = {
  // Identity-first set (default story for visitor-facing webapps).
  minimal_html_page: minimalHtmlPage,
  multi_route_dispatch: multiRouteDispatch,
  visitor_session_personalized: visitorSessionPersonalized,
  signin_with_robutler: signinWithRobutler,
  session_check: sessionCheck,
  kv_visitor_state: kvVisitorState,
  json_api_endpoint: jsonApiEndpoint,
  csrf_protected_form: csrfProtectedForm,
  logout: logout,
  // Google API access (NOT identity).
  oauth_google_login: oauthGoogleLogin,
  oauth_google_callback: oauthGoogleCallback,
  // Agent-to-agent.
  agent_to_agent_endpoint: agentToAgentEndpoint,
  call_other_agent: callOtherAgent,
  // Widget.
  widget_personalized_card: widgetPersonalizedCard,
} as const;

export type WebappTemplateName = keyof typeof WEBAPP_TEMPLATES;

export const WEBAPP_TEMPLATE_NAMES: readonly WebappTemplateName[] = Object.keys(
  WEBAPP_TEMPLATES,
) as WebappTemplateName[];

export function getWebappTemplate(name: WebappTemplateName): WebappTemplate {
  return WEBAPP_TEMPLATES[name];
}

export function listWebappTemplates(): Array<{
  name: WebappTemplateName;
  description: string;
}> {
  return WEBAPP_TEMPLATE_NAMES.map((name) => ({
    name,
    description: WEBAPP_TEMPLATES[name].description,
  }));
}
