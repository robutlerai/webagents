/**
 * `CustomWebsocketSkill` — v2 placeholder.
 *
 * v2 follow-up: non-UAMP WS upgrade branch in `server.ts`, per-connection
 * sandbox lifecycle, per-message timeout, idle/heartbeat, back-pressure.
 *
 * The skill class ships as a no-op stub in v1 so the manifest validator
 * can return `WS_NOT_YET_SUPPORTED` with a stable error code, and so the
 * `agent_configs.skills.customWs` slot is reserved without colliding
 * with v1 dispatcher routes.
 */

import { Skill } from '../../core/skill';
import { prompt } from '../../core/decorators';

export interface CustomWebsocketEndpointEntry {
  id: string;
  use: string;
  path: string;
  enabled?: boolean;
  description?: string;
}

export interface CustomWebsocketSkillConfig {
  endpoints?: CustomWebsocketEndpointEntry[];
}

export class CustomWebsocketSkill extends Skill {
  readonly name = 'custom_websocket';
  readonly dependencies = ['function-runtime'] as const;

  constructor(private readonly cfg: CustomWebsocketSkillConfig = {}) {
    super();
  }

  /**
   * Returns the configured endpoints unchanged. v1 routes them all to
   * a `WS_NOT_YET_SUPPORTED` close frame at the dispatcher edge.
   */
  endpoints(): CustomWebsocketEndpointEntry[] {
    return this.cfg.endpoints ?? [];
  }

  @prompt({ name: 'customWebsocketGuide' })
  async customWebsocketGuide(): Promise<string> {
    if ((this.cfg.endpoints ?? []).length === 0) return '';
    return [
      '## Custom WebSocket endpoints (v2 preview)',
      '',
      'WebSocket-style functions are reserved for v2. v1 surfaces these as `WS_NOT_YET_SUPPORTED`.',
    ].join('\n');
  }
}
