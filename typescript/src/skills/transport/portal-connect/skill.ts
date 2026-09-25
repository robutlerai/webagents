/**
 * Portal Connect Skill — the reverse UAMP WebSocket bridge as a SKILL.
 *
 * The TypeScript counterpart of `webagents.agents.skills.robutler.portal_connect`.
 * Attach it to an agent and serve that agent normally; the skill reads
 * `WEBAGENTS_PORTAL_URL` and `WEBAGENTS_AGENT_TOKEN` itself, so attaching it
 * IS the whole configuration.
 *
 * TOKEN OPTIONAL SINCE 2026-09-23. `serve()` hands the agent the identity it
 * persisted for it (`agent.identity`, before `initialize()`), and with no
 * token configured the bridge SIGNS the `/ws` handshake with it (RFC 9421,
 * the same Web Bot Auth signature every platform call carries) instead of
 * presenting a key a human had to mint first. A configured token is used
 * exactly as before and wins over the identity. The skill refuses at
 * `start()` only when it has NEITHER, or an identity the platform could not
 * fetch a key set from (src/portal/connect.ts, `checkPortalCredential`).
 *
 * WHY A SKILL AND NOT A `connect(agent)` FUNCTION: a bridged agent cannot run
 * until it is connected, and agents initialise their skills lazily — a real
 * lifecycle deadlock, which the old `connect()` wrapper papered over by
 * initialising and starting things by hand for the one caller that used it.
 * Fixing the lifecycle instead means every path gets it: `serve()` starts
 * attached skills at boot, and `initialize()` opens the socket for anything
 * that only ever builds the agent.
 */

import { Skill } from '../../../core/skill';
import type { IAgent, SkillConfig } from '../../../core/types';
import type { SigningIdentity } from '../../../crypto/http-signature';
import {
  checkPortalCredential,
  resolvePortalWsUrl,
  runPortalBridge,
  type PortalBridgeOptions,
  type TerminalRouterLike,
} from '../../../portal/connect';

export interface PortalConnectConfig extends SkillConfig {
  /** Portal URL — http(s) accepted; `/ws` appended to a bare origin. Falls back to WEBAGENTS_PORTAL_URL / PORTAL_WS_URL. */
  portalUrl?: string;
  /** Per-agent token. Falls back to WEBAGENTS_AGENT_TOKEN. Optional since 2026-09-23: with none, `identity` signs the handshake. */
  token?: string;
  /**
   * The identity that signs the handshake when no token is configured. Under
   * `serve()` this needs no setting: the server hands the agent the identity
   * it persisted for it (`agent.identity`), which the skill reads at
   * `start()`. Set it for a bridge run outside `serve()` whose key set is
   * served somewhere the platform can fetch it. A configured identity wins
   * over the agent's.
   */
  identity?: SigningIdentity;
  /** Workspace `terminal` node support (see PortalBridgeOptions.terminal). */
  terminal?: boolean | TerminalRouterLike;
  /** Reconnect on an unexpected close (default true). */
  autoReconnect?: boolean;
  /** Seconds between reconnect attempts (default 5). */
  reconnectDelayS?: number;
  /** Max consecutive reconnect attempts before giving up (default 10). */
  maxReconnectAttempts?: number;
  /**
   * Open the connection from `initialize()` (default true). `false` keeps
   * registration-only behaviour and WARNS — a skill that is initialised and
   * never started is indistinguishable from a healthy one from the outside,
   * which is precisely the silent failure this skill exists to prevent.
   */
  autostart?: boolean;
}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

export class PortalConnectSkill extends Skill {
  private agent: IAgent | null = null;
  private abort: AbortController | null = null;
  private running: Promise<void> | null = null;

  /** The resolved WS endpoint (config, then env, then the public default). */
  readonly portalWsUrl: string;

  constructor(config: PortalConnectConfig = {}) {
    super({ ...config, name: config.name || 'portal' });
    this.portalWsUrl = resolvePortalWsUrl(config.portalUrl);
  }

  /** Called by `BaseAgent` when the skill is registered. */
  setAgent(agent: IAgent): void {
    this.agent = agent;
  }

  /** The token this skill will present, config first then the environment. */
  get token(): string | undefined {
    return (this.config as PortalConnectConfig).token ?? envVar('WEBAGENTS_AGENT_TOKEN');
  }

  /**
   * The identity that signs the handshake when there is no token: the
   * configured one first, then the one the host handed the agent
   * (`IAgent.identity`, set by `serve()` and `WebAgentsServer.addAgent()`
   * before `initialize()` from the identity whose key set they serve). Read
   * at `start()`, not at construction: the host sets it after the skill is
   * built and before the agent initialises, which is what starts this skill.
   */
  get identity(): SigningIdentity | undefined {
    return (
      (this.config as PortalConnectConfig).identity ??
      (this.agent as { identity?: SigningIdentity } | null)?.identity ??
      undefined
    );
  }

  /** True once the bridge loop is running. */
  get isStarted(): boolean {
    return this.running !== null;
  }

  async initialize(): Promise<void> {
    const autostart = (this.config as PortalConnectConfig).autostart ?? true;
    if (!autostart) {
      console.warn(
        `[webagents] PortalConnectSkill initialised with autostart=false and NOT started: ` +
          'no socket is open and no turn can arrive until start() is called.',
      );
      return;
    }
    await this.start();
  }

  /**
   * Open the portal connection (idempotent).
   *
   * Refuses HERE, before any socket is opened, when the credential cannot
   * work: a configured token that is an owner-subject key with no `agent_id`
   * binding connects successfully and then never receives a single turn
   * (F-045); an identity with a loopback issuer signs a handshake the
   * platform cannot verify; neither credential at all. That check
   * used to live in the `connect()` wrapper, where it guarded exactly one
   * entry point; on the skill it guards all of them.
   */
  async start(): Promise<void> {
    if (this.running) return;
    if (!this.agent) {
      throw new Error(
        'PortalConnectSkill has no agent. Pass it in `new BaseAgent({ skills: [...] })`.',
      );
    }
    const cfg = this.config as PortalConnectConfig;
    const token = this.token;
    const identity = this.identity;
    checkPortalCredential({ token, identity });

    this.abort = new AbortController();
    const options: PortalBridgeOptions = {
      portalUrl: this.portalWsUrl,
      ...(token ? { token } : {}),
      ...(identity ? { identity } : {}),
      signal: this.abort.signal,
      ...(cfg.terminal !== undefined ? { terminal: cfg.terminal } : {}),
      ...(cfg.autoReconnect !== undefined ? { autoReconnect: cfg.autoReconnect } : {}),
      ...(cfg.reconnectDelayS !== undefined ? { reconnectDelayS: cfg.reconnectDelayS } : {}),
      ...(cfg.maxReconnectAttempts !== undefined
        ? { maxReconnectAttempts: cfg.maxReconnectAttempts }
        : {}),
    };
    this.running = runPortalBridge(this.agent, options).finally(() => {
      this.running = null;
    });
    console.log(`[webagents] portal bridge for ${this.agent.name} -> ${this.portalWsUrl}`);
  }

  /** Close the socket and wait for the bridge loop to unwind. */
  async stop(): Promise<void> {
    this.abort?.abort();
    this.abort = null;
    const running = this.running;
    this.running = null;
    if (running) await running;
  }

  async cleanup(): Promise<void> {
    await this.stop();
  }
}
