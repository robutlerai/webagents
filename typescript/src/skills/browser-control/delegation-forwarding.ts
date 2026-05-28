/**
 * Delegation forwarding helper — re-emits a child agent's
 * `HtmlContent.live` block into the parent's UAMP stream as a
 * `tool_progress { kind: 'forward_live' }` envelope. Plan v3-02 (B8).
 *
 * Wire (Plan 1 L8 extension):
 *   parent.tool_progress {
 *     kind: 'forward_live',
 *     data: {
 *       liveId,
 *       child_agent_slug,
 *       transport,
 *       display_hint: 'compact-chip' | 'inline',
 *       content_id,
 *     }
 *   }
 *
 * The chat-view dispatcher (Plan 1) recognises `kind:'forward_live'`
 * and renders a clickable compact chip in the parent's thread; click
 * opens the original child block in a subchat or spawns it onto the
 * canvas (when `delegate({ spawnOnCanvas: true })`).
 *
 * Invocation point: NLI delegation skill at
 *   `webagents/typescript/src/skills/nli/skill.ts`
 * inside the loop that consumes the child UAMP stream — when a child
 * envelope of type `HtmlContent` carries a `live` field, call
 * `forwardChildLiveBlock({ ... })` to mirror it into the parent stream
 * BEFORE forwarding the rest of the message. (TODO Plan v3-02: actual
 * call site wiring — gated behind a feature flag during rollout.)
 */

export interface ChildLiveBlock {
  contentId?: string;
  liveId?: string;
  /** Plan 1 transport tag. */
  transport: 'iframe-url' | 'webrtc' | 'portal-relay';
  /** Slug of the child agent (e.g. `@robutler.browserbase`). */
  childAgentSlug: string;
  /** When set, parent should request the canvas spawn hook. */
  spawnOnCanvas?: boolean;
}

export interface ForwardLiveEnvelope {
  type: 'tool_progress';
  kind: 'forward_live';
  data: {
    liveId: string;
    child_agent_slug: string;
    transport: ChildLiveBlock['transport'];
    display_hint: 'compact-chip' | 'inline';
    content_id?: string;
    spawn_on_canvas: boolean;
  };
}

/**
 * Build the `forward_live` envelope. Pure function — testable in
 * isolation, and the caller decides how to emit it (push onto the
 * NLI parent stream, or fan out via the agent runtime's outbox).
 */
export function buildForwardLiveEnvelope(block: ChildLiveBlock): ForwardLiveEnvelope {
  const liveId = block.liveId ?? (block.contentId ? `content:${block.contentId}` : '');
  if (!liveId) {
    throw new Error('forward_live: liveId or contentId required');
  }
  return {
    type: 'tool_progress',
    kind: 'forward_live',
    data: {
      liveId,
      child_agent_slug: block.childAgentSlug,
      transport: block.transport,
      display_hint: block.spawnOnCanvas ? 'inline' : 'compact-chip',
      content_id: block.contentId,
      spawn_on_canvas: Boolean(block.spawnOnCanvas),
    },
  };
}

export interface ForwardChildLiveBlockArgs extends ChildLiveBlock {
  /**
   * Caller-supplied emit hook — typically the parent agent runtime's
   * UAMP outbox. Kept abstract so this helper has no dependency on
   * `core/` types.
   */
  emit: (envelope: ForwardLiveEnvelope) => void | Promise<void>;
}

/**
 * Forward a single child live block to the parent stream. Idempotency
 * (don't emit twice for the same child block) is the caller's
 * responsibility — the parent NLI loop should de-dupe by `liveId`
 * because the child may republish on each heartbeat.
 */
export async function forwardChildLiveBlock(args: ForwardChildLiveBlockArgs): Promise<ForwardLiveEnvelope> {
  const envelope = buildForwardLiveEnvelope(args);
  await args.emit(envelope);
  return envelope;
}
