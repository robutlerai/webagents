/**
 * The agent loop's trace lines, and whether the host wants them (2026-09-24).
 *
 * `core/agent.ts` wrote its trace with bare `console.log`: the loop starting,
 * every iteration, every streamed text chunk, every tool call. The portal
 * runtime reads those lines in its pod logs, so for every host the default
 * here is unchanged: ON, to stdout.
 *
 * The CLI is a different host. `webagents -p "..." --output-format json`
 * printed three trace lines on STDOUT ahead of the JSON document, so the one
 * machine-readable mode could not be piped into `jq`, and a first-time user's
 * very first reply arrived wrapped in loop internals. The CLI therefore turns
 * the trace off, or sends it to stderr when `WEBAGENTS_DEBUG` is set.
 *
 * What the lines CONTAIN is decided separately: reply text and tool
 * arguments appear only under `LOG_LOOP_DEBUG=1` (`traceContent` below,
 * S-227 in the portal's security log).
 */

export interface AgentTraceOptions {
  /** Whether trace lines are written at all. */
  enabled?: boolean;
  /** Where they go when enabled. Defaults to `console.log`. */
  sink?: (line: string) => void;
}

let enabled = true;
let sink: (line: string) => void = (line) => console.log(line);

/** Set once by the host at startup. Fields left out keep their current value. */
export function setAgentTrace(options: AgentTraceOptions): void {
  if (options.enabled !== undefined) enabled = options.enabled;
  if (options.sink) sink = options.sink;
}

/** One trace line from the agent loop. */
export function agentTrace(line: string): void {
  if (enabled) sink(line);
}

/**
 * Whether trace lines may carry CONTENT: reply text and tool arguments
 * (2026-09-24, S-227). Off unless `LOG_LOOP_DEBUG=1`, the flag the loop's other
 * content lines already used. Two lines did not ask: one logged every streamed
 * text chunk and one up to 500 characters of every tool call's arguments, so
 * the portal's pod logs held the replies of portal-hosted agents and whatever
 * their tools were handed. They log lengths now.
 */
export function traceContent(): boolean {
  return typeof process !== 'undefined' && process.env?.LOG_LOOP_DEBUG === '1';
}
