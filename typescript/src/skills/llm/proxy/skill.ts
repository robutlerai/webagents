/**
 * LLM Proxy Skill
 *
 * Routes LLM inference through the UAMP proxy (portal-hosted).
 * Same uniform interface as direct skills but uses WebSocket transport.
 * Sets _llm_capabilities and _llm_usage on context for PaymentSkill.
 */

import { Skill } from '../../../core/skill';
import { agentTrace, traceContent } from '../../../core/trace';
import { handoff } from '../../../core/decorators';
import type { Context, SkillConfig } from '../../../core/types';
import type { ClientEvent, ServerEvent, SessionCreateEvent, InputTextEvent } from '../../../uamp/events';
import {
  generateEventId,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
} from '../../../uamp/events';
import { UAMPClient, type UAMPInBandBuyer } from '../../../uamp/client';
import type { ContentItem, ToolDefinition, UsageStats } from '../../../uamp/types';
import type { UAMPUsage } from '../../../adapters/types';

export type ThinkingLevel = 'off' | 'low' | 'medium' | 'high';

export interface LLMProxySkillConfig extends SkillConfig {
  proxyUrl?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  enabledTools?: Record<string, unknown>;
  /**
   * Canonical thinking effort. Adapters at the proxy map this onto each
   * provider's native parameter (Google `thinkingConfig`, OpenAI
   * `reasoning_effort`, Anthropic `thinking.budget_tokens`,
   * xAI `reasoning_effort`).
   *
   * `boolean` is accepted as a legacy compatibility shim:
   *   - `true`   → leave the model's catalog default in effect (no override sent)
   *   - `false`  → equivalent to `'off'`
   * New callers should use a `ThinkingLevel` string.
   */
  thinking?: ThinkingLevel | boolean;
  /**
   * The MPP buyer (2026-09-19), `MppBuyer` from `skills/payments`, set once
   * by the operator. When the token this session pays with runs dry, the
   * platform's `/llm` socket names where to buy more: a `payment.required`
   * whose `mpp` entry carries `purchase_url` and no challenge (nobody on that
   * socket was verified, so none could be minted). With a buyer the UAMP
   * client follows it (`purchaseAt`: the buyer's own signed request to the
   * purchase URL, paid under its policy), and the token negotiation below
   * then runs once more against the funded balance. `MppBuyer` follows a
   * pointer only when its policy sets `dailyCapCents`; with none it refuses
   * (`pointer_needs_daily_cap`) and the event is handled as it always was,
   * which is also what happens without a buyer.
   */
  mppBuyer?: UAMPInBandBuyer;
  /**
   * THE SIGNED-IN PERSON (2026-09-24). A CLI agent with no provider key runs
   * here on the token `webagents login` stored: sent as the `Authorization`
   * session extension when there is no payment token, and funded by the
   * platform from that person's credits (portal `lib/llm/cli-bearer-funding.ts`).
   * A function is called per run, so a fresh login reaches an agent already
   * built. Never read from the environment.
   */
  platformToken?: string | (() => string | null | undefined | Promise<string | null | undefined>);
  /**
   * OTHER CALLERS' TURNS (S-327, 2026-09-28). `serve`, `mcp serve`, the
   * daemon and `cron run` build this skill with `callersPay`: each call is
   * paid by the payment token the caller's request carries, never by a
   * sign-in (a `platformToken` is ignored), and a call with no token is
   * refused with 402 (`CALLERS_PAY_REFUSAL`) before anything is dialled. The
   * Python skill's `callers_pay` does the same.
   */
  callersPay?: boolean;
}

/**
 * What a caller of a served agent reads when its request carried no payment
 * token and the agent runs on Robutler's models (S-327): the turn is the
 * caller's, so the caller pays. The shared fixture
 * `cli/final_sdk_serve_model.json` (`no_payment_token`) holds the words.
 */
export const CALLERS_PAY_REFUSAL =
  "This agent runs on Robutler's models, and each caller pays for its own turns: " +
  'send a payment token in the X-Payment-Token header.';


/**
 * The `X-Payment-Token` a transport put on the `session.create` extensions.
 * The completions transport sets it from the request header or the context
 * (`skills/transport/completions/skill.ts`), and the UAMP client sets the same
 * key, so this is the wire-level carrier rather than a fourth convention.
 */
function paymentTokenFromEvents(events: readonly unknown[]): string | undefined {
  for (const event of events) {
    const e = event as { type?: string; session?: { extensions?: Record<string, unknown> } };
    if (e?.type !== 'session.create') continue;
    const ext = e.session?.extensions;
    const token = ext?.['X-Payment-Token'] ?? ext?.['X-PAYMENT'];
    if (typeof token === 'string' && token) return token;
  }
  return undefined;
}

export class LLMProxySkill extends Skill {
  private proxyUrl: string;
  private modelConfig: LLMProxySkillConfig;

  constructor(config: LLMProxySkillConfig = {}) {
    super({ ...config, name: config.name || 'llm-proxy' });
    this.proxyUrl = config.proxyUrl
      || (typeof process !== 'undefined' && process.env?.ROBUTLER_LLM_PROXY_URL)
      || 'wss://robutler.ai/llm';
    this.modelConfig = config;
  }

  @handoff({ name: 'llm-proxy', priority: 10 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield {
      type: 'response.created' as const,
      event_id: generateEventId(),
      response_id: responseId,
    };

    const conversation = context.get<Array<{
      role: string;
      content: string | null;
      content_items?: ContentItem[];
      tool_calls?: unknown[];
      tool_call_id?: string;
    }>>('_agentic_messages') || [];

    const tools: ToolDefinition[] = [...(context.get<ToolDefinition[]>('_agentic_tools') || [])];
    // THE TOKEN ARRIVES THREE WAYS AND ONLY ONE WAS READ (S-206, 2026-09-21).
    //
    // `context.payment.token` is set when a PAYMENT SKILL verified a token for
    // this run. It is NOT set on the path that matters most: the machine door.
    // There, `applyServedToSdkContext` (portal, lib/payments/machine-door-http.ts)
    // writes the door's minted token onto the PER-REQUEST context, the
    // completions transport reads it back and forwards it as the
    // `X-Payment-Token` session extension — and this skill, running against the
    // AGENT's own context, saw none of it. So the LLM rail was dialled with no
    // credential, closed the socket with 4001, and the completion failed.
    //
    // The caller still received 200, which is what made it invisible: the door
    // had reserved a budget (`token_lock`), nothing was ever charged against
    // it, and the reservation was handed back untouched when the token expired.
    // Observed identically on local and on dev.
    //
    // Read all three, nearest first. `payment_token` is the same context key
    // the completions transport and the portal payment skill already use.
    const paymentToken =
      context.payment?.token
      ?? context.get<string>('payment_token')
      ?? paymentTokenFromEvents(events);

    if (conversation.length === 0) {
      for (const event of events) {
        if (event.type === 'session.create') {
          const createEvent = event as SessionCreateEvent;
          if (createEvent.session.instructions) {
            conversation.push({ role: 'system', content: createEvent.session.instructions });
          }
          if (createEvent.session.tools) {
            tools.push(...createEvent.session.tools);
          }
        } else if (event.type === 'input.text') {
          const inputEvent = event as InputTextEvent;
          conversation.push({ role: inputEvent.role || 'user', content: inputEvent.text });
        }
      }
    }

    const model = this.modelConfig.model || 'auto/balanced';

    // Set capabilities so PaymentSkill can size the lock
    context.set?.('_llm_capabilities', {
      model,
      provider: 'proxy',
      maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
      pricing: { inputPer1k: 0, outputPer1k: 0 },
    });

    const _first = conversation[0];
    const _preview = _first?.content_items
      ? _first.content_items.map((ci: any) => ci.type === 'text' ? ci.text?.slice(0, 100) : `[${ci.type}]`).join(' ')
      : (typeof _first?.content === 'string' ? _first.content.slice(0, 200) : '(null)');
    // The first message's text only under LOG_LOOP_DEBUG=1 (S-233, S-227's rule);
    // it went to the pod log for every hosted conversation.
    agentTrace(`[llm-proxy-skill] processUAMP: ${conversation.length} messages, ${tools.length} tools, paymentToken=${paymentToken ? 'yes' : 'no'}, url=${this.proxyUrl}${traceContent() ? `, firstMsg=${_preview}` : ''}`);

    // Normalize thinking config into a single canonical extension. Boolean
    // legacy: `true` omits the override (use catalog default), `false` →
    // `'off'`. String values pass through as-is and are clamped server-side
    // against the model's declared supported levels.
    const rawThinking = this.modelConfig.thinking;
    const thinkingExt: Partial<Record<string, unknown>> = {};
    if (typeof rawThinking === 'string') {
      thinkingExt.thinking_level = rawThinking;
    } else if (rawThinking === false) {
      thinkingExt.thinking_level = 'off';
      // Also send the legacy bool for older proxy versions during rollout.
      thinkingExt.thinking_enabled = false;
    }

    // Nothing to pay with, and never the sign-in (S-327): refused here, with
    // the caller's words and a status the server answers (`details.shown`,
    // `server/error-reply.ts` `shownResponseError`), rather than by the
    // platform with its protocol's (B11).
    const callersPay = this.modelConfig.callersPay === true;
    if (callersPay && !paymentToken) {
      agentTrace('[llm-proxy-skill] refused: callers pay, and the request carried no payment token');
      yield createResponseErrorEvent('payment_required', CALLERS_PAY_REFUSAL, responseId, { status: 402, shown: true });
      return;
    }

    let bearer: string | undefined;
    if (!paymentToken && this.modelConfig.platformToken && !callersPay) {
      const source = this.modelConfig.platformToken;
      bearer = (typeof source === 'function' ? await source() : source) || undefined;
    }

    const clientConfig = {
      url: this.proxyUrl,
      paymentToken,
      signal: context.signal,
      extensions: {
        ...(bearer ? { Authorization: `Bearer ${bearer}` } : {}),
        ...(context.metadata?.chatId ? { 'X-Chat-Id': context.metadata.chatId } : {}),
        ...(context.metadata?.agentId ? { 'X-Agent-Id': context.metadata.agentId } : {}),
        ...(this.modelConfig.enabledTools ? { enabled_tools: this.modelConfig.enabledTools } : {}),
        ...thinkingExt,
      },
      session: {
        modalities: ['text'] as ['text'],
      },
      ...(this.modelConfig.mppBuyer ? { buyer: this.modelConfig.mppBuyer } : {}),
    };

    const collectedOutput: ContentItem[] = [];
    let usage: UsageStats | undefined;
    let preExecutedRounds: import('../../../uamp/events').PreExecutedRound[] | undefined;
    let fullText = '';
    let error: Error | null = null;
    let done = false;
    let wasCancelled = false;
    // Why the provider stopped (`response.done.finish_reason`), and whether
    // the PROMPT was blocked; read below when the completion is empty.
    let finish: { reason?: string; blocked: boolean } = { blocked: false };
    // ONE RETRY ON A BROKEN TOOL CALL (2026-09-27). Gemini answers a turn it
    // could not turn into a tool call with an EMPTY completion and the finish
    // reason MALFORMED_FUNCTION_CALL (UNEXPECTED_TOOL_CALL for a tool it was
    // never given). That is a sampling accident, not a property of the
    // prompt, so the request goes once more on a fresh socket; the reason
    // rides on the `response.done` this skill yields, so the chat can say what
    // happened when the second attempt is empty too. Nothing else is retried:
    // an empty `STOP` was measured to repeat (the platform's own note), and a
    // retry there only bills a second prompt. The Python proxy skill does the
    // same.
    let attempt = 0;
    // The first attempt's tokens when there was a second: the person paid for both.
    let spent: UsageStats | undefined;

    const pendingEvents: ServerEvent[] = [];
    let notifyPending: (() => void) | null = null;

    for (;;) {
      attempt += 1;
      const client = new UAMPClient(clientConfig);

      client.on('delta', (text) => {
        fullText += text;
        pendingEvents.push(createResponseDeltaEvent(responseId, { type: 'text', text }));
        notifyPending?.();
      });

      client.on('toolCall', (tc: { id: string; name: string; arguments: string }) => {
        pendingEvents.push(createResponseDeltaEvent(responseId, {
          type: 'tool_call',
          tool_call: tc,
        }));
        notifyPending?.();
      });

      client.on('toolResult', (tr: Record<string, unknown>) => {
        pendingEvents.push(createResponseDeltaEvent(responseId, {
          type: 'tool_result',
          tool_result: tr as unknown as { call_id: string; result: string; status?: string },
        }));
        notifyPending?.();
      });

      client.on('file', (fileData: Record<string, unknown>) => {
        agentTrace(`[llm-proxy-skill] file event: content_id=${fileData.content_id} filename=${fileData.filename}`);
        pendingEvents.push(createResponseDeltaEvent(responseId, fileData as any));
        notifyPending?.();
      });

      client.on('done', (response) => {
        collectedOutput.push(...response.output);
        usage = response.usage;
        finish = { reason: response.finish_reason, blocked: response.finish_blocked === true };
        if (response.pre_executed_rounds && response.pre_executed_rounds.length > 0) {
          preExecutedRounds = response.pre_executed_rounds;
          if (process.env.LOG_LOOP_DEBUG === '1' || process.env.LOG_LLM_PAYLOAD === '1') {
            agentTrace(`[loop-debug] proxy-skill forwarded pre_executed_rounds=${preExecutedRounds.length} to agent`);
          }
        }

        // Copy proxy usage to context for PaymentSkill settlement
        if (usage) {
          const isByok = (response as Record<string, unknown>).is_byok === true
            || ((response.usage as unknown as Record<string, unknown>)?.is_byok === true);
          context.set?.('_llm_usage', {
            model,
            provider: 'proxy',
            input_tokens: usage.input_tokens ?? 0,
            output_tokens: usage.output_tokens ?? 0,
            is_byok: isByok,
          } satisfies UAMPUsage);
        }

        done = true;
        notifyPending?.();
      });

      client.on('error', (err) => {
        error = err;
        done = true;
        notifyPending?.();
      });

      client.on('thinking', (data) => {
        pendingEvents.push({
          type: 'thinking' as const,
          event_id: generateEventId(),
          content: data.content,
          stage: data.stage,
          redacted: data.redacted,
          is_delta: data.is_delta,
        } as ServerEvent);
        notifyPending?.();
      });

      client.on('cancelled', () => {
        wasCancelled = true;
        done = true;
        notifyPending?.();
      });

      // payment.required negotiation. A resubmit only makes sense when the
      // token CHANGED (fresh token, or a re-signed one after a top-up):
      // resubmitting a token the proxy just refused, unchanged, produced six
      // identical retries in ~350ms before the proxy gave up (F-023). One
      // unchanged fallback submit is allowed (covers proxies that raced a
      // settle); after that, fail the turn with the proxy's stated reason.
      let lastSubmittedPayment: string | null = null;
      client.on('paymentRequired', (req) => {
        // The buyer has just bought through the platform's purchase pointer:
        // what stands behind the token changed, so the one unchanged submit is
        // allowed again (a fresh token from `refreshToken` still goes first).
        if (req.purchased) lastSubmittedPayment = null;
        const refreshToken = context.payment?.refreshToken;
        const submit = (token: string) => {
          lastSubmittedPayment = token;
          client.sendPayment({ scheme: 'token', amount: req.amount, token });
        };
        const giveUp = () => {
          error = new Error(
            (req as { reason?: string }).reason || 'Insufficient balance for this request',
          );
          done = true;
          notifyPending?.();
          client.cancel().catch(() => {});
        };
        if (refreshToken) {
          refreshToken({ amount: req.amount }).then((newToken) => {
            if (newToken && newToken !== lastSubmittedPayment) submit(newToken);
            else if (paymentToken && paymentToken !== lastSubmittedPayment) submit(paymentToken);
            else giveUp();
          }).catch(() => {
            if (paymentToken && paymentToken !== lastSubmittedPayment) submit(paymentToken);
            else giveUp();
          });
        } else if (paymentToken && paymentToken !== lastSubmittedPayment) {
          submit(paymentToken);
        } else {
          giveUp();
        }
      });

      try {
        await client.connect();

        await client.sendResponse({
          messages: conversation,
          model,
          tools: tools.length > 0 ? tools : undefined,
          temperature: this.modelConfig.temperature ?? 0.7,
          max_tokens: this.modelConfig.max_tokens ?? 4096,
        });

        while (!done) {
          while (pendingEvents.length > 0) {
            yield pendingEvents.shift()!;
          }
          if (!done) {
            await new Promise<void>(resolve => { notifyPending = resolve; });
            notifyPending = null;
          }
        }
        while (pendingEvents.length > 0) {
          yield pendingEvents.shift()!;
        }
      } catch (err) {
        // The platform's OWN reason first (2026-09-24). A refused session gets
        // a `response.error` ("Not enough credits...", "Invalid or expired
        // payment token") and then a close; the close failed the connection
        // attempt and overwrote that reason with "WebSocket closed unexpectedly
        // (code=4002)", which is all the person saw. The close stays in the
        // text, after the reason: the portal's agent completions route reads
        // `(code=4001)` there to answer an unfunded caller 402
        // (`unfundedRunRefused`).
        const failure = err instanceof Error ? err : new Error(String(err));
        // Set by the `error` listener, which the compiler cannot see from here.
        const earlier = error as Error | null;
        error = earlier && earlier !== failure ? new Error(`${earlier.message} [${failure.message}]`) : failure;
        agentTrace(`[llm-proxy-skill] error: ${error.message}`);
      } finally {
        client.close();
      }

      // Set by listeners, which the compiler cannot see from here.
      const outcome = { error: error as Error | null, cancelled: wasCancelled as boolean, finish: finish as { reason?: string; blocked: boolean } };
      const empty = !fullText.trim() && !collectedOutput.some((item) => item.type === 'text' || item.type === 'tool_call');
      if (
        attempt === 1 && !outcome.error && !outcome.cancelled && empty
        && outcome.finish.reason !== undefined && RETRY_FINISH_REASONS.has(outcome.finish.reason) && !outcome.finish.blocked
      ) {
        agentTrace(`[llm-proxy-skill] ${model} returned an empty completion (${outcome.finish.reason}); sending the request once more`);
        spent = usage;
        collectedOutput.length = 0;
        fullText = '';
        usage = undefined;
        preExecutedRounds = undefined;
        done = false;
        continue;
      }
      break;
    }

    if (spent && usage) {
      const sum = (a?: number, b?: number) => (a ?? 0) + (b ?? 0);
      usage = {
        ...usage,
        input_tokens: sum(usage.input_tokens, spent.input_tokens),
        output_tokens: sum(usage.output_tokens, spent.output_tokens),
        total_tokens: sum(usage.total_tokens, spent.total_tokens),
      };
      context.set?.('_llm_usage', {
        model,
        provider: 'proxy',
        input_tokens: usage.input_tokens ?? 0,
        output_tokens: usage.output_tokens ?? 0,
        is_byok: false,
      } satisfies UAMPUsage);
    }

    if (error) {
      // Through the trace, not `console.error`: the error is yielded, and the
      // host shows or logs it. In the CLI chat the raw line landed in the
      // middle of the rendered reply.
      agentTrace(`[llm-proxy-skill] yielding proxy_error: ${error.message}`);
      yield createResponseErrorEvent('proxy_error', error.message, responseId);
      return;
    }

    const output: ContentItem[] = collectedOutput.length > 0
      ? collectedOutput
      : fullText
        ? [{ type: 'text' as const, text: fullText }]
        : [];

    const status = wasCancelled ? 'cancelled' : 'completed';
    const doneEvent = createResponseDoneEvent(responseId, output, status, usage, preExecutedRounds);
    // Why the provider stopped, for the chat's line when the reply is empty
    // (`cli/failures.ts`, `presentEmptyReply`).
    const lastFinish = finish as { reason?: string; blocked: boolean };
    Object.assign(doneEvent.response, {
      ...(lastFinish.reason ? { finish_reason: lastFinish.reason } : {}),
      ...(lastFinish.blocked ? { finish_blocked: true } : {}),
      ...(attempt > 1 ? { finish_retried: true } : {}),
    });
    yield doneEvent;
  }
}

/**
 * Finish reasons worth one more request when the completion was empty (the
 * Python proxy skill keeps the same list).
 */
export const RETRY_FINISH_REASONS: ReadonlySet<string> = new Set(['MALFORMED_FUNCTION_CALL', 'UNEXPECTED_TOOL_CALL']);
