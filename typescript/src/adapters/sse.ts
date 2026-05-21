/**
 * Shared SSE stream reader used by all provider adapters.
 * Reads response body as text, splits on newlines, yields parsed `data:` lines.
 */
export async function* readSSEStream(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<unknown> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  // Track the most recent `event:` header within an SSE frame so the adapter
  // can distinguish e.g. `event: error` from `event: message`. Anthropic
  // emits errors mid-stream via `event: error\ndata: {...}` on an otherwise
  // HTTP-200 response; if we drop the event name we silently discard the
  // error and the caller just sees an empty completion. State lives outside
  // the read loop because the `event:` and matching `data:` lines can arrive
  // in different network chunks.
  let pendingEvent: string | undefined;

  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          // Blank line terminates an SSE frame; reset the pending event.
          pendingEvent = undefined;
          continue;
        }
        if (trimmed.startsWith('event:')) {
          pendingEvent = trimmed.slice(6).trim();
          continue;
        }
        if (!trimmed.startsWith('data: ')) continue;
        const data = trimmed.slice(6);
        if (data === '[DONE]') {
          pendingEvent = undefined;
          continue;
        }

        try {
          const parsed = JSON.parse(data);
          if (parsed && typeof parsed === 'object' && pendingEvent && !('__sseEvent' in parsed)) {
            (parsed as Record<string, unknown>).__sseEvent = pendingEvent;
          }
          yield parsed;
        } catch {
          // partial JSON or non-JSON line; skip
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
