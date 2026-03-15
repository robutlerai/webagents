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
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const data = trimmed.slice(6);
        if (data === '[DONE]') continue;

        try {
          yield JSON.parse(data);
        } catch {
          // partial JSON or non-JSON line; skip
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
