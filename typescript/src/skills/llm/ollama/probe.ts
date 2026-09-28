/**
 * Whether Ollama answers, and which models it serves (2026-09-26, plan item
 * 2.8): what `webagents doctor` and `webagents models` ask before they call
 * a local model "ready". One GET of the OpenAI-compatible model list, with a
 * short timeout, because a machine without Ollama must not make either
 * command wait. The Python CLI's `ollama/probe.py` asks the same way, and
 * the words are pinned by `python/tests/fixtures/w2ops/models.json`.
 */

export interface OllamaProbe {
  /** Whether anything answered at the address. */
  ok: boolean;
  /** The model ids it serves (`llama3.2:latest`), when it answered. */
  models: string[];
}

/** GET `<baseUrl>/models`; a refused, timed out or malformed answer is `ok: false`. */
export async function probeOllama(baseUrl: string, timeoutMs = 1500): Promise<OllamaProbe> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${baseUrl.replace(/\/+$/, '')}/models`, { signal: controller.signal });
    if (!response.ok) return { ok: false, models: [] };
    const body = (await response.json()) as { data?: Array<{ id?: unknown }> } | null;
    const models = Array.isArray(body?.data)
      ? body!.data.map((row) => (typeof row?.id === 'string' ? row.id : '')).filter(Boolean)
      : [];
    return { ok: true, models };
  } catch {
    return { ok: false, models: [] };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Whether a served id counts as the wanted model: the same id, or, for a
 * wanted name with no tag, the served name's part before the colon
 * (`llama3.2` is served as `llama3.2:latest` or `llama3.2:3b`).
 */
export function servesModel(served: readonly string[], wanted: string): boolean {
  return served.some((id) => id === wanted || (!wanted.includes(':') && id.split(':')[0] === wanted));
}

/**
 * The `model` check of `webagents doctor` for an `ollama/<model>` agent, from
 * what the probe found: the fixture's words for answering with the model,
 * answering without it, and not answering.
 */
export function ollamaModelCheck(
  model: string,
  baseUrl: string,
  probe: OllamaProbe,
): { status: 'ok' | 'warn' | 'fail'; detail: string; fix?: string } {
  const bare = model.includes('/') ? model.slice(model.indexOf('/') + 1) : model;
  if (!probe.ok) {
    return {
      status: 'fail',
      detail: `${model}, but nothing answers at ${baseUrl}`,
      fix: `Start Ollama (\`ollama serve\`) and pull the model (\`ollama pull ${bare}\`), or set OLLAMA_BASE_URL`,
    };
  }
  if (!servesModel(probe.models, bare)) {
    return { status: 'warn', detail: `${model}, at ${baseUrl}, but the model is not pulled`, fix: `\`ollama pull ${bare}\`` };
  }
  return { status: 'ok', detail: `${model}, at ${baseUrl}` };
}
