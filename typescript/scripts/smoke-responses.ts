/**
 * Smoke test for the OpenAI/xAI Responses-API adapters.
 *
 * Runs three live calls against real provider endpoints to validate:
 *   1. openai gpt-5.5  + web_search
 *   2. xai    grok-4.3 + web_search
 *   3. openai gpt-5.5  + code_interpreter
 *
 * Usage:
 *   npx tsx webagents/typescript/scripts/smoke-responses.ts
 *
 * Reads OPENAI_API_KEY / XAI_API_KEY from env (or from
 * infrastructure/secrets/local.env / .env via dotenv).
 */
import { config } from 'dotenv';
import { resolve } from 'path';
import { openaiAdapter, xaiAdapter } from '../src/adapters/responses';
import type { AdapterChunk, AdapterRequestParams, LLMAdapter } from '../src/adapters/types';

config({ path: resolve(process.cwd(), 'infrastructure/secrets/local.env') });
config({ path: resolve(process.cwd(), '.env') });

const OPENAI_KEY = process.env.OPENAI_API_KEY;
const XAI_KEY = process.env.XAI_API_KEY;

interface CollectedStream {
  text: string;
  thinking: string;
  toolStarts: Array<{ name: string; id: string }>;
  toolResults: Array<{ name: string; result: unknown }>;
  images: number;
  usage: { input: number; output: number; reasoning?: number } | null;
  encryptedReasoning: number;
}

async function collectStream(adapter: LLMAdapter, response: Response): Promise<CollectedStream> {
  const out: CollectedStream = {
    text: '',
    thinking: '',
    toolStarts: [],
    toolResults: [],
    images: 0,
    usage: null,
    encryptedReasoning: 0,
  };
  for await (const chunk of adapter.parseStream(response) as AsyncIterable<AdapterChunk>) {
    switch (chunk.type) {
      case 'text':
        out.text += chunk.text;
        break;
      case 'thinking':
        out.thinking += chunk.text;
        break;
      case 'tool_call_start':
        out.toolStarts.push({ name: chunk.name, id: chunk.id });
        break;
      case 'tool_result':
        out.toolResults.push({ name: chunk.name, result: chunk.result });
        break;
      case 'image':
        out.images += 1;
        break;
      case 'usage': {
        out.usage = {
          input: chunk.input,
          output: chunk.output,
          reasoning: (chunk as { reasoning?: number }).reasoning,
        };
        const enc = (chunk as { _encryptedReasoning?: string[] })._encryptedReasoning;
        if (Array.isArray(enc)) out.encryptedReasoning = enc.length;
        break;
      }
    }
  }
  return out;
}

async function runOne(
  label: string,
  adapter: LLMAdapter,
  params: AdapterRequestParams,
): Promise<void> {
  const banner = `\n=== ${label} ===`;
  console.log(banner);
  const req = adapter.buildRequest(params);
  const t0 = Date.now();
  const res = await fetch(req.url, {
    method: 'POST',
    headers: req.headers,
    body: req.body,
  });
  if (!res.ok) {
    const body = await res.text();
    console.log(`  HTTP ${res.status}: ${body.slice(0, 600)}`);
    return;
  }
  const collected = await collectStream(adapter, res);
  const ms = Date.now() - t0;
  console.log(`  duration: ${ms} ms`);
  console.log(`  text:     ${JSON.stringify(collected.text.slice(0, 200))}${collected.text.length > 200 ? '…' : ''}`);
  if (collected.thinking) console.log(`  thinking: ${collected.thinking.length} chars`);
  if (collected.toolStarts.length) {
    console.log(`  tool_call_start: ${collected.toolStarts.map(t => t.name).join(', ')}`);
  }
  if (collected.toolResults.length) {
    for (const tr of collected.toolResults) {
      const summary =
        tr.result && typeof tr.result === 'object'
          ? Object.keys(tr.result as Record<string, unknown>).join(',')
          : typeof tr.result;
      console.log(`  tool_result:     ${tr.name} -> {${summary}}`);
    }
  }
  if (collected.images) console.log(`  images:          ${collected.images}`);
  if (collected.usage) console.log(`  usage:           ${JSON.stringify(collected.usage)}`);
  if (collected.encryptedReasoning) console.log(`  _encryptedReasoning: ${collected.encryptedReasoning} blob(s)`);
}

async function main(): Promise<void> {
  if (!OPENAI_KEY) {
    console.error('OPENAI_API_KEY not set — skipping OpenAI smoke');
  }
  if (!XAI_KEY) {
    console.error('XAI_API_KEY not set — skipping xAI smoke');
  }

  if (OPENAI_KEY) {
    // 1. gpt-5.5 web_search
    await runOne('openai gpt-5.5  + web_search', openaiAdapter, {
      apiKey: OPENAI_KEY,
      model: 'gpt-5.5',
      stream: true,
      maxTokens: 800,
      messages: [
        {
          role: 'user',
          content:
            'Use the web_search tool to find one fresh, dated headline from today and report the source. Be concise.',
        },
      ],
      tools: [{ type: 'web_search' }],
    });
  }

  if (XAI_KEY) {
    // 2. grok-4.3 web_search
    await runOne('xai grok-4.3   + web_search', xaiAdapter, {
      apiKey: XAI_KEY,
      model: 'grok-4.3',
      stream: true,
      maxTokens: 800,
      messages: [
        {
          role: 'user',
          content:
            'Use the web_search tool to find one fresh, dated headline from today and report the source. Be concise.',
        },
      ],
      tools: [{ type: 'web_search' }],
    });
  }

  if (OPENAI_KEY) {
    // 3. gpt-5.5 code_interpreter
    await runOne('openai gpt-5.5  + code_interpreter', openaiAdapter, {
      apiKey: OPENAI_KEY,
      model: 'gpt-5.5',
      stream: true,
      maxTokens: 800,
      messages: [
        {
          role: 'user',
          content:
            'Use the code_interpreter tool to compute the 30th Fibonacci number. Show the code and the answer.',
        },
      ],
      tools: [{ type: 'code_interpreter', container: { type: 'auto' } }],
    });
  }
}

main().catch((e) => {
  console.error('Smoke run failed:', e);
  process.exit(1);
});
