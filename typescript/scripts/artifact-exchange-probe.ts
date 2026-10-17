/**
 * Two agents exchange a folder of large files with no platform in between.
 *
 * A producer agent publishes a folder (`files.publish`) and serves it from its
 * `/artifacts/blobs/sha256:<hex>` endpoint; a consumer agent fetches the
 * manifest and every file with signed, ranged requests, is cut off part way
 * through the largest file, resumes from the chunks it already verified, and
 * checks every byte. Both agents are served from this process (the producer
 * verifies the consumer's signature against the key set the consumer serves),
 * or, with `--producer python`, the producer is the Python SDK's
 * `webagents.artifacts.probe_server` in a child process.
 *
 * `process.memoryUsage()` is sampled every 100 ms; the peaks are printed. The
 * point is that they do not grow with the file.
 *
 *   tsx scripts/artifact-exchange-probe.ts [--size-mb=1024] [--producer=ts|python] [--cut-at=0.6] [--keep]
 */

import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, createWriteStream, createReadStream, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash, randomFillSync } from 'node:crypto';
import { createServer } from 'node:net';
import { networkInterfaces } from 'node:os';
import { spawn, type ChildProcess } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

process.env.ROBUTLER_AGENT_URL_ALLOW_PRIVATE = '1';

import { BaseAgent } from '../src/core/agent';
import { AgentIdentity } from '../src/crypto/identity';
import { serve } from '../src/server/node';
import { AccessSkill } from '../src/skills/access/skill';
import { parseAccess } from '../src/access/policy';
import { ArtifactRegistry, ArtifactsSkill, fetchArtifactSet, fetchManifest, files, type ArtifactSet } from '../src/artifacts';

const args = process.argv.slice(2);
const arg = (name: string): string | undefined => args.find((a) => a.startsWith(`--${name}=`))?.split('=')[1];
const SIZE = Number(arg('size-mb') ?? 1024) * 1024 * 1024;
const PRODUCER = arg('producer') ?? 'ts';
const CUT_AT = Number(arg('cut-at') ?? 0.6);
const KEEP = args.includes('--keep');
const MiB = 1024 * 1024;
const fmt = (n: number) => `${(n / MiB).toFixed(1)} MiB`;
const HERE = path.dirname(fileURLToPath(import.meta.url));

type Usage = { heapUsed: number; external: number; arrayBuffers: number; rss: number };
const snap = (): Usage => { const u = process.memoryUsage(); return { heapUsed: u.heapUsed, external: u.external, arrayBuffers: u.arrayBuffers, rss: u.rss }; };
class Sampler {
  peak = snap();
  timer = setInterval(() => this.tick(), 100);
  tick() { const u = snap(); for (const k of Object.keys(u) as Array<keyof Usage>) this.peak[k] = Math.max(this.peak[k], u[k]); }
  reset() { this.tick(); this.peak = snap(); }
  stop() { clearInterval(this.timer); }
  line(label: string) { const p = this.peak; return `${label.padEnd(30)} heapUsed ${fmt(p.heapUsed).padStart(10)}  external ${fmt(p.external).padStart(10)}  arrayBuffers ${fmt(p.arrayBuffers).padStart(10)}  rss ${fmt(p.rss).padStart(10)}`; }
}

/**
 * The signer refuses a loopback agent URL (a platform could never fetch a key
 * set from it), so both agents are addressed by this machine's own
 * non-loopback IPv4, with plaintext allowed for the private range.
 */
function ownAddress(): string {
  for (const list of Object.values(networkInterfaces())) {
    for (const entry of list ?? []) {
      if (entry.family === 'IPv4' && !entry.internal) return entry.address;
    }
  }
  throw new Error('no non-loopback IPv4 address on this machine; the signer refuses loopback agent URLs');
}
const HOST = ownAddress();

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.listen(0, HOST, () => { const port = (s.address() as { port: number }).port; s.close(() => resolve(port)); });
    s.on('error', reject);
  });
}

async function writeRandom(file: string, size: number): Promise<string> {
  const hash = createHash('sha256');
  const buf = Buffer.allocUnsafe(MiB);
  async function* gen() {
    let left = size;
    while (left > 0) {
      const n = Math.min(left, buf.length);
      randomFillSync(buf, 0, n);
      const out = Buffer.from(buf.subarray(0, n));
      hash.update(out);
      left -= n;
      yield out;
    }
  }
  await pipeline(Readable.from(gen()), createWriteStream(file, { highWaterMark: MiB }));
  return hash.digest('hex');
}

async function hashPath(file: string): Promise<string> {
  const h = createHash('sha256');
  for await (const c of createReadStream(file, { highWaterMark: MiB })) h.update(c as Buffer);
  return h.digest('hex');
}

async function main() {
  const t0 = Date.now();
  const sampler = new Sampler();
  const root = mkdtempSync(path.join(tmpdir(), 'artifact-exchange-'));
  const folder = path.join(root, 'out');
  mkdirSync(path.join(folder, 'gds'), { recursive: true });
  mkdirSync(path.join(folder, 'reports'), { recursive: true });
  console.log(`[probe] producer=${PRODUCER} size=${fmt(SIZE)} root=${root} node ${process.version}`);
  console.log(sampler.line('baseline'));
  const expected: Record<string, string> = {};
  expected['gds/top.gds'] = await writeRandom(path.join(folder, 'gds/top.gds'), SIZE);
  writeFileSync(path.join(folder, 'reports/drc.txt'), 'clean\n');
  expected['reports/drc.txt'] = createHash('sha256').update('clean\n').digest('hex');
  writeFileSync(path.join(folder, 'summary.json'), JSON.stringify({ ok: true }));
  expected['summary.json'] = createHash('sha256').update(JSON.stringify({ ok: true })).digest('hex');
  console.log(`[probe] source written (${((Date.now() - t0) / 1000).toFixed(1)} s)`);

  // The consumer: an identity whose key set is served, so the producer can verify its signature.
  const consumerPort = await freePort();
  const consumerBase = `http://${HOST}:${consumerPort}`;
  const consumerUrl = `${consumerBase}/agents/consumer`;
  const consumerIdentity = new AgentIdentity({ agentId: 'consumer', issuer: consumerUrl });
  await consumerIdentity.initialize();
  const consumerAgent = new BaseAgent({ name: 'consumer', skills: [] });
  const consumerServer = await serve(consumerAgent, { port: consumerPort, hostname: HOST, basePath: '/agents/consumer', identity: consumerIdentity, publicUrl: consumerBase, logging: false, heartbeat: false });

  // The producer.
  const producerPort = await freePort();
  const producerBase = `http://${HOST}:${producerPort}`;
  let producerUrl: string;
  let manifestUrl: string;
  let manifestSha: string;
  let child: ChildProcess | null = null;
  let producerServer: Awaited<ReturnType<typeof serve>> | null = null;
  const task = 'probe-task';
  sampler.reset();
  let t = Date.now();
  if (PRODUCER === 'ts') {
    producerUrl = `${producerBase}/agents/producer`;
    const producerIdentity = new AgentIdentity({ agentId: 'producer', issuer: producerUrl });
    await producerIdentity.initialize();
    const registry = new ArtifactRegistry();
    const producerAgent = new BaseAgent({
      name: 'producer',
      skills: [new AccessSkill({ policy: parseAccess({ default: 'public' }), publicUrl: producerBase }), new ArtifactsSkill({ registry })],
    });
    producerServer = await serve(producerAgent, { port: producerPort, hostname: HOST, basePath: '/agents/producer', identity: producerIdentity, publicUrl: producerBase, logging: false, heartbeat: false });
    const published = await files.publish(folder, { producer: { agent: producerUrl, task }, registry, baseUrl: producerUrl, audience: [consumerUrl], allowHttp: true });
    manifestUrl = published.manifestUrl!;
    manifestSha = published.sha256;
  } else {
    producerUrl = `${producerBase}/producer`;
    const python = path.resolve(HERE, '../../python/.venv/bin/python');
    child = spawn(python, ['-m', 'webagents.artifacts.probe_server', '--folder', folder, '--port', String(producerPort), '--host', HOST, '--public-url', producerBase, '--agent', 'producer', '--audience', consumerUrl, '--task', task], {
      cwd: path.resolve(HERE, '../../python'),
      // The producer verifies the consumer's signature by fetching its key set
      // from this machine; a proxy configured in the environment must not be
      // in that path.
      env: { ...process.env, ROBUTLER_AGENT_URL_ALLOW_PRIVATE: '1', WEBAGENTS_PUBLIC_URL: producerBase, NO_PROXY: HOST, no_proxy: HOST },
      stdio: ['ignore', 'pipe', 'inherit'],
    });
    const line = await new Promise<string>((resolve, reject) => {
      let buf = '';
      child!.stdout!.on('data', (d) => {
        buf += String(d);
        const m = /^PUBLISHED (.*)$/m.exec(buf);
        if (m) resolve(m[1]);
      });
      child!.on('exit', (code) => reject(new Error(`python producer exited ${code}`)));
      setTimeout(() => reject(new Error('python producer did not publish in 120 s')), 120_000);
    });
    const info = JSON.parse(line) as { manifestUrl: string; sha256: string };
    manifestUrl = info.manifestUrl;
    manifestSha = info.sha256;
    // The producer prints before its server listens; wait for the door to open.
    const deadline = Date.now() + 30_000;
    for (;;) {
      try {
        const res = await fetch(`${producerBase}/health`);
        if (res.ok) break;
      } catch {
        /* not yet */
      }
      if (Date.now() > deadline) throw new Error('the python producer never started listening');
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  console.log(sampler.line(`publish (${((Date.now() - t) / 1000).toFixed(1)} s)`));

  // The consumer fetches: manifest, then files; the big file is cut once.
  const dest = path.join(root, 'in');
  const common = { identity: consumerIdentity, allowHttp: true, task, concurrency: 4 };
  sampler.reset();
  t = Date.now();
  const manifest = await fetchManifest(manifestUrl, { ...common, expectedSha256: manifestSha });
  const set: ArtifactSet = manifest.set;
  const big = set.entries.find((e) => e.path === 'gds/top.gds')!;
  if (big.sha256 !== expected['gds/top.gds']) throw new Error('the manifest names a different hash for the big file');
  let streamed = 0;
  let cut = false;
  const cuttingFetch: typeof fetch = async (input, init) => {
    const res = await fetch(input, init);
    const url = String(input instanceof Request ? input.url : input);
    if (cut || !url.includes(big.sha256) || !res.body) return res;
    const reader = res.body.getReader();
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const { value, done } = await reader.read();
        if (done) { controller.close(); return; }
        streamed += value.byteLength;
        if (!cut && streamed >= SIZE * CUT_AT) {
          cut = true;
          await reader.cancel().catch(() => {});
          controller.error(new Error('connection cut by the probe'));
          return;
        }
        controller.enqueue(value);
      },
      cancel() { void reader.cancel().catch(() => {}); },
    });
    return new Response(body, { status: res.status, headers: res.headers });
  };
  let firstError: unknown = null;
  try {
    await fetchArtifactSet(set, { ...common, dest, fetch: cuttingFetch });
  } catch (err) {
    firstError = err;
  }
  if (!firstError) throw new Error('the cut never happened');
  const partState = JSON.parse(readFileSync(path.join(dest, 'gds/top.gds.part.json'), 'utf8')) as { done: number[] };
  console.log(sampler.line(`fetch until cut at ${Math.round(CUT_AT * 100)} % (${((Date.now() - t) / 1000).toFixed(1)} s)`));
  console.log(`[probe] cut after ${fmt(streamed)} streamed; ${partState.done.length} chunks verified and kept`);

  sampler.reset();
  t = Date.now();
  const result = await fetchArtifactSet(set, { ...common, dest });
  const resumedFile = result.files.find((f) => f.path === 'gds/top.gds')!;
  console.log(sampler.line(`resume and finish (${((Date.now() - t) / 1000).toFixed(1)} s, ${(resumedFile.bytesFetched / MiB / ((Date.now() - t) / 1000)).toFixed(0)} MiB/s)`));
  console.log(`[probe] resumed fetch moved ${fmt(resumedFile.bytesFetched)} of ${fmt(SIZE)} (${partState.done.length} chunks were kept)`);

  for (const [rel, sha] of Object.entries(expected)) {
    const got = await hashPath(path.join(dest, rel));
    if (got !== sha) throw new Error(`${rel}: fetched bytes hash to ${got}, source was ${sha}`);
  }
  if (statSync(path.join(dest, 'gds/top.gds')).size !== SIZE) throw new Error('size differs');
  sampler.stop();
  console.log(`[probe] every byte verified; peak heapUsed ${fmt(sampler.peak.heapUsed)}, peak rss ${fmt(sampler.peak.rss)}`);
  console.log(`[probe] OK in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

  await consumerServer.close?.();
  await producerServer?.close?.();
  child?.kill();
  if (!KEEP) rmSync(root, { recursive: true, force: true });
  process.exit(0);
}

main().catch((err) => {
  console.error('[probe] FAILED:', err instanceof Error ? err.stack ?? err.message : err);
  process.exit(1);
});
