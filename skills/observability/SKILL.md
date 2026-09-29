---
name: observability
description: "Logs, metrics, and traces as complementary signals, high-cardinality dimensions, and the instrumentation budget."
license: MIT-0
metadata:
  author: "ivangdavila"
  source: "https://clawhub.ai/ivangdavila/skills/observability"
  version: "1.0.0"
  homepage: "https://clawic.com/skills/observability"
---

## Mental model: three signals, one budget

- Metrics answer "is it broken, and where": pre-aggregated, cheap to query, low cardinality, the signal you alert on. Logs answer "why": per-event detail, queryable by field. Traces answer "where in the request path, and how long": the span tree across services.
- The signals are complementary, not redundant: a metric shows p99 latency rose; a trace shows which span; a log shows the error string. Reaching for one to do another's job is the cardinal sin, a metric per `request_id`, a log you count instead of a counter.
- Exemplars bridge metrics to traces: a histogram bucket carrying a `trace_id` exemplar turns "latency is high" into "here is one slow request." Reach for an exemplar before you add a label.
- The instrumentation budget: every span, log line, and series costs ingestion + storage + query. Budget telemetry like memory; a service emitting 1000 spans/request is the new memory leak, just billed monthly.
- Cardinality multiplies, it does not add: 10 labels of 10 values each is not 100 series, it is 10^10. This single fact is the difference between a cheap pipeline and a six-figure bill.
- Cost = ingestion (bytes in) + storage (bytes retained) + query (bytes scanned). You cut cost at ingestion, by sampling, not at query; once it is stored, you have paid.
- Observability vs monitoring: monitoring covers known-unknowns (alerts you wrote); observability lets you investigate unknown-unknowns (a novel failure) by querying high-cardinality fields retroactively. A pipeline with only pre-aggregated metrics is monitoring, not observability.

## Metrics: types, aggregation, the percentile trap

- Four types: counter (monotonic, `rate()` it), gauge (current value, never rate), histogram (bucketed distribution, the workhorse), summary (client-side quantiles, legacy, not aggregatable across instances). Use histogram, not summary: summaries cannot be averaged across pods.
- The four golden signals: latency, traffic, errors, saturation. RED (Rate, Errors, Duration) for request-driven services; USE (Utilization, Saturation, Errors) for resources, CPU, disk, network. RED for the service, USE for the box.
- Percentiles, not averages: alert and report on p99 latency, never the mean. An average of 50ms hides a p99 of 5s; the mean is fine while the tail is on fire. `avg(p99)` across instances is meaningless: average the buckets, not the quantiles.
- Set histogram buckets to your SLO boundary, not the defaults: Prometheus default buckets are tuned for seconds and lie about sub-100ms APIs. Buckets at [25, 50, 100, 250, 500, 1000, 2500, 5000]ms cover a web service; the bucket above your SLO threshold is the one you alert on.
- `histogram_quantile(0.99, rate(http_duration_bucket[5m]))` is the standard query; it reads bucket boundaries, not raw values. `rate` needs >= 2 samples, a range >= 2x the scrape interval. A 1m range at 15s scrape = 4 samples, noisy; use 5m.
- Recording rules pre-compute heavy queries: if a dashboard query takes > 1s, move it to a recording rule that writes a new metric every 30-60s. The p99-per-service query is the canonical candidate.
- `rate()` handles counter resets on restart; `increase()` over a long range can overcount across resets. Trust `rate` for trends, not `increase` for exact counts.
- Count what you alert on: if you alert on `5xx / total`, emit both as counters from the code; do not derive the total from log lines. Deriving metrics from logs is logs doing metrics' job at ~100x cost.
- Gauges for current state (queue depth, active connections, memory); counters for cumulative events (requests, errors). Putting a request count in a gauge loses the rate and the reset handling.

## High cardinality: the multiplicative cost

- Cardinality = number of distinct label-value combinations. Cost grows as the product across labels: `user_id` (1M values) x `endpoint` (50) = 50M series. One bad label sinks the pipeline.
- The blacklist, never a metric label: `user_id`, `email`, `request_id`, `session_id`, `trace_id`, `ip`, raw URL. These belong on spans and logs, or as exemplars, never as metric labels.
- Sane budget: < 10 labels per metric, < 100k active series per service. Past ~1M series per backend, query latency and storage cost degrade non-linearly; the cliff, not the slope, is what bites.
- Audit with `/api/v1/status/tsdb` (Prometheus) or the equivalent top-series endpoint: it lists labels by distinct value count. Run it monthly; a label with 100k+ values is the smoking gun.
- A metric that needs a per-user breakdown is using metrics as logs: move the dimension to a span attribute (sampled, so bounded) or a structured log field (indexed, queryable), not a label.
- The cardinality cliff is a step, not a ramp: cost grows linearly then jumps when you exceed a shard limit or trigger an index rehash. Watch the series-count trend, not just the bill; the trend breaks before the bill does.
- Drop high-cardinality attributes at the collector before they reach the backend: an OTel processor strips `http.target` (raw URL) and keeps `http.route` (templated). The collector is the last free transform.
- A new label is a cost decision made at PR time, not at bill time. Review label additions like schema changes; the reviewer who asks "how many distinct values can this take?" is doing the job.

## Logs: structure, levels, the trace bridge

- Structured logs are mandatory: JSON or logfmt, never free text. Every line carries `timestamp`, `level`, `message`, `trace_id`, `span_id`, plus contextual fields. Free text is unqueryable at scale.
- Templated messages, variable fields: `user logged in` + `{user_id: 12345}` is one log stream; `user 12345 logged in` is 1M streams. Log aggregation indexes the template, not the values; high-cardinality messages = index bloat.
- Levels as a sampling decision: ERROR means "someone should page"; if it is not page-worthy, it is WARN. Default INFO in prod, DEBUG only for an active investigation. A service in DEBUG in prod is a volume leak.
- `trace_id` on every log line is non-negotiable: it is the bridge from metric (count rose) -> trace (which request) -> logs (what it said). Without it, you have three disconnected silos.
- Do not log and metric the same event: if you count it, emit a counter; if you need the detail, log it. Logging a line per request and then counting log matches is ~100x the cost of a counter.
- Volume target: 1-5 log lines per request in steady state, not 50. A service logging 1KB/req at 10k req/s = 10MB/s = ~800GB/day. Past that, the logs are doing metrics' or tracing's job.
- PII and secrets: never log tokens, passwords, full card numbers, raw PII. Put a redaction processor at the collector; pipeline redaction is cheaper than a breach and catches what code missed.
- Levels do not substitute for sampling: INFO at full volume is still too much. Use tail-based sampling at the trace layer to drop whole requests, keeping their logs only if their trace is kept.

## Traces: spans, propagation, tail sampling

- A trace is a tree of spans, one per service hop. A span carries `operation`, `start`/`duration`, `status`, `attributes`, `span_id`, `parent_span_id`, `trace_id`. The tree shape is the request's actual path.
- Propagation is the most common failure: W3C `traceparent` must be injected on send and extracted on receive at every boundary, HTTP, queue, RPC. A missing hop = a broken trace = two disconnected halves.
- Head-based sampling (decide at the root, before execution) is cheap and deterministic but drops before it knows the outcome: it throws away the errors. Use it only for low-volume or when tail sampling is infeasible.
- Tail-based sampling (decide after the trace completes) keeps 100% of errors and a sample of successes: keep `status != ok` at 100%, keep `latency > p99` at 100%, sample the rest at 1-10%. This is how you get full failure signal without full volume.
- Span attributes are the high-cardinality channel: `user_id`, `request_id`, `http.method`, `db.statement`, `http.status_code` go on spans, not metric labels. Sampling bounds the cardinality: a 5% sample makes a 1M-user label ~50k spans.
- Templated span names, not raw: `GET /users/{id}`, not `GET /users/12345`. Raw URLs explode span-name cardinality and break aggregation; the route template is what you group by.
- Do not over-span: a span per DB row in a loop = 1000 spans/req. Span the loop once with a `rows` count attribute, or sample the inner spans. Over-spanning is the trace equivalent of a memory leak.
- The critical path, not the longest span, governs latency: a 200ms span parallel to a 50ms span costs 200ms; a 50ms span in series with another 50ms costs 100ms. Read the tree's longest chain, not the biggest node.
- Tail sampling needs a collector holding the full trace, which adds latency and memory proportional to trace duration. Cap trace duration (drop spans > 30s) or the collector OOMs under load, exactly when you need it most.

## SLOs and error budgets

- SLI = a user-facing quality metric (e.g. % of requests with latency < 500ms and status < 500). SLO = the target for it (99.9% over 28 days). Error budget = 1 - SLO (0.1%): the allowed unreliability you can spend on risk.
- The SLI must be user-perceived: request latency and availability, not CPU or disk. CPU is a supporting metric for dashboards; making it an SLI optimizes the wrong thing.
- 28-day rolling window (not 30): aligns to weeks, smooths weekday/weekend variance, the SRE convention. A 30-day window shifts its day-of-week over time and drifts.
- Alert on burn rate, not raw SLI: a 28-day window hides a 1-hour outage (0.15% of the window). Burn rate = how fast you consume the budget; 14.4x burn over 1h = the whole budget in ~1h.
- Multi-window multi-burn-rate (Google SRE): page if burn > 14.4x over 1h AND 5m, or > 6x over 6h AND 30m; ticket if burn > 1x over 3d AND 6h. The AND of both windows cuts false positives; fast burn pages, slow burn tickets.
- Error budget policy: when the budget is exhausted, freeze feature work and fix reliability. Without the freeze, the SLO is theater; teams ship through a red budget and learn the SLO does not bind.
- Set the SLO from user need, not current performance: if users notice 500ms latency, the SLO is < 500ms even if the system does 200ms today. A comfortable SLO is a lie that hides degradation.
- One SLI per user journey, not per endpoint: "checkout succeeds in < 1s" beats 50 endpoint-level SLOs. The user does not care about endpoint N; they care about the journey.

## Alerting: symptoms, burn rate, fatigue

- Page on symptoms (user-visible), not causes: error rate and latency, not CPU or disk. High CPU that does not move latency is a dashboard tile, not a page.
- Saturation is the leading indicator: CPU > 85-90%, memory near limit, queue depth rising. Saturation alerts fire before latency does; they are the early warning, latency is the confirmation.
- Alert fatigue is the silent killer: every alert that fires without action erodes trust. Target < 1 page/person/shift, < 5 tickets/person/day. An alert firing > 1x/week with no runbook action is noise: delete or tune it.
- Every alert carries a runbook link: the on-call's first click. The runbook lists what to check, in order, and the resolution. An alert without a runbook is half an alert; the on-call improvises, slowly.
- AND vs OR in multi-window alerts: AND of two windows (both must burn) cuts false positives, use for pages; OR (either burns) increases them, use for tickets. Flipping them is a common misconfiguration.
- Alert on the SLO burn rate, not component health: the user-visible SLO is the page; the component dashboard is the investigation. Inverting this pages you on noise and misses real outages.
- Avoid alerting on raw counters: alert on `rate(errors[5m]) / rate(total[5m])` (a ratio), not on `errors` (which scales with traffic). A traffic spike doubles errors without any bug.

## The pipeline: OTel, collector, cost economics

- OpenTelemetry is the one standard to invest in: one SDK, one wire format (OTLP), backends plural. Instrument once with OTel, ship to Prometheus, Tempo, Jaeger, or Datadog. Vendor SDKs lock you to one bill.
- The collector is the transform layer between services and the backend: tail sampling, attribute dropping, redaction, metric aggregation, all before you pay for storage. Never send spans straight to a SaaS: you pay per byte and lose the transform.
- Collector placement: an agent per node (DaemonSet) collects; a gateway cluster does tail sampling and fan-out. Tail sampling at the gateway sees whole traces; at the agent you only see the node's spans.
- Retention by signal: metrics 3-6 months (downsampled after 1-2 weeks), logs 7-30 days (sampled), traces 1-7 days full + 30 days errors-only. Raw traces beyond a week are rarely re-queried; keep the errors, sample the rest.
- Auto-instrumentation (OTel SDKs, eBPF) reduces code burden but adds noise: it captures every HTTP span including health checks and metrics scrapes. Drop `/healthz`, `/readyz`, `/metrics` paths at the collector or they dominate trace volume.
- The quarterly audit, three lists: top 10 metrics by series count, top 10 log templates by volume, top 10 span names by count. Slow growth on any list is the cost killer; the spike after a deploy is the bug.
- Vendor vs self-hosted backend: vendor wins on time-to-value and query UX until ingest crosses ~1 TB/day; past that, self-hosted (Mimir + Tempo + Loki, or ClickHouse) wins on cost. The crossover is real; re-evaluate yearly as volume grows.

## Situations

| Situation | Play |
|---|---|
| p99 latency spiked, no obvious cause | Pull an exemplar `trace_id` from the latency histogram; open the trace; read the critical path, not the longest span. |
| Metrics storage bill jumped overnight | Audit cardinality via `/api/v1/status/tsdb`; a new high-card label multiplied across others. Roll back the label to a span attribute. |
| Traces break at one service boundary | Propagation: W3C `traceparent` not injected/extracted. Check the queue consumer or RPC client; the trace splits where context drops. |
| On-call pages nightly, no one acts | Alert fatigue: delete alerts firing > 1x/week without a runbook action; re-baseline on SLO burn, not CPU. |
| Debug one user's failed request | Do not add a metric label. Search logs by `trace_id` (on every line) -> open the trace -> read the spans. |
| Service logs ~800GB/day | Metric the per-request count and errors; log only errors and warnings; tail-sample the rest at 1-5%. |
| Dashboard query times out | Move it to a recording rule (pre-compute p99 per service every 30s); the dashboard reads the new metric. |
| Need all errors kept, sample successes | Tail-based sampling: keep `status != ok` at 100%, slow traces at 100%, sample the rest 1-10%, at the collector. |
| Cardinality spike after a deploy | A raw URL or `user_id` landed as a label. Strip it at the collector, move it to a span attribute. |
| SLI green but users complain | Wrong SLI: averaging latency, or measuring uptime not user-perceived speed. Re-derive from what users notice. |

## Where camps disagree

- **Metrics-first vs logs-first vs traces-first**: metrics-first (Prometheus culture) wins on cost and alerting; logs-first (ELK culture) wins on ad-hoc investigation and retroactive query; traces-first wins on distributed systems where the request crosses 10+ services. OTel convergence ships all three from one pipeline, but the budget still forces a primary signal: pick by failure mode, not by team habit.
- **Head vs tail sampling**: head is cheap, deterministic, stateless, and drops the failures you most want. Tail keeps errors and slow traces at 100%, costs a stateful collector that adds latency. Frontier: head for low-volume or greenfield; tail for any production where you debug by failure.
- **Averages vs percentiles for the SLI**: percentiles (p99) for user-perceived quality, the right SLI; averages for capacity and resource planning, where the mean models the load. Using average latency as an SLI hides the tail that users feel.
- **Vendor vs self-hosted backend**: vendor wins on time-to-value, query UX, and correlation until ingest crosses ~1 TB/day; self-hosted (Mimir/Tempo/Loki or ClickHouse) wins on cost past it. The crossover is volume-driven, not preference-driven; re-evaluate as ingest grows.
- **Strict SLO vs aspirational SLO**: strict (freeze feature work on budget exhaustion) in mature reliability cultures; aspirational (track but do not enforce) where the freeze has no political backing. A strict SLO without the freeze is the worst case: it looks like a contract and is not.
- **Structured JSON vs logfmt**: JSON for nested or complex events (a trace span as a log); logfmt for flat high-volume lines (a request log). Both are structured; the disagreement is parse cost and human readability, and both beat free text.
