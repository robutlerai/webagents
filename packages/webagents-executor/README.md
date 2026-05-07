# webagents-executor

Standalone function-execution daemon for the WebAgents SDK, intended for
SDK consumers running outside our Kubernetes cluster (e.g. embedded in a
host process, a one-off laptop demo, or a third-party deployment).

Implements the same `/invoke` + `/validate` HTTP protocol as the cloud-deployed `function-executor` pod, so a single `npm install -g webagents-executor` is enough to run agent functions locally with the same isolation guarantees.

> **Robutler-internal note:** `./admin.sh local up` deploys the
> function-executor into the local Kubernetes cluster (via
> `infrastructure/applications/function-executor/local`) — mirroring the
> cloud topology. You only need this npm package if you're running the
> SDK *outside* that flow.

## Install

```bash
npm install -g webagents-executor
```

## Run

```bash
webagents-executor --port 7070 --dev
```

Then point your local agent at the daemon:

```bash
export WEBAGENTS_EXECUTOR_URL=http://127.0.0.1:7070
webagents serve ./my-agent
```

## Runtimes

- `js-v1` — V8 isolate via `isolated-vm` (always enabled in v1).
- `python-pyodide-v1` — deferred per ADR-0008. Manifests pinning this runtime fail validation with `RUNTIME_DISABLED`.
- `wasm-v1` — reserved slot, ships disabled.

## Dev mode

`--dev` enables:

- `file://` codeRefs (point at local source).
- Local-kind folder bindings (read/write to your filesystem).
- Larger default limits.

Never enable `--dev` on a shared host.

## Configuration

| Env                          | Default | Notes                                    |
| ---                          | ---     | ---                                      |
| `PORT`                       | 7070    | HTTP port                                |
| `EXECUTOR_OVERSUBSCRIBE`     | 8       | Workers = oversubscribe × CPU count      |
| `EXECUTOR_CPU_PRESSURE_PCT`  | 85      | Admission gate threshold (% of CPU loadavg). Set to `0` to disable — recommended for containerized dev where `loadavg` reflects the host, not the container cgroup. |
| `WEBAGENTS_EXECUTOR_DEV`     | -       | `1` enables dev flags (same as `--dev`)  |
