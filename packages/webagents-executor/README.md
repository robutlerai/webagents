# webagents-executor

Local function-execution daemon for the WebAgents SDK.

Implements the same `/invoke` + `/validate` HTTP protocol as the cloud-deployed `function-executor` pod, so a single `npm install -g webagents-executor` is enough to run agent functions locally with the same isolation guarantees.

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

- `js-v1` — V8 isolate via `isolated-vm` (always enabled).
- `python-pyodide-v1` — CPython on WebAssembly (enabled if `pyodide` optional dep is present).
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
| `EXECUTOR_CPU_PRESSURE_PCT`  | 85      | Admission gate threshold                 |
| `WEBAGENTS_EXECUTOR_DEV`     | -       | `1` enables dev flags (same as `--dev`)  |
