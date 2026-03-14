---
title: CLI Test Coverage Assessment
---

# CLI Test Coverage Assessment

## Python CLI

### Tested

| Command | Test File | Notes |
|---------|-----------|-------|
| `--help` | `tests/cli/test_commands.py` | |
| `version` (import) | `tests/cli/test_commands.py` | Checks `__version__`, not `webagents version` |
| `list` | `tests/cli/test_commands.py` | Basic |
| `init` (4 variants) | `tests/cli/test_commands.py` | Default, named, context, already-exists |
| `skill list` | `tests/cli/test_commands.py` | |
| `daemon status` | `tests/cli/test_commands.py` | |
| `auth whoami` | `tests/cli/test_commands.py` | Not-logged-in only |
| `config --help` | `tests/cli/test_commands.py` | Help only |
| `config sandbox --help` | `tests/cli/test_commands.py` | Help only |
| `template use` | `tests/cli/test_template.py` | Bundled + cached |
| `template pull` | `tests/cli/test_template.py` | Mocked |
| REPL basic commands | `tests/cli/test_repl.py` | Registry, session, token stats |
| `DaemonClient.is_running` | `tests/cli/client/test_daemon_client.py` | |
| `DaemonClient.list_agents` | `tests/cli/client/test_daemon_client.py` | |
| `DaemonClient.register_agent` | `tests/cli/client/test_daemon_client.py` | |
| `DaemonClient.get_agent` | `tests/cli/client/test_daemon_client.py` | |

### Gaps (priority order)

1. **`webagents serve`** -- Start server and handle requests. Needs: start in background, send a request, verify response, shut down.
2. **`webagents run`** -- Single-turn agent execution. Needs: mock LLM, verify output.
3. **`webagents connect`** -- Connect to a running agent. Needs: mock daemon.
4. **`auth login` / `auth logout`** -- Auth flow. Needs: mock OAuth redirect or API key input.
5. **`agent` subcommands** -- `run`, `stop`, `restart`, `logs`, `debug`, `info`. Needs: mock daemon.
6. **`session` subcommands** -- `new`, `history`, `save`, `load`, `list`, `clear`. Needs: mock session manager.
7. **`config` subcommands** -- `get`, `set`, `edit`, `reset`. Needs: temp config dir.
8. **REPL slash command execution** -- Only registry presence is tested; actual execution not tested.
9. **`DaemonClient.chat` / `chat_stream`** -- Needs: mock server.
10. **`discover`, `register`, `namespace`, `cron`** -- Various subcommands with no tests.

---

## TypeScript CLI

### Tested

| Command | Test File | Notes |
|---------|-----------|-------|
| `--help` | `tests/e2e/cli.test.ts` | Gated behind `RUN_E2E=true` |
| `--version` | `tests/e2e/cli.test.ts` | Gated behind `RUN_E2E=true` |
| `info` | `tests/e2e/cli.test.ts` | Gated behind `RUN_E2E=true` |
| `models` | `tests/e2e/cli.test.ts` | Gated behind `RUN_E2E=true` |
| REPL slash parsing | `tests/e2e/cli.test.ts` | Only `/help` and `/model` parsing |

### Infrastructure issue

`cli.test.ts` uses Vitest but lives in `tests/e2e/` which Vitest excludes. Playwright only runs `**/*.spec.ts` files. **These tests are effectively dead** -- they don't run in either test runner during normal CI.

**Fix:** Either rename to `cli.spec.ts` (for Playwright) or move to `tests/unit/cli/` (for Vitest).

### Gaps (priority order)

1. **Test runner fix** -- Make the 4 existing tests actually run in CI.
2. **`webagents serve`** -- Start server, verify health endpoint, shut down.
3. **`webagents chat`** (default command) -- Interactive REPL with mock agent.
4. **`webagents connect`** -- Connect to remote agent.
5. **`webagents login` / `logout`** -- Auth flow with mock.
6. **`webagents daemon`** -- Lifecycle commands.
7. **`webagents init`** -- Project scaffolding.
8. **`webagents discover`** -- Agent discovery.
9. **`webagents config`** -- get, set, path.
10. **REPL execution** -- `/chat`, `/model`, `/history`, `/clear`, `/save`, `/load`, `/tools`, `/exit`.

---

## Recommendations

1. **Immediate**: Fix the TS `cli.test.ts` runner mismatch so existing tests actually execute.
2. **High value**: Add `serve` command tests for both SDKs -- this is the most user-facing command.
3. **Medium**: Add `login` mock tests (both SDKs) and `run` tests (Python).
4. **Lower priority**: Session/config/daemon subcommands can be backfilled incrementally.
