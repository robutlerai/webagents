#!/usr/bin/env node
/**
 * `webagents-executor` CLI entrypoint.
 *
 * Re-exports `cli()` from the SDK's executor module so the binary picks
 * up runtime upgrades automatically.
 */

import { cli } from 'webagents/executor';

cli().catch((e) => {
  console.error('[webagents-executor] fatal:', e);
  process.exit(1);
});
