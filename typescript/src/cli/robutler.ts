#!/usr/bin/env node
/**
 * Robutler CLI
 * 
 * Alias for `webagents connect` - starts interactive session with the default robutler agent.
 */

import { InteractiveREPL } from './app.js';

async function main() {
  const repl = new InteractiveREPL({ 
    model: 'default',
    agentName: 'robutler'
  });
  await repl.run();
}

main().catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});
