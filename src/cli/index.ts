#!/usr/bin/env node
/**
 * WebAgents CLI
 * 
 * Command-line interface for WebAgents.
 * Based on Gemini CLI architecture.
 */

import { Command } from 'commander';
import { InteractiveREPL } from './app.js';
import { WebAgentsDaemon } from '../daemon/server.js';

const version = '0.1.0';

const program = new Command();

program
  .name('webagents')
  .description('TypeScript SDK for in-browser AI agents')
  .version(version);

// Interactive mode (default)
program
  .command('chat', { isDefault: true })
  .description('Start interactive chat session')
  .option('-m, --model <model>', 'Model to use', 'default')
  .option('-p, --prompt <prompt>', 'Initial prompt (non-interactive)')
  .option('--output-format <format>', 'Output format: text, json, stream-json', 'text')
  .option('--no-streaming', 'Disable streaming output')
  .action(async (options) => {
    if (options.prompt) {
      // Non-interactive mode
      const repl = new InteractiveREPL({ model: options.model });
      await repl.initialize();
      
      const response = await repl.sendMessage(options.prompt);
      
      if (options.outputFormat === 'json') {
        console.log(JSON.stringify(response, null, 2));
      } else {
        console.log(response.content);
      }
      
      process.exit(0);
    } else {
      // Interactive mode
      const repl = new InteractiveREPL({ model: options.model });
      await repl.run();
    }
  });

// Daemon command
program
  .command('daemon')
  .description('Start the WebAgents daemon')
  .option('-p, --port <port>', 'Port to listen on', '8080')
  .option('-h, --host <host>', 'Hostname to bind to', '0.0.0.0')
  .option('-w, --watch <dir>', 'Directory to watch for agent files')
  .option('--no-cron', 'Disable cron scheduler')
  .action(async (options) => {
    const daemon = new WebAgentsDaemon({
      port: parseInt(options.port, 10),
      hostname: options.host,
      watchDir: options.watch,
      cron: options.cron,
    });
    
    await daemon.start();
    
    // Handle shutdown
    process.on('SIGINT', () => {
      daemon.stop();
      process.exit(0);
    });
  });

// Agent info command
program
  .command('info')
  .description('Show agent information')
  .option('-u, --url <url>', 'Agent URL')
  .action(async (options) => {
    if (options.url) {
      try {
        const response = await fetch(`${options.url}/info`);
        const info = await response.json();
        console.log(JSON.stringify(info, null, 2));
      } catch (error) {
        console.error('Failed to fetch agent info:', (error as Error).message);
        process.exit(1);
      }
    } else {
      console.log({
        name: 'webagents-cli',
        version,
        description: 'TypeScript SDK for in-browser AI agents',
      });
    }
  });

// List models command
program
  .command('models')
  .description('List available models')
  .action(() => {
    const models = [
      { id: 'webllm', provider: 'webllm', description: 'In-browser LLM via WebGPU' },
      { id: 'transformers', provider: 'transformers.js', description: 'In-browser LLM via Transformers.js' },
      { id: 'gpt-4o', provider: 'openai', description: 'OpenAI GPT-4o' },
      { id: 'claude-3-5-sonnet', provider: 'anthropic', description: 'Anthropic Claude 3.5 Sonnet' },
      { id: 'gemini-1.5-pro', provider: 'google', description: 'Google Gemini 1.5 Pro' },
      { id: 'grok-2', provider: 'xai', description: 'xAI Grok 2' },
    ];
    
    console.log('\nAvailable Models:\n');
    for (const model of models) {
      console.log(`  ${model.id.padEnd(20)} ${model.provider.padEnd(15)} ${model.description}`);
    }
    console.log();
  });

program.parse();
