#!/usr/bin/env node
/**
 * WebAgents CLI
 *
 * Full command-line interface for WebAgents (17 command groups).
 */

import { Command } from 'commander';
import { InteractiveREPL } from './app.js';
import { WebAgentsDaemon } from '../daemon/server.js';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

const version = '0.1.0';
const CONFIG_DIR = path.join(os.homedir(), '.webagents');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');
const AUTH_FILE = path.join(CONFIG_DIR, 'auth.json');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function loadConfig(): Record<string, unknown> {
  try {
    return JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

function saveConfig(config: Record<string, unknown>): void {
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
}

function loadAuth(): { token?: string; portalUrl?: string; userId?: string } {
  try {
    return JSON.parse(fs.readFileSync(AUTH_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

function saveAuth(data: Record<string, unknown>): void {
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(AUTH_FILE, JSON.stringify(data, null, 2));
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

const program = new Command();

program
  .name('webagents')
  .description('TypeScript SDK for building and running AI agents')
  .version(version);

// ============================================================================
// 1. chat (default)
// ============================================================================

async function chatAction(options: { model?: string; prompt?: string; outputFormat?: string; agent?: string }) {
  if (options.prompt) {
    const repl = new InteractiveREPL({ model: options.model || 'default', agentName: options.agent });
    await repl.initialize();
    const response = await repl.sendMessage(options.prompt);
    if (options.outputFormat === 'json') {
      console.log(JSON.stringify(response, null, 2));
    } else {
      console.log(response.content);
    }
    process.exit(0);
  } else {
    const repl = new InteractiveREPL({ model: options.model || 'default', agentName: options.agent });
    await repl.run();
  }
}

program
  .command('chat', { isDefault: true })
  .description('Start interactive chat session')
  .option('-m, --model <model>', 'Model to use', 'default')
  .option('-a, --agent <agent>', 'Agent name')
  .option('-p, --prompt <prompt>', 'Non-interactive prompt')
  .option('--output-format <format>', 'Output: text, json, stream-json', 'text')
  .option('--no-streaming', 'Disable streaming')
  .action(chatAction);

// ============================================================================
// 2. connect
// ============================================================================

program
  .command('connect')
  .description('Start interactive session (alias for chat)')
  .option('-m, --model <model>', 'Model to use', 'default')
  .option('-a, --agent <agent>', 'Agent name')
  .option('-p, --prompt <prompt>', 'Non-interactive prompt')
  .option('--output-format <format>', 'Output format', 'text')
  .action(chatAction);

// ============================================================================
// 3. serve
// ============================================================================

program
  .command('serve')
  .description('Serve an agent on HTTP')
  .argument('[path]', 'Path to agent config file', '.')
  .option('-p, --port <port>', 'Port', '3000')
  .option('-h, --host <host>', 'Host', '0.0.0.0')
  .option('--multi', 'Multi-agent mode (load all agents in directory)')
  .action(async (agentPath, options) => {
    const { serve } = await import('../server/node.js');
    const { BaseAgent } = await import('../core/agent.js');

    const configPath = path.resolve(agentPath);
    let agentConfig: Record<string, unknown> = { name: 'agent' };

    try {
      const raw = fs.readFileSync(
        fs.statSync(configPath).isDirectory()
          ? path.join(configPath, 'agent.json')
          : configPath,
        'utf-8',
      );
      agentConfig = JSON.parse(raw);
    } catch {
      console.log('No agent.json found, serving default agent');
    }

    const agent = new BaseAgent({
      name: (agentConfig.name as string) ?? 'agent',
      description: (agentConfig.description as string),
      instructions: (agentConfig.instructions as string),
    });
    await agent.initialize();

    await serve(agent, { port: parseInt(options.port, 10), hostname: options.host });
  });

// ============================================================================
// 4. daemon
// ============================================================================

program
  .command('daemon')
  .description('Start the WebAgents daemon')
  .option('-p, --port <port>', 'Port', '8080')
  .option('-h, --host <host>', 'Host', '0.0.0.0')
  .option('-w, --watch <dir>', 'Watch directory')
  .option('--no-cron', 'Disable cron')
  .action(async (options) => {
    const daemon = new WebAgentsDaemon({
      port: parseInt(options.port, 10),
      hostname: options.host,
      watchDir: options.watch,
      cron: options.cron,
    });
    await daemon.start();
    process.on('SIGINT', () => { daemon.stop(); process.exit(0); });
  });

// ============================================================================
// 5. login / logout
// ============================================================================

program
  .command('login')
  .description('Authenticate with the portal')
  .option('-u, --url <url>', 'Portal URL', 'https://robutler.ai')
  .option('-t, --token <token>', 'API token (or use interactive flow)')
  .action(async (options) => {
    if (options.token) {
      saveAuth({ token: options.token, portalUrl: options.url });
      console.log('Authenticated successfully.');
    } else {
      const portalUrl = options.url;
      console.log(`\nVisit: ${portalUrl}/settings/api-keys to get your API key.\n`);
      const readline = await import('readline');
      const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
      rl.question('Paste your API key: ', (token) => {
        saveAuth({ token: token.trim(), portalUrl });
        console.log('Authenticated successfully.');
        rl.close();
      });
    }
  });

program
  .command('logout')
  .description('Remove stored credentials')
  .action(() => {
    try { fs.unlinkSync(AUTH_FILE); } catch {}
    console.log('Logged out.');
  });

// ============================================================================
// 6. sync
// ============================================================================

program
  .command('sync')
  .description('Sync agents with the portal')
  .option('--push', 'Push local agents to portal')
  .option('--pull', 'Pull agents from portal')
  .action(async (options) => {
    const auth = loadAuth();
    if (!auth.token) { console.error('Not logged in. Run `webagents login` first.'); process.exit(1); }

    const direction = options.push ? 'push' : options.pull ? 'pull' : 'pull';
    console.log(`Syncing agents (${direction}) with ${auth.portalUrl}...`);

    const res = await fetch(`${auth.portalUrl}/api/agents/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${auth.token}` },
      body: JSON.stringify({ direction }),
    });

    if (!res.ok) { console.error(`Sync failed: ${res.status}`); process.exit(1); }
    const data = await res.json();
    console.log(`Synced: ${JSON.stringify(data, null, 2)}`);
  });

// ============================================================================
// 7. discover
// ============================================================================

program
  .command('discover')
  .description('Search for agents on the platform')
  .argument('<query>', 'Search query')
  .option('-n, --limit <n>', 'Max results', '10')
  .action(async (query, options) => {
    const auth = loadAuth();
    const portalUrl = auth.portalUrl ?? 'https://robutler.ai';

    const res = await fetch(`${portalUrl}/api/intents/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(auth.token ? { Authorization: `Bearer ${auth.token}` } : {}) },
      body: JSON.stringify({ intent: query, top_k: parseInt(options.limit, 10) }),
    });

    if (!res.ok) { console.error(`Discovery failed: ${res.status}`); process.exit(1); }
    const data = await res.json() as { results: Array<{ name: string; agentUrl: string; description: string; score: number }> };

    console.log('\nDiscovered Agents:\n');
    for (const r of data.results ?? []) {
      console.log(`  ${r.name.padEnd(25)} ${r.score.toFixed(2)}  ${r.description?.slice(0, 60) ?? ''}`);
      console.log(`  ${' '.repeat(25)} ${r.agentUrl}`);
    }
    console.log();
  });

// ============================================================================
// 8. info
// ============================================================================

program
  .command('info')
  .description('Show agent information')
  .option('-u, --url <url>', 'Agent URL')
  .action(async (options) => {
    if (options.url) {
      try {
        const response = await fetch(`${options.url}/info`);
        console.log(JSON.stringify(await response.json(), null, 2));
      } catch (error) {
        console.error('Failed:', (error as Error).message);
        process.exit(1);
      }
    } else {
      console.log(JSON.stringify({ name: 'webagents-cli', version }, null, 2));
    }
  });

// ============================================================================
// 9. models
// ============================================================================

program
  .command('models')
  .description('List available models')
  .action(() => {
    const models = [
      { id: 'webllm', provider: 'webllm', description: 'In-browser LLM via WebGPU' },
      { id: 'transformers', provider: 'transformers.js', description: 'In-browser via Transformers.js' },
      { id: 'gpt-4o', provider: 'openai', description: 'OpenAI GPT-4o' },
      { id: 'claude-3-5-sonnet', provider: 'anthropic', description: 'Anthropic Claude 3.5 Sonnet' },
      { id: 'gemini-1.5-pro', provider: 'google', description: 'Google Gemini 1.5 Pro' },
      { id: 'grok-2', provider: 'xai', description: 'xAI Grok 2' },
    ];
    console.log('\nAvailable Models:\n');
    for (const m of models) {
      console.log(`  ${m.id.padEnd(22)} ${m.provider.padEnd(16)} ${m.description}`);
    }
    console.log();
  });

// ============================================================================
// 10. skills
// ============================================================================

const skillsCmd = program.command('skills').description('Manage skills');

skillsCmd
  .command('list')
  .description('List all available built-in skills')
  .action(() => {
    const skills = [
      'openai', 'anthropic', 'google', 'xai', 'webllm', 'transformers',
      'completions-transport', 'portal-transport', 'uamp-transport', 'realtime-transport', 'a2a-transport', 'acp-transport',
      'browser', 'speech', 'nli', 'discovery', 'testrunner',
      'auth', 'payments', 'filesystem', 'shell', 'mcp', 'routing',
      'storage-kv', 'storage-json', 'storage-files',
      'session', 'checkpoint', 'todo', 'rag', 'sandbox', 'plugin',
      'chats', 'notifications', 'publish', 'portal-connect', 'portal-ws',
    ];
    console.log('\nAvailable Skills:\n');
    for (const s of skills) console.log(`  ${s}`);
    console.log(`\nTotal: ${skills.length} skills\n`);
  });

skillsCmd
  .command('info <name>')
  .description('Show skill details')
  .action((name) => {
    console.log(`Skill: ${name}\n`);
    console.log('Use in code:\n');
    console.log(`  import { ${name.split('-').map((p: string) => p.charAt(0).toUpperCase() + p.slice(1)).join('')}Skill } from 'webagents';\n`);
  });

// ============================================================================
// 11. templates
// ============================================================================

const templatesCmd = program.command('templates').description('Agent templates');

templatesCmd
  .command('list')
  .description('List available templates')
  .action(() => {
    const templates = [
      { name: 'chatbot', description: 'Simple chatbot with LLM' },
      { name: 'tool-agent', description: 'Agent with custom tools' },
      { name: 'rag-agent', description: 'Retrieval-augmented generation agent' },
      { name: 'multi-agent', description: 'Multi-agent system with routing' },
      { name: 'browser-agent', description: 'In-browser agent with WebLLM' },
      { name: 'mcp-agent', description: 'Agent with MCP tool integration' },
    ];
    console.log('\nAvailable Templates:\n');
    for (const t of templates) {
      console.log(`  ${t.name.padEnd(20)} ${t.description}`);
    }
    console.log('\nUse: webagents init --template <name>\n');
  });

// ============================================================================
// 12. config
// ============================================================================

const configCmd = program.command('config').description('Manage configuration');

configCmd
  .command('get [key]')
  .description('Get configuration value')
  .action((key) => {
    const config = loadConfig();
    if (key) {
      console.log(config[key] ?? '(not set)');
    } else {
      console.log(JSON.stringify(config, null, 2));
    }
  });

configCmd
  .command('set <key> <value>')
  .description('Set configuration value')
  .action((key, value) => {
    const config = loadConfig();
    config[key] = value;
    saveConfig(config);
    console.log(`Set ${key} = ${value}`);
  });

configCmd
  .command('path')
  .description('Show config file path')
  .action(() => console.log(CONFIG_FILE));

// ============================================================================
// 13. init
// ============================================================================

program
  .command('init')
  .description('Initialize a new agent project')
  .argument('[name]', 'Project name', 'my-agent')
  .option('-t, --template <template>', 'Template to use', 'chatbot')
  .action(async (name, options) => {
    const dir = path.resolve(name);
    if (fs.existsSync(dir)) {
      console.error(`Directory ${name} already exists.`);
      process.exit(1);
    }

    fs.mkdirSync(dir, { recursive: true });

    const agentJson = {
      name,
      description: `A ${options.template} agent`,
      template: options.template,
      model: 'gpt-4o',
      skills: options.template === 'chatbot' ? ['openai'] : ['openai', 'filesystem', 'shell'],
    };

    fs.writeFileSync(path.join(dir, 'agent.json'), JSON.stringify(agentJson, null, 2));

    fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({
      name,
      version: '0.1.0',
      type: 'module',
      scripts: { start: 'webagents serve', dev: 'webagents serve --port 3000' },
      dependencies: { webagents: `^${version}` },
    }, null, 2));

    fs.writeFileSync(path.join(dir, 'instructions.md'), `# ${name}\n\nYou are a helpful assistant.\n`);

    console.log(`\nCreated agent project: ${name}/`);
    console.log(`  agent.json`);
    console.log(`  package.json`);
    console.log(`  instructions.md`);
    console.log(`\nNext steps:`);
    console.log(`  cd ${name}`);
    console.log(`  npm install`);
    console.log(`  webagents serve\n`);
  });

// ============================================================================
// 14. publish
// ============================================================================

program
  .command('publish')
  .description('Publish agent to the portal')
  .argument('[path]', 'Path to agent config', '.')
  .action(async (agentPath) => {
    const auth = loadAuth();
    if (!auth.token) { console.error('Not logged in.'); process.exit(1); }

    const configPath = path.resolve(agentPath);
    let agentConfig: Record<string, unknown>;
    try {
      const raw = fs.readFileSync(
        fs.statSync(configPath).isDirectory()
          ? path.join(configPath, 'agent.json')
          : configPath,
        'utf-8',
      );
      agentConfig = JSON.parse(raw);
    } catch {
      console.error('Could not read agent.json');
      process.exit(1);
    }

    const res = await fetch(`${auth.portalUrl}/api/agents/publish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${auth.token}` },
      body: JSON.stringify(agentConfig),
    });

    if (!res.ok) { console.error(`Publish failed: ${res.status}`); process.exit(1); }
    const data = await res.json();
    console.log(`Published: ${JSON.stringify(data, null, 2)}`);
  });

// ============================================================================
// 15. test
// ============================================================================

program
  .command('test')
  .description('Run agent tests')
  .argument('[path]', 'Test file or directory', 'tests/')
  .option('--timeout <ms>', 'Test timeout', '30000')
  .action(async (testPath, options) => {
    console.log(`Running tests from ${testPath} (timeout: ${options.timeout}ms)...`);
    const { execSync } = await import('child_process');
    try {
      execSync(`npx vitest run ${testPath} --timeout ${options.timeout}`, { stdio: 'inherit' });
    } catch {
      process.exit(1);
    }
  });

// ============================================================================
// 16. logs
// ============================================================================

program
  .command('logs')
  .description('View agent logs')
  .option('-a, --agent <name>', 'Agent name')
  .option('-f, --follow', 'Follow log output')
  .option('-n, --lines <n>', 'Number of lines', '50')
  .action(async (options) => {
    const auth = loadAuth();
    const portalUrl = auth.portalUrl ?? 'https://robutler.ai';
    const qs = new URLSearchParams({ lines: options.lines });
    if (options.agent) qs.set('agent', options.agent);

    if (options.follow) {
      console.log('Streaming logs (Ctrl+C to stop)...\n');
      const res = await fetch(`${portalUrl}/api/logs/stream?${qs}`, {
        headers: auth.token ? { Authorization: `Bearer ${auth.token}` } : {},
      });
      if (!res.ok || !res.body) { console.error('Failed to stream logs'); process.exit(1); }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          process.stdout.write(decoder.decode(value, { stream: true }));
        }
      } catch {
        // Stream closed
      }
    } else {
      const res = await fetch(`${portalUrl}/api/logs?${qs}`, {
        headers: auth.token ? { Authorization: `Bearer ${auth.token}` } : {},
      });
      if (!res.ok) { console.error('Failed to fetch logs'); process.exit(1); }
      const data = await res.json() as { logs?: string[] };
      for (const line of data.logs ?? []) console.log(line);
    }
  });

// ============================================================================
// 17. version / update
// ============================================================================

program
  .command('update')
  .description('Check for updates')
  .action(async () => {
    try {
      const res = await fetch('https://registry.npmjs.org/webagents/latest');
      const data = await res.json() as { version?: string };
      const latest = data.version ?? 'unknown';
      if (latest === version) {
        console.log(`You're up to date! (v${version})`);
      } else {
        console.log(`Update available: v${version} → v${latest}`);
        console.log(`Run: npm install -g webagents@latest`);
      }
    } catch {
      console.log(`Current version: v${version}`);
      console.log('Could not check for updates.');
    }
  });

program.parse();
