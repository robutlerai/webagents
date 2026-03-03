/**
 * Interactive REPL
 * 
 * Interactive chat session with slash commands.
 * Based on Gemini CLI architecture.
 */

import * as readline from 'readline';
import { BaseAgent } from '../core/agent.js';
import { OpenAISkill } from '../skills/llm/openai/skill.js';
import type { RunResponse } from '../core/types.js';
import type { Message } from '../uamp/types.js';

/**
 * REPL configuration
 */
export interface REPLConfig {
  /** Model to use */
  model?: string;
  /** System instructions */
  instructions?: string;
  /** Enable streaming */
  streaming?: boolean;
  /** Agent name to connect to (defaults to 'robutler') */
  agentName?: string;
}

/**
 * Slash command handler
 */
interface SlashCommand {
  name: string;
  description: string;
  handler: (args: string) => Promise<void>;
}

/**
 * Interactive REPL for agent conversations
 */
export class InteractiveREPL {
  private config: REPLConfig;
  private agent: BaseAgent | null = null;
  private messages: Message[] = [];
  private commands: Map<string, SlashCommand> = new Map();
  private rl: readline.Interface | null = null;
  private running = false;
  
  constructor(config: REPLConfig = {}) {
    this.config = {
      model: 'gpt-4o',
      streaming: true,
      ...config,
    };
    
    this.setupCommands();
  }
  
  /**
   * Set up slash commands
   */
  private setupCommands(): void {
    this.registerCommand({
      name: 'help',
      description: 'Show available commands',
      handler: async () => {
        console.log('\nAvailable commands:\n');
        for (const cmd of this.commands.values()) {
          console.log(`  /${cmd.name.padEnd(15)} ${cmd.description}`);
        }
        console.log();
      },
    });
    
    this.registerCommand({
      name: 'chat',
      description: 'Start a new conversation',
      handler: async () => {
        this.messages = [];
        console.log('\nStarted new conversation.\n');
      },
    });
    
    this.registerCommand({
      name: 'model',
      description: 'Change or show current model',
      handler: async (args) => {
        if (args) {
          this.config.model = args.trim();
          console.log(`\nModel changed to: ${this.config.model}\n`);
        } else {
          console.log(`\nCurrent model: ${this.config.model}\n`);
        }
      },
    });
    
    this.registerCommand({
      name: 'history',
      description: 'Show conversation history',
      handler: async () => {
        if (this.messages.length === 0) {
          console.log('\nNo messages in history.\n');
          return;
        }
        
        console.log('\nConversation history:\n');
        for (const msg of this.messages) {
          const role = msg.role.toUpperCase();
          const content = msg.content?.slice(0, 100) || '';
          console.log(`  [${role}] ${content}${content.length >= 100 ? '...' : ''}`);
        }
        console.log();
      },
    });
    
    this.registerCommand({
      name: 'clear',
      description: 'Clear the screen',
      handler: async () => {
        console.clear();
      },
    });
    
    this.registerCommand({
      name: 'save',
      description: 'Save conversation to file',
      handler: async (args) => {
        const filename = args.trim() || `conversation_${Date.now()}.json`;
        const fs = await import('fs');
        fs.writeFileSync(filename, JSON.stringify(this.messages, null, 2));
        console.log(`\nConversation saved to: ${filename}\n`);
      },
    });
    
    this.registerCommand({
      name: 'load',
      description: 'Load conversation from file',
      handler: async (args) => {
        if (!args.trim()) {
          console.log('\nUsage: /load <filename>\n');
          return;
        }
        
        const fs = await import('fs');
        try {
          const content = fs.readFileSync(args.trim(), 'utf-8');
          this.messages = JSON.parse(content);
          console.log(`\nLoaded ${this.messages.length} messages from: ${args.trim()}\n`);
        } catch (error) {
          console.error(`\nFailed to load file: ${(error as Error).message}\n`);
        }
      },
    });
    
    this.registerCommand({
      name: 'exit',
      description: 'Exit the REPL',
      handler: async () => {
        this.running = false;
        console.log('\nGoodbye!\n');
      },
    });
    
    this.registerCommand({
      name: 'tools',
      description: 'List available tools',
      handler: async () => {
        if (!this.agent) {
          console.log('\nNo agent initialized.\n');
          return;
        }
        
        const tools = (this.agent as unknown as { toolRegistry: Map<string, { name: string; description?: string }> }).toolRegistry;
        if (!tools || tools.size === 0) {
          console.log('\nNo tools available.\n');
          return;
        }
        
        console.log('\nAvailable tools:\n');
        for (const tool of tools.values()) {
          console.log(`  ${tool.name.padEnd(20)} ${tool.description || ''}`);
        }
        console.log();
      },
    });
  }
  
  /**
   * Register a slash command
   */
  registerCommand(command: SlashCommand): void {
    this.commands.set(command.name, command);
  }
  
  /**
   * Initialize the agent
   */
  async initialize(): Promise<void> {
    // Load agent configuration
    let agentName = this.config.agentName || 'robutler';
    let instructions = this.config.instructions;
    
    // If using robutler or no custom instructions, load from embedded ROBUTLER.md
    if (agentName === 'robutler' || !instructions) {
      try {
        const { getRobutlerContent, parseAgentMarkdown } = await import('../agents/index.js');
        const content = getRobutlerContent();
        const parsed = parseAgentMarkdown(content);
        
        if (!instructions) {
          instructions = parsed.instructions;
        }
        if (agentName === 'robutler') {
          agentName = parsed.name;
        }
      } catch (error) {
        // Fallback if embedded agent not available
        console.warn('Could not load embedded robutler agent:', (error as Error).message);
      }
    }
    
    // Create agent with OpenAI skill (can be extended to support other providers)
    const skill = new OpenAISkill({ model: this.config.model });
    await skill.initialize();
    
    this.agent = new BaseAgent({
      name: agentName,
      instructions,
      skills: [skill],
    });
    
    await this.agent.initialize();
  }
  
  /**
   * Send a message and get response
   */
  async sendMessage(content: string): Promise<RunResponse> {
    if (!this.agent) {
      throw new Error('Agent not initialized');
    }
    
    // Add user message
    this.messages.push({ role: 'user', content });
    
    // Get response
    const response = await this.agent.run(this.messages);
    
    // Add assistant message
    this.messages.push({ role: 'assistant', content: response.content });
    
    return response;
  }
  
  /**
   * Send message with streaming
   */
  async *sendMessageStreaming(content: string): AsyncGenerator<string, RunResponse, unknown> {
    if (!this.agent) {
      throw new Error('Agent not initialized');
    }
    
    // Add user message
    this.messages.push({ role: 'user', content });
    
    let fullContent = '';
    let response: RunResponse | undefined;
    
    for await (const chunk of this.agent.runStreaming(this.messages)) {
      if (chunk.type === 'delta' && chunk.delta) {
        fullContent += chunk.delta;
        yield chunk.delta;
      } else if (chunk.type === 'done' && chunk.response) {
        response = chunk.response;
      }
    }
    
    // Add assistant message
    this.messages.push({ role: 'assistant', content: fullContent });
    
    return response || { content: fullContent };
  }
  
  /**
   * Handle input line
   */
  private async handleInput(line: string): Promise<void> {
    const trimmed = line.trim();
    
    if (!trimmed) {
      return;
    }
    
    // Check for slash command
    if (trimmed.startsWith('/')) {
      const parts = trimmed.slice(1).split(/\s+/);
      const cmdName = parts[0].toLowerCase();
      const args = parts.slice(1).join(' ');
      
      const command = this.commands.get(cmdName);
      if (command) {
        await command.handler(args);
      } else {
        console.log(`\nUnknown command: /${cmdName}. Type /help for available commands.\n`);
      }
      return;
    }
    
    // Send message to agent
    try {
      if (this.config.streaming) {
        process.stdout.write('\nAssistant: ');
        
        for await (const delta of this.sendMessageStreaming(trimmed)) {
          process.stdout.write(delta);
        }
        
        console.log('\n');
      } else {
        console.log('\nThinking...');
        const response = await this.sendMessage(trimmed);
        console.log(`\nAssistant: ${response.content}\n`);
      }
    } catch (error) {
      console.error(`\nError: ${(error as Error).message}\n`);
    }
  }
  
  /**
   * Run the interactive REPL
   */
  async run(): Promise<void> {
    await this.initialize();
    
    const agentName = this.agent?.name || 'cli-agent';
    console.log(`\nWebAgents CLI - Connected to ${agentName}`);
    console.log('Type /help for available commands, or start chatting.\n');
    
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    
    this.running = true;
    
    const prompt = () => {
      if (!this.running) {
        this.rl?.close();
        return;
      }
      
      this.rl?.question('You: ', async (line) => {
        await this.handleInput(line);
        prompt();
      });
    };
    
    prompt();
    
    // Handle Ctrl+C
    this.rl.on('close', () => {
      this.running = false;
      console.log('\n');
    });
  }
}
