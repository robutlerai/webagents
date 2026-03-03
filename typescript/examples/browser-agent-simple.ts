/**
 * Simple Browser Agent Example
 * 
 * Demonstrates a basic browser-based agent using WebLLM
 * for text generation with the webagents-ts SDK.
 * 
 * Run in browser via the example HTML page or bundler.
 */

import { BaseAgent } from '../src/core/agent.js';
import { WebLLMSkill } from '../src/skills/llm/webllm/skill.js';

/**
 * Simple text-based browser agent
 */
async function createSimpleAgent() {
  // Create agent with WebLLM skill
  const agent = new BaseAgent({
    id: 'simple-browser-agent',
    name: 'Simple Browser Agent',
    description: 'A simple text-based agent running in the browser',
  });

  // Add WebLLM skill for text generation
  const llmSkill = new WebLLMSkill({
    model: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
  });

  agent.addSkill(llmSkill);

  // Initialize agent
  await agent.initialize();

  return agent;
}

/**
 * Example usage
 */
async function main() {
  console.log('Creating simple browser agent...');
  
  const agent = await createSimpleAgent();
  
  console.log('Agent created:', agent.id);
  console.log('Capabilities:', agent.capabilities);

  // Simple chat interaction
  const response = await agent.chat([
    { role: 'user', content: 'Hello! What can you help me with?' }
  ]);

  console.log('Response:', response);

  // Streaming example
  console.log('\nStreaming response:');
  for await (const chunk of agent.chatStream([
    { role: 'user', content: 'Tell me a short joke.' }
  ])) {
    process.stdout.write(chunk.content || '');
  }
  console.log('\n');

  // Cleanup
  await agent.cleanup();
}

// Export for use in browser
export { createSimpleAgent, main };

// Run if executed directly
if (typeof window === 'undefined') {
  main().catch(console.error);
}
