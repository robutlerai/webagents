/**
 * Builtin embedded agents.
 * 
 * These agents are bundled with the webagents package and serve as defaults
 * when no local agent files are present.
 */

import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Get the path to the embedded ROBUTLER.md agent.
 */
export function getRobutlerPath(): string {
  return join(__dirname, 'ROBUTLER.md');
}

/**
 * Get the content of the embedded ROBUTLER.md agent.
 */
export function getRobutlerContent(): string {
  return readFileSync(getRobutlerPath(), 'utf-8');
}

/**
 * Parse agent frontmatter from markdown content.
 */
export function parseAgentMarkdown(content: string): {
  name: string;
  description: string;
  skills: string[];
  instructions: string;
} {
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  
  if (!frontmatterMatch) {
    return {
      name: 'unknown',
      description: '',
      skills: [],
      instructions: content,
    };
  }
  
  const [, frontmatter, body] = frontmatterMatch;
  
  // Simple YAML parsing
  const lines = frontmatter.split('\n');
  let name = 'unknown';
  let description = '';
  const skills: string[] = [];
  let inSkills = false;
  
  for (const line of lines) {
    if (line.startsWith('name:')) {
      name = line.replace('name:', '').trim();
    } else if (line.startsWith('description:')) {
      description = line.replace('description:', '').trim();
    } else if (line.startsWith('skills:')) {
      inSkills = true;
    } else if (inSkills && line.trim().startsWith('-')) {
      skills.push(line.trim().replace(/^-\s*/, ''));
    } else if (!line.startsWith(' ') && !line.startsWith('-')) {
      inSkills = false;
    }
  }
  
  return {
    name,
    description,
    skills,
    instructions: body.trim(),
  };
}
