/**
 * Test specification parser
 * 
 * Parses Markdown test files with YAML frontmatter into structured test specs.
 */

import * as yaml from 'yaml';
import type { TestSpec, TestCase, HttpRequestConfig, AgentDefinition } from './types';

/**
 * Parser for Markdown compliance test files.
 * 
 * Parses files with structure:
 * - YAML frontmatter
 * - # Title
 * - Description paragraph
 * - ## Setup
 * - ## Test Cases
 * - ### N. Case Name
 */
export class TestParser {
  /**
   * Parse a test file content into a structured spec.
   */
  parse(content: string, filename: string = 'test'): TestSpec {
    // Extract frontmatter
    const { frontmatter, body } = this.extractFrontmatter(content);

    // Parse body sections
    const { description } = this.extractTitleDescription(body);
    const setup = this.extractSection(body, 'Setup');
    const agents = setup ? this.parseAgents(setup) : [];
    const cases = this.parseTestCases(body);

    return {
      name: (frontmatter.name as string) || this.getBasename(filename),
      version: (frontmatter.version as string) || '1.0',
      transport: (frontmatter.transport as string) || 'completions',
      type: (frontmatter.type as 'single-agent' | 'multi-agent') || 'single-agent',
      tags: (frontmatter.tags as string[]) || [],
      timeout: (frontmatter.timeout as number) || 60,
      dependsOn: (frontmatter.depends_on as string[]) || [],
      description,
      setup: setup || '',
      agents,
      cases,
    };
  }

  /**
   * Extract YAML frontmatter from markdown.
   */
  private extractFrontmatter(content: string): { frontmatter: Record<string, unknown>; body: string } {
    const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    
    if (match) {
      try {
        const frontmatter = yaml.parse(match[1]) || {};
        return { frontmatter, body: match[2] };
      } catch {
        return { frontmatter: {}, body: match[2] };
      }
    }
    
    return { frontmatter: {}, body: content };
  }

  /**
   * Extract title and description from body.
   */
  private extractTitleDescription(body: string): { title: string; description: string } {
    // Find H1 title
    const titleMatch = body.match(/^# (.+)$/m);
    const title = titleMatch ? titleMatch[1] : '';

    // Description is text between title and first ## section
    let description = '';
    if (titleMatch) {
      const afterTitle = body.slice(titleMatch.index! + titleMatch[0].length);
      const descMatch = afterTitle.match(/^\n+(.+?)(?=\n## |\n*$)/s);
      description = descMatch ? descMatch[1].trim() : '';
    }

    return { title, description };
  }

  /**
   * Extract content of a ## section.
   */
  private extractSection(body: string, sectionName: string): string | null {
    const pattern = new RegExp(`^## ${sectionName}\\s*\\n([\\s\\S]*?)(?=\\n## |$)`, 'm');
    const match = body.match(pattern);
    return match ? match[1].trim() : null;
  }

  /**
   * Parse agent definitions from setup section.
   */
  private parseAgents(setup: string): AgentDefinition[] {
    const agents: AgentDefinition[] = [];

    // Find ### Agent: name sections
    const agentPattern = /### Agent: (\w+)\s*\n([\s\S]*?)(?=\n### |$)/g;
    let match;

    while ((match = agentPattern.exec(setup)) !== null) {
      const agentId = match[1];
      const agentContent = match[2];

      const agent: AgentDefinition = { id: agentId };

      // Parse bullet points
      for (const line of agentContent.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.startsWith('- Name:')) {
          agent.name = trimmed.split(':', 2)[1].trim().replace(/`/g, '');
        } else if (trimmed.startsWith('- Instructions:')) {
          agent.instructions = trimmed.split(':', 2)[1].trim().replace(/"/g, '');
        } else if (trimmed.startsWith('- Handoffs:')) {
          const handoffsStr = trimmed.split(':', 2)[1].trim();
          agent.handoffs = handoffsStr
            .replace(/[\[\]]/g, '')
            .split(',')
            .map(h => h.trim())
            .filter(Boolean);
        } else if (trimmed.startsWith('- Tools:')) {
          const toolsStr = trimmed.split(':', 2)[1].trim();
          agent.tools = toolsStr
            .replace(/[\[\]]/g, '')
            .split(',')
            .map(t => t.trim())
            .filter(Boolean);
        }
      }

      agents.push(agent);
    }

    return agents;
  }

  /**
   * Parse test cases from body.
   */
  private parseTestCases(body: string): TestCase[] {
    const cases: TestCase[] = [];

    // Find ## Test Cases section
    const casesSection = this.extractSection(body, 'Test Cases');
    if (!casesSection) {
      return cases;
    }

    // Find ### N. Case Name sections
    const casePattern = /### \d+\. (.+?)\s*\n([\s\S]*?)(?=\n### |$)/g;
    let match;

    while ((match = casePattern.exec(casesSection)) !== null) {
      const caseName = match[1].trim();
      const caseContent = match[2];

      const testCase: TestCase = {
        name: caseName,
        assertions: [],
        expected: 'success',
      };

      // Parse Request section
      const request = this.parseRequest(caseContent);
      if (request) {
        testCase.request = request;
      }

      // Parse Flow section (for multi-agent)
      const flow = this.parseFlow(caseContent);
      if (flow) {
        testCase.flow = flow;
      }

      // Parse Assertions
      testCase.assertions = this.parseAssertions(caseContent);

      // Parse Strict block
      const strict = this.parseStrict(caseContent);
      if (strict) {
        testCase.strict = strict;
      }

      // Parse Expected
      const expectedMatch = caseContent.match(/\*\*Expected:\*\*\s*(\w+)/);
      if (expectedMatch) {
        testCase.expected = expectedMatch[1] as 'success' | 'failure';
      }

      cases.push(testCase);
    }

    return cases;
  }

  /**
   * Parse request section from test case.
   */
  private parseRequest(content: string): HttpRequestConfig | null {
    const match = content.match(/\*\*Request:\*\*\s*\n([\s\S]*?)(?=\n\*\*|$)/);
    if (!match) {
      return null;
    }

    const requestContent = match[1];

    // Extract method and path
    const methodMatch = requestContent.match(/(GET|POST|PUT|DELETE|PATCH)\s+`?([^`\s]+)`?/);
    if (!methodMatch) {
      return null;
    }

    const request: HttpRequestConfig = {
      method: methodMatch[1],
      path: methodMatch[2],
    };

    // Extract JSON body
    const jsonMatch = requestContent.match(/```json\s*\n([\s\S]*?)\n```/);
    if (jsonMatch) {
      try {
        request.body = yaml.parse(jsonMatch[1]);
      } catch {
        // Body parsing failed, skip
      }
    }

    // Check for streaming
    if (
      requestContent.toLowerCase().includes('(streaming)') ||
      requestContent.includes('"stream": true')
    ) {
      request.stream = true;
    }

    return request;
  }

  /**
   * Parse flow section for multi-agent tests.
   */
  private parseFlow(content: string): string[] | null {
    const match = content.match(/\*\*Flow:\*\*\s*\n([\s\S]*?)(?=\n\*\*|$)/);
    if (!match) {
      return null;
    }

    const flow: string[] = [];
    for (const line of match[1].split('\n')) {
      const trimmed = line.trim();
      if (/^\d+\./.test(trimmed)) {
        flow.push(trimmed.replace(/^\d+\.\s*/, ''));
      }
    }

    return flow.length > 0 ? flow : null;
  }

  /**
   * Parse natural language assertions.
   */
  private parseAssertions(content: string): string[] {
    const match = content.match(/\*\*Assertions:\*\*\s*\n([\s\S]*?)(?=\n\*\*|$)/);
    if (!match) {
      return [];
    }

    const assertions: string[] = [];
    for (const line of match[1].split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('- ')) {
        assertions.push(trimmed.slice(2));
      }
    }

    return assertions;
  }

  /**
   * Parse strict assertions YAML block.
   */
  private parseStrict(content: string): Record<string, unknown> | null {
    const match = content.match(/\*\*Strict:\*\*\s*\n```yaml\s*\n([\s\S]*?)\n```/);
    if (!match) {
      return null;
    }

    try {
      return yaml.parse(match[1]);
    } catch {
      return null;
    }
  }

  /**
   * Get basename from filename.
   */
  private getBasename(filename: string): string {
    const parts = filename.split('/');
    const name = parts[parts.length - 1];
    return name.replace(/\.[^.]+$/, '');
  }
}
