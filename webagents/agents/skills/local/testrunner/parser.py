"""
Test specification parser.

Parses Markdown test files with YAML frontmatter into structured test specs.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A single test case."""
    name: str
    request: Optional[Dict[str, Any]] = None
    flow: Optional[List[str]] = None
    assertions: List[str] = field(default_factory=list)
    strict: Optional[Dict[str, Any]] = None
    expected: str = "success"


@dataclass
class TestSpec:
    """A parsed test specification."""
    name: str
    version: str
    transport: str
    type: str = "single-agent"
    tags: List[str] = field(default_factory=list)
    timeout: int = 60
    depends_on: List[str] = field(default_factory=list)
    description: str = ""
    setup: str = ""
    agents: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[TestCase] = field(default_factory=list)


class TestParser:
    """
    Parser for Markdown compliance test files.
    
    Parses files with structure:
    - YAML frontmatter
    - # Title
    - Description paragraph
    - ## Setup
    - ## Test Cases
    - ### N. Case Name
    """
    
    def parse(self, path: Path) -> Dict[str, Any]:
        """
        Parse a test file into a structured spec.
        
        Args:
            path: Path to the markdown file
            
        Returns:
            Parsed test specification as dict
        """
        content = path.read_text()
        
        # Extract frontmatter
        frontmatter, body = self._extract_frontmatter(content)
        
        # Parse body sections
        title, description = self._extract_title_description(body)
        setup = self._extract_section(body, "Setup")
        agents = self._parse_agents(setup) if setup else []
        cases = self._parse_test_cases(body)
        
        spec = TestSpec(
            name=frontmatter.get("name", path.stem),
            version=frontmatter.get("version", "1.0"),
            transport=frontmatter.get("transport", "completions"),
            type=frontmatter.get("type", "single-agent"),
            tags=frontmatter.get("tags", []),
            timeout=frontmatter.get("timeout", 60),
            depends_on=frontmatter.get("depends_on", []),
            description=description,
            setup=setup or "",
            agents=agents,
            cases=cases,
        )
        
        return self._spec_to_dict(spec)
    
    def _extract_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from markdown."""
        match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = match.group(2)
        else:
            frontmatter = {}
            body = content
        return frontmatter, body
    
    def _extract_title_description(self, body: str) -> tuple[str, str]:
        """Extract title and description from body."""
        # Find H1 title
        title_match = re.search(r'^# (.+)$', body, re.MULTILINE)
        title = title_match.group(1) if title_match else ""
        
        # Description is text between title and first ## section
        if title_match:
            after_title = body[title_match.end():]
            desc_match = re.match(r'\n+(.+?)(?=\n## |\Z)', after_title, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""
        else:
            description = ""
        
        return title, description
    
    def _extract_section(self, body: str, section_name: str) -> Optional[str]:
        """Extract content of a ## section."""
        pattern = rf'^## {section_name}\s*\n(.*?)(?=\n## |\Z)'
        match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _parse_agents(self, setup: str) -> List[Dict[str, Any]]:
        """Parse agent definitions from setup section."""
        agents = []
        
        # Find ### Agent: name sections
        agent_pattern = r'### Agent: (\w+)\s*\n(.*?)(?=\n### |\Z)'
        for match in re.finditer(agent_pattern, setup, re.DOTALL):
            agent_id = match.group(1)
            agent_content = match.group(2)
            
            agent = {"id": agent_id}
            
            # Parse bullet points
            for line in agent_content.split('\n'):
                line = line.strip()
                if line.startswith('- Name:'):
                    agent["name"] = line.split(':', 1)[1].strip().strip('`')
                elif line.startswith('- Instructions:'):
                    agent["instructions"] = line.split(':', 1)[1].strip().strip('"')
                elif line.startswith('- Handoffs:'):
                    handoffs_str = line.split(':', 1)[1].strip()
                    agent["handoffs"] = [h.strip() for h in handoffs_str.strip('[]').split(',')]
                elif line.startswith('- Tools:'):
                    tools_str = line.split(':', 1)[1].strip()
                    agent["tools"] = [t.strip() for t in tools_str.strip('[]').split(',')]
            
            agents.append(agent)
        
        return agents
    
    def _parse_test_cases(self, body: str) -> List[TestCase]:
        """Parse test cases from body."""
        cases = []
        
        # Find ## Test Cases section
        cases_section = self._extract_section(body, "Test Cases")
        if not cases_section:
            return cases
        
        # Find ### N. Case Name sections
        case_pattern = r'### \d+\. (.+?)\s*\n(.*?)(?=\n### |\Z)'
        for match in re.finditer(case_pattern, cases_section, re.DOTALL):
            case_name = match.group(1).strip()
            case_content = match.group(2)
            
            case = TestCase(name=case_name)
            
            # Parse Request section
            request = self._parse_request(case_content)
            if request:
                case.request = request
            
            # Parse Flow section (for multi-agent)
            flow = self._parse_flow(case_content)
            if flow:
                case.flow = flow
            
            # Parse Assertions
            case.assertions = self._parse_assertions(case_content)
            
            # Parse Strict block
            strict = self._parse_strict(case_content)
            if strict:
                case.strict = strict
            
            # Parse Expected
            expected_match = re.search(r'\*\*Expected:\*\*\s*(\w+)', case_content)
            if expected_match:
                case.expected = expected_match.group(1)
            
            cases.append(case)
        
        return cases
    
    def _parse_request(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse request section from test case."""
        match = re.search(
            r'\*\*Request:\*\*\s*\n(.*?)(?=\n\*\*|\Z)',
            content,
            re.DOTALL
        )
        if not match:
            return None
        
        request_content = match.group(1)
        
        # Extract method and path
        method_match = re.search(r'(GET|POST|PUT|DELETE|PATCH)\s+`?([^`\s]+)`?', request_content)
        if not method_match:
            return None
        
        request = {
            "method": method_match.group(1),
            "path": method_match.group(2),
        }
        
        # Extract JSON body
        json_match = re.search(r'```json\s*\n(.*?)\n```', request_content, re.DOTALL)
        if json_match:
            try:
                request["body"] = yaml.safe_load(json_match.group(1))
            except:
                request["body_raw"] = json_match.group(1)
        
        # Check for streaming
        if '(streaming)' in request_content.lower() or '"stream": true' in request_content:
            request["stream"] = True
        
        return request
    
    def _parse_flow(self, content: str) -> Optional[List[str]]:
        """Parse flow section for multi-agent tests."""
        match = re.search(r'\*\*Flow:\*\*\s*\n(.*?)(?=\n\*\*|\Z)', content, re.DOTALL)
        if not match:
            return None
        
        flow = []
        for line in match.group(1).split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                flow.append(re.sub(r'^\d+\.\s*', '', line))
        
        return flow if flow else None
    
    def _parse_assertions(self, content: str) -> List[str]:
        """Parse natural language assertions."""
        match = re.search(r'\*\*Assertions:\*\*\s*\n(.*?)(?=\n\*\*|\Z)', content, re.DOTALL)
        if not match:
            return []
        
        assertions = []
        for line in match.group(1).split('\n'):
            line = line.strip()
            if line.startswith('- '):
                assertions.append(line[2:])
        
        return assertions
    
    def _parse_strict(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse strict assertions YAML block."""
        match = re.search(r'\*\*Strict:\*\*\s*\n```yaml\s*\n(.*?)\n```', content, re.DOTALL)
        if not match:
            return None
        
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    
    def _spec_to_dict(self, spec: TestSpec) -> Dict[str, Any]:
        """Convert TestSpec to dict."""
        return {
            "name": spec.name,
            "version": spec.version,
            "transport": spec.transport,
            "type": spec.type,
            "tags": spec.tags,
            "timeout": spec.timeout,
            "depends_on": spec.depends_on,
            "description": spec.description,
            "setup": spec.setup,
            "agents": spec.agents,
            "cases": [
                {
                    "name": case.name,
                    "request": case.request,
                    "flow": case.flow,
                    "assertions": case.assertions,
                    "strict": case.strict,
                    "expected": case.expected,
                }
                for case in spec.cases
            ],
        }
