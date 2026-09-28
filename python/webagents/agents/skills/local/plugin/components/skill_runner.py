"""
Skill Runner - SKILL.md Parser and Executor

Parses SKILL.md files with YAML frontmatter and executes them
with $ARGUMENTS substitution.

Claude Code compatible SKILL.md format:
---
name: skill-name
description: Skill description
disable-model-invocation: false
allowed-tools: [tool1, tool2]
context: inline  # or "fork"
---
# Skill content with $ARGUMENTS.param placeholders
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SkillMD:
    """Parsed SKILL.md representation.
    
    Attributes:
        name: Skill identifier
        description: Human-readable description
        content: Markdown content (after frontmatter)
        frontmatter: Raw frontmatter dict
        disable_model_invocation: If True, skill runs without LLM
        allowed_tools: Whitelist of tools for forked execution
        context: Execution context - "inline" or "fork"
        path: Source file path
    """
    name: str
    description: str
    content: str
    frontmatter: Dict[str, Any]
    disable_model_invocation: bool = False
    allowed_tools: Optional[List[str]] = None
    context: str = "inline"
    path: Optional[Path] = None


class SkillRunner:
    """Parse and execute SKILL.md files.
    
    Features:
    - YAML frontmatter parsing
    - $ARGUMENTS.key substitution
    - Inline and forked execution modes
    - Tool restriction support
    """
    
    # Regex patterns
    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL
    )
    ARGUMENT_PATTERN = re.compile(r'\$ARGUMENTS\.(\w+)')
    ARGUMENT_BLOCK_PATTERN = re.compile(r'\$ARGUMENTS\[([\'"]?)(\w+)\1\]')
    
    def parse(self, path: Path) -> SkillMD:
        """Parse SKILL.md file.

        The name falls back to the FOLDER's name, as the Agent Skills
        specification defines it (2026-09-26, plan item 1.4); it used to be
        `path.stem`, which is "SKILL" for every skill.

        Args:
            path: Path to SKILL.md file

        Returns:
            Parsed SkillMD instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")

        fallback = path.parent.name if path.name.lower() == "skill.md" else path.stem
        parsed = self.parse_string(path.read_bytes().decode("utf-8", errors="replace"), fallback)
        parsed.path = path
        return parsed

    def parse_string(self, content: str, name: str = "unnamed") -> SkillMD:
        """Parse SKILL.md content from string.

        The frontmatter is split and read by the SKILL.md loader
        (`agents/skills/local/skillmd/skillmd_loader.py`, 2026-09-26, plan
        item 1.4) rather than by a regex that needed a newline after the
        closing `---` and broke on CRLF; a description with an unquoted colon
        is retried quoted; `allowed-tools` splits on spaces as well as commas.
        A frontmatter that is not YAML is still read as `{}`, with a warning,
        as this runner always did: the plugin skill is lenient by design.

        Args:
            content: Skill content string
            name: Skill name (used if not in frontmatter)

        Returns:
            Parsed SkillMD instance
        """
        from ...skillmd.skillmd_loader import parse_yaml_frontmatter, split_frontmatter

        frontmatter: Dict[str, Any] = {}
        front, body, problem = split_frontmatter(content)
        if problem == "no_frontmatter":
            body = content
        elif problem == "unclosed":
            logger.warning("Failed to parse frontmatter: the --- block never closes")
            body = content
        else:
            data, reason = parse_yaml_frontmatter(front or "")
            if data is None:
                logger.warning(f"Failed to parse frontmatter: {reason}")
            else:
                frontmatter = data
            content = body

        allowed_tools = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
        if isinstance(allowed_tools, str):
            allowed_tools = [t for t in re.split(r"[\s,]+", allowed_tools) if t]

        return SkillMD(
            name=str(frontmatter.get("name") or name),
            description=str(frontmatter.get("description") or ""),
            content=content if problem is None else body,
            frontmatter=frontmatter,
            disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
            allowed_tools=allowed_tools,
            context=str(frontmatter.get("context", "inline")),
        )
    
    def substitute_arguments(self, content: str, arguments: Dict[str, Any]) -> str:
        """Replace $ARGUMENTS.key with actual values.
        
        Supports two syntaxes:
        - $ARGUMENTS.key - dot notation
        - $ARGUMENTS['key'] or $ARGUMENTS["key"] - bracket notation
        
        Args:
            content: Content with argument placeholders
            arguments: Dict of argument values
            
        Returns:
            Content with substituted values
        """
        def replace_dot(match):
            key = match.group(1)
            value = arguments.get(key)
            if value is None:
                # Return original placeholder if not found
                return f"$ARGUMENTS.{key}"
            return str(value)
        
        def replace_bracket(match):
            key = match.group(2)
            value = arguments.get(key)
            if value is None:
                quote = match.group(1) or "'"
                return f"$ARGUMENTS[{quote}{key}{quote}]"
            return str(value)
        
        # Apply both substitution patterns
        result = self.ARGUMENT_PATTERN.sub(replace_dot, content)
        result = self.ARGUMENT_BLOCK_PATTERN.sub(replace_bracket, result)
        
        return result
    
    def get_required_arguments(self, content: str) -> List[str]:
        """Extract list of argument placeholders from content.
        
        Args:
            content: Skill content
            
        Returns:
            List of argument names
        """
        args = set()
        
        # Find dot notation
        for match in self.ARGUMENT_PATTERN.finditer(content):
            args.add(match.group(1))
        
        # Find bracket notation
        for match in self.ARGUMENT_BLOCK_PATTERN.finditer(content):
            args.add(match.group(2))
        
        return sorted(args)
    
    async def execute(
        self,
        skill: SkillMD,
        arguments: Dict[str, Any],
        agent: Any = None
    ) -> str:
        """Execute skill content.
        
        Args:
            skill: Parsed SkillMD instance
            arguments: Arguments for substitution
            agent: Agent instance for context
            
        Returns:
            Executed skill content or result
        """
        # Substitute arguments
        content = self.substitute_arguments(skill.content, arguments)
        
        # Execute based on context mode
        if skill.context == "fork":
            return await self._execute_forked(content, skill, agent)
        else:
            return await self._execute_inline(content, skill, agent)
    
    async def _execute_inline(
        self,
        content: str,
        skill: SkillMD,
        agent: Any
    ) -> str:
        """Execute skill inline in current agent context.
        
        For inline execution, the content is returned to be injected
        into the agent's context/conversation.
        
        Args:
            content: Substituted skill content
            skill: Skill metadata
            agent: Agent instance
            
        Returns:
            Content to inject into context
        """
        logger.debug(f"Executing skill inline: {skill.name}")
        return content
    
    async def _execute_forked(
        self,
        content: str,
        skill: SkillMD,
        agent: Any
    ) -> str:
        """Execute skill in a forked subagent context.
        
        Creates a subagent with restricted tools if allowed_tools
        is specified in the skill frontmatter.
        
        Args:
            content: Substituted skill content
            skill: Skill metadata with tool restrictions
            agent: Parent agent instance
            
        Returns:
            Subagent execution result
        """
        logger.debug(f"Executing skill in fork: {skill.name}")
        
        if skill.allowed_tools:
            logger.debug(f"Skill restricts tools to: {skill.allowed_tools}")
        
        # TODO: Implement actual subagent forking with tool restrictions
        # For now, return content with metadata
        return content
    
    def validate_skill(self, skill: SkillMD) -> List[str]:
        """Validate a parsed skill.
        
        Args:
            skill: Skill to validate
            
        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []
        
        if not skill.name:
            warnings.append("Skill has no name")
        
        if not skill.content.strip():
            warnings.append("Skill has no content")
        
        if skill.context not in ("inline", "fork"):
            warnings.append(f"Invalid context '{skill.context}', must be 'inline' or 'fork'")
        
        if skill.allowed_tools and skill.context != "fork":
            warnings.append("allowed-tools only applies when context is 'fork'")
        
        return warnings
