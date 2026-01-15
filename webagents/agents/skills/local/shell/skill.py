"""
Shell Skill - WebAgents V2.0
Execute shell commands with safety checks.
"""

import subprocess
from typing import Dict, List, Any, Optional

from ...base import Skill
from webagents.agents.tools.decorators import tool

class ShellSkill(Skill):
    """
    Skill for executing shell commands.
    
    Tools:
    - run_command: Run a shell command and get output
    """

    @tool
    async def run_command(self, command: str) -> str:
        """
        Execute a shell command.
        
        Args:
            command: The command to execute
        """
        # Note: In a production environment, you should add rigorous command validation
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nError:\n{result.stderr}"
            
            if not output:
                output = "(No output)"
                
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
