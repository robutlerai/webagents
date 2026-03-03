"""
Todo Skill

Local tool to manage subtasks for complex requests, matching Gemini CLI specification.
"""

from typing import List, Dict, Any, Optional, Union
from ...base import Skill
from webagents.agents.tools.decorators import tool

class TodoSkill(Skill):
    """Subtask management for complex agent workflows"""
    
    def __init__(self, session=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.session = session

    @tool
    async def write_todos(self, todos: Union[List[Dict[str, str]], str]) -> str:
        """Create and manage a list of subtasks for complex requests.
        
        Use this tool to break down multi-step requests into a clear plan and track progress.
        
        Args:
            todos: The complete list of todo items. 
                  Can be a list of objects or a JSON string representing the list.
                  Each item must have:
                  - description: Task description
                  - status: 'pending', 'in_progress', 'completed', or 'cancelled'
            
        Returns:
            Status message.
        """
        import json
        
        # Handle string input (LLM might pass JSON string instead of list)
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                # Fallback: try parsing YAML-like format often produced by LLMs
                try:
                    import yaml
                    parsed = yaml.safe_load(todos)
                    if isinstance(parsed, list):
                        todos = parsed
                    else:
                        raise ValueError("Not a list")
                except Exception as e:
                    return f"Error parsing todos. Please provide valid JSON list. Error: {str(e)}"
        
        if not isinstance(todos, list):
            return "Error: todos must be a list of task objects."
            
        # Validate structure
        valid_todos = []
        for t in todos:
            if not isinstance(t, dict):
                continue
            if 'description' not in t or 'status' not in t:
                continue
            valid_todos.append(t)
            
        todos = valid_todos
        
        # Find in_progress task for confirmation
        in_progress = next((t['description'] for t in todos if t['status'] == 'in_progress'), None)
        
        # Update local session if available (for non-daemon mode)
        if self.session:
            # Check if session is a dict (daemon context) or object (CLI session)
            if hasattr(self.session, 'todos'):
                self.session.todos = todos
        
        status_msg = f"Updated todo list with {len(todos)} items."
        if in_progress:
            status_msg += f" Currently working on: {in_progress}"
            
        return status_msg
