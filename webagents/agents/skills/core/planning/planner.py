"""
PlannerSkill - Simple Task Planning and Todo Management

A straightforward planner that matches the TypeScript implementation.
Maintains a single plan state across multiple tool calls for consistent UX.
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from ...base import Skill
from ....tools.decorators import tool


@dataclass
class PlannerTask:
    """Represents a single task in a plan"""
    id: str
    title: str
    description: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, cancelled
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass 
class PlannerState:
    """Represents the complete planner state"""
    plan_id: str
    title: str
    description: Optional[str] = None
    tasks: List[PlannerTask] = None
    created_at: str = None
    updated_at: str = None
    status: str = "active"  # active, completed, cancelled

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


class PlannerSkill(Skill):
    """
    Simple task planning and todo management skill that matches the TypeScript implementation.
    
    Features:
    - Single plan state management
    - Task status tracking (pending, in_progress, completed, cancelled)
    - State persistence across tool calls
    - Matches TypeScript planner_tool behavior
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def initialize(self, agent_reference):
        """Initialize with agent reference"""
        await super().initialize(agent_reference)
        self.agent = agent_reference

    def _generate_id(self) -> str:
        """Generate a random ID"""
        return str(uuid.uuid4())[:8]

    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.utcnow().isoformat()

    def _get_action_message(self, action: str, title: str = None, total_tasks: int = 0, completed_tasks: int = 0) -> str:
        """Generate action message"""
        title_part = f' "{title}"' if title else ''
        
        if action == 'create_plan':
            return f"Created new plan{title_part} with {total_tasks} tasks. Start by marking the first task as 'in_progress'."
        elif action == 'add_task':
            return f"Added task{title_part} to the plan"
        elif action == 'update_task':
            return f"Started task{title_part}. Remember to mark this task as 'completed' when you finish explaining/doing this step."
        elif action == 'complete_task':
            next_task_part = '. Move to the next task.' if completed_tasks < total_tasks else ''
            return f"✅ Completed task{title_part}. Progress: {completed_tasks}/{total_tasks}{next_task_part}"
        elif action == 'cancel_task':
            return f"Cancelled task{title_part}"
        elif action == 'update_plan':
            return f"Updated plan{title_part}"
        else:
            return 'Plan updated'

    @tool(description="Create and manage a single task plan to break down complex work into manageable steps. Use this tool to organize your approach and track progress. CRITICAL: Always pass the complete 'state' object from previous tool results as 'current_state' to maintain the same plan across multiple calls.", scope="all")
    async def planner_tool(
        self,
        action: str,  # create_plan, add_task, update_task, complete_task, cancel_task, update_plan
        title: Optional[str] = None,
        description: Optional[str] = None,
        task_id: Optional[str] = None,
        tasks: Optional[List[Dict[str, str]]] = None,  # [{"title": "...", "description": "..."}]
        status: Optional[str] = None,  # pending, in_progress, completed, cancelled
        current_state: Optional[Dict[str, Any]] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Create and manage task plans to break down complex work into manageable steps.
        
        Actions:
        - create_plan: Create a new plan with initial tasks
        - add_task: Add a task to existing plan
        - update_task: Update task status
        - complete_task: Mark task as completed
        - cancel_task: Mark task as cancelled
        - update_plan: Update plan details
        
        Args:
            action: The action to perform
            title: Title for plan or task
            description: Description for plan or task
            task_id: ID of task to update (required for task operations)
            tasks: List of initial tasks when creating plan
            status: New status for task updates
            current_state: CRITICAL - Complete state object from previous calls
        """
        try:
            timestamp = self._get_current_timestamp()
            
            # Handle state management
            if action == 'create_plan':
                # Create new plan
                state = PlannerState(
                    plan_id=self._generate_id(),
                    title=title or 'New Plan',
                    description=description or '',
                    created_at=timestamp,
                    updated_at=timestamp,
                    status='active'
                )
                
                # Add initial tasks
                if tasks:
                    for task_data in tasks:
                        task = PlannerTask(
                            id=self._generate_id(),
                            title=task_data.get('title', ''),
                            description=task_data.get('description'),
                            status='pending',
                            created_at=timestamp,
                            updated_at=timestamp
                        )
                        state.tasks.append(task)
            else:
                # For all other actions, require current_state
                if not current_state:
                    return {
                        'success': False,
                        'action': action,
                        'error': f"Action '{action}' requires current_state from previous planner_tool result. Please pass the complete 'state' object from the previous call.",
                        'state': None
                    }
                
                # Reconstruct state from current_state
                state = PlannerState(
                    plan_id=current_state['plan_id'],
                    title=current_state['title'],
                    description=current_state.get('description'),
                    status=current_state['status'],
                    created_at=current_state['created_at'],
                    updated_at=current_state['updated_at']
                )
                
                # Reconstruct tasks
                state.tasks = []
                for task_data in current_state.get('tasks', []):
                    task = PlannerTask(
                        id=task_data['id'],
                        title=task_data['title'],
                        description=task_data.get('description'),
                        status=task_data['status'],
                        created_at=task_data['created_at'],
                        updated_at=task_data['updated_at']
                    )
                    state.tasks.append(task)

            # Execute action
            if action == 'create_plan':
                # Already handled above
                pass
            
            elif action == 'add_task':
                if not title:
                    raise Exception('Task title is required')
                
                new_task = PlannerTask(
                    id=self._generate_id(),
                    title=title,
                    description=description,
                    status='pending',
                    created_at=timestamp,
                    updated_at=timestamp
                )
                state.tasks.append(new_task)
                state.updated_at = timestamp
            
            elif action == 'update_task':
                if not task_id:
                    raise Exception('Task ID is required for task updates')
                
                task = next((t for t in state.tasks if t.id == task_id), None)
                if not task:
                    raise Exception(f'Task with ID {task_id} not found')
                
                if title:
                    task.title = title
                if description is not None:
                    task.description = description
                if status:
                    task.status = status
                task.updated_at = timestamp
                state.updated_at = timestamp
            
            elif action == 'complete_task':
                if not task_id:
                    raise Exception('Task ID is required to complete a task')
                
                task = next((t for t in state.tasks if t.id == task_id), None)
                if not task:
                    raise Exception(f'Task with ID {task_id} not found')
                
                task.status = 'completed'
                task.updated_at = timestamp
                state.updated_at = timestamp
                
                # Check if all tasks are completed
                all_completed = all(t.status in ['completed', 'cancelled'] for t in state.tasks)
                if all_completed and state.tasks:
                    state.status = 'completed'
            
            elif action == 'cancel_task':
                if not task_id:
                    raise Exception('Task ID is required to cancel a task')
                
                task = next((t for t in state.tasks if t.id == task_id), None)
                if not task:
                    raise Exception(f'Task with ID {task_id} not found')
                
                task.status = 'cancelled'
                task.updated_at = timestamp
                state.updated_at = timestamp
            
            elif action == 'update_plan':
                if title:
                    state.title = title
                if description is not None:
                    state.description = description
                if status and status in ['active', 'completed', 'cancelled']:
                    state.status = status
                state.updated_at = timestamp
            
            else:
                raise Exception(f'Unknown action: {action}')

            # Calculate summary stats
            total_tasks = len(state.tasks)
            completed_tasks = len([t for t in state.tasks if t.status == 'completed'])
            in_progress_tasks = len([t for t in state.tasks if t.status == 'in_progress'])
            pending_tasks = len([t for t in state.tasks if t.status == 'pending'])
            cancelled_tasks = len([t for t in state.tasks if t.status == 'cancelled'])

            # Convert state to dict for JSON serialization
            state_dict = {
                'plan_id': state.plan_id,
                'title': state.title,
                'description': state.description,
                'tasks': [
                    {
                        'id': task.id,
                        'title': task.title,
                        'description': task.description,
                        'status': task.status,
                        'created_at': task.created_at,
                        'updated_at': task.updated_at
                    }
                    for task in state.tasks
                ],
                'created_at': state.created_at,
                'updated_at': state.updated_at,
                'status': state.status
            }

            return {
                'success': True,
                'action': action,
                'state': state_dict,
                'summary': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'in_progress_tasks': in_progress_tasks,
                    'pending_tasks': pending_tasks,
                    'cancelled_tasks': cancelled_tasks,
                    'progress_percentage': round((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0
                },
                'message': self._get_action_message(action, title, total_tasks, completed_tasks),
                '_reminder': "IMPORTANT: Use this 'state' object as 'current_state' parameter in your next planner_tool call to maintain the same plan!"
            }
            
        except Exception as e:
            return {
                'success': False,
                'action': action,
                'error': str(e),
                'state': current_state
            }

    def get_dependencies(self) -> List[str]:
        """Return list of dependencies"""
        return []

    def get_skill_info(self) -> Dict[str, Any]:
        """Get comprehensive skill information"""
        return {
            "name": "PlannerSkill",
            "description": "Simple task planning and todo management that matches TypeScript implementation",
            "version": "3.0.0",
            "capabilities": [
                "Single plan state management",
                "Task status tracking", 
                "State persistence across calls",
                "TypeScript compatibility"
            ],
            "tools": [
                "planner_tool"
            ]
        }