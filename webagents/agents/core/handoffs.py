"""
Agent Handoffs - WebAgents V2.0

LocalAgentHandoff implementation for same-instance agent transfers.
Provides basic agent-to-agent handoff functionality within a single server instance.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from ..skills.base import HandoffResult
from ...server.context.context_vars import get_context, create_context, set_context


@dataclass
class HandoffExecution:
    """Record of handoff execution"""
    timestamp: datetime
    source_agent: str
    target_agent: str
    handoff_type: str
    context_data: Dict[str, Any]
    result: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None


class LocalAgentHandoff:
    """
    LocalAgentHandoff - Same-instance agent handoff system
    
    Features:
    - Agent-to-agent transfers within the same server instance
    - Context preservation and transfer
    - Handoff execution tracking and history
    - Automatic context cleanup and management
    """
    
    def __init__(self, agents: Dict[str, 'BaseAgent']):
        """
        Initialize LocalAgentHandoff with available agents
        
        Args:
            agents: Dictionary of agent_name -> BaseAgent instances
        """
        self.agents = agents
        self.handoff_history: List[HandoffExecution] = []
        self.active_handoffs: Dict[str, str] = {}  # handoff_id -> target_agent
        
    def register_agent(self, agent_name: str, agent: 'BaseAgent') -> None:
        """Register an agent for handoffs"""
        self.agents[agent_name] = agent
        
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents for handoff"""
        return list(self.agents.keys())
    
    async def execute_handoff(
        self,
        source_agent: str,
        target_agent: str,
        handoff_data: Dict[str, Any] = None,
        preserve_context: bool = True
    ) -> HandoffResult:
        """
        Execute handoff from source agent to target agent
        
        Args:
            source_agent: Name of the source agent
            target_agent: Name of the target agent
            handoff_data: Additional data to pass to target agent
            preserve_context: Whether to preserve request context
            
        Returns:
            HandoffResult with execution details
        """
        start_time = datetime.utcnow()
        handoff_id = f"handoff_{int(time.time())}_{source_agent}_to_{target_agent}"
        
        try:
            # Validate agents exist
            if source_agent not in self.agents:
                raise ValueError(f"Source agent '{source_agent}' not found")
            if target_agent not in self.agents:
                raise ValueError(f"Target agent '{target_agent}' not found")
            
            source = self.agents[source_agent]
            target = self.agents[target_agent]
            
            # Get current context
            current_context = get_context()
            if not current_context:
                raise ValueError("No active context for handoff")
            
            # Prepare handoff context data
            context_data = {
                'source_agent': source_agent,
                'target_agent': target_agent,
                'handoff_id': handoff_id,
                'handoff_timestamp': start_time.isoformat(),
                'handoff_data': handoff_data or {}
            }
            
            # Create new context for target agent if preserving context
            if preserve_context:
                target_context = create_context(
                    request_id=current_context.request_id,
                    peer_user_id=current_context.peer_user_id,
                    payment_user_id=current_context.payment_user_id,
                    origin_user_id=current_context.origin_user_id,
                    agent_owner_user_id=current_context.agent_owner_user_id,
                    messages=current_context.messages.copy(),
                    stream=current_context.stream
                )
                
                # Add handoff context data
                target_context.set("handoff_context", context_data)
                target_context.update_agent_context(target, target_agent)
                set_context(target_context)
            
            # Track active handoff
            self.active_handoffs[handoff_id] = target_agent
            
            # Execute source agent's handoff hooks (if any)
            if hasattr(source, '_execute_hooks'):
                current_context.set("handoff_data", context_data)
                await source._execute_hooks("before_handoff", current_context)
            
            # Create handoff message for target agent
            handoff_message = {
                "role": "system",
                "content": f"Handoff from {source_agent}: {handoff_data.get('reason', 'Agent transfer requested')}"
            }
            
            # Add handoff context to messages
            messages_with_handoff = current_context.messages + [handoff_message]
            if handoff_data and handoff_data.get('user_message'):
                messages_with_handoff.append({
                    "role": "user", 
                    "content": handoff_data['user_message']
                })
            
            # Execute target agent with handoff context
            # Note: This is a basic implementation - could be enhanced with specific handoff methods
            response = await target.run(
                messages=messages_with_handoff,
                tools=handoff_data.get('tools', []),
                stream=False
            )
            
            # Execute target agent's handoff hooks (if any)
            if hasattr(target, '_execute_hooks'):
                if preserve_context:
                    target_context.set("handoff_result", response)
                    await target._execute_hooks("after_handoff", target_context)
            
            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record successful handoff
            execution = HandoffExecution(
                timestamp=start_time,
                source_agent=source_agent,
                target_agent=target_agent,
                handoff_type="local_agent",
                context_data=context_data,
                result=response,
                duration_ms=duration_ms,
                success=True
            )
            self.handoff_history.append(execution)
            
            # Remove from active handoffs
            if handoff_id in self.active_handoffs:
                del self.active_handoffs[handoff_id]
            
            # Return handoff result
            return HandoffResult(
                result=response,
                handoff_type="local_agent",
                success=True,
                metadata={
                    'handoff_id': handoff_id,
                    'source_agent': source_agent,
                    'target_agent': target_agent,
                    'duration_ms': duration_ms,
                    'context_preserved': preserve_context
                }
            )
            
        except Exception as e:
            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record failed handoff
            execution = HandoffExecution(
                timestamp=start_time,
                source_agent=source_agent,
                target_agent=target_agent,
                handoff_type="local_agent",
                context_data=context_data if 'context_data' in locals() else {},
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
            self.handoff_history.append(execution)
            
            # Clean up active handoff
            if handoff_id in self.active_handoffs:
                del self.active_handoffs[handoff_id]
            
            # Return error result
            return HandoffResult(
                result=None,
                handoff_type="local_agent",
                success=False,
                metadata={
                    'handoff_id': handoff_id if 'handoff_id' in locals() else 'unknown',
                    'error': str(e),
                    'duration_ms': duration_ms
                }
            )
    
    def get_handoff_history(self, limit: Optional[int] = None) -> List[HandoffExecution]:
        """Get handoff execution history"""
        history = sorted(self.handoff_history, key=lambda x: x.timestamp, reverse=True)
        if limit:
            history = history[:limit]
        return history
    
    def get_active_handoffs(self) -> Dict[str, str]:
        """Get currently active handoffs"""
        return self.active_handoffs.copy()
    
    def get_handoff_stats(self) -> Dict[str, Any]:
        """Get handoff statistics"""
        if not self.handoff_history:
            return {
                'total_handoffs': 0,
                'success_rate': 0.0,
                'average_duration_ms': 0.0,
                'most_common_source': None,
                'most_common_target': None
            }
        
        successful = [h for h in self.handoff_history if h.success]
        failed = [h for h in self.handoff_history if not h.success]
        
        # Calculate statistics
        total = len(self.handoff_history)
        success_rate = len(successful) / total if total > 0 else 0.0
        avg_duration = sum(h.duration_ms for h in successful) / len(successful) if successful else 0.0
        
        # Most common source and target
        sources = [h.source_agent for h in self.handoff_history]
        targets = [h.target_agent for h in self.handoff_history]
        
        most_common_source = max(set(sources), key=sources.count) if sources else None
        most_common_target = max(set(targets), key=targets.count) if targets else None
        
        return {
            'total_handoffs': total,
            'successful_handoffs': len(successful),
            'failed_handoffs': len(failed),
            'success_rate': success_rate,
            'average_duration_ms': avg_duration,
            'most_common_source': most_common_source,
            'most_common_target': most_common_target
        }


# Factory function for easy creation
def create_local_handoff_system(agents: Dict[str, 'BaseAgent']) -> LocalAgentHandoff:
    """
    Factory function to create LocalAgentHandoff system
    
    Args:
        agents: Dictionary of agent_name -> BaseAgent instances
        
    Returns:
        Configured LocalAgentHandoff instance
    """
    return LocalAgentHandoff(agents) 