"""
Cron Scheduler

Schedule agent execution using cron expressions.
"""

import asyncio
from typing import Optional, Dict, List, Callable
from datetime import datetime
from dataclasses import dataclass
import uuid

from croniter import croniter


@dataclass
class CronJob:
    """A scheduled cron job."""
    id: str
    agent_name: str
    schedule: str  # Cron expression
    next_run: datetime
    last_run: Optional[datetime] = None
    status: str = "active"  # active, paused
    run_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "schedule": self.schedule,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "status": self.status,
            "run_count": self.run_count,
        }


class CronScheduler:
    """Schedule agent execution with cron expressions."""
    
    # Human-readable shortcuts
    SHORTCUTS = {
        "@yearly": "0 0 1 1 *",
        "@annually": "0 0 1 1 *",
        "@monthly": "0 0 1 * *",
        "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *",
        "@midnight": "0 0 * * *",
        "@hourly": "0 * * * *",
    }
    
    def __init__(self, agent_manager=None):
        """Initialize scheduler.
        
        Args:
            agent_manager: AgentManager for executing agents
        """
        self.manager = agent_manager
        self.jobs: Dict[str, CronJob] = {}
        self._running = False
    
    def add_job(
        self,
        agent_name: str,
        schedule: str,
        job_id: Optional[str] = None,
    ) -> CronJob:
        """Add a cron job.
        
        Args:
            agent_name: Agent to run
            schedule: Cron expression or shortcut (@daily, @hourly, etc.)
            job_id: Optional custom job ID
            
        Returns:
            Created CronJob
        """
        # Convert shortcuts
        if schedule in self.SHORTCUTS:
            schedule = self.SHORTCUTS[schedule]
        
        # Validate cron expression
        try:
            cron = croniter(schedule, datetime.utcnow())
            next_run = cron.get_next(datetime)
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {schedule}") from e
        
        job_id = job_id or str(uuid.uuid4())[:8]
        
        job = CronJob(
            id=job_id,
            agent_name=agent_name,
            schedule=schedule,
            next_run=next_run,
        )
        
        self.jobs[job_id] = job
        return job
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a cron job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if removed
        """
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a cron job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if paused
        """
        if job_id in self.jobs:
            self.jobs[job_id].status = "paused"
            return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if resumed
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = "active"
            # Recalculate next run
            cron = croniter(job.schedule, datetime.utcnow())
            job.next_run = cron.get_next(datetime)
            return True
        return False
    
    def get_job(self, job_id: str) -> Optional[CronJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[CronJob]:
        """List all jobs."""
        return list(self.jobs.values())
    
    async def run(self):
        """Run the scheduler loop."""
        self._running = True
        
        while self._running:
            now = datetime.utcnow()
            
            for job in self.jobs.values():
                if job.status != "active":
                    continue
                
                if job.next_run and now >= job.next_run:
                    # Execute job
                    await self._execute_job(job)
                    
                    # Calculate next run
                    cron = croniter(job.schedule, now)
                    job.next_run = cron.get_next(datetime)
                    job.last_run = now
                    job.run_count += 1
            
            # Check every second
            await asyncio.sleep(1)
    
    async def _execute_job(self, job: CronJob):
        """Execute a cron job.
        
        Args:
            job: Job to execute
        """
        try:
            if self.manager:
                # Check if agent is already running
                if job.agent_name not in self.manager.get_running_agents():
                    await self.manager.start(job.agent_name)
            else:
                # No manager, just log
                print(f"[CRON] Would execute: {job.agent_name}")
        except Exception as e:
            print(f"[CRON] Error executing {job.agent_name}: {e}")
    
    async def trigger(self, job_id: str):
        """Trigger immediate job execution.
        
        Args:
            job_id: Job ID
        """
        job = self.get_job(job_id)
        if job:
            await self._execute_job(job)
            job.last_run = datetime.utcnow()
            job.run_count += 1
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
    
    def sync_from_registry(self, registry):
        """Sync jobs from agent registry.
        
        Args:
            registry: DaemonRegistry with agents
        """
        for agent in registry.get_agents_with_cron():
            # Check if job already exists for this agent
            existing = None
            for job in self.jobs.values():
                if job.agent_name == agent.name:
                    existing = job
                    break
            
            if existing:
                # Update schedule if changed
                if existing.schedule != agent.cron:
                    existing.schedule = agent.cron
                    cron = croniter(agent.cron, datetime.utcnow())
                    existing.next_run = cron.get_next(datetime)
            else:
                # Add new job
                self.add_job(agent.name, agent.cron)
