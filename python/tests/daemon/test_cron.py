"""
Cron Scheduler Tests

Test scheduled job functionality.
"""

import pytest
from datetime import datetime, timedelta

from webagents.cli.daemon.cron import CronScheduler, CronJob


class TestCronScheduler:
    """Test cron scheduler."""
    
    def test_add_job(self):
        """Test adding a cron job."""
        scheduler = CronScheduler()
        
        job = scheduler.add_job("my-agent", "0 9 * * *")
        
        assert job.agent_name == "my-agent"
        assert job.schedule == "0 9 * * *"
        assert job.status == "active"
        assert job.next_run is not None
    
    def test_add_job_shortcut(self):
        """Test adding job with shortcut."""
        scheduler = CronScheduler()
        
        job = scheduler.add_job("agent", "@daily")
        
        assert job.schedule == "0 0 * * *"
    
    def test_invalid_schedule(self):
        """Test invalid cron expression."""
        scheduler = CronScheduler()
        
        with pytest.raises(ValueError):
            scheduler.add_job("agent", "invalid cron")
    
    def test_remove_job(self):
        """Test removing a job."""
        scheduler = CronScheduler()
        job = scheduler.add_job("agent", "@hourly")
        
        result = scheduler.remove_job(job.id)
        
        assert result == True
        assert scheduler.get_job(job.id) is None
    
    def test_pause_resume_job(self):
        """Test pausing and resuming a job."""
        scheduler = CronScheduler()
        job = scheduler.add_job("agent", "@hourly")
        
        scheduler.pause_job(job.id)
        assert scheduler.get_job(job.id).status == "paused"
        
        scheduler.resume_job(job.id)
        assert scheduler.get_job(job.id).status == "active"
    
    def test_list_jobs(self):
        """Test listing jobs."""
        scheduler = CronScheduler()
        scheduler.add_job("a", "@hourly")
        scheduler.add_job("b", "@daily")
        scheduler.add_job("c", "@weekly")
        
        jobs = scheduler.list_jobs()
        
        assert len(jobs) == 3


class TestCronJob:
    """Test CronJob model."""
    
    def test_job_creation(self):
        """Test creating a cron job."""
        job = CronJob(
            id="123",
            agent_name="test",
            schedule="0 9 * * *",
            next_run=datetime.utcnow() + timedelta(hours=1),
        )
        
        assert job.id == "123"
        assert job.agent_name == "test"
        assert job.status == "active"
        assert job.run_count == 0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        job = CronJob(
            id="123",
            agent_name="test",
            schedule="@daily",
            next_run=datetime.utcnow(),
        )
        
        data = job.to_dict()
        
        assert data["id"] == "123"
        assert data["agent_name"] == "test"
        assert "next_run" in data
