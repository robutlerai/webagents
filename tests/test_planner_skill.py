"""
PlannerSkill Unit Tests
"""

import pytest
from unittest.mock import Mock, patch

try:
    from webagents.agents.skills.core.planner.skill import PlannerSkill
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False
    PlannerSkill = None

pytestmark = pytest.mark.skipif(not HAS_PLANNER, reason="PlannerSkill not available")


class TestPlannerSkillInit:
    """Test PlannerSkill initialization."""
    
    def test_skill_instantiation(self):
        """Test skill can be instantiated."""
        skill = PlannerSkill(config={})
        assert skill is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
