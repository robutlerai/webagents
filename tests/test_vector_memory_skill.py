"""
VectorMemorySkill Unit Tests

Tests for the vector memory skill using Milvus/ChromaDB backend.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

# Try to import, skip if dependencies missing
try:
    from webagents.agents.skills.core.memory.vector_memory.skill import VectorMemorySkill
    HAS_VECTOR_MEMORY = True
except ImportError as e:
    HAS_VECTOR_MEMORY = False
    VectorMemorySkill = None

pytestmark = pytest.mark.skipif(not HAS_VECTOR_MEMORY, reason=f"VectorMemorySkill not available")


class TestVectorMemorySkillInit:
    """Test VectorMemorySkill initialization."""
    
    def test_skill_instantiation(self):
        """Test skill can be instantiated."""
        skill = VectorMemorySkill(config={})
        assert skill is not None
        # VectorMemorySkill stores config as an attribute
        assert skill.config == {}


class TestVectorMemorySkillOperations:
    """Test VectorMemorySkill operations."""
    
    @pytest.mark.asyncio
    async def test_skill_has_store_method(self):
        """Test skill has store capability."""
        skill = VectorMemorySkill(config={})
        # Check for expected methods - actual method is upload_instruction
        assert hasattr(skill, 'upload_instruction')
    
    @pytest.mark.asyncio
    async def test_skill_has_search_method(self):
        """Test skill has search capability."""
        skill = VectorMemorySkill(config={})
        # Actual method is fetch_instructions_tool
        assert hasattr(skill, 'fetch_instructions_tool')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
