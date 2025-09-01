"""
Test suite for PlannerSkill - Simple todo management
"""

import pytest
import json
import sys
import os

# Add webagents to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.mark.asyncio
async def test_planner_skill_import():
    """Test that PlannerSkill can be imported successfully"""
    from webagents.agents.skills.core.planning import PlannerSkill
    assert PlannerSkill is not None
    print("✅ PlannerSkill import successful")


@pytest.mark.asyncio
async def test_planner_skill_initialization():
    """Test PlannerSkill initialization"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    # Default initialization
    skill = PlannerSkill()
    assert skill is not None
    assert skill.max_items == 10
    assert len(skill.active_plans) == 0
    
    # Custom configuration
    config = {'max_items': 5}
    skill_custom = PlannerSkill(config)
    assert skill_custom.max_items == 5
    
    print("✅ PlannerSkill initialization test passed")


@pytest.mark.asyncio
async def test_create_plan():
    """Test creating a simple plan"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Create a plan
    result = await skill.create_plan(
        title="Test Project",
        description="Complete the test project tasks",
        todo_items=["Design the system", "Implement features", "Test everything"]
    )
    
    result_data = json.loads(result)
    
    # Check result structure
    assert "plan_id" in result_data
    assert "title" in result_data
    assert "description" in result_data
    assert "total_items" in result_data
    assert "todos" in result_data
    assert "progress" in result_data
    
    # Check values
    assert result_data["title"] == "Test Project"
    assert result_data["total_items"] == 3
    assert len(result_data["todos"]) == 3
    assert result_data["progress"] == "0/3"
    
    # Verify plan was stored
    plan_id = result_data["plan_id"]
    assert plan_id in skill.active_plans
    
    print("✅ Create plan test passed")


@pytest.mark.asyncio
async def test_complete_todo():
    """Test completing a todo item"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Create a plan first
    create_result = await skill.create_plan(
        title="Simple Plan",
        description="A simple test plan",
        todo_items=["Task 1", "Task 2"]
    )
    
    create_data = json.loads(create_result)
    plan_id = create_data["plan_id"]
    
    # Complete first todo
    complete_result = await skill.complete_todo(plan_id, "todo_1")
    complete_data = json.loads(complete_result)
    
    # Check result
    assert "completed_todo" in complete_data
    assert complete_data["completed_todo"]["status"] == "completed"
    assert complete_data["progress"] == "1/2"
    assert complete_data["plan_status"] == "active"
    
    # Complete second todo
    await skill.complete_todo(plan_id, "todo_2")
    
    # Check if plan is complete
    plan = skill.active_plans[plan_id]
    assert plan.status == "completed"
    
    print("✅ Complete todo test passed")


@pytest.mark.asyncio
async def test_add_todo():
    """Test adding a todo to an existing plan"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Create a plan
    create_result = await skill.create_plan(
        title="Expandable Plan",
        description="Plan that can grow",
        todo_items=["Initial task"]
    )
    
    create_data = json.loads(create_result)
    plan_id = create_data["plan_id"]
    
    # Add a new todo
    add_result = await skill.add_todo(plan_id, "New additional task")
    add_data = json.loads(add_result)
    
    # Check result
    assert "added_todo" in add_data
    assert add_data["added_todo"]["description"] == "New additional task"
    assert add_data["total_items"] == 2
    assert add_data["progress"] == "0/2"
    
    print("✅ Add todo test passed")


@pytest.mark.asyncio
async def test_update_todo_status():
    """Test updating todo status"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Create a plan
    create_result = await skill.create_plan(
        title="Status Test Plan",
        description="Test status updates",
        todo_items=["Status test task"]
    )
    
    create_data = json.loads(create_result)
    plan_id = create_data["plan_id"]
    
    # Update to in_progress
    update_result = await skill.update_todo_status(plan_id, "todo_1", "in_progress")
    update_data = json.loads(update_result)
    
    assert update_data["updated_todo"]["status"] == "in_progress"
    
    # Update to completed
    update_result = await skill.update_todo_status(plan_id, "todo_1", "completed")
    update_data = json.loads(update_result)
    
    assert update_data["updated_todo"]["status"] == "completed"
    assert update_data["plan_status"] == "completed"
    
    print("✅ Update todo status test passed")


@pytest.mark.asyncio
async def test_get_plan():
    """Test getting plan details"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Create a plan
    create_result = await skill.create_plan(
        title="Detail Test Plan",
        description="Test getting plan details",
        todo_items=["Detail task 1", "Detail task 2"]
    )
    
    create_data = json.loads(create_result)
    plan_id = create_data["plan_id"]
    
    # Get plan details
    get_result = await skill.get_plan(plan_id)
    get_data = json.loads(get_result)
    
    # Check structure
    assert "plan_id" in get_data
    assert "title" in get_data
    assert "description" in get_data
    assert "status" in get_data
    assert "progress" in get_data
    assert "todos" in get_data
    
    # Check values
    assert get_data["title"] == "Detail Test Plan"
    assert len(get_data["todos"]) == 2
    assert get_data["progress"] == "0/2"
    
    print("✅ Get plan test passed")


@pytest.mark.asyncio
async def test_list_plans():
    """Test listing all plans"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Initially no plans
    list_result = await skill.list_plans()
    list_data = json.loads(list_result)
    assert list_data["total_plans"] == 0
    
    # Create a couple of plans
    await skill.create_plan("Plan 1", "First plan", ["Task 1"])
    await skill.create_plan("Plan 2", "Second plan", ["Task 2"])
    
    # List plans again
    list_result = await skill.list_plans()
    list_data = json.loads(list_result)
    
    assert list_data["total_plans"] == 2
    assert len(list_data["plans"]) == 2
    
    # Check plan summaries
    plan_titles = [plan["title"] for plan in list_data["plans"]]
    assert "Plan 1" in plan_titles
    assert "Plan 2" in plan_titles
    
    print("✅ List plans test passed")


@pytest.mark.asyncio
async def test_delete_plan():
    """Test deleting a plan"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    
    # Create a plan
    create_result = await skill.create_plan(
        title="Delete Test Plan",
        description="Plan to be deleted",
        todo_items=["Temporary task"]
    )
    
    create_data = json.loads(create_result)
    plan_id = create_data["plan_id"]
    
    # Verify plan exists
    assert plan_id in skill.active_plans
    
    # Delete the plan
    delete_result = await skill.delete_plan(plan_id)
    delete_data = json.loads(delete_result)
    
    # Check result
    assert "message" in delete_data
    assert "deleted successfully" in delete_data["message"]
    
    # Verify plan is removed
    assert plan_id not in skill.active_plans
    
    print("✅ Delete plan test passed")


@pytest.mark.asyncio
async def test_skill_info():
    """Test getting skill information"""
    from webagents.agents.skills.core.planning import PlannerSkill
    
    skill = PlannerSkill()
    info = skill.get_skill_info()
    
    # Check required fields
    assert "name" in info
    assert "description" in info
    assert "version" in info
    assert "capabilities" in info
    assert "tools" in info
    assert "active_plans" in info
    assert "config" in info
    
    # Check specific values
    assert info["name"] == "PlannerSkill"
    assert len(info["tools"]) == 7  # All the tool methods
    assert info["active_plans"] == 0
    
    print("✅ Skill info test passed")


@pytest.mark.asyncio
async def test_data_structures():
    """Test TodoItem and TaskPlan data structures"""
    from webagents.agents.skills.core.planning import TodoItem, TaskPlan
    
    # Test TodoItem
    todo = TodoItem(
        id="test_todo",
        description="Test todo item"
    )
    
    assert todo.id == "test_todo"
    assert todo.description == "Test todo item"
    assert todo.status == "pending"
    assert todo.created_at is not None
    
    # Test TaskPlan
    plan = TaskPlan(
        id="test_plan",
        title="Test Plan",
        description="Test plan description",
        todos=[todo]
    )
    
    assert plan.id == "test_plan"
    assert plan.title == "Test Plan"
    assert plan.total_items == 1
    assert plan.completed_items == 0
    assert plan.progress == "0/1"
    
    print("✅ Data structures test passed")


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_planner_tests():
        print("🧪 Running PlannerSkill Tests...")
        
        tests = [
            test_planner_skill_import,
            test_planner_skill_initialization,
            test_create_plan,
            test_complete_todo,
            test_add_todo,
            test_update_todo_status,
            test_get_plan,
            test_list_plans,
            test_delete_plan,
            test_skill_info,
            test_data_structures
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                continue
        
        print("🎯 PlannerSkill Tests Complete!")
    
    asyncio.run(run_planner_tests()) 