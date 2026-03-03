"""
Tests for FilesystemSkill sandboxing
"""

import pytest
from pathlib import Path
from webagents.agents.skills.local.filesystem.skill import FilesystemSkill


@pytest.mark.asyncio
async def test_filesystem_skill_read_allowed(tmp_path):
    """Test reading from allowed directory"""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")
    
    # Create skill with whitelist
    skill = FilesystemSkill({
        "whitelist": [str(tmp_path)]
    })
    
    # Read file
    content = await skill.read_file(str(test_file))
    assert "Hello, World!" in content


@pytest.mark.asyncio
async def test_filesystem_skill_read_denied(tmp_path):
    """Test reading from denied directory"""
    # Create test file outside allowed directory
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    test_file = other_dir / "secret.txt"
    test_file.write_text("Secret data")
    
    # Create skill with limited whitelist
    skill = FilesystemSkill({
        "whitelist": [str(tmp_path / "allowed")]
    })
    
    # Try to read file
    result = await skill.read_file(str(test_file))
    assert "Access denied" in result


@pytest.mark.asyncio
async def test_filesystem_skill_blacklist(tmp_path):
    """Test blacklist blocking access"""
    # Create test file
    blocked_dir = tmp_path / "secrets"
    blocked_dir.mkdir()
    test_file = blocked_dir / "key.txt"
    test_file.write_text("Secret key")
    
    # Create skill with blacklist
    skill = FilesystemSkill({
        "whitelist": [str(tmp_path)],
        "blacklist": [str(blocked_dir)]
    })
    
    # Try to read file
    result = await skill.read_file(str(test_file))
    assert "Access denied" in result


@pytest.mark.asyncio
async def test_filesystem_skill_write_allowed(tmp_path):
    """Test writing to allowed directory"""
    test_file = tmp_path / "output.txt"
    
    skill = FilesystemSkill({
        "whitelist": [str(tmp_path)]
    })
    
    # Write file
    result = await skill.write_file(str(test_file), "Test content")
    assert "Successfully" in result
    assert test_file.exists()
    assert test_file.read_text() == "Test content"


@pytest.mark.asyncio
async def test_filesystem_skill_list_files(tmp_path):
    """Test listing files"""
    # Create test files
    (tmp_path / "file1.txt").write_text("1")
    (tmp_path / "file2.txt").write_text("2")
    (tmp_path / "subdir").mkdir()
    
    skill = FilesystemSkill({
        "whitelist": [str(tmp_path)]
    })
    
    # List files (method is list_directory, not list_files)
    result = await skill.list_directory(str(tmp_path))
    assert "file1.txt" in result
    assert "file2.txt" in result
    assert "subdir" in result


@pytest.mark.asyncio
async def test_filesystem_skill_search_files(tmp_path):
    """Test searching files with glob pattern"""
    # Create test files
    (tmp_path / "test.py").write_text("1")
    (tmp_path / "test.txt").write_text("2")
    (tmp_path / "main.py").write_text("3")
    
    skill = FilesystemSkill({
        "whitelist": [str(tmp_path)]
    })
    
    # Search for Python files (method is glob, not search_files)
    result = await skill.glob("*.py", str(tmp_path))
    assert "test.py" in result
    assert "main.py" in result
    assert "test.txt" not in result
