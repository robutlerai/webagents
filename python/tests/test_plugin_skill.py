"""
Plugin Skill Unit Tests

Tests for the Plugin skill components:
- MarketplaceClient fuzzy search
- PluginLoader manifest validation
- SkillRunner frontmatter parsing
- SkillRunner $ARGUMENTS substitution
"""

import pytest
import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# ===== MarketplaceClient Tests =====

class TestMarketplaceClient:
    """Test MarketplaceClient fuzzy search and caching."""
    
    def test_marketplace_client_initialization(self):
        """Test MarketplaceClient initialization."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        
        assert client._index == {}
        assert client._last_refresh == 0
        assert client._github_token is None
    
    def test_marketplace_client_with_github_token(self):
        """Test MarketplaceClient with GitHub token."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient(github_token="test-token")
        
        assert client._github_token == "test-token"
    
    def test_needs_refresh_initially_true(self):
        """needs_refresh should return True initially."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        
        assert client.needs_refresh() == True
    
    def test_needs_refresh_after_refresh(self):
        """needs_refresh should return False after recent refresh."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._last_refresh = time.time()  # Just refreshed
        
        assert client.needs_refresh() == False
    
    def test_search_empty_index_returns_empty(self):
        """search on empty index should return empty list."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        
        results = client.search("test query")
        
        assert results == []
    
    def test_simple_search_fallback(self):
        """Test simple substring search when rapidfuzz unavailable."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._index = {
            "test-plugin": {
                "name": "test-plugin",
                "description": "A test plugin for testing",
                "stars": 100
            },
            "other-plugin": {
                "name": "other-plugin",
                "description": "Something else",
                "stars": 50
            }
        }
        
        # Use simple search fallback
        results = client._simple_search("test", limit=10)
        
        assert len(results) >= 1
        assert results[0]["name"] == "test-plugin"
    
    def test_search_with_rapidfuzz(self):
        """Test fuzzy search with rapidfuzz."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._index = {
            "code-formatter": {
                "name": "code-formatter",
                "description": "Format code automatically",
                "keywords": ["format", "lint"],
                "stars": 500
            },
            "code-linter": {
                "name": "code-linter",
                "description": "Lint code for errors",
                "keywords": ["lint", "check"],
                "stars": 300
            },
            "data-analyzer": {
                "name": "data-analyzer",
                "description": "Analyze data",
                "keywords": ["data"],
                "stars": 100
            }
        }
        
        try:
            from rapidfuzz import fuzz
            results = client.search("code format", limit=5)
            
            # Should find code-related plugins
            names = [r["name"] for r in results]
            assert "code-formatter" in names or "code-linter" in names
        except ImportError:
            # Fall back to simple search
            results = client._simple_search("code", limit=5)
            assert len(results) >= 1
    
    def test_search_star_ranking_boost(self):
        """Test that star count boosts ranking."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._index = {
            "popular-plugin": {
                "name": "popular-plugin",
                "description": "test plugin",
                "keywords": ["test"],
                "stars": 10000
            },
            "unpopular-plugin": {
                "name": "unpopular-plugin",
                "description": "test plugin",
                "keywords": ["test"],
                "stars": 5
            }
        }
        
        results = client._simple_search("test", limit=10)
        
        # Popular plugin should rank higher (sorted by stars as tiebreaker)
        assert results[0]["name"] == "popular-plugin"
    
    def test_get_plugin_by_name(self):
        """Test get() returns plugin by exact name."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._index = {
            "my-plugin": {"name": "my-plugin", "version": "1.0.0"}
        }
        
        result = client.get("my-plugin")
        assert result["name"] == "my-plugin"
        
        result = client.get("nonexistent")
        assert result is None
    
    def test_get_completions(self):
        """Test get_completions returns all plugin names."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._index = {
            "plugin-a": {},
            "plugin-b": {},
            "plugin-c": {}
        }
        
        completions = client.get_completions()
        
        assert len(completions) == 3
        assert "plugin-a" in completions
        assert "plugin-b" in completions
        assert "plugin-c" in completions
    
    def test_get_index_stats(self):
        """Test get_index_stats returns statistics."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._index = {"a": {}, "b": {}}
        client._last_refresh = time.time()
        
        stats = client.get_index_stats()
        
        assert stats["total_plugins"] == 2
        assert stats["last_refresh"] > 0
        assert stats["needs_refresh"] == False
    
    def test_load_cached_index_missing_file(self):
        """load_cached_index should return False if no cache file."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        client = MarketplaceClient()
        client._cache_file = Path("/nonexistent/cache.json")
        
        result = client.load_cached_index()
        
        assert result == False
    
    def test_load_cached_index_success(self):
        """load_cached_index should load from valid cache file."""
        from webagents.agents.skills.local.plugin import MarketplaceClient
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            cache_file.write_text(json.dumps({
                "timestamp": time.time(),
                "plugins": {
                    "cached-plugin": {"name": "cached-plugin"}
                }
            }))
            
            client = MarketplaceClient()
            client._cache_file = cache_file
            
            result = client.load_cached_index()
            
            assert result == True
            assert "cached-plugin" in client._index


# ===== PluginLoader Tests =====

class TestPluginLoader:
    """Test PluginLoader manifest validation and loading."""
    
    def test_plugin_loader_initialization(self):
        """Test PluginLoader initialization."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        loader = PluginLoader()
        
        assert loader.plugins_dir.exists()
    
    def test_plugin_loader_custom_dir(self):
        """Test PluginLoader with custom plugins directory."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PluginLoader({"plugins_dir": tmpdir})
            
            assert loader.plugins_dir == Path(tmpdir)
    
    def test_load_local_missing_manifest(self):
        """load_local should raise if plugin.json missing."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        loader = PluginLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No plugin.json"):
                loader.load_local(Path(tmpdir))
    
    def test_load_local_valid_plugin(self):
        """load_local should load valid plugin."""
        from webagents.agents.skills.local.plugin import PluginLoader, Plugin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "plugin.json").write_text(json.dumps({
                "name": "test-plugin",
                "version": "1.0.0",
                "description": "A test plugin"
            }))
            
            loader = PluginLoader()
            plugin = loader.load_local(plugin_dir)
            
            assert isinstance(plugin, Plugin)
            assert plugin.name == "test-plugin"
            assert plugin.version == "1.0.0"
            assert plugin.description == "A test plugin"
    
    def test_list_installed_empty(self):
        """list_installed should return empty list if no plugins."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PluginLoader({"plugins_dir": tmpdir})
            
            plugins = loader.list_installed()
            
            assert plugins == []
    
    def test_list_installed_with_plugins(self):
        """list_installed should return installed plugins."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create plugin directory
            plugin_dir = Path(tmpdir) / "my-plugin"
            plugin_dir.mkdir()
            (plugin_dir / "plugin.json").write_text(json.dumps({
                "name": "my-plugin",
                "version": "1.0.0"
            }))
            
            loader = PluginLoader({"plugins_dir": tmpdir})
            plugins = loader.list_installed()
            
            assert len(plugins) == 1
            assert plugins[0].name == "my-plugin"
    
    def test_uninstall_nonexistent(self):
        """uninstall should return False for nonexistent plugin."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PluginLoader({"plugins_dir": tmpdir})
            
            result = loader.uninstall("nonexistent")
            
            assert result == False
    
    def test_uninstall_existing(self):
        """uninstall should remove existing plugin."""
        from webagents.agents.skills.local.plugin import PluginLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "remove-me"
            plugin_dir.mkdir()
            (plugin_dir / "plugin.json").write_text("{}")
            
            loader = PluginLoader({"plugins_dir": tmpdir})
            result = loader.uninstall("remove-me")
            
            assert result == True
            assert not plugin_dir.exists()


# ===== Manifest Validation Tests =====

class TestManifestValidation:
    """Test plugin manifest validation."""
    
    def test_validate_manifest_missing_name(self):
        """validate_manifest should raise for missing name."""
        from webagents.agents.skills.local.plugin import validate_manifest
        from webagents.agents.skills.local.plugin.schema import ManifestValidationError
        
        with pytest.raises(ManifestValidationError, match="Missing required field: name"):
            validate_manifest({})
    
    def test_validate_manifest_empty_name(self):
        """validate_manifest should raise for empty name."""
        from webagents.agents.skills.local.plugin import validate_manifest
        from webagents.agents.skills.local.plugin.schema import ManifestValidationError
        
        with pytest.raises(ManifestValidationError, match="Missing required field: name"):
            validate_manifest({"name": ""})
    
    def test_validate_manifest_invalid_name_format(self):
        """validate_manifest should raise for invalid name format."""
        from webagents.agents.skills.local.plugin import validate_manifest
        from webagents.agents.skills.local.plugin.schema import ManifestValidationError
        
        with pytest.raises(ManifestValidationError, match="Invalid plugin name"):
            validate_manifest({"name": "123-invalid"})  # Must start with letter
        
        with pytest.raises(ManifestValidationError, match="Invalid plugin name"):
            validate_manifest({"name": "has spaces"})
    
    def test_validate_manifest_valid_names(self):
        """validate_manifest should accept valid names."""
        from webagents.agents.skills.local.plugin import validate_manifest
        
        # These should all be valid
        validate_manifest({"name": "my-plugin"})
        validate_manifest({"name": "MyPlugin"})
        validate_manifest({"name": "plugin_v2"})
        validate_manifest({"name": "a"})
    
    def test_validate_manifest_invalid_version(self):
        """validate_manifest should raise for invalid version format."""
        from webagents.agents.skills.local.plugin import validate_manifest
        from webagents.agents.skills.local.plugin.schema import ManifestValidationError
        
        with pytest.raises(ManifestValidationError, match="Invalid version"):
            validate_manifest({"name": "test", "version": "invalid"})
    
    def test_validate_manifest_valid_version(self):
        """validate_manifest should accept valid semver versions."""
        from webagents.agents.skills.local.plugin import validate_manifest
        
        validate_manifest({"name": "test", "version": "1.0.0"})
        validate_manifest({"name": "test", "version": "0.0.1"})
        validate_manifest({"name": "test", "version": "10.20.30"})
        validate_manifest({"name": "test", "version": "1.0.0-beta"})  # Pre-release
    
    def test_validate_manifest_invalid_dependencies(self):
        """validate_manifest should raise for invalid dependencies."""
        from webagents.agents.skills.local.plugin import validate_manifest
        from webagents.agents.skills.local.plugin.schema import ManifestValidationError
        
        with pytest.raises(ManifestValidationError, match="dependencies"):
            validate_manifest({"name": "test", "dependencies": "not-a-list"})
        
        with pytest.raises(ManifestValidationError, match="Invalid dependency"):
            validate_manifest({"name": "test", "dependencies": [123]})
    
    def test_validate_manifest_valid_dependencies(self):
        """validate_manifest should accept valid dependencies."""
        from webagents.agents.skills.local.plugin import validate_manifest
        
        manifest = validate_manifest({
            "name": "test",
            "dependencies": ["requests>=2.0", "pyyaml"]
        })
        
        assert manifest.dependencies == ["requests>=2.0", "pyyaml"]
    
    def test_validate_manifest_invalid_keywords(self):
        """validate_manifest should raise for invalid keywords."""
        from webagents.agents.skills.local.plugin import validate_manifest
        from webagents.agents.skills.local.plugin.schema import ManifestValidationError
        
        with pytest.raises(ManifestValidationError, match="keywords"):
            validate_manifest({"name": "test", "keywords": "not-a-list"})


# ===== PluginManifest Tests =====

class TestPluginManifest:
    """Test PluginManifest dataclass."""
    
    def test_manifest_from_dict(self):
        """Test PluginManifest.from_dict()."""
        from webagents.agents.skills.local.plugin import PluginManifest
        
        manifest = PluginManifest.from_dict({
            "name": "my-plugin",
            "version": "1.2.3",
            "description": "Test plugin",
            "author": "Test Author",
            "repository": "https://github.com/test/plugin"
        })
        
        assert manifest.name == "my-plugin"
        assert manifest.version == "1.2.3"
        assert manifest.description == "Test plugin"
        assert manifest.author == "Test Author"
        assert manifest.repository == "https://github.com/test/plugin"
    
    def test_manifest_default_values(self):
        """Test PluginManifest default values."""
        from webagents.agents.skills.local.plugin import PluginManifest
        
        manifest = PluginManifest.from_dict({"name": "test"})
        
        assert manifest.version == "0.0.0"
        assert manifest.commands == "./commands/"
        assert manifest.skills == "./skills/"
        assert manifest.agents == "./agents/"
        assert manifest.dependencies == []
        assert manifest.keywords == []
    
    def test_manifest_to_dict(self):
        """Test PluginManifest.to_dict()."""
        from webagents.agents.skills.local.plugin import PluginManifest
        
        manifest = PluginManifest.from_dict({
            "name": "test",
            "version": "1.0.0"
        })
        
        result = manifest.to_dict()
        
        assert result["name"] == "test"
        assert result["version"] == "1.0.0"
        assert "commands" in result
        assert "skills" in result


# ===== SkillRunner Tests =====

class TestSkillRunnerFrontmatter:
    """Test SkillRunner frontmatter parsing."""
    
    def test_parse_skill_without_frontmatter(self):
        """parse_string should handle content without frontmatter."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        skill = runner.parse_string("# Just markdown content\n\nNo frontmatter here.")
        
        assert skill.name == "unnamed"
        assert skill.description == ""
        assert skill.content == "# Just markdown content\n\nNo frontmatter here."
    
    def test_parse_skill_with_frontmatter(self):
        """parse_string should parse YAML frontmatter."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = """---
name: my-skill
description: A test skill
context: inline
---
# Skill Content

This is the skill body.
"""
        skill = runner.parse_string(content)
        
        assert skill.name == "my-skill"
        assert skill.description == "A test skill"
        assert skill.context == "inline"
        assert "# Skill Content" in skill.content
    
    def test_parse_skill_allowed_tools_list(self):
        """parse_string should parse allowed-tools as list."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = """---
name: restricted-skill
allowed-tools: [read_file, write_file]
context: fork
---
Content
"""
        skill = runner.parse_string(content)
        
        assert skill.allowed_tools == ["read_file", "write_file"]
        assert skill.context == "fork"
    
    def test_parse_skill_allowed_tools_string(self):
        """parse_string should parse allowed-tools as comma-separated string."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = """---
name: restricted-skill
allowed-tools: "read_file, write_file, execute"
---
Content
"""
        skill = runner.parse_string(content)
        
        assert skill.allowed_tools == ["read_file", "write_file", "execute"]
    
    def test_parse_skill_disable_model_invocation(self):
        """parse_string should parse disable-model-invocation flag."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = """---
name: no-llm-skill
disable-model-invocation: true
---
Run without LLM
"""
        skill = runner.parse_string(content)
        
        assert skill.disable_model_invocation == True
    
    def test_parse_skill_from_file(self):
        """parse should load skill from file."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""---
name: file-skill
description: From file
---
# File Content
""")
            f.flush()
            
            runner = SkillRunner()
            skill = runner.parse(Path(f.name))
            
            assert skill.name == "file-skill"
            assert skill.path == Path(f.name)
    
    def test_parse_missing_file_raises(self):
        """parse should raise for missing file."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        
        with pytest.raises(FileNotFoundError):
            runner.parse(Path("/nonexistent/skill.md"))


class TestSkillRunnerArgumentSubstitution:
    """Test SkillRunner $ARGUMENTS substitution."""
    
    def test_substitute_dot_notation(self):
        """substitute_arguments should replace $ARGUMENTS.key."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "Hello, $ARGUMENTS.name! Your file is $ARGUMENTS.filename."
        
        result = runner.substitute_arguments(content, {
            "name": "World",
            "filename": "test.py"
        })
        
        assert result == "Hello, World! Your file is test.py."
    
    def test_substitute_bracket_notation_single_quotes(self):
        """substitute_arguments should replace $ARGUMENTS['key']."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "Value: $ARGUMENTS['key1'] and $ARGUMENTS['key2']"
        
        result = runner.substitute_arguments(content, {
            "key1": "first",
            "key2": "second"
        })
        
        assert result == "Value: first and second"
    
    def test_substitute_bracket_notation_double_quotes(self):
        """substitute_arguments should replace $ARGUMENTS["key"]."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = 'Path: $ARGUMENTS["filepath"]'
        
        result = runner.substitute_arguments(content, {"filepath": "/home/user"})
        
        assert result == "Path: /home/user"
    
    def test_substitute_missing_argument_preserved(self):
        """substitute_arguments should preserve unmatched placeholders."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "Found: $ARGUMENTS.found, Missing: $ARGUMENTS.missing"
        
        result = runner.substitute_arguments(content, {"found": "yes"})
        
        assert result == "Found: yes, Missing: $ARGUMENTS.missing"
    
    def test_substitute_mixed_notations(self):
        """substitute_arguments should handle mixed notations."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = """
Use $ARGUMENTS.name to process $ARGUMENTS['file'].
The output goes to $ARGUMENTS["output_dir"].
"""
        
        result = runner.substitute_arguments(content, {
            "name": "processor",
            "file": "input.txt",
            "output_dir": "/tmp/out"
        })
        
        assert "processor" in result
        assert "input.txt" in result
        assert "/tmp/out" in result
    
    def test_substitute_non_string_values(self):
        """substitute_arguments should convert non-string values."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "Count: $ARGUMENTS.count, Flag: $ARGUMENTS.enabled"
        
        result = runner.substitute_arguments(content, {
            "count": 42,
            "enabled": True
        })
        
        assert result == "Count: 42, Flag: True"


class TestSkillRunnerGetRequiredArguments:
    """Test SkillRunner.get_required_arguments()."""
    
    def test_get_required_arguments_empty(self):
        """get_required_arguments should return empty for no placeholders."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        
        args = runner.get_required_arguments("No placeholders here")
        
        assert args == []
    
    def test_get_required_arguments_dot_notation(self):
        """get_required_arguments should find dot notation placeholders."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "$ARGUMENTS.name and $ARGUMENTS.value"
        
        args = runner.get_required_arguments(content)
        
        assert sorted(args) == ["name", "value"]
    
    def test_get_required_arguments_bracket_notation(self):
        """get_required_arguments should find bracket notation placeholders."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "$ARGUMENTS['first'] and $ARGUMENTS[\"second\"]"
        
        args = runner.get_required_arguments(content)
        
        assert sorted(args) == ["first", "second"]
    
    def test_get_required_arguments_deduplication(self):
        """get_required_arguments should deduplicate repeated placeholders."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = "$ARGUMENTS.name appears twice: $ARGUMENTS.name"
        
        args = runner.get_required_arguments(content)
        
        assert args == ["name"]


class TestSkillRunnerValidation:
    """Test SkillRunner.validate_skill()."""
    
    def test_validate_skill_valid(self):
        """validate_skill should return no warnings for valid skill."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner, SkillMD
        
        runner = SkillRunner()
        skill = SkillMD(
            name="valid-skill",
            description="A valid skill",
            content="# Content here",
            frontmatter={},
            context="inline"
        )
        
        warnings = runner.validate_skill(skill)
        
        assert warnings == []
    
    def test_validate_skill_missing_name(self):
        """validate_skill should warn about missing name."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner, SkillMD
        
        runner = SkillRunner()
        skill = SkillMD(
            name="",
            description="",
            content="Content",
            frontmatter={}
        )
        
        warnings = runner.validate_skill(skill)
        
        assert any("no name" in w for w in warnings)
    
    def test_validate_skill_empty_content(self):
        """validate_skill should warn about empty content."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner, SkillMD
        
        runner = SkillRunner()
        skill = SkillMD(
            name="empty",
            description="",
            content="   ",  # Whitespace only
            frontmatter={}
        )
        
        warnings = runner.validate_skill(skill)
        
        assert any("no content" in w for w in warnings)
    
    def test_validate_skill_invalid_context(self):
        """validate_skill should warn about invalid context."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner, SkillMD
        
        runner = SkillRunner()
        skill = SkillMD(
            name="test",
            description="",
            content="Content",
            frontmatter={},
            context="invalid"
        )
        
        warnings = runner.validate_skill(skill)
        
        assert any("Invalid context" in w for w in warnings)
    
    def test_validate_skill_allowed_tools_wrong_context(self):
        """validate_skill should warn if allowed_tools used without fork context."""
        from webagents.agents.skills.local.plugin.components.skill_runner import SkillRunner, SkillMD
        
        runner = SkillRunner()
        skill = SkillMD(
            name="test",
            description="",
            content="Content",
            frontmatter={},
            context="inline",
            allowed_tools=["tool1", "tool2"]
        )
        
        warnings = runner.validate_skill(skill)
        
        assert any("allowed-tools" in w and "fork" in w for w in warnings)


# ===== PluginExecutor Tests =====

class TestPluginExecutor:
    """Test PluginExecutor."""
    
    def test_executor_initialization(self):
        """Test PluginExecutor initialization."""
        from webagents.agents.skills.local.plugin import PluginExecutor
        
        executor = PluginExecutor()
        
        assert executor.sandbox_enabled == False
        assert executor.timeout == 30
    
    def test_executor_custom_config(self):
        """Test PluginExecutor with custom configuration."""
        from webagents.agents.skills.local.plugin import PluginExecutor
        
        executor = PluginExecutor({
            "sandbox": True,
            "timeout": 60,
            "python_path": "/custom/python"
        })
        
        assert executor.sandbox_enabled == True
        assert executor.timeout == 60
        assert executor.python_path == "/custom/python"


# ===== Plugin Class Tests =====

class TestPlugin:
    """Test Plugin dataclass."""
    
    def test_plugin_get_tools_empty(self):
        """get_tools should return empty list if no commands/skills."""
        from webagents.agents.skills.local.plugin.loader import Plugin
        from webagents.agents.skills.local.plugin.schema import PluginManifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = PluginManifest.from_dict({
                "name": "empty-plugin",
                "version": "1.0.0"
            })
            
            plugin = Plugin(
                name="empty-plugin",
                version="1.0.0",
                description="",
                path=Path(tmpdir),
                manifest=manifest
            )
            
            tools = plugin.get_tools()
            
            assert tools == []
    
    def test_plugin_get_tools_with_commands(self):
        """get_tools should discover Python commands."""
        from webagents.agents.skills.local.plugin.loader import Plugin
        from webagents.agents.skills.local.plugin.schema import PluginManifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            commands_dir = plugin_dir / "commands"
            commands_dir.mkdir()
            (commands_dir / "my_command.py").write_text("# command")
            (commands_dir / "_private.py").write_text("# private")
            
            manifest = PluginManifest.from_dict({
                "name": "test-plugin",
                "commands": "./commands/"
            })
            
            plugin = Plugin(
                name="test-plugin",
                version="1.0.0",
                description="",
                path=plugin_dir,
                manifest=manifest
            )
            
            tools = plugin.get_tools()
            
            # Should find my_command but not _private
            assert len(tools) == 1
            assert tools[0]["name"] == "test-plugin:my_command"
            assert tools[0]["type"] == "command"
    
    def test_plugin_get_tools_with_skills(self):
        """get_tools should discover SKILL.md files."""
        from webagents.agents.skills.local.plugin.loader import Plugin
        from webagents.agents.skills.local.plugin.schema import PluginManifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            skills_dir = plugin_dir / "skills"
            skills_dir.mkdir()
            (skills_dir / "my_skill.md").write_text("# skill")
            
            manifest = PluginManifest.from_dict({
                "name": "test-plugin",
                "skills": "./skills/"
            })
            
            plugin = Plugin(
                name="test-plugin",
                version="1.0.0",
                description="",
                path=plugin_dir,
                manifest=manifest
            )
            
            tools = plugin.get_tools()
            
            assert len(tools) == 1
            assert tools[0]["name"] == "test-plugin:my_skill"
            assert tools[0]["type"] == "skill"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
