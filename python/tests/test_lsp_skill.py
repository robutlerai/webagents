"""
LSP Skill Unit Tests

Tests for the Language Server Protocol skill:
- Language detection from file extension
- is_supported() function
- LSPSkill initialization
- Document symbols parsing (mocked)
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Check if multilspy is available
try:
    import multilspy
    HAS_MULTILSPY = True
except ImportError:
    HAS_MULTILSPY = False

# Skip marker for tests that require multilspy
requires_multilspy = pytest.mark.skipif(
    not HAS_MULTILSPY,
    reason="multilspy not installed - pip install multilspy"
)


# ===== Language Detection Tests =====

class TestLanguageDetection:
    """Test language detection from file extensions."""
    
    def test_detect_python_language(self):
        """Detect Python from .py extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("main.py") == "python"
        assert detect_language("script.pyw") == "python"
        assert detect_language("types.pyi") == "python"
        assert detect_language("/path/to/project/src/main.py") == "python"
    
    def test_detect_typescript_language(self):
        """Detect TypeScript from .ts/.tsx extensions."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("app.ts") == "typescript"
        assert detect_language("Component.tsx") == "typescript"
        assert detect_language("src/index.ts") == "typescript"
    
    def test_detect_javascript_language(self):
        """Detect JavaScript from .js/.jsx/.mjs/.cjs extensions."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("main.js") == "javascript"
        assert detect_language("Component.jsx") == "javascript"
        assert detect_language("module.mjs") == "javascript"
        assert detect_language("config.cjs") == "javascript"
    
    def test_detect_rust_language(self):
        """Detect Rust from .rs extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("main.rs") == "rust"
        assert detect_language("lib.rs") == "rust"
    
    def test_detect_go_language(self):
        """Detect Go from .go extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("main.go") == "go"
        assert detect_language("handler_test.go") == "go"
    
    def test_detect_java_language(self):
        """Detect Java from .java extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("Main.java") == "java"
        assert detect_language("com/example/Service.java") == "java"
    
    def test_detect_csharp_language(self):
        """Detect C# from .cs extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("Program.cs") == "csharp"
    
    def test_detect_kotlin_language(self):
        """Detect Kotlin from .kt/.kts extensions."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("Main.kt") == "kotlin"
        assert detect_language("build.gradle.kts") == "kotlin"
    
    def test_detect_ruby_language(self):
        """Detect Ruby from .rb extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("app.rb") == "ruby"
    
    def test_detect_dart_language(self):
        """Detect Dart from .dart extension."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("main.dart") == "dart"
    
    def test_detect_language_case_insensitive(self):
        """Language detection should be case-insensitive for extensions."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        assert detect_language("Main.PY") == "python"
        assert detect_language("App.TS") == "typescript"
        assert detect_language("Module.JS") == "javascript"
    
    def test_detect_unsupported_language_raises(self):
        """Unsupported extension should raise ValueError."""
        from webagents.agents.skills.local.lsp.languages import detect_language
        
        with pytest.raises(ValueError, match="Cannot detect language"):
            detect_language("file.xyz")
        
        with pytest.raises(ValueError, match="Cannot detect language"):
            detect_language("README.md")
        
        with pytest.raises(ValueError, match="Cannot detect language"):
            detect_language("no_extension")


# ===== is_supported Tests =====

class TestIsSupported:
    """Test is_supported() function."""
    
    def test_supported_languages_return_true(self):
        """Supported languages should return True."""
        from webagents.agents.skills.local.lsp.languages import is_supported
        
        assert is_supported("main.py") == True
        assert is_supported("app.ts") == True
        assert is_supported("index.js") == True
        assert is_supported("Main.java") == True
        assert is_supported("lib.rs") == True
        assert is_supported("main.go") == True
    
    def test_unsupported_languages_return_false(self):
        """Unsupported languages should return False."""
        from webagents.agents.skills.local.lsp.languages import is_supported
        
        assert is_supported("README.md") == False
        assert is_supported("config.yaml") == False
        assert is_supported("style.css") == False
        assert is_supported("index.html") == False
        assert is_supported("no_extension") == False
    
    def test_is_supported_case_insensitive(self):
        """is_supported should be case-insensitive."""
        from webagents.agents.skills.local.lsp.languages import is_supported
        
        assert is_supported("Main.PY") == True
        assert is_supported("App.TS") == True


# ===== SUPPORTED_LANGUAGES Tests =====

class TestSupportedLanguages:
    """Test SUPPORTED_LANGUAGES constant."""
    
    def test_all_supported_languages_exist(self):
        """Verify all documented languages are supported."""
        from webagents.agents.skills.local.lsp.languages import SUPPORTED_LANGUAGES
        
        expected_languages = [
            "python", "typescript", "javascript", "java",
            "rust", "go", "csharp", "dart", "ruby", "kotlin"
        ]
        
        for lang in expected_languages:
            assert lang in SUPPORTED_LANGUAGES, f"Missing language: {lang}"
    
    def test_extension_map_completeness(self):
        """Verify extension map covers all extensions."""
        from webagents.agents.skills.local.lsp.languages import (
            SUPPORTED_LANGUAGES,
            EXTENSION_MAP
        )
        
        # All extensions should be in the map
        for lang, exts in SUPPORTED_LANGUAGES.items():
            for ext in exts:
                assert ext in EXTENSION_MAP, f"Missing extension mapping: {ext}"
                assert EXTENSION_MAP[ext] == lang


class TestGetSupportedLanguages:
    """Test get_supported_languages function."""
    
    def test_returns_language_list(self):
        """get_supported_languages should return list of language names."""
        from webagents.agents.skills.local.lsp.languages import get_supported_languages
        
        languages = get_supported_languages()
        
        assert isinstance(languages, list)
        assert "python" in languages
        assert "typescript" in languages
        assert len(languages) >= 10


class TestGetExtensionsForLanguage:
    """Test get_extensions_for_language function."""
    
    def test_returns_extensions_for_python(self):
        """get_extensions_for_language should return Python extensions."""
        from webagents.agents.skills.local.lsp.languages import get_extensions_for_language
        
        exts = get_extensions_for_language("python")
        
        assert ".py" in exts
        assert ".pyw" in exts
        assert ".pyi" in exts
    
    def test_returns_extensions_for_javascript(self):
        """get_extensions_for_language should return JavaScript extensions."""
        from webagents.agents.skills.local.lsp.languages import get_extensions_for_language
        
        exts = get_extensions_for_language("javascript")
        
        assert ".js" in exts
        assert ".jsx" in exts
        assert ".mjs" in exts
        assert ".cjs" in exts
    
    def test_unsupported_language_raises(self):
        """Unsupported language should raise ValueError."""
        from webagents.agents.skills.local.lsp.languages import get_extensions_for_language
        
        with pytest.raises(ValueError, match="Unsupported language"):
            get_extensions_for_language("haskell")


# ===== LSPSkill Initialization Tests =====

@requires_multilspy
class TestLSPSkillInitialization:
    """Test LSPSkill initialization."""
    
    def test_lsp_skill_default_config(self):
        """Test LSPSkill with default configuration."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill()
        
        assert skill.project_root == Path(".").resolve()
        assert skill._servers == {}
    
    def test_lsp_skill_custom_project_root(self):
        """Test LSPSkill with custom project root."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/custom/path"})
        
        assert skill.project_root == Path("/custom/path").resolve()
    
    @pytest.mark.asyncio
    async def test_lsp_skill_initialize(self):
        """Test LSPSkill initialize method."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/tmp"})
        mock_agent = Mock()
        
        await skill.initialize(mock_agent)
        
        assert skill.agent == mock_agent
    
    @pytest.mark.asyncio
    async def test_lsp_skill_cleanup(self):
        """Test LSPSkill cleanup shuts down servers."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill()
        
        # Mock a server
        mock_server = Mock()
        mock_server.shutdown = Mock()
        skill._servers["python"] = mock_server
        
        await skill.cleanup()
        
        mock_server.shutdown.assert_called_once()
        assert len(skill._servers) == 0


# ===== LSPSkill Server Management Tests =====

@requires_multilspy
class TestLSPSkillServerManagement:
    """Test LSPSkill language server management."""
    
    def test_get_server_unsupported_language(self):
        """_get_server should raise for unsupported language."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill()
        
        with pytest.raises(ValueError, match="Unsupported language"):
            skill._get_server("haskell")
    
    def test_get_server_creates_server(self):
        """_get_server should create server on first call."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/tmp"})
        
        with patch("webagents.agents.skills.local.lsp.skill.SyncLanguageServer") as mock_ls:
            mock_server = Mock()
            mock_ls.create.return_value = mock_server
            
            with patch("webagents.agents.skills.local.lsp.skill.MultilspyConfig") as mock_config:
                mock_config.from_dict.return_value = Mock()
                
                server = skill._get_server("python")
                
                assert server == mock_server
                assert "python" in skill._servers
    
    def test_get_server_reuses_existing(self):
        """_get_server should reuse existing server."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill()
        mock_server = Mock()
        skill._servers["python"] = mock_server
        
        server = skill._get_server("python")
        
        assert server == mock_server


# ===== LSPSkill Path Resolution Tests =====

@requires_multilspy
class TestLSPSkillPathResolution:
    """Test LSPSkill path resolution methods."""
    
    def test_resolve_file_path_relative(self):
        """_resolve_file_path should resolve relative paths."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/project"})
        
        result = skill._resolve_file_path("src/main.py")
        
        assert result == str(Path("/project/src/main.py").resolve())
    
    def test_resolve_file_path_absolute(self):
        """_resolve_file_path should handle absolute paths."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/project"})
        
        result = skill._resolve_file_path("/absolute/path/main.py")
        
        assert result == str(Path("/absolute/path/main.py").resolve())
    
    def test_make_relative_from_uri(self):
        """_make_relative should convert file URI to relative path."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/project"})
        
        result = skill._make_relative("file:///project/src/main.py")
        
        assert result == "src/main.py"
    
    def test_make_relative_from_absolute(self):
        """_make_relative should convert absolute path to relative."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/project"})
        
        result = skill._make_relative("/project/src/main.py")
        
        assert result == "src/main.py"
    
    def test_make_relative_outside_project(self):
        """_make_relative should return original if outside project."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/project"})
        
        result = skill._make_relative("/other/path/main.py")
        
        assert result == "/other/path/main.py"


# ===== LSPSkill Tool Methods Tests (Mocked) =====

@requires_multilspy
class TestLSPSkillTools:
    """Test LSPSkill tool methods with mocked servers."""
    
    @pytest.fixture
    def skill_with_mock_server(self):
        """Create LSPSkill with mocked server."""
        from webagents.agents.skills.local.lsp import LSPSkill
        
        skill = LSPSkill({"project_root": "/project"})
        
        mock_server = Mock()
        skill._servers["python"] = mock_server
        
        # Patch _get_server to return mock
        original_get_server = skill._get_server
        def patched_get_server(lang):
            if lang == "python":
                return mock_server
            return original_get_server(lang)
        
        skill._get_server = patched_get_server
        
        return skill, mock_server
    
    @pytest.mark.asyncio
    async def test_goto_definition_found(self, skill_with_mock_server):
        """Test goto_definition when definition is found."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_definition.return_value = [{
            "uri": "file:///project/src/lib.py",
            "range": {
                "start": {"line": 10, "character": 4},
                "end": {"line": 10, "character": 20}
            }
        }]
        
        result = await skill.goto_definition("src/main.py", 5, 10, "python")
        
        assert result["found"] == True
        assert result["file"] == "src/lib.py"
        assert result["line"] == 11  # 1-indexed
        assert result["column"] == 5  # 1-indexed
    
    @pytest.mark.asyncio
    async def test_goto_definition_not_found(self, skill_with_mock_server):
        """Test goto_definition when no definition found."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_definition.return_value = None
        
        result = await skill.goto_definition("src/main.py", 5, 10, "python")
        
        assert result["found"] == False
        assert "No definition found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_find_references(self, skill_with_mock_server):
        """Test find_references returns list of references."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_references.return_value = [
            {"uri": "file:///project/src/a.py", "range": {"start": {"line": 5, "character": 0}}},
            {"uri": "file:///project/src/b.py", "range": {"start": {"line": 10, "character": 4}}},
        ]
        
        result = await skill.find_references("src/main.py", 5, 10, language="python")
        
        assert result["count"] == 2
        assert len(result["references"]) == 2
    
    @pytest.mark.asyncio
    async def test_find_references_empty(self, skill_with_mock_server):
        """Test find_references when no references found."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_references.return_value = []
        
        result = await skill.find_references("src/main.py", 5, 10, language="python")
        
        assert result["count"] == 0
        assert result["references"] == []
    
    @pytest.mark.asyncio
    async def test_get_document_symbols(self, skill_with_mock_server):
        """Test get_document_symbols returns flattened symbols."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_document_symbols.return_value = [
            {
                "name": "MyClass",
                "kind": 5,  # Class
                "range": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "my_method",
                        "kind": 6,  # Method
                        "range": {"start": {"line": 2, "character": 4}}
                    }
                ]
            },
            {
                "name": "helper_function",
                "kind": 12,  # Function
                "range": {"start": {"line": 10, "character": 0}}
            }
        ]
        
        result = await skill.get_document_symbols("src/main.py", language="python")
        
        assert result["count"] == 3  # Class + Method + Function
        
        symbols = result["symbols"]
        assert symbols[0]["name"] == "MyClass"
        assert symbols[0]["kind"] == "Class"
        assert symbols[1]["name"] == "my_method"
        assert symbols[1]["parent"] == "MyClass"
        assert symbols[2]["name"] == "helper_function"
    
    @pytest.mark.asyncio
    async def test_get_completions(self, skill_with_mock_server):
        """Test get_completions returns completion items."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_completions.return_value = [
            {"label": "print", "kind": 3, "detail": "function"},
            {"label": "len", "kind": 3, "detail": "function"},
        ]
        
        result = await skill.get_completions("src/main.py", 5, 10, language="python")
        
        assert result["count"] == 2
        assert result["completions"][0]["label"] == "print"
    
    @pytest.mark.asyncio
    async def test_get_hover(self, skill_with_mock_server):
        """Test get_hover returns hover information."""
        skill, mock_server = skill_with_mock_server
        
        mock_server.request_hover.return_value = {
            "contents": {"value": "(function) print(*args) -> None"}
        }
        
        result = await skill.get_hover("src/main.py", 5, 10, language="python")
        
        assert result["found"] == True
        assert "print" in result["content"]
    
    @pytest.mark.asyncio
    async def test_lsp_status_command(self, skill_with_mock_server):
        """Test /lsp status command."""
        skill, mock_server = skill_with_mock_server
        
        result = await skill.lsp_status()
        
        assert "project_root" in result
        assert "active_servers" in result
        assert "supported_languages" in result
        assert "python" in result["active_servers"]


# ===== Helper Function Tests =====

@requires_multilspy
class TestHelperFunctions:
    """Test LSP skill helper functions."""
    
    def test_symbol_kind_name_known(self):
        """Test _symbol_kind_name for known kinds."""
        from webagents.agents.skills.local.lsp.skill import _symbol_kind_name
        
        assert _symbol_kind_name(5) == "Class"
        assert _symbol_kind_name(6) == "Method"
        assert _symbol_kind_name(12) == "Function"
        assert _symbol_kind_name(13) == "Variable"
    
    def test_symbol_kind_name_unknown(self):
        """Test _symbol_kind_name for unknown kinds."""
        from webagents.agents.skills.local.lsp.skill import _symbol_kind_name
        
        result = _symbol_kind_name(999)
        assert "Unknown" in result
    
    def test_extract_documentation_string(self):
        """Test _extract_documentation with string input."""
        from webagents.agents.skills.local.lsp.skill import _extract_documentation
        
        assert _extract_documentation("Simple docs") == "Simple docs"
        assert _extract_documentation(None) is None
    
    def test_extract_documentation_dict(self):
        """Test _extract_documentation with dict input."""
        from webagents.agents.skills.local.lsp.skill import _extract_documentation
        
        result = _extract_documentation({"value": "Doc content"})
        assert result == "Doc content"
    
    def test_extract_hover_content_list(self):
        """Test _extract_hover_content with list input."""
        from webagents.agents.skills.local.lsp.skill import _extract_hover_content
        
        result = _extract_hover_content([
            "First part",
            {"value": "Second part"}
        ])
        
        assert "First part" in result
        assert "Second part" in result
    
    def test_flatten_symbols(self):
        """Test _flatten_symbols flattens nested structure."""
        from webagents.agents.skills.local.lsp.skill import _flatten_symbols
        
        nested = [
            {
                "name": "Parent",
                "kind": 5,
                "range": {"start": {"line": 0}},
                "children": [
                    {"name": "Child", "kind": 6, "range": {"start": {"line": 2}}}
                ]
            }
        ]
        
        flat = _flatten_symbols(nested)
        
        assert len(flat) == 2
        assert flat[0]["name"] == "Parent"
        assert flat[1]["name"] == "Child"
        assert flat[1]["parent"] == "Parent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
