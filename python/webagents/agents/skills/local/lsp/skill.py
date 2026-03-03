"""
LSP Skill - Language Server Protocol Code Intelligence

Provides code intelligence capabilities using Microsoft multilspy:
- Go to definition
- Find references
- Code completions
- Hover information
- Document symbols

Supports: Python, TypeScript, JavaScript, Java, Rust, Go, C#, Dart, Ruby, Kotlin
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from multilspy import SyncLanguageServer
    from multilspy.multilspy_config import MultilspyConfig
    LSP_AVAILABLE = True
except ImportError:
    LSP_AVAILABLE = False

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command, tool

from .languages import SUPPORTED_LANGUAGES, detect_language, is_supported


class LSPSkill(Skill):
    """Code intelligence via Language Server Protocol.
    
    Wraps Microsoft multilspy to provide LSP features with:
    - Automatic language detection from file extension
    - Lazy initialization of language servers
    - Proper lifecycle management
    
    Configuration:
        project_root: Path to the project root directory (default: ".")
    
    Example:
        skill = LSPSkill({"project_root": "/path/to/project"})
        await skill.initialize(agent)
        
        # Find definition
        result = await skill.goto_definition("src/main.py", 10, 5)
        
        # Cleanup when done
        await skill.cleanup()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not LSP_AVAILABLE:
            raise ImportError(
                "LSP skill requires multilspy. "
                "Install with: pip install webagents[local]"
            )
        self.project_root = Path(self.config.get("project_root", ".")).resolve()
        self._servers: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, agent) -> None:
        """Initialize skill with agent reference.
        
        Args:
            agent: The BaseAgent instance
        """
        await super().initialize(agent)
        self.logger.info(f"LSP Skill initialized for project: {self.project_root}")
    
    async def cleanup(self) -> None:
        """Shutdown all active language servers."""
        for lang, server in self._servers.items():
            try:
                server.shutdown()
                self.logger.info(f"Shutdown {lang} language server")
            except Exception as e:
                self.logger.warning(f"Error shutting down {lang} server: {e}")
        self._servers.clear()
    
    def _get_server(self, language: str) -> SyncLanguageServer:
        """Get or create a language server for the given language.
        
        Language servers are lazily initialized on first use.
        
        Args:
            language: Language identifier (e.g., "python", "typescript")
            
        Returns:
            SyncLanguageServer instance
            
        Raises:
            ValueError: If language is not supported
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        if language not in self._servers:
            config = MultilspyConfig.from_dict({"code_language": language})
            self._servers[language] = SyncLanguageServer.create(
                config, self.logger, str(self.project_root)
            )
            self.logger.info(f"Started {language} language server")
        
        return self._servers[language]
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language identifier
        """
        return detect_language(file_path)
    
    def _resolve_file_path(self, file_path: str) -> str:
        """Resolve file path relative to project root.
        
        Args:
            file_path: Relative or absolute file path
            
        Returns:
            Absolute file path as string
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        return str(path.resolve())
    
    def _make_relative(self, uri: str) -> str:
        """Convert URI or absolute path to relative path.
        
        Args:
            uri: File URI or path
            
        Returns:
            Path relative to project root
        """
        # Handle file:// URIs
        if uri.startswith("file://"):
            uri = uri[7:]
        
        try:
            return str(Path(uri).relative_to(self.project_root))
        except ValueError:
            return uri
    
    @tool(description="Go to the definition of a symbol at a given position")
    async def goto_definition(
        self,
        file_path: str,
        line: int,
        column: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find the definition of a symbol.
        
        Args:
            file_path: Path to the file (relative to project root)
            line: Line number (1-indexed)
            column: Column number (1-indexed)
            language: Language (auto-detected if not provided)
        
        Returns:
            Definition location with file, line, column, or not found message
        """
        language = language or self._detect_language(file_path)
        server = self._get_server(language)
        
        abs_path = self._resolve_file_path(file_path)
        
        # Convert to 0-indexed for LSP
        result = server.request_definition(
            abs_path,
            line - 1,
            column - 1
        )
        
        if not result:
            return {"found": False, "message": "No definition found"}
        
        # Handle result - can be dict or list
        if isinstance(result, list):
            result = result[0] if result else None
        
        if not result:
            return {"found": False, "message": "No definition found"}
        
        # Extract location info
        uri = result.get("uri", result.get("targetUri", ""))
        range_info = result.get("range", result.get("targetRange", {}))
        start = range_info.get("start", {})
        
        rel_path = self._make_relative(uri)
        result_line = start.get("line", 0) + 1
        result_col = start.get("character", 0) + 1
        
        return {
            "found": True,
            "file": rel_path,
            "line": result_line,
            "column": result_col,
            "display": f"Definition at {rel_path}:{result_line}:{result_col}"
        }
    
    @tool(description="Find all references to a symbol at a given position")
    async def find_references(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find all references to a symbol.
        
        Args:
            file_path: Path to the file (relative to project root)
            line: Line number (1-indexed)
            column: Column number (1-indexed)
            include_declaration: Include the declaration in results
            language: Language (auto-detected if not provided)
        
        Returns:
            List of reference locations with count
        """
        language = language or self._detect_language(file_path)
        server = self._get_server(language)
        
        abs_path = self._resolve_file_path(file_path)
        
        # Convert to 0-indexed for LSP
        results = server.request_references(
            abs_path,
            line - 1,
            column - 1,
            include_declaration
        )
        
        if not results:
            return {"count": 0, "references": [], "message": "No references found"}
        
        refs = []
        for ref in results:
            uri = ref.get("uri", "")
            range_info = ref.get("range", {})
            start = range_info.get("start", {})
            
            refs.append({
                "file": self._make_relative(uri),
                "line": start.get("line", 0) + 1,
                "column": start.get("character", 0) + 1,
            })
        
        return {
            "count": len(refs),
            "references": refs,
            "display": f"Found {len(refs)} reference(s)"
        }
    
    @tool(description="Get code completions at a given position")
    async def get_completions(
        self,
        file_path: str,
        line: int,
        column: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get code completions at cursor position.
        
        Args:
            file_path: Path to the file (relative to project root)
            line: Line number (1-indexed)
            column: Column number (1-indexed)
            language: Language (auto-detected if not provided)
        
        Returns:
            List of completion items (limited to 20)
        """
        language = language or self._detect_language(file_path)
        server = self._get_server(language)
        
        abs_path = self._resolve_file_path(file_path)
        
        # Convert to 0-indexed for LSP
        results = server.request_completions(
            abs_path,
            line - 1,
            column - 1
        )
        
        if not results:
            return {"count": 0, "completions": []}
        
        # Handle CompletionList vs list of CompletionItems
        items = results
        if isinstance(results, dict):
            items = results.get("items", [])
        
        completions = []
        for item in items[:20]:  # Limit to 20 items
            completions.append({
                "label": item.get("label"),
                "kind": item.get("kind"),
                "detail": item.get("detail"),
                "documentation": _extract_documentation(item.get("documentation")),
            })
        
        return {
            "count": len(completions),
            "completions": completions,
            "display": f"Found {len(completions)} completion(s)"
        }
    
    @tool(description="Get hover information for a symbol at a given position")
    async def get_hover(
        self,
        file_path: str,
        line: int,
        column: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get hover information (type info, docs) for a symbol.
        
        Args:
            file_path: Path to the file (relative to project root)
            line: Line number (1-indexed)
            column: Column number (1-indexed)
            language: Language (auto-detected if not provided)
        
        Returns:
            Hover information including type and documentation
        """
        language = language or self._detect_language(file_path)
        server = self._get_server(language)
        
        abs_path = self._resolve_file_path(file_path)
        
        # Convert to 0-indexed for LSP
        result = server.request_hover(
            abs_path,
            line - 1,
            column - 1
        )
        
        if not result:
            return {"found": False, "message": "No hover information"}
        
        contents = result.get("contents", {})
        value = _extract_hover_content(contents)
        
        if not value:
            return {"found": False, "message": "No hover information"}
        
        return {
            "found": True,
            "content": value,
            "display": value[:500] + "..." if len(value) > 500 else value
        }
    
    @tool(description="Get all symbols defined in a document")
    async def get_document_symbols(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get all symbols (functions, classes, variables) in a file.
        
        Args:
            file_path: Path to the file (relative to project root)
            language: Language (auto-detected if not provided)
        
        Returns:
            List of symbols with names, kinds, and locations
        """
        language = language or self._detect_language(file_path)
        server = self._get_server(language)
        
        abs_path = self._resolve_file_path(file_path)
        
        results = server.request_document_symbols(abs_path)
        
        if not results:
            return {"count": 0, "symbols": []}
        
        symbols = _flatten_symbols(results)
        
        return {
            "count": len(symbols),
            "symbols": symbols,
            "display": f"Found {len(symbols)} symbol(s)"
        }
    
    @command("/lsp", description="Show LSP skill status and active servers")
    async def lsp_status(self) -> Dict[str, Any]:
        """Show LSP skill status."""
        active_servers = list(self._servers.keys())
        return {
            "project_root": str(self.project_root),
            "active_servers": active_servers,
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "display": f"""**LSP Status**
Project: {self.project_root}
Active servers: {', '.join(active_servers) or 'None'}
Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"""
        }


# Helper functions

def _extract_documentation(doc: Any) -> Optional[str]:
    """Extract documentation string from various formats."""
    if doc is None:
        return None
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return doc.get("value", str(doc))
    return str(doc)


def _extract_hover_content(contents: Any) -> str:
    """Extract hover content from various formats."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return contents.get("value", "")
    if isinstance(contents, list):
        parts = []
        for c in contents:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                parts.append(c.get("value", str(c)))
            else:
                parts.append(str(c))
        return "\n".join(parts)
    return str(contents)


def _flatten_symbols(symbols: List[Dict], parent: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flatten nested document symbols into a flat list.
    
    Args:
        symbols: List of document symbols (may have children)
        parent: Parent symbol name for nested symbols
        
    Returns:
        Flat list of symbol dictionaries
    """
    flat = []
    for sym in symbols:
        # Extract range - handle both DocumentSymbol and SymbolInformation formats
        range_info = sym.get("range", sym.get("location", {}).get("range", {}))
        start = range_info.get("start", {})
        
        flat.append({
            "name": sym.get("name"),
            "kind": _symbol_kind_name(sym.get("kind")),
            "parent": parent,
            "line": start.get("line", 0) + 1,
        })
        
        # Recursively flatten children
        if "children" in sym:
            flat.extend(_flatten_symbols(sym["children"], sym.get("name")))
    
    return flat


def _symbol_kind_name(kind: Optional[int]) -> str:
    """Convert LSP SymbolKind to human-readable name."""
    SYMBOL_KINDS = {
        1: "File",
        2: "Module",
        3: "Namespace",
        4: "Package",
        5: "Class",
        6: "Method",
        7: "Property",
        8: "Field",
        9: "Constructor",
        10: "Enum",
        11: "Interface",
        12: "Function",
        13: "Variable",
        14: "Constant",
        15: "String",
        16: "Number",
        17: "Boolean",
        18: "Array",
        19: "Object",
        20: "Key",
        21: "Null",
        22: "EnumMember",
        23: "Struct",
        24: "Event",
        25: "Operator",
        26: "TypeParameter",
    }
    return SYMBOL_KINDS.get(kind, f"Unknown({kind})")
