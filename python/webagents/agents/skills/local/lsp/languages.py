"""
Language Detection and Configuration for LSP Skill

Supports: Python, TypeScript, JavaScript, Java, Rust, Go, C#, Dart, Ruby, Kotlin
"""

from pathlib import Path
from typing import Dict, List

# Supported languages and their file extensions
SUPPORTED_LANGUAGES: Dict[str, List[str]] = {
    "python": [".py", ".pyw", ".pyi"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "java": [".java"],
    "rust": [".rs"],
    "go": [".go"],
    "csharp": [".cs"],
    "dart": [".dart"],
    "ruby": [".rb"],
    "kotlin": [".kt", ".kts"],
}

# Reverse mapping: extension -> language
EXTENSION_MAP: Dict[str, str] = {}
for lang, exts in SUPPORTED_LANGUAGES.items():
    for ext in exts:
        EXTENSION_MAP[ext] = lang


def detect_language(file_path: str) -> str:
    """Detect language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language identifier string
        
    Raises:
        ValueError: If file extension is not supported
    """
    ext = Path(file_path).suffix.lower()
    
    if ext not in EXTENSION_MAP:
        supported = sorted(EXTENSION_MAP.keys())
        raise ValueError(
            f"Cannot detect language for extension '{ext}'. "
            f"Supported extensions: {supported}"
        )
    
    return EXTENSION_MAP[ext]


def is_supported(file_path: str) -> bool:
    """Check if file language is supported.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file's language is supported
    """
    ext = Path(file_path).suffix.lower()
    return ext in EXTENSION_MAP


def get_supported_languages() -> List[str]:
    """Get list of supported language identifiers.
    
    Returns:
        List of language names
    """
    return list(SUPPORTED_LANGUAGES.keys())


def get_extensions_for_language(language: str) -> List[str]:
    """Get file extensions for a language.
    
    Args:
        language: Language identifier
        
    Returns:
        List of file extensions
        
    Raises:
        ValueError: If language is not supported
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    return SUPPORTED_LANGUAGES[language]
