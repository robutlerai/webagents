"""
GitHub Template Fetcher

Pull templates from GitHub repositories.
"""

import re
from pathlib import Path
from typing import Optional
import httpx


GITHUB_RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


async def fetch_template(
    url: str,
    branch: str = "main",
    path: str = "TEMPLATE.md",
) -> Optional[str]:
    """Fetch template from GitHub.
    
    Args:
        url: GitHub URL or user/repo shorthand
        branch: Branch name
        path: Path to template file in repo
        
    Returns:
        Template content or None
    """
    owner, repo = _parse_github_url(url)
    if not owner or not repo:
        return None
    
    raw_url = GITHUB_RAW_URL.format(
        owner=owner,
        repo=repo,
        branch=branch,
        path=path,
    )
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(raw_url)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass
    
    return None


def _parse_github_url(url: str) -> tuple:
    """Parse GitHub URL to owner/repo.
    
    Args:
        url: Full URL or user/repo shorthand
        
    Returns:
        Tuple of (owner, repo)
    """
    # Shorthand: user/repo
    if "/" in url and not url.startswith("http"):
        parts = url.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
    
    # Full URL: https://github.com/user/repo
    match = re.match(
        r'https?://github\.com/([^/]+)/([^/]+)',
        url
    )
    if match:
        return match.group(1), match.group(2)
    
    return None, None


async def pull_template(
    url: str,
    dest_dir: Path = None,
    branch: str = "main",
    path: str = "TEMPLATE.md",
) -> Optional[Path]:
    """Pull template and save to local cache.
    
    Args:
        url: GitHub URL or shorthand
        dest_dir: Destination directory (defaults to cache)
        branch: Branch name
        path: Path in repo
        
    Returns:
        Path to saved template or None
    """
    content = await fetch_template(url, branch=branch, path=path)
    if not content:
        return None
    
    # Save to cache
    if dest_dir is None:
        from ..state.local import get_state
        dest_dir = get_state().get_templates_dir()
    
    owner, repo = _parse_github_url(url)
    template_dir = dest_dir / f"{owner}_{repo}"
    template_dir.mkdir(parents=True, exist_ok=True)
    
    template_file = template_dir / "TEMPLATE.md"
    template_file.write_text(content)
    
    return template_file


def list_cached_templates(cache_dir: Path = None) -> list:
    """List templates in cache.
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        List of template paths
    """
    if cache_dir is None:
        from ..state.local import get_state
        cache_dir = get_state().get_templates_dir()
    
    templates = []
    for template_dir in cache_dir.iterdir():
        if template_dir.is_dir():
            template_file = template_dir / "TEMPLATE.md"
            if template_file.exists():
                templates.append({
                    "name": template_dir.name,
                    "path": str(template_file),
                })
    
    return templates


def clear_cache(cache_dir: Path = None):
    """Clear template cache.
    
    Args:
        cache_dir: Cache directory
    """
    import shutil
    
    if cache_dir is None:
        from ..state.local import get_state
        cache_dir = get_state().get_templates_dir()
    
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
