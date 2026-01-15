import asyncio
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typer.testing import CliRunner

from webagents.cli.commands.template import app
from webagents.cli.templates import github

runner = CliRunner()

@pytest.fixture
def mock_github_pull():
    with patch("webagents.cli.templates.github.pull_template", new_callable=MagicMock) as mock:
        yield mock

@pytest.fixture
def mock_state():
    with patch("webagents.cli.state.local.get_state") as mock:
        mock.return_value.get_templates_dir.return_value = Path("/tmp/webagents/templates")
        yield mock

def test_pull_template(mock_github_pull, mock_state, tmp_path):
    # Setup mock return
    mock_template_path = tmp_path / "templates" / "owner_repo" / "TEMPLATE.md"
    mock_template_path.parent.mkdir(parents=True)
    mock_template_path.write_text("content")
    
    # Mock the coroutine return - side_effect is safer for asyncio.run compatibility
    async def mock_coro(*args, **kwargs):
        return mock_template_path
    
    mock_github_pull.side_effect = mock_coro
    
    with patch("webagents.cli.commands.template.use") as mock_use:
        result = runner.invoke(app, ["pull", "owner/repo", "--apply"])
        
        assert result.exit_code == 0
        assert "Pulling template" in result.stdout
        assert "Template saved" in result.stdout
        
        mock_github_pull.assert_called_once()
        mock_use.assert_called_once_with(
            template="owner_repo",
            name=None,
            keep=False
        )

def test_use_bundled_template(tmp_path):
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["use", "assistant", "--name", "test-agent"])
        
        assert result.exit_code == 0
        assert "Created AGENT-test-agent.md" in result.stdout
        
        output_file = tmp_path / "AGENT-test-agent.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert "name: test-agent" in content
        assert "model: openai/gpt-4o-mini" in content

def test_use_cached_template(mock_state, tmp_path):
    # Setup cache
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    template_dir = cache_dir / "custom-template"
    template_dir.mkdir()
    (template_dir / "TEMPLATE.md").write_text("---\nname: original\n---\n# Custom")
    
    mock_state.return_value.get_templates_dir.return_value = cache_dir
    
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["use", "custom-template", "--name", "custom"])
        
        assert result.exit_code == 0
        assert "Created AGENT-custom.md from cached template" in result.stdout
        
        output_file = tmp_path / "AGENT-custom.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert "name: custom" in content  # Replaced name
        assert "# Custom" in content
