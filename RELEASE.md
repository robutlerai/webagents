# Release Instructions

This document outlines the process for releasing new versions of the WebAgents package to PyPI.

## Prerequisites

1. **PyPI Account**: Ensure you have a PyPI account with access to the `webagents` package
2. **GitHub Repository Secrets**: The following secrets must be configured in the GitHub repository:
   - `PYPI_API_TOKEN`: Your PyPI API token with upload permissions

## Release Methods

### Method 1: Automated Release via Git Tags (Recommended)

1. **Update Version**: Manually update the version in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Update to your desired version
   ```

2. **Commit Changes**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git push origin main
   ```

3. **Create and Push Tag**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **Automatic Publishing**: The GitHub Action will automatically:
   - Build the package
   - Run quality checks
   - Publish to PyPI
   - Create a GitHub release

### Method 2: Manual Workflow Dispatch

1. **Go to GitHub Actions**: Navigate to the "Actions" tab in your GitHub repository

2. **Run Workflow**: 
   - Select "Publish to PyPI" workflow
   - Click "Run workflow"
   - Enter the version number (e.g., `0.2.0`)
   - Click "Run workflow"

3. **Monitor Progress**: The workflow will automatically update the version and publish

## Local Development Release (Manual)

For testing or when automated methods aren't available:

1. **Install Build Tools**:
   ```bash
   pip install build twine
   ```

2. **Update Version** in `pyproject.toml`

3. **Build Package**:
   ```bash
   python -m build
   ```

4. **Check Package**:
   ```bash
   twine check dist/*
   ```

5. **Upload to Test PyPI** (optional):
   ```bash
   twine upload --repository testpypi dist/*
   ```

6. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Pre-release Checklist

Before releasing, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is properly formatted (`black .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Documentation is updated
- [ ] `CHANGELOG.md` is updated (if applicable)
- [ ] Version number is updated in `pyproject.toml`

## Post-release

After a successful release:

1. **Verify Installation**: Test installing the new version:
   ```bash
   pip install webagents==0.2.0
   ```

2. **Update Documentation**: Ensure all documentation reflects the new version

3. **Announce**: Consider announcing the release on relevant channels

## Troubleshooting

### Common Issues

1. **PyPI Token Issues**: Ensure the `PYPI_API_TOKEN` secret is correctly set and has upload permissions

2. **Version Conflicts**: If a version already exists on PyPI, you must increment the version number

3. **Build Failures**: Check the GitHub Actions logs for specific error messages

4. **Permission Errors**: Ensure your PyPI account has maintainer access to the `webagents` package

### Getting Help

- Check GitHub Actions logs for detailed error messages
- Verify PyPI token permissions
- Ensure all required files are present and correctly formatted 