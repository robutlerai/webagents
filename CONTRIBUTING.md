# Contributing to WebAgents

Thanks for your interest in contributing! Please take a moment to review this guide.

## Getting Started

1. Fork the repository and create a feature branch:
   - Branch naming: `feat/<short-name>`, `fix/<short-name>`, or `docs/<short-name>`
2. Setup Python 3.10+ and install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run the test suite:
   ```bash
   pytest -q
   ```

## Development

- Use type hints and keep functions small and focused
- Match existing code style (Black, Ruff, MyPy in `pyproject.toml`)
  ```bash
  black .
  ruff check .
  mypy webagents
  ```
- Prefer clear naming (no abbreviations) and early returns

## Commit & PR Guidelines

- Write clear, atomic commits (present tense): `fix: correct NLI scope check`
- Open a Pull Request with:
  - A concise description
  - Screenshots or logs when relevant
  - Tests for new behavior
- Link related issues (e.g., `Fixes #123`)

## Testing

- Add unit tests under `tests/`
- For HTTP endpoints, include positive and negative cases
- Keep tests deterministic; avoid network calls (mock where needed)

## Documentation

- Update `docs/` for user-visible changes
- Keep admonitions minimal (2–3 per page)
- Ensure links are relative and valid (`skills/platform/...` etc.)
- Build docs locally and fix warnings

## Code of Conduct

Be respectful and constructive. We value inclusive, collaborative contributions.

## Questions?

Open a discussion or issue on GitHub. Thank you for helping improve WebAgents!
