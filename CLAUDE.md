# LROPT Project Guidelines

## Build & Test Commands

- Install dependencies: `uv pip install -e ".[dev]"`
- Run all tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/test_file.py::test_function_name`
- Build documentation: `cd docs && make html`

## Code Style

- Line length: 100 characters
- Use meaningful variable and function names
- Keep functions small with 2-3 arguments ideally
- Prefix Python commands with "uv run"
- Self-explanatory code over comments
- Exception handling: use try/except blocks appropriately

## Formatting & Linting

- Linter: `uv run ruff check .`
- Fix linting issues: `uv run ruff check --fix .`
- Follow existing import style (standard lib, third-party, local)
- Type hints: use for function parameters and return values
- Follow PEP8 naming conventions (snake_case for variables/functions)
