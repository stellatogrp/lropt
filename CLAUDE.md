# LRopt Development Guide

## Build & Test Commands
- Install dev dependencies: `pip install -e ".[dev]"`
- Run all tests: `pytest tests`
- Run a specific test: `pytest tests/test_file.py::test_function_name`
- Run with verbosity: `pytest -v tests`
- Build documentation: `cd docs && make html`
- Lint code: `ruff check .`
- Format code: `ruff format .`

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, project-specific last
- **Formatting**: 4-space indentation, 100 char line length, single quotes
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Types**: Use type annotations; modern syntax with pipe operator for Union
- **Functions**: Small, focused functions (< 10 lines); descriptive names
- **Error handling**: Use exceptions with descriptive messages, not error codes
- **Documentation**: Detailed docstrings with reStructuredText formatting
- **Testing**: Test methods prefixed with `test_`; uses CLARABEL solver

Follow principles in CONVENTIONS.md: readability, simplicity, meaningful names,
and self-explanatory code over comments.
