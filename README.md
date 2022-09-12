# Learning for Optimization under Uncertainty


To develop:

1. Create a new python or conda environment
2. Install this package in development mode with `pip install -e ".[dev]"`
3. Try stuff!

To run pre-commit checks on files, you need to install pre-commit with `pre-commit install`.
Now, it will run reformatting and linting checks at every commit. If they fail, you can retry the commit and they should work. If `flake8` fails, you have to fix the linting errors and run it again.

To run unittests, you can run `pytest tests/`.


## Things to generalize later
- [ ] Make canonicalization work when not DPP
