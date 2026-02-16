# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CVXRO (Convex Optimization under Robust Optimization) is a Python package for decision-making under uncertainty, built on CVXPY. It solves robust optimization problems of the form:

```
minimize f(x) subject to g(x,u) ≤ 0 for all u ∈ U(θ)
```

**Dual mode operation**: Users can either define uncertainty sets explicitly OR pass historical data and let CVXRO learn the optimal uncertainty set parameters through gradient-based training.

## Commands

All commands use `uv run` to ensure the virtual environment is active:

```bash
# Run core + integration tests (safe, fast)
uv run python run_tests_safe.py tests/core/ tests/integration/ -q

# Run only core tests (~2s)
uv run pytest tests/core/ -q

# Run specific test file or test
uv run pytest tests/core/test_simple_opt.py
uv run pytest tests/core/test_simple_opt.py::test_name -v

# Lint check / auto-fix
uv run ruff check cvxro/
uv run ruff check --fix cvxro/

# Install in development mode
uv pip install -e ".[dev]"
```

**WARNING**: Never run `tests/learning/` tests directly — they are extremely RAM-heavy and will crash the machine. Use `run_tests_safe.py` which enforces memory limits via `RLIMIT_RSS`.

## Architecture

### Solution Pipeline

```
RobustProblem(objective, constraints)
    ↓ [UncertainCanonicalization]
Canonicalized Problem (separates certain/uncertain parts)
    ↓ [RemoveUncertainty - applies duality]
Standard CVXPY Problem (no uncertain parameters)
    ↓
Solver (Clarabel, SCS, etc.)
```

The key mechanism is **duality reformulation**: for each uncertain constraint `g(x,u) ≤ 0`, dual variables are introduced and the uncertainty set's conjugate function transforms it into a tractable deterministic constraint.

### Learning Pipeline

When data is provided, CVXRO learns optimal uncertainty set parameters:

```
RobustProblem + UncertaintySet.data
    ↓
Trainer.train()
    ├─ Canonicalize to standard CVXPY
    ├─ TorchExpressionGenerator (CVXPY → PyTorch)
    ├─ CVXPyLayer (differentiable optimization)
    └─ Gradient descent on A, b, rho parameters
    ↓
Updated uncertainty set (shape A, center b, size rho)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `robust_problem.py` | `RobustProblem` class extending CVXPY Problem |
| `uncertain_parameter.py` | `UncertainParameter` (CVXPY Parameter + uncertainty_set) |
| `parameter.py` | `ContextParameter`, `Parameter`, `ShapeParameter`, `SizeParameter` |
| `train/trainer.py` | `Trainer` orchestrating learning with CVXPyLayers |
| `train/settings.py` | `TrainerSettings` (45+ hyperparameters) |
| `train/parameter.py` | Backward-compat re-exports from `parameter.py` |
| `uncertainty_sets/` | Set implementations with `conjugate()` methods |
| `uncertain_canon/` | Duality-based reformulation reductions |
| `torch_expression_generator.py` | CVXPY → PyTorch expression conversion |

### Import Architecture

The codebase separates **core** (no torch required) from **learning** (torch-dependent):

- `__init__.py`: Core imports are eager; training imports (`Trainer`, `TrainerSettings`, predictors, etc.) are **lazy** via `__getattr__` — `import cvxro` does not load torch.
- `robust_problem.py`: No module-level torch/training imports. Training-related imports happen lazily inside methods.
- `parameter.py`: Canonical location for all parameter types. `train/parameter.py` re-exports for backward compatibility.

### Uncertainty Set Hierarchy

```
UncertaintySet (base)
├─ Norm (p-norm family)
│  ├─ Ellipsoidal (‖z‖₂ ≤ ρ)
│  └─ Box (‖z‖∞ ≤ ρ)
├─ Polyhedral (Cz ≤ d)
├─ Budget (dual norm constraints)
├─ MRO (Wasserstein/data-driven)
└─ Scenario (satisfy all scenarios)
```

Each set defines: `rho` (size), `a` (shape matrix), `b` (center), and optional support constraints (`c`, `d`, `ub`, `lb`).

### Public API

**Core**: `RobustProblem`, `UncertainParameter`, `Trainer`, `TrainerSettings`, `ContextParameter`, `Parameter`

**Uncertainty sets**: `Ellipsoidal`, `Box`, `Norm`, `Polyhedral`, `Budget`, `MRO`, `Scenario`

**Predictors** (contextual learning): `LinearPredictor`, `NNPredictor`, `CovPredictor`, `DeepNormalModel`

**Utilities**: `max_of_uncertain`, `sum_of_max_of_uncertain`, `Simulator`

## Typical Usage Patterns

**Explicit uncertainty set:**
```python
u = cvxro.UncertainParameter(dim, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))
prob = cvxro.RobustProblem(objective, constraints)
prob.solve()
```

**Learning from data** (explicit training step required):
```python
u = cvxro.UncertainParameter(dim, uncertainty_set=cvxro.Ellipsoidal())
u.uncertainty_set.data = historical_data
prob = cvxro.RobustProblem(objective, constraints)
settings = cvxro.TrainerSettings()
cvxro.Trainer(prob).train(settings)  # explicit training step
prob.solve()
```

**Contextual learning** (uncertainty depends on state):
```python
x = cvxro.ContextParameter(data=X_train)
settings = cvxro.TrainerSettings(contextual=True, predictor=cvxro.NNPredictor())
```

## Code Style

From CONVENTIONS.md:
- Keep functions small and focused on one task
- Use meaningful names that reveal intent
- Prefer fewer arguments (2-3 max)
- Code should be self-explanatory; avoid comments unless necessary
- Use exceptions rather than error codes

Ruff config: line-length 100, Python 3.12+ target, rules E/F/I enabled.

## Test Structure

```
tests/
├── conftest.py           # Fixtures (rng), pytest markers
├── settings.py           # SOLVER, RTOL, ATOL
├── core/                 # 73 tests, fast (~2s), no torch needed
├── learning/             # Training tests (SLOW, RAM-heavy — use run_tests_safe.py)
└── integration/          # 18 tests, evaluate/portfolio
```

Markers: `@pytest.mark.slow` for learning tests, `@pytest.mark.core` for core tests.

## Known Gotchas

- Several training modules (`predictors/linear.py`, `predictors/nn.py`, etc.) call `torch.set_default_dtype(torch.double)` at module level. With lazy loading, this side effect doesn't run at import time. Any code creating tensors for evaluation must explicitly use `dtype=torch.float64` (not `torch.Tensor(value)`, use `torch.tensor(value, dtype=torch.float64)`).
- `solve()` does **not** auto-train. Users must call `Trainer(prob).train(settings)` explicitly before solving.

## Dependencies

Core: cvxpy, torch, scipy, scikit-learn, pandas, diffcp, cvxpylayers, cvxtorch, tqdm

Dev: pytest, ruff, sphinx, jupyterlab, marimo, pyscipopt, hydra-core
