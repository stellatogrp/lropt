# Backward-compatible re-exports from cvxro.parameter
# Deprecated: import directly from cvxro.parameter instead.
import warnings as _warnings

from cvxro.parameter import (
    ContextParameter,
    Parameter,
    ShapeParameter,
    SizeParameter,
)

__all__ = ["ContextParameter", "Parameter", "ShapeParameter", "SizeParameter"]

_warnings.warn(
    "cvxro.train.parameter is deprecated. "
    "Import from cvxro.parameter instead.",
    DeprecationWarning,
    stacklevel=2,
)
