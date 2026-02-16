from cvxro._version import __version__

# Core (always available, no torch required at import time)
from cvxro.parameter import ContextParameter, Parameter
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_canon.max_of_uncertain import max_of_uncertain, sum_of_max_of_uncertain
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.box import Box
from cvxro.uncertainty_sets.budget import Budget
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal
from cvxro.uncertainty_sets.mro import MRO
from cvxro.uncertainty_sets.norm import Norm
from cvxro.uncertainty_sets.polyhedral import Polyhedral
from cvxro.uncertainty_sets.scenario import Scenario

# Learning exports (lazy-loaded to avoid importing torch at module level)
_LAZY_IMPORTS = {
    'Trainer': 'cvxro.train.trainer',
    'TrainerSettings': 'cvxro.train.settings',
    'OPTIMIZERS': 'cvxro.train.settings',
    'Simulator': 'cvxro.train.simulator',
    'LinearPredictor': 'cvxro.train.predictors.linear',
    'NNPredictor': 'cvxro.train.predictors.nn',
    'CovPredictor': 'cvxro.train.predictors.covpred',
    'DeepNormalModel': 'cvxro.train.predictors.deep',
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'cvxro' has no attribute {name!r}")
