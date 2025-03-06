import unittest
import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

ATOL = 1e-4
RTOL = 1e-4
SOLVER = cp.CLARABEL
SOLVER_SETTINGS = { "equilibrate_enable": False, "verbose": False }

class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        c = np.random.rand(self.n)
        self.b = 1.5
        self.x = cp.Variable(self.n, name="x")
        param_val = cp.Parameter()
        param_val.value = 0
        self.param_val = param_val
        self.objective = cp.Minimize(c @ self.x + param_val)
        # Robust set
        self.rho = 0.2
        self.p = 2

    @unittest.SkipTest
    def test_evaluate(self):
        b, x, objective, n, rho, p = \
            self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x + self.param_val <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value


        # npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)