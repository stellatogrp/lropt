import unittest

import cvxpy as cp
# import matplotlib.pyplot as plt
import numpy as np
# import numpy.random as npr
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# import pandas as pd
# import torch


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        c = np.random.rand(self.n)
        self.b = 10.
        self.x = cp.Variable(self.n)
        self.objective = cp.Minimize(c @ self.x)
        # Robust set
        self.rho = 0.2
        self.p = 2

    def test_ellipsoidal(self):
        """Test uncertain variable"""
        u = UncertainParameter(uncertainty_set=Ellipsoidal(rho=3.5))
        assert u.uncertainty_set.dual_norm() == 2.0

    def test_robust_norm_lp(self):
        b, x, objective, n, rho, p = \
            self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=p) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho))
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_affine_transform(self):
        # import ipdb
        # ipdb.set_trace()
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        constraints = [-b_unc @ x + rho * cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [1*-(A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_simple_ellipsoidal(self):
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust constraints with lropt
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [2 * (A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
