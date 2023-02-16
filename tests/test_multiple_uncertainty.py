import unittest

import cvxpy as cp
# import matplotlib.pyplot as plt
import numpy as np

from lro.robust_problem import RobustProblem
from lro.uncertain import UncertainParameter
from lro.uncertainty_sets.ellipsoidal import Ellipsoidal


class TestMultipleUncertainty(unittest.TestCase):

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

    def test_simple_ellipsoidal_2u(self):
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        P = np.eye(n)
        a = np.zeros(n)
        # Formulate robust constraints with lro
        unc_set = Ellipsoidal(rho=rho)
        u_1 = UncertainParameter(n,
                                 uncertainty_set=unc_set)

        u_2 = UncertainParameter(n,
                                 uncertainty_set=unc_set)
        constraints = [(P @ u_1 + a) @ x + x @ u_2 <= b]

        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve()
        # import ipdb
        # ipdb.set_trace()
