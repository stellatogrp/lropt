import unittest

import cvxpy as cp
import numpy as np

from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.box import Box


class TestSimpleOpt(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(seed=1234)
        self.n = 7
        self.m = 5
        self.gamma = 9
        self.TOL = 1e-4
        self.low = 0
        self.high = 8

        self.x = cp.Variable(self.n)
        self.uncertainty_set = Box()
        self.t = UncertainParameter(uncertainty_set=self.uncertainty_set)
        self.z = UncertainParameter(self.n, uncertainty_set=self.uncertainty_set)
        self.a = np.random.uniform(low=self.low, high=self.high, size=self.n)
        self.b = np.random.uniform(low=self.low, high=self.high, size=self.n)
        self.A = np.random.uniform(low=self.low, high=self.high, size=(self.m, self.n))
        self.B = np.random.uniform(low=self.low, high=self.high, size=(self.m, self.n))
        self.I = np.ones((self.m, self.n))
        self.i = np.ones(self.n)

    def check_result(self, objective, constraints, target):
        """
        This helper function checks if a given test passes.
        """
        prob = RobustProblem(objective=objective, constraints=constraints)
        prob.solve()
        self.assertAlmostEqual(prob.value, target, delta=self.TOL)

    def test_simple(self):
        objective = cp.Minimize(self.a@self.x)
        constraints =   [
                            self.x >= 1,
                        ]
        self.check_result(objective, constraints, np.sum(self.a))

    def test_max(self):
        objective = cp.Minimize(cp.maximum(self.a@self.x, self.b@self.x))
        constraints =   [
                            self.x >= 1,
                        ]

        self.check_result(objective, constraints, np.max((np.sum(self.a), np.sum(self.b))))

    def test_max_constraint(self):
        objective = cp.Maximize(self.a@self.x)
        constraints =   [
                            self.i@self.x == 1,
                            self.x >= 0,
                        ]
        self.check_result(objective, constraints, np.max(self.a))

    def test_unc_simple(self):
        objective = cp.Maximize(self.z@self.x)
        constraints =   [
                            self.x >= 0,
                        ]
        self.check_result(objective, constraints, 0)

    def test_unc_max(self):
        objective = cp.Maximize(self.i@self.x)
        constraints =   [
                            self.x >= 0,
                            self.x + self.t <= 1,
                        ]
        self.check_result(objective, constraints, 0)
