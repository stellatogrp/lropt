import unittest

import cvxpy as cp
import numpy as np

from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.box import Box
from cvxro.uncertainty_sets.budget import Budget


class TestRobustPortfolio(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(seed=1234)
        self.n = 150
        self.i = np.arange(1, self.n+1)
        self.p = 1.15 + self.i*0.05/self.n
        self.delta = np.diag(0.05/450 * (2*self.i*self.n*(self.n+1))**0.5)
        self.Gamma = 5
        self.TOL = 1e-4

    def test_robust_portfolio(self):
        x = cp.Variable(self.n)
        uncertainty_set = Budget(rho1=1, rho2=self.Gamma)
        z = UncertainParameter(self.n, uncertainty_set=uncertainty_set)

        t = cp.Variable()
        objective = cp.Minimize(-t)
        constraints = [
                        (self.p + self.delta@z) @ x >= t,
                        cp.sum(x)==1,
                        x>=0,
                      ]
        prob1 = RobustProblem(objective=objective, constraints=constraints)
        res1 = prob1.solve()

        objective = cp.Minimize(-((self.p + self.delta@z) @ x))
        constraints = [
                        cp.sum(x)==1,
                        x>=0,
                      ]
        prob2 = RobustProblem(objective=objective, constraints=constraints)
        res2 = prob2.solve()

        self.assertAlmostEqual(res1, res2, delta=self.TOL)

    def test_objective_uncertainty(self):
        cp.Variable(1)
        uncertainty_set = Box(rho=5)
        u = UncertainParameter(1, uncertainty_set=uncertainty_set)

        objective = cp.Minimize(u)
        constraints = []
        prob1 = RobustProblem(objective=objective, constraints=constraints)
        res1 = prob1.solve()

        t = cp.Variable()
        objective = cp.Minimize(t)
        constraints = [u<=t]
        prob2 = RobustProblem(objective=objective, constraints=constraints)
        res2 = prob2.solve()

        self.assertAlmostEqual(res1, res2, delta=self.TOL)


    def test_objective_uncertainty_flip(self):
        cp.Variable(1)
        uncertainty_set = Box(rho=5)
        u = UncertainParameter(1, uncertainty_set=uncertainty_set)

        objective = cp.Maximize(-u)
        constraints = []
        prob1 = RobustProblem(objective=objective, constraints=constraints)
        res1 = prob1.solve()

        t = cp.Variable()
        objective = cp.Minimize(t)
        constraints = [u<=t]
        prob2 = RobustProblem(objective=objective, constraints=constraints)
        res2 = prob2.solve()

        self.assertAlmostEqual(res1, res2, delta=self.TOL)
