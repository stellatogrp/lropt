import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

from lropt.family_parameter import FamilyParameter
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

# import numpy.testing as npt

# from tests.settings import SOLVER
# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL

# import pandas as pd
# import torch


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):

        self.n = 4
        self.N = 100
        norms = npr.multivariate_normal(np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)

    def test_simple_learn(self):
        # Setup
        n = self.n
        num_instances = 10
        y_instances = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

        # Problem
        y = FamilyParameter(n, instances=y_instances)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = npr.randint(-3, 3, n)
        b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        z = cp.Variable(n)

        objective = cp.Maximize(a @ x)
        constraints = [x @ u + z @ b + y @ x <= c]

        prob = RobustProblem(objective, constraints)
        prob.train()
        prob.solve()
