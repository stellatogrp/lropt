import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch

from lropt.parameter import Parameter
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

    # @unittest.skip('learning not ready')
    def test_simple_learn(self):
        # Setup
        n = self.n
        num_instances = 10
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

        # Problem
        # y = np.ones(n)
        y = Parameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = npr.randint(3, 5, n)
        # b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        # z = cp.Variable(n)

        objective = cp.Maximize(a @ x)

        # y_tch = torch.tensor(y, dtype = float)
        a_tch = torch.tensor(a, dtype=float)
        c_tch = torch.tensor(c, dtype=float)

        constraints = [x @ (u + y) <= c, cp.norm(x) <= 2*c]

        def f_tch(x,y,u):
            return a_tch @ x
        def g_tch(x,y,u):
            return x @ u + x @ y - c_tch

        prob = RobustProblem(objective, constraints,
                             objective_torch=f_tch, constraints_torch=[g_tch])
        prob.train(lr = 0.001, step=2, momentum = 0.8, optimizer = "SGD")
        # prob.solve()
