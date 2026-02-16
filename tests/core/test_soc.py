import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.testing as npt

# from tests.settings import SOLVER
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL

ATOL = 1e-5
RTOL = 1e-5


class TestSOC(unittest.TestCase):

    def setUp(self):

        self.n = 4
        self.N = 20
        norms = npr.multivariate_normal(
            np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)
        self.ATOL=ATOL
        self.RTOL=RTOL

    # @unittest.skip('learning not ready')
    def test_soc(self):
        # Setup
        n = self.n

        # Problem
        # y = np.ones(n)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal())

        # b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        z = cp.Variable(n)

        objective = cp.Maximize(cp.sum(z))

        constraints = [x @ u <= c, cp.norm(z) <= 2*c]

        prob = RobustProblem(objective, constraints)
        prob.solve()
        npt.assert_allclose(np.linalg.norm(z.value),2*c, rtol=RTOL, atol=ATOL)
