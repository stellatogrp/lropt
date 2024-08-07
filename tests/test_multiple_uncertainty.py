import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.testing as npt

from lropt import max_of_uncertain
from lropt.robust_problem import RobustProblem
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.box import Box
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL


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

    # @unittest.skip("not currently implementing multiple sets")
    def test_simple_ellipsoidal_2u(self):

        # SETUP
        n = 5
        x_lropt = cp.Variable(n)
        c = npr.rand(n)
        b = 10.
        P = npr.randint(-1, 5, size=(n, n))
        a = np.zeros(n)
        rho_1 = 0.2
        rho_2 = 0.5

        objective_1 = cp.Minimize(c @ x_lropt)
        # Formulate robust constraints with lropt
        unc_set_1 = Ellipsoidal(rho=rho_1)
        unc_set_2 = Box(rho=rho_2)
        u_1 = UncertainParameter(n,
                                 uncertainty_set=unc_set_1)

        u_2 = UncertainParameter(n,
                                 uncertainty_set=unc_set_2)
        constraints_1 = [max_of_uncertain(
            [u_1 @ P @ x_lropt, a @ x_lropt]) + u_2 @ x_lropt <= b]

        prob_robust = RobustProblem(objective_1, constraints_1)
        prob_robust.solve()

        # Robust problem 2

        x_rob2 = cp.Variable(n)
        objective_2 = cp.Minimize(x_rob2 @ c)

        constraints_2 = [u_1 @ P @ x_rob2 + u_2 @ x_rob2  <= b]
        constraints_2 += [a @ x_rob2 + u_2 @ x_rob2 <= b]

        prob_rob2 = RobustProblem(objective_2, constraints_2)
        prob_rob2.solve()

        npt.assert_allclose(x_lropt.value, x_rob2.value, rtol=RTOL, atol=ATOL)


        # Cvxpy problem
        x_cvx = cp.Variable(n)
        objective_3 = cp.Minimize(x_cvx @ c)
        constraints_3 = []
        constraints_3 += [-b + rho_1*cp.norm(P @ x_cvx,2) + rho_2*cp.norm(x_cvx,1) <= 0 ]
        constraints_3 += [-b + rho_2*cp.norm(x_cvx,1) + a @ x_cvx <= 0]
        prob_cvx = cp.Problem(objective_3, constraints_3)
        prob_cvx.solve()
        npt.assert_allclose(x_lropt.value, x_cvx.value, rtol=RTOL, atol=ATOL)
