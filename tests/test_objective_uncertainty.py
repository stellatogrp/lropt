import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.testing as npt

# import numpy.random as npr
from lropt.robust_problem import RobustProblem
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.box import Box
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal


class TestObjectiveUncertainty(unittest.TestCase):

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

    @unittest.skip("not currently implementing objective uncertainty")
    def test_objective_uncertainty(self):

        n = 5
        x_lropt = cp.Variable(n)
        c = npr.rand(n)
        b = 10.
        P = npr.randint(-1, 5, size=(n, n))
        a = np.zeros(n)
        rho_1 = 0.2
        rho_2 = 0.5

        # objective_1 = cp.Minimize(c @ x_lropt)
        # Formulate robust constraints with lropt
        unc_set_1 = Ellipsoidal(rho=rho_1)
        unc_set_2 = Box(rho=rho_2)
        u_1 = UncertainParameter(n,
                                 uncertainty_set=unc_set_1)

        objective_1 = cp.Minimize(u_1 @ x_lropt + x_lropt @ c)
        u_2 = UncertainParameter(n,
                                 uncertainty_set=unc_set_2)
        constraints_1 = [cp.maximum(
            u_1 @ P @ x_lropt, a @ x_lropt) + u_2 @ x_lropt <= b]

        prob_robust = RobustProblem(objective_1, constraints_1)
        prob_robust.solve()

        # CVXPY problem

        x_cvx = cp.Variable(n)

        tau_1 = cp.Variable()
        tau_2 = cp.Variable()
        # tau_3 = cp.Variable()

        u_1 = UncertainParameter(n,
                                 uncertainty_set=unc_set_1)
        u_2 = UncertainParameter(n,
                                 uncertainty_set=unc_set_2)

        objective_2 = cp.Minimize(u_1 @ x_cvx + x_cvx @ c)

        # constraints_2 = [x_cvx @ u_1 <= tau_3]
        constraints_2 = [tau_1 + tau_2 - b <= 0]
        constraints_2 += [cp.maximum(u_1 @ P @ x_cvx, a @ x_cvx) <= tau_1]
        constraints_2 += [u_2 @ x_cvx <= tau_2]

        prob_cvx = RobustProblem(objective_2, constraints_2)
        prob_cvx.solve()

        npt.assert_allclose(x_cvx.value, x_lropt.value)
