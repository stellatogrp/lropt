import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

import cvxro
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# ATOL = 1e-4
# RTOL = 1e-4
# SOLVER = cp.CLARABEL

class TestMROUncertainty(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        self.c = np.random.rand(self.n)
        self.b = 10.
        self.x = cp.Variable(self.n)
        self.objective = cp.Minimize(self.c @ self.x)
        # Robust set
        self.rho = 0.2
        self.p = 2

    def test_ellip_equal_mro(self):
        # Ellipsoidal uncertainty set example
        m = 5
        data = np.random.normal(0,1,size = (100,m))
        ellip_u = cvxro.UncertainParameter(m,
                                          uncertainty_set = \
                                            cvxro.Ellipsoidal(p= 2, rho=2.,
                                                 b = np.mean(data, axis = 0)))
        n = 4

        # formulate cvxpy variable
        x_r = cp.Variable(4)

        # formulate problem constants
        P = 3. * np.eye(m)[:n, :]
        a = 0.1 * np.random.rand(n)
        c = np.random.rand(n)

        # formulate objective
        objective = cp.Minimize(c@x_r)

        # formulate constraints
        constraints = [(P@ellip_u +a)@ x_r <= 10]

        # formulate Robust Problem
        prob_robust = cvxro.RobustProblem(objective, constraints)
        # solve
        prob_robust.solve()

        #
        mro_u = cvxro.UncertainParameter(m,
                                        uncertainty_set = cvxro.MRO(rho=2.,
                                                    K = 1, data = data,
                                                    train = False))
        n = 4

        # formulate cvxpy variable
        x_m = cp.Variable(4)

        # formulate objective
        objective = cp.Minimize(c@x_m)

        # formulate constraints
        constraints = [(P@mro_u +a)@ x_m <= 10]

        # formulate Robust Problem
        prob_robust = cvxro.RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()


        # formulate cvxpy variable
        x = cp.Variable(4)

        # formulate objective
        objective = cp.Minimize(c@x)

        # formulate constraints
        constraints = [a@x + np.mean(data, axis = 0)@(P.T@x) + 2*cp.norm(P.T@x,2) <= 10]

        # formulate problem
        prob_cvxpy = cp.Problem(objective, constraints)

        # solve
        prob_cvxpy.solve()

        # assert x values are equal
        npt.assert_allclose(x_r.value, x.value, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(x_r.value, x_m.value, rtol=RTOL, atol=ATOL)
