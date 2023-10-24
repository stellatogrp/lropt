import unittest

import cvxpy as cp
import numpy as np
import scipy as sc
from sklearn import datasets
import numpy.testing as npt


from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertain_atoms.quad_form import quad_form
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from lropt.uncertainty_sets.mro import MRO
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL


class TestQuad(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)

    def test_quad(self):
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2, rho=0.5))
        # formulate cvxpy variables
        x_r = cp.Variable(n)
        t = cp.Variable()

        # formulate problem constants
        P = {}
        P_inv = {}
        for i in range(n):
            P[i] = datasets.make_spd_matrix(n, random_state=i)
            P_inv[i] = sc.linalg.sqrtm(np.linalg.inv(P[i]))

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [cp.sum([-0.5*quad_form(u, P[i]*x_r[i])
                              for i in range(n)]) <= t]
        constraints += [cp.sum(x_r) == 4]
        constraints += [x_r >= 0.6, x_r <= 1]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()

        # formulate using cvxpy
        x_cvxpy = cp.Variable(n)
        t = cp.Variable()
        z = cp.Variable(n)
        y = cp.Variable((n, n))

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [cp.sum([cp.quad_over_lin(y[i]@P_inv[i], 2*x_cvxpy[i])
                              for i in range(n)]) + 0.5*cp.norm(z, 2) <= t]
        constraints += [cp.sum(y, axis=1) == z]
        constraints += [cp.sum(x_cvxpy) == 4]
        constraints += [x_cvxpy >= 0.6, x_cvxpy <= 1]

        # formulate problem
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve()

        # assert x values are equal
        npt.assert_allclose(x_r.value, x_cvxpy.value, rtol=RTOL, atol=ATOL)

    