import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.polyhedral import Polyhedral
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL


class TestPolyhedralUncertainty(unittest.TestCase):

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

    def test_polyhedral_lp_new(self):
        b, x, n, objective, _ = \
            self.b, self.x, self.n, self.objective, self.p

        # Polyhedral constraint (make a box)
        n_poly = 2 * n
        A_poly = np.vstack((np.eye(n), -np.eye(n)))
        b_poly = np.concatenate((.1 * np.ones(n), .1 * np.ones(n)))

        # Formulate robust problem explicitly with cvxpy
        p = cp.Variable(n_poly)
        constraints = [p @ b_poly <= b,
                       p.T @ A_poly == x,
                       p >= 0]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve()
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        unc_set = Polyhedral(d=b_poly,
                             D=A_poly)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve()
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_polyhedral_lp(self):
        # import ipdb
        # ipdb.set_trace()

        b, x, n, objective, p = \
            self.b, self.x, self.n, self.objective, self.p
        # Robust set
        # Affine transform
        m_unc = n
        A_unc = 3. * np.eye(m_unc)[:n, :]
        b_unc = 0.1 * np.random.rand(n)

        # Polyhedral constraint (make a box)
        n_poly = 2 * m_unc
        A_poly = np.vstack((np.eye(m_unc), -2*np.eye(m_unc)))
        b_poly = np.concatenate((.1 * np.ones(m_unc), .1 * np.ones(m_unc)))

        # Formulate robust problem explicitly with cvxpy
        p = cp.Variable(n_poly)
        constraints = [b_unc @ x + p @ b_poly <= b,
                       p.T @ A_poly == A_unc.T @ x,
                       p >= 0]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve()
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        unc_set = Polyhedral(d=b_poly,
                             D=A_poly)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [(A_unc @ a + b_unc) @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve()
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    #  def test_reverse_inequality(self):
    #  def test_uncertainty_in_objective(self):
    def test_poly1(self):
        m = 5
        D = np.vstack((np.eye(m), -2*np.eye(m)))
        d = np.concatenate((0.1*np.ones(m), 0.1*np.ones(m)))
        poly_u = UncertainParameter(m,
                                    uncertainty_set=Polyhedral(
                                        D=D,
                                        d=d))
        n = 4
        # formulate cvxpy variable
        x = cp.Variable(n)

        # formulate problem constants
        P1 = 0.5 * np.eye(m)[:n, :]
        P2 = 3*np.random.rand(n, m)
        a = 0.1 * np.random.rand(n)
        c = np.random.rand(n)

        # formulate objective
        objective = cp.Minimize(-c@x)

        # formulate constraints
        constraints = [(P1@poly_u + a) @ x <= 10, x <= 5]
        constraints += [(P2@poly_u) @ x <= 5]
        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)
        # solve
        prob_robust.solve()
