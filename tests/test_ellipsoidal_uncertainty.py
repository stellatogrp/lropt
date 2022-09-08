import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lro.robust_problem import RobustProblem
from lro.uncertain import UncertainParameter
from lro.uncertainty_sets.ellipsoidal import Ellipsoidal
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        c = np.random.rand(self.n)
        self.b = 10.
        self.x = cp.Variable(self.n)
        self.objective = cp.Minimize(c * self.x)
        # Robust set
        self.rho = 0.2
        self.p = 2

    def test_ellipsoidal(self):
        """Test uncertain variable"""
        u = UncertainParameter(uncertainty_set=Ellipsoidal(p=3, rho=3.5))
        assert u.uncertainty_set.dual_norm() == 1.5

    def test_robust_norm_lp(self):
        b, x, objective, n, rho, p = \
            self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=p) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lro
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=p, rho=rho))
        constraints = [a * x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_affine_transform(self):
        b, x, n, objective, rho, p = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        m_unc = 8
        A_unc = 3. * np.eye(m_unc)[:n, :]
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        constraints = [b_unc * x + rho * cp.norm(A_unc.T * x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lro
        unc_set = Ellipsoidal(p=p, rho=rho,
                              affine_transform={'A': A_unc, 'b': b_unc})
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [a * x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    #  def test_reverse_inequality(self):
    #  def test_uncertainty_in_objective(self):
