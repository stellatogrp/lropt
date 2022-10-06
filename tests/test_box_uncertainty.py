import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lro.robust_problem import RobustProblem
from lro.uncertain import UncertainParameter
from lro.uncertainty_sets.box import Box
from lro.uncertainty_sets.polyhedral import Polyhedral
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# import pytest


print("Hello")


class TestBoxUncertainty(unittest.TestCase):

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

    def test_polyhedral_equal_box_norm(self):
        b, x, n, objective = self.b, self.x, self.n, self.objective

        # Robust set
        # Affine transform

        m_unc = 8
        A_unc = 3. * np.eye(m_unc)[:n, :]
        b_unc = 0.1 * np.random.rand(n)

        # Polyhedral constraint (make a box)
        A_poly = np.vstack((np.eye(m_unc),
                            -np.eye(m_unc)))
        b_poly = np.concatenate((0.1 * np.ones(m_unc),
                                 0.1 * np.ones(m_unc)))

        # Formulate robust problem using box constraints in cvxpy
        constraints = [-2*b_unc @ x + 0.1 * cp.norm(-2*A_unc.T @ x, p=1) <= b]
        prob_cvxpy_box = cp.Problem(objective, constraints)
        prob_cvxpy_box.solve(solver=SOLVER)
        x_cvxpy_box = x.value

        # Formulate robust problem using box constraints in lro
        unc_set = Box(rho=0.1,
                      affine_transform={'A': A_unc, 'b': np.zeros(n)})

        a = UncertainParameter(n, uncertainty_set=unc_set)
        constraints = [-2*(b_unc + np.eye(n)@a) @ x <= b]
        prob_robust_box = RobustProblem(objective, constraints)
        prob_robust_box.solve(solver=SOLVER)
        x_robust_box = x.value

        # Formulate robust problem using equivalent polyhedral constraint
        unc_set = Polyhedral(d=b_poly,
                             D=A_poly,
                             affine_transform={'A': -2*A_unc, 'b': -2*b_unc})
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [a @ x <= b]
        prob_robust_poly = RobustProblem(objective, constraints)
        prob_robust_poly.solve(solver=SOLVER)
        x_robust_poly = x.value

        npt.assert_allclose(x_cvxpy_box, x_robust_box, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(x_robust_box, x_robust_poly, rtol=RTOL, atol=ATOL)

    # @pytest.mark.skip(reason="Need to add scalar multiplication")
    def test_inf_norm_scaler(self):
        # import ipdb
        # ipdb.set_trace()
        x = cp.Variable()
        objective = cp.Minimize(-10 * x)
        u = UncertainParameter(
            uncertainty_set=Box(center=0., rho=2.)
        )
        constraints = [0 <= x, x <= 10,
                       u * x*-0.5 <= 2]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, 2.0, rtol=RTOL, atol=ATOL)

    def test_inf_norm1(self):
        # import ipdb
        # ipdb.set_trace()
        x = cp.Variable()
        objective = cp.Minimize(-10 * x)
        u = UncertainParameter(
            uncertainty_set=Box(center=0., rho=2.)
        )
        constraints = [0 <= x, x <= 10,
                       -(0 - 1*u) * x + u*x >= -2]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, 0.5, rtol=RTOL, atol=ATOL)

    # @pytest.mark.skip(reason="Need to add scalar multiplication")
    def test_inf_norm1_flip(self):
        x = cp.Variable()
        objective = cp.Minimize(-10 * x)
        u = UncertainParameter(
            uncertainty_set=Box(rho=2., affine_transform={'A': 1., 'b': 0.})
        )
        constraints = [0 <= x, x <= 10,
                       (2*-u*x)*-1 <= 2]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, 0.5, rtol=RTOL, atol=ATOL)

    def test_mat_multiply(self):
        n = 5
        x = cp.Variable(n)
        c = np.ones(n)
        A_unc = np.eye(n)
        b_unc = 3*np.ones(n)

        objective = cp.Minimize(c @ x)
        u = UncertainParameter(n,
                               uncertainty_set=Box(rho=2., affine_transform={'A': A_unc, 'b': b_unc}))

        constraints = [0 <= x, x <= 10,
                       2 * (u @ x)*-1 <= 2]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)

    # @pytest.mark.skip(reason="Need to add quad")
    def test_isolate_scalar(self):
        x = cp.Variable()
        objective = cp.Minimize(-10 * x)
        u = UncertainParameter(
            uncertainty_set=Box(rho=2., affine_transform={'A': 1., 'b': 4.})
        )
        constraints = [0 <= x, x <= 10,
                       x <= u]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, 2, rtol=RTOL, atol=ATOL)
    #  def test_reverse_inequality(self):
    #  def test_uncertainty_in_objective(self):

    def test_isolate_vec(self):
        x = cp.Variable(4)
        objective = cp.Minimize(-10*cp.sum(x))
        # import ipdb
        # ipdb.set_trace()
        u = UncertainParameter(4,
                               uncertainty_set=Box(rho=2., affine_transform={'A': np.eye(4), 'b': [4., 2., 6, 3]})
                               )
        constraints = [0 <= x, x <= 10,
                       x <= u]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, [2, 0, 4, 1], rtol=RTOL, atol=ATOL)
    #  def test_reverse_inequality(self):
    #  def test_uncertainty_in_objective(self):
