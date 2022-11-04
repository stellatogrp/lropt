import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
# import pytest
import torch

from lro.robust_problem import RobustProblem
from lro.uncertain import UncertainParameter
from lro.uncertainty_sets.box import Box
from lro.uncertainty_sets.budget import Budget
from lro.uncertainty_sets.polyhedral import Polyhedral
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

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
        constraints = [-12 <= x, x <= 10,
                       x <= -2*u]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, -12, rtol=RTOL, atol=ATOL)
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

    def test_param_eps(self):
        # import ipdb
        # ipdb.set_trace()
        x = cp.Variable()
        y = cp.Parameter()
        objective = cp.Minimize(-10 * x)

        def loss_f(x, data_eval):
            return torch.tensor(-10)*x + 20*torch.mean(
                torch.maximum(torch.tensor(data_eval)*x - torch.tensor(2), torch.tensor(0))), -10*x

        data = np.array([[1.], [1.], [2.], [1.], [2.]])
        u = UncertainParameter(
            uncertainty_set=Box(data=data, loss=loss_f)
        )
        constraints = [0 <= x, x <= y,
                       u*x <= 2]
        prob = RobustProblem(objective, constraints)
        y.value = 10
        df = prob.train(eps=True)
        prob.solve(solver=SOLVER)
        print(x.value)
        print(df)
        # npt.assert_allclose(x.value, 1, rtol=RTOL, atol=ATOL)

    def test_param_matrix(self):
        # import ipdb
        # ipdb.set_trace()
        x = cp.Variable()
        y = cp.Parameter()
        objective = cp.Minimize(-10 * x)

        def loss_f(x, data_eval):
            return torch.tensor(-10)*x \
                + 20*torch.mean(torch.maximum(
                    torch.tensor(data_eval)*x - torch.tensor(2), torch.tensor(0))), -10*x

        data = np.array([[1.], [1.], [2.], [1.], [2.]])
        u = UncertainParameter(
            uncertainty_set=Box(data=data, loss=loss_f)
        )
        constraints = [0 <= x, x <= y,
                       u*x <= 2]
        prob = RobustProblem(objective, constraints)
        y.value = 10
        df = prob.train()
        prob.solve(solver=SOLVER)
        print(x.value)
        print(df)
        # npt.assert_allclose(x.value, 1, rtol=RTOL, atol=ATOL)

    def test_boxe(self):
        # formulate the box set
        m = 5
        box_u = UncertainParameter(m,
                                   uncertainty_set=Box(center=0.1*np.ones(m),
                                                       side=0.01*np.array([1, 2, 3, 4, 5]),
                                                       rho=2.))
        n = 5

        # formulate cvxpy variable
        x = cp.Variable(n)

        # formulate problem constants
        P = np.random.rand(n, m)
        c = np.random.rand(n)

        # formulate objective
        objective = cp.Minimize(-c@x)

        # formulate constraints
        constraints = [P@box_u @ x <= 10 + 2.5*box_u@x, x >= 0, x <= 1]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()

    def test_budget(self):
        # import ipdb
        # ipdb.set_trace()
        # restate the ellipsoidal set
        m = 5
        budget_u = UncertainParameter(m,
                                      uncertainty_set=Budget(rho1=2.,
                                                             rho2=1.))
        n = 5
        # formulate cvxpy variable
        x = cp.Variable(n)

        # formulate problem constants
        P = 3. * np.eye(m)[:n, :]
        P1 = 3*np.random.rand(n, m)
        a = 0.1 * np.random.rand(n)
        c = np.random.rand(n)

        # formulate objective
        objective = cp.Minimize(c@x)

        # formulate constraints
        constraints = [(P@budget_u + a) @ x <= 10]
        constraints += [x >= P1@budget_u]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()

        print("LRO objective value: ", prob_robust.objective.value, "\nLRO x: ", x.value)
