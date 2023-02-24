import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.box import Box
from lropt.uncertainty_sets.budget import Budget
from lropt.uncertainty_sets.polyhedral import Polyhedral
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# import pytest
# import torch


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

        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)

        # Polyhedral constraint (make a box)
        A_poly = np.vstack((np.eye(n),
                            -np.eye(n)))
        b_poly = np.concatenate((0.1 * np.ones(n),
                                 0.1 * np.ones(n)))

        # Formulate robust problem using box constraints in cvxpy
        constraints = [-2*b_unc @ x + 0.1 * cp.norm(-2*A_unc.T @ x, p=1) <= b]
        prob_cvxpy_box = cp.Problem(objective, constraints)
        prob_cvxpy_box.solve(solver=SOLVER)
        x_cvxpy_box = x.value

        # Formulate robust problem using box constraints in lropt
        unc_set = Box(rho=0.1)

        a = UncertainParameter(n, uncertainty_set=unc_set)
        constraints = [-2*(b_unc + np.eye(n)@(A_unc @ a)) @ x <= b]
        prob_robust_box = RobustProblem(objective, constraints)
        prob_robust_box.solve(solver=SOLVER)
        x_robust_box = x.value

        # Formulate robust problem using equivalent polyhedral constraint
        unc_set = Polyhedral(d=b_poly,
                             D=A_poly)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [(-2 * A_unc @ a - 2 * b_unc) @ x <= b]
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
            uncertainty_set=Box(rho=2.)
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
            uncertainty_set=Box(rho=2.)
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
            uncertainty_set=Box(rho=2)
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
                               uncertainty_set=Box(rho=2.))

        constraints = [0 <= x, x <= 10,
                       2 * ((A_unc @ u + b_unc) @ x)*-1 <= 2]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)

    def test_box_std(self):
        n = 5
        x = cp.Variable(n)
        c = np.ones(n)
        A_unc = np.eye(n)
        b_unc = 3*np.ones(n)

        objective = cp.Minimize(c @ x)
        u = UncertainParameter(n,
                               uncertainty_set=Box(rho=2.))

        constraints = [0 <= x, x <= 10,
                       2 * ((A_unc @ u + b_unc) @ x)*-1 <= 2]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)

    # @pytest.mark.skip(reason="Need to add quad")
    def test_isolate_scalar(self):
        x = cp.Variable()
        objective = cp.Minimize(-10 * x)
        u = UncertainParameter(
            uncertainty_set=Box(rho=2.)
        )
        constraints = [-12 <= x, x <= 10,
                       x <= -2*(u + 4)]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, -12, rtol=RTOL, atol=ATOL)
    #  def test_reverse_inequality(self):
    #  def test_uncertainty_in_objective(self):

    def test_isolate_vec(self):
        x = cp.Variable(4)
        objective = cp.Minimize(-10*cp.sum(x))
        A_unc = np.eye(4)
        b_unc = [4., 2., 6, 3]
        # import ipdb
        # ipdb.set_trace()
        u = UncertainParameter(4,
                               uncertainty_set=Box(rho=2.)
                               )
        constraints = [0 <= x, x <= 10,
                       x <= (A_unc @ u + b_unc)]
        prob = RobustProblem(objective, constraints)
        prob.solve(solver=SOLVER)
        npt.assert_allclose(x.value, [2, 0, 4, 1], rtol=RTOL, atol=ATOL)
    #  def test_reverse_inequality(self):
    #  def test_uncertainty_in_objective(self):

    # def test_param_eps(self):
    #     # import ipdb
    #     # ipdb.set_trace()
    #     x = cp.Variable()
    #     y = cp.Parameter()
    #     objective = cp.Minimize(-10 * x)

    #     def loss_f(x, data_eval):
    #         return torch.tensor(-10)*x + 20*torch.mean(
    #             torch.maximum(torch.tensor(data_eval)*x - torch.tensor(2), torch.tensor(0))), -10*x, torch.mean(
    #             torch.maximum(torch.tensor(data_eval)*x - torch.tensor(2), torch.tensor(0)))

    #     data = np.array([[1.], [1.], [2.], [1.], [2.]])
    #     u = UncertainParameter(
    #         uncertainty_set=Box(data=data, loss=loss_f)
    #     )
    #     constraints = [0 <= x, x <= y,
    #                    u*x <= 2]
    #     prob = RobustProblem(objective, constraints)
    #     y.value = 10
    #     df = prob.train(eps=True)
    #     prob.solve(solver=SOLVER)
    #     print(x.value)
    #     print(df)
    #     # npt.assert_allclose(x.value, 1, rtol=RTOL, atol=ATOL)

    # def test_param_matrix(self):
    #     # import ipdb
    #     # ipdb.set_trace()
    #     x = cp.Variable()
    #     y = cp.Parameter()
    #     objective = cp.Minimize(-10 * x)

    #     def loss_f(x, data_eval):
    #         return torch.tensor(-10)*x \
    #             + 20*torch.mean(torch.maximum(
    #                 torch.tensor(data_eval)*x - torch.tensor(2), torch.tensor(0))), -10*x, torch.mean(torch.maximum(
    #                     torch.tensor(data_eval)*x - torch.tensor(2), torch.tensor(0)))

    #     data = np.array([[1.], [1.], [2.], [1.], [2.]])
    #     u = UncertainParameter(
    #         uncertainty_set=Box(data=data, loss=loss_f)
    #     )
    #     constraints = [0 <= x, x <= y,
    #                    u*x <= 2]
    #     prob = RobustProblem(objective, constraints)
    #     y.value = 10
    #     df = prob.train()
    #     prob.solve(solver=SOLVER)
    #     print(x.value)
    #     print(df)
    #     # npt.assert_allclose(x.value, 1, rtol=RTOL, atol=ATOL)

    def test_boxe(self):
        m = 5
        center = 0.5*np.ones(m)
        side = 0.1*np.array([1, 2, 3, 4, 5])
        box_u = UncertainParameter(m,
                                   uncertainty_set=Box(rho=2.))
        n = 5

        # formulate cvxpy variable
        x = cp.Variable(n)
        # x_2 = cp.Variable(n)

        # formulate problem constants
        P = 3*np.random.rand(n, m)
        c = np.random.rand(n)
        A = np.diag(0.5*side)
        b = center

        # formulate objective
        objective = cp.Minimize(-c@x)

        # formulate constraints
        constraints = [(P@(A @ box_u + b)) @ x - 2*(A @ box_u + b)@x <= 10, x >= 0, x <= 1]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)
        # solve
        prob_robust.solve()

        # formulate in cvxpy

        x_cvx = cp.Variable(5)
        # formulate objective
        objective = cp.Minimize(-c@x_cvx)

        # formulate constraints
        constraints = [(P@b)@x_cvx - 2*b@x_cvx + 2*cp.norm((P@A).T@x_cvx -
                                                           2*A.T@x_cvx, p=1) <= 10, x_cvx >= 0, x_cvx <= 1]

        # formulate Robust Problem
        prob_cvxpy = cp.Problem(objective, constraints)

        # solve
        prob_cvxpy.solve()

        npt.assert_allclose(x.value, x_cvx.value, rtol=RTOL, atol=ATOL)

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

        print("LROPT objective value: ", prob_robust.objective.value, "\nLROPT x: ", x.value)
