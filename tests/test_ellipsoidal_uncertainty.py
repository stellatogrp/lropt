import unittest

import cvxpy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import torch

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
        self.objective = cp.Minimize(c @ self.x)
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
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_affine_transform(self):
        # import ipdb
        # ipdb.set_trace()
        b, x, n, objective, rho, p = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        m_unc = 8
        A_unc = 3. * np.eye(m_unc)[:n, :]
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        constraints = [-b_unc @ x + rho * cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lro
        unc_set = Ellipsoidal(p=p, rho=rho,
                              affine_transform={'A': A_unc, 'b': b_unc})
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [1*-a @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    # @unittest.skip('need to combine with new code')
    def test_ellipsoidal_learning(self):
        # import ipdb
        # ipdb.set_trace()
        torch.seed()
        cov_scale = 5
        data_dim = 10
        data_num = 100
        data_mean = torch.zeros(data_dim)
        cov = cov_scale * torch.eye(data_dim)
        X = npr.multivariate_normal(data_mean, cov, data_num)
        c = npr.rand(data_dim)
        c_tch = torch.tensor(c, requires_grad=True)

        b = 3

        # def cvar_loss(x_soln, data, alpha, lmbda = 1):
        #     tau = cp.variable()
        #     y = cp.variable()

        #     obj = y
        #     constr = []
        #     constr += [tau*(1-1/alpha) -
        #   (torch.mean(torch.maximum(data @ x_soln - b, torch.zeros(data_dim))))/alpha <= y]
        #     constr += [tau <= y]
        #     problem = cp.Problem(obj, constr)
        #     problem.solve()

        #     return c @ x_soln + lmbda * problem.value

        def violation_loss(x_soln, data, lmbda=1):
            # import ipdb
            # ipdb.set_trace()
            npt.assert_equal(x_soln.shape[0], data.shape[1])
            return c_tch @ x_soln + lmbda * torch.mean(
                torch.maximum(torch.tensor(data, requires_grad=True) @
                              x_soln - b, torch.tensor(0., requires_grad=True))), \
                c_tch @ x_soln

        unc_set = Ellipsoidal(data=X, loss=violation_loss)
        u = UncertainParameter(data_dim, uncertainty_set=unc_set)
        x = cp.Variable(data_dim)
        objective = cp.Minimize(-c @ x)
        constraints = [u @ x <= b, x >= 0, x <= 5]

        prob_robust = RobustProblem(objective, constraints)
        df = prob_robust.train()
        print(df)
        prob_robust.solve(solver=SOLVER)
        print(x.value)

        # Need prob_robust.train
