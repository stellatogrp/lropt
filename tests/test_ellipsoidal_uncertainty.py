import unittest

import cvxpy as cp
# import matplotlib.pyplot as plt
import numpy as np
# import numpy.random as npr
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# import pandas as pd
# import torch


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
        u = UncertainParameter(uncertainty_set=Ellipsoidal(rho=3.5))
        assert u.uncertainty_set.dual_norm() == 2.0

    def test_robust_norm_lp(self):
        b, x, objective, n, rho, p = \
            self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=p) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho))
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_affine_transform(self):
        # import ipdb
        # ipdb.set_trace()
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        constraints = [-b_unc @ x + rho * cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [1*-(A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_simple_ellipsoidal(self):
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust constraints with lropt
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [2 * (A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)

    # @unittest.skip('need to combine with new code')

    # def test_ellipsoidal_learning(self):
    #     # import ipdb
    #     # ipdb.set_trace()
    #     torch.seed()
    #     cov_scale = 5
    #     data_dim = 10
    #     data_num = 100
    #     data_mean = torch.zeros(data_dim)
    #     cov = cov_scale * torch.eye(data_dim)
    #     X = npr.multivariate_normal(data_mean, cov, data_num)
    #     c = npr.rand(data_dim)
    #     c_tch = torch.tensor(c, requires_grad=True)

    #     b = 3

    #     # def cvar_loss(x_soln, data, alpha, lmbda = 1):
    #     #     tau = cp.variable()
    #     #     y = cp.variable()

    #     #     obj = y
    #     #     constr = []
    #     #     constr += [tau*(1-1/alpha) -
    #     #   (torch.mean(torch.maximum(data @ x_soln - b, torch.zeros(data_dim))))/alpha <= y]
    #     #     constr += [tau <= y]
    #     #     problem = cp.Problem(obj, constr)
    #     #     problem.solve()

    #     #     return c @ x_soln + lmbda * problem.value

    #     def violation_loss(x_soln, data, lmbda=1):
    #         # import ipdb
    #         # ipdb.set_trace()
    #         npt.assert_equal(x_soln.shape[0], data.shape[1])
    #         return c_tch @ x_soln + lmbda * torch.mean(
    #             torch.maximum(torch.tensor(data, requires_grad=True) @
    #                           x_soln - b, torch.tensor(0., requires_grad=True))), \
    #             c_tch @ x_soln, torch.mean(
    #             torch.maximum(torch.tensor(data, requires_grad=True) @
    #                           x_soln - b, torch.tensor(0., requires_grad=True)))

    #     unc_set = Ellipsoidal(data=X, loss=violation_loss)
    #     u = UncertainParameter(data_dim, uncertainty_set=unc_set)
    #     x = cp.Variable(data_dim)
    #     objective = cp.Minimize(-c @ x)
    #     constraints = [u @ x <= b, x >= 0, x <= 5]

    #     prob_robust = RobustProblem(objective, constraints)
    #     result = prob_robust.train(eps=True, lr=0.005, step=46)
    #     print(result.df)
    #     result.reform_problem.solve(solver=SOLVER)
    #     print(x.value)

    #     result = prob_robust.train(lr=0.002, step=46)
    #     print(result.df)
    #     result.reform_problem.solve(solver=SOLVER)
    #     print(x.value)

    #     # plt.figure(figsize=(9, 5))
    #     # plt.plot(df['steps'], df['Loss_val'], color="tab:blue", label="Eps")
    #     # plt.plot(df['steps'], df['Eval_val'], linestyle='--', color="tab:blue")
    #     # plt.plot(df1['steps'], df1['Loss_val'], color="tab:orange", label="Reshape")
    #     # plt.plot(df1['steps'], df1['Eval_val'], linestyle='--', color="tab:orange")
    #     # plt.legend()
    #     # plt.savefig("plot")

    #     # Need prob_robust.train

    # def test_portfolio_learning(self):
    #     # import ipdb
    #     # ipdb.set_trace()
    #     torch.seed()
    #     sp500 = pd.read_csv('docs/examples/experiments/stock_data/prices_sp500.csv').to_numpy()

    #     num_stocks = 100
    #     num_rets = 500

    #     npr.seed(1)
    #     stock_idxs = np.random.choice(sp500.shape[1], num_stocks, replace=False)

    #     sp_prices = sp500[:num_rets+1, stock_idxs]
    #     sp_rets = (sp_prices[1:, :] - sp_prices[:-1, :])/sp_prices[1:, :]
    #     sp_rets = sp_rets + 0.05
    #     # for i in range(sp_rets.shape[0]):
    #     #     u = np.random.uniform()
    #     #     if u > 0.8:
    #     #         sp_rets[i] += np.random.normal(0,0.05, num_stocks)

    #     def violation_loss(t_soln, x_soln, data, lmbda=1):
    #         # import ipdb
    #         # ipdb.set_trace()
    #         # print(x_soln.shape)
    #         npt.assert_equal(x_soln.shape[0], data.shape[1])
    #         return t_soln + lmbda * torch.mean(
    #             torch.maximum(-data @ x_soln - t_soln, torch.tensor(0., requires_grad=True))), t_soln, torch.mean(
    #             torch.maximum(-data @ x_soln - t_soln, torch.tensor(0., requires_grad=True)))

    #     unc_set = Ellipsoidal(data=sp_rets, loss=violation_loss)
    #     u = UncertainParameter(num_stocks, uncertainty_set=unc_set)

    #     x = cp.Variable(num_stocks)
    #     t = cp.Variable()

    #     objective = cp.Minimize(t)
    #     cons = [-u @ x <= t]
    #     cons += [cp.sum(x) == 1, x >= 0]

    #     # import ipdb
    #     # ipdb.set_trace()
    #     prob_robust = RobustProblem(objective, cons)
    #     result = prob_robust.train(eps=True, lr=0.05, step=100, momentum=0.8, optimizer="SGD", initeps=5)
    #     result.reform_problem.solve(solver=SOLVER)
    #     result1 = prob_robust.train(lr=0.05, step=100, momentum=0.8, optimizer="SGD")
    #     result1.reform_problem.solve(solver=SOLVER)
    #     result2 = prob_robust.grid(epslst=np.linspace(0.1, 5, 20))
    #     print(result.df, result1.df, result2.df)
    #     # df.to_csv('df_eps.csv')
    #     # df1.to_csv('df_R.csv')
    #     # dfgrid.to_csv('df_grid.csv')
    #     # plt.figure(figsize=(9, 5))
    #     # plt.plot(df['step'], df['Loss_val'], color="tab:blue", label=r"$/epsilon$ training")
    #     # plt.plot(df['step'], df['Eval_val'], linestyle='--', color="tab:blue", label=r"$/epsilon$ testing")
    #     # plt.plot(df1['step'], df1['Loss_val'], color="tab:orange", label="Reshape training")
    #     # plt.plot(df1['step'], df1['Eval_val'], linestyle='--', color="tab:orange", label="Reshape testing")
    #     # plt.legend()
    #     # plt.xlabel("Iterations")
    #     # plt.ylabel("Training and Testing Loss")
    #     # plt.savefig("plot.pdf")

    # def test_tut(self):
    #     m = 5
    #     n = 4
    #     ellip_unc = Ellipsoidal(rho=2.)
    #     ellip_u = UncertainParameter(m, uncertainty_set=ellip_unc)
    #     x = cp.Variable(4)
    #     P = 3. * np.eye(m)[:n, :]
    #     a = 0.1 * np.random.rand(n)
    #     c = np.random.rand(n)
    #     objective = cp.Minimize(c@x)
    #     constraints = [(P@ellip_u + a) @ x <= 10]
    #     prob_robust = RobustProblem(objective, constraints)
    #     prob_robust.solve()
