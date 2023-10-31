import time
import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy as sc
import torch
from sklearn.model_selection import train_test_split

from lropt.parameter import Parameter
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

# from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# import pandas as pd
# import torch


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):

        self.n = 4
        self.N = 100
        norms = npr.multivariate_normal(
            np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)

    # @unittest.skip('learning not ready')
    def test_simple_learn(self):
        # Setup
        n = self.n
        num_instances = 5
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

        # Problem
        # y = np.ones(n)
        y = Parameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = npr.randint(3, 5, n)
        # b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        # z = cp.Variable(n)

        objective = cp.Maximize(a @ x.T)

        # y_tch = torch.tensor(y, dtype = float)
        a_tch = torch.tensor(a, dtype=float)
        c_tch = torch.tensor(c, dtype=float)

        constraints = [x @ (u + y).T <= c, cp.norm(x) <= 2*c]

        def f_tch(x, y, u):
            # x is a tensor that represents the cp.Variable x.
            return a_tch @ x.T

        def g_tch(x, y, u):
            # x,y,u are tensors that represent the cp.Variable x and cp.Parameter y and u.
            # The cp.Constant c is converted to a tensor
            return x @ u.T + x @ y.T - c_tch

        prob = RobustProblem(objective, constraints,
                             objective_torch=f_tch, constraints_torch=[g_tch])
        prob.train(lr=0.001, num_iter=2, momentum=0.8, optimizer="SGD")
        # prob.solve()

    def test_portfolio_intro(self):
        timestart = time.time()
        n = 2
        seed = 15
        np.random.seed(seed)
        dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20,
                20, 15, 10, 10, 10, 10, 10, 10])/10)[:n]

        y_data = np.random.dirichlet(dist, 10)
        y = Parameter(n, data=y_data)

        def gen_demand_intro(N, seed):
            np.random.seed(seed)
            sig = np.array([[0.3, -0.4], [-0.5, 0.1]])
            mu = np.array((0.3, 0.3))
            d_train = np.random.multivariate_normal(mu, sig, N)
            # d_train = np.exp(d_train)
            return d_train

        def f_tch(t, x, y, u):
            # x is a tensor that represents the cp.Variable x.
            return t + 0.2*torch.linalg.vector_norm(x-y, 1)

        def g_tch(t, x, y, u):
            # x,y,u are tensors that represent the cp.Variable x and
            # cp.Parameter y and u.
            # The cp.Constant c is converted to a tensor
            return -x @ u.T - t

        def eval_tch(t, x, y, u):
            return -x @ u.T + 0.2*torch.linalg.vector_norm(x-y, 1)

        data = gen_demand_intro(600, seed=15)
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2,
                                                           data=data))
        # Formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()

        objective = cp.Minimize(t + 0.2*cp.norm(x - y, 1))
        constraints = [-x@u.T <= t, cp.sum(x) == 1, x >= 0]

        prob = RobustProblem(objective, constraints,
                             objective_torch=f_tch,
                             constraints_torch=[
                                 g_tch], eval_torch=eval_tch)
        test_p = 0.1
        s = 5
        train, _ = train_test_split(data, test_size=int(
            data.shape[0]*test_p), random_state=s)
        init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
        -init@np.mean(train, axis=0)
        np.random.seed(15)
        initn = np.random.rand(n, n) + 0.1*init + 0.5*np.eye(n)
        init_bvaln = -initn@(np.mean(train, axis=0) - 0.3*np.ones(n))

        # Train A and b
        result = prob.train(lr=0.01, num_iter=5, momentum=0.8,
                            optimizer="SGD",
                            seed=s, init_A=initn, init_b=init_bvaln,
                            init_lam=0.5, init_mu=0.01,
                            mu_multiplier=1.001, init_alpha=0., test_percentage=test_p, kappa=-0.01,
                            n_jobs=8, random_init=True, num_random_init=2)
        timefin = time.time()
        timefin - timestart
        df = result.df
        npt.assert_allclose(np.array(
            result.df["Violations_train"])[-1], 0.18101063,
            rtol=RTOL, atol=ATOL)

        print(df)
        # # Grid search epsilon
        # result4 = prob.grid(epslst=np.linspace(0.01, 5, 10), \
        # init_A=init,
        #                     init_b=init_bval, seed=s,
        #                     init_alpha=0., test_percentage=test_p)
        # dfgrid = result4.df

        # result5 = prob.grid(epslst=np.linspace(0.01, 5, 10), \
        # init_A=result.A, init_b=result.b, seed=s,
        #                     init_alpha=0., test_percentage=test_p)
        # dfgrid2 = result5.df
        # print(dfgrid, dfgrid2)
        pass
