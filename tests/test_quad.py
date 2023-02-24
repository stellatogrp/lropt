import unittest

import cvxpy as cp
import numpy as np
# import numpy.testing as npt
import scipy as sc
# import pytest
# import torch
from sklearn import datasets

from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertain_atoms.quad_form import quad_form
# from lropt.uncertainty_sets.box import Box
# from lropt.uncertainty_sets.budget import Budget
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from lropt.uncertainty_sets.mro import MRO
# from lropt.uncertainty_sets.polyhedral import Polyhedral
from tests.settings import SOLVER

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL


class TestQuad(unittest.TestCase):

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

    def test_quad_simple(self):
        m = 5
        n = 5
        x = cp.Variable(n)
        t = cp.Variable()
        u = UncertainParameter(m,
                               uncertainty_set=Ellipsoidal(p=2,
                                                           rho=2.))
        A = {}
        Ainv = {}
        for i in range(m):
            A[i] = datasets.make_spd_matrix(m, random_state=i)
            Ainv[i] = sc.linalg.sqrtm(np.linalg.inv(A[i]))

        objective = cp.Minimize(t)
        constraints = [cp.sum([-0.5*quad_form(u, A[i]*x[i]) for i in range(m)]) <= t]
        constraints += [x >= 0, x <= 1]
        # import ipdb
        # ipdb.set_trace()
        prob = RobustProblem(objective, constraints)
        newprob = prob.dualize_constraints()
        newprob.solve(solver=SOLVER)
        print(x.value)

    # def test_quad_simple_2(self):
    #     m = 5
    #     n = 5
    #     x = cp.Variable(n)
    #     t = cp.Variable()
    #     u = UncertainParameter(m,
    #                            uncertainty_set=Ellipsoidal(p=2,
    #                                                        rho=2.))
    #     X = cp.variable((n,n))

    #     objective = cp.Minimize(t)
    #     constraints = [-quad_form(u,X) <= t]
    #     # import ipdb
    #     # ipdb.set_trace()
    #     prob = RobustProblem(objective, constraints)
    #     newprob = prob.dualize_constraints()
    #     newprob.solve(solver=SOLVER)
    #     print(x.value)

    def test_max(self):
        n = 2
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2,
                                                           rho=1))
        # formulate cvxpy variables
        x = cp.Variable(n)
        t = cp.Variable()

        # formulate constants
        a = np.array([2, 3])
        d = np.array([3, 4])

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [cp.maximum(a@x - d@x, a@x - d@u) <= t]
        constraints += [x >= 0]
        # import ipdb
        # ipdb.set_trace()
        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        new_prob = prob_robust.dualize_constraints()
        new_prob.solve()

    def test_hydro(self):
        import math
        T = 5
        # data = data_modes(N,m,[10,20,30])
        # restate the ellipsoidal set
        u = UncertainParameter(T,
                               uncertainty_set=Ellipsoidal(p=2,
                                                           rho=0.00001))

        # formulate cvxpy variable
        tau = cp.Variable()
        # lam = cp.Variable(T)
        x0 = cp.Variable(T)
        x = {}
        for t in range(T):
            x[t] = cp.Variable(t+1)

        # formulate problem constants
        P = {}
        for t in range(T):
            P[t] = np.eye(T)[0:t+1, :]
        l0 = 1
        lhigh = 5
        llow = 1
        c = np.zeros(T)
        for i in range(T):
            c[i] = 10 + 5*math.sin(math.pi*(1-(i+1))/3)

        # formulate objective
        objective = cp.Minimize(tau)

        # formulate constraints
        constraints = [-cp.sum([c[t]*x0[t] + c[t]*x[t]@P[t]@u for t in range(T)]) <= tau]
        for t in range(T):
            constraints += [l0 - lhigh + np.ones(t+1)@P[t]@u - cp.sum([x0[i] + x[i]@P[i]@u for i in range(t+1)]) <= 0]
            constraints += [llow - l0 - np.ones(t+1)@P[t]@u + cp.sum([x0[i] + x[i]@P[i]@u for i in range(t+1)]) <= 0]
            constraints += [-x0[t] - x[t]@P[t]@u <= 0]

        # formulate Robust Problem
        prob = RobustProblem(objective, constraints)

        # solve
        # Train only epsilon
        # newprob = prob.dualize_constraints()
        prob.solve()

    def test_mro(self):

        def normal_returns_scaled(N, m, scale):
            R = np.vstack([np.random.normal(
                i*0.03*scale, np.sqrt((0.02**2+(i*0.025)**2)), N) for i in range(1, m+1)])
            return (R.transpose())

        def data_modes(N, m, scales):
            modes = len(scales)
            d = np.zeros((N+100, m))
            weights = int(np.ceil(N/modes))
            for i in range(modes):
                d[i*weights:(i+1)*weights,
                  :] = normal_returns_scaled(weights, m, scales[i])
            return d[0:N, :]

        # Generate data
        num_stocks = 5
        N = 100
        data = data_modes(N, num_stocks, [1, 2, 3])
        m = 5
        mro_u = UncertainParameter(m,
                                   uncertainty_set=MRO(rho=2., K=1, data=data, train=False))
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
        constraints = [(P@mro_u + a) @ x_r <= 10]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()
        print("LROPT objective value: ", prob_robust.objective.value, "\nLROPT x: ", x_r.value)

    # def test_train_mro(self):
    #     # Formulate constants
    #     n = 2
    #     N = 50
    #     # k = npr.uniform(1,4,n)
    #     # p = k + npr.uniform(2,5,n)
    #     k = np.array([2., 3.])
    #     p = np.array([3, 4.5])
    #     k_tch = torch.tensor(k, requires_grad=True)
    #     p_tch = torch.tensor(p, requires_grad=True)

    #     # Formulate loss function
    #     def loss(t, x, data, la=5):
    #         return t + la*torch.mean(torch.maximum(
    #             torch.maximum(k_tch@x - data@p_tch, k_tch@x - p_tch@x) - t,
    #             torch.tensor(0., requires_grad=True))), t, torch.mean(torch.maximum(
    #                 torch.maximum(k_tch@x - data@p_tch, k_tch@x - p_tch@x) - t,
    #                 torch.tensor(0., requires_grad=True)))

    #     def gen_demand(n, N):
    #         F = np.random.normal(size=(n, 2))
    #         sig = 0.1*F@(F.T)
    #         mu = np.random.uniform(-0.2, 3, n)
    #         norms = np.random.multivariate_normal(mu, sig, N)
    #         d_train = np.exp(norms)
    #         return d_train

    #     # Generate data
    #     data = gen_demand(n, N)
    #     # Formulate uncertainty set
    #     # u = lropt.UncertainParameter(n,
    #     #                         uncertainty_set=lropt.Ellipsoidal(p=2,
    #     #                                                     data=data, loss = loss))

    #     u = UncertainParameter(n,
    #                            uncertainty_set=MRO(K=5, p=2,
    #                                                data=data, loss=loss, uniqueA=True))
    #     # Formulate the Robust Problem
    #     x_r = cp.Variable(n)
    #     t = cp.Variable()

    #     objective = cp.Minimize(t)

    #     constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]
    #     constraints += [x_r >= 0]

    #     prob = RobustProblem(objective, constraints)
    #     prob.solve()

    #     # s = 13
    #     # # Train only epsilon
    #     # result = prob.train(eps = True, lr = 0.001, step=50, \
    #     # momentum = 0.8, optimizer = "SGD", initeps = 1, seed = s)
    #     # df_eps = result.df
    #     # # Train A and b
    #     # df1, newprob, A_fin, b_fin = prob.train(lr = 0.01,\
    #     #  step=50, momentum = 0.8, optimizer = "SGD", seed = s, initeps=1)

    #     # # Grid search epsilon
    #     # dfgrid, newprob = prob.grid(epslst = np.linspace(0.8, 2, 40),\
    #     #  seed = s)
    #     # init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(data.T)))
    #     # init_b = -init@np.mean(data, axis=0)
    #     # # Train A and b
    #     # result1 = prob.train(lr = 0.001, step=50, \
    #     # momentum = 0.8, optimizer = "SGD", seed = s, \
    #     # initA = init, initb = init_b, fixb = False)
    #     # df1 = result1.df
    #     # A_fin = result1.A
    #     # b_fin = result1.b
    #     # result2 = prob.train(eps = True, lr = 0.001, step=50, \
    #     # momentum = 0.8, optimizer = "SGD", seed = s, \
    #     # initA = A_fin, initb = b_fin)
    #     # df_r1 = result2.df
    #     # result4 = prob.grid(epslst = np.linspace(0.01, 3, 40),\
    #     #  initA = init, initb = init_b, seed = s)
    #     # dfgrid = result4.df
