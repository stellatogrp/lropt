import unittest

import cvxpy as cp
import scipy as sc

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
from sklearn.model_selection import train_test_split


from lropt.parameter import Parameter
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

# import numpy.testing as npt

# from tests.settings import SOLVER
# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL

# import pandas as pd
# import torch


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):

        self.n = 4
        self.N = 100
        norms = npr.multivariate_normal(np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)

    @unittest.skip('learning not ready')
    def test_simple_learn(self):
        # Setup
        n = self.n
        num_instances = 10
        y_instances = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

        # Problem
        y = Parameter(n, instances=y_instances)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = npr.randint(-3, 3, n)
        b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        z = cp.Variable(n)

        objective = cp.Maximize(a @ x)
        constraints = [x @ u + z @ b + y @ x <= c]

        prob = RobustProblem(objective, constraints)
        prob.train()
        prob.solve()

    def test_news(self):
        # Formulate constants
        n = 5
        N = 300
        np.random.seed(399)
        k = npr.uniform(2,5,n)
        p = k + npr.uniform(0.5,2,n)
        k1 = npr.uniform(1,3,n)
        p1 = k1 + npr.uniform(0,2,n)
        k_tch = torch.tensor(k, requires_grad = True)
        # p_tch = torch.tensor(p, requires_grad = True)
        k1_tch = torch.tensor(k1, requires_grad = True)
        p1_tch = torch.tensor(p1, requires_grad = True)



        def loss(t,y,x_r, x_1, p_tch, alpha, data,l=1000, quantile = 0.95, target = 0.):
            if len(torch.tensor(data).shape) < 2:
                Nsample = 1
            else:
                Nsample = data.shape[0]
            sums =  torch.mean(torch.maximum(torch.maximum(
                torch.maximum(k_tch@x_r -data@p_tch, k_tch@x_r - p_tch@x_r) - t-alpha, 
                torch.tensor(0.,requires_grad = True)), 
                torch.maximum(k1_tch@x_1 -data@p1_tch, k1_tch@x_1 - p1_tch@x_1) - y-alpha))
            sums = sums/(1-quantile) + alpha
            sums1 = torch.mean(torch.maximum(k_tch@torch.tensor(x_r) -torch.tensor(data)@p_tch, k_tch@torch.tensor(x_r) - p_tch@torch.tensor(x_r))+ torch.maximum(k1_tch@torch.tensor(x_1) -torch.tensor(data)@p1_tch, k1_tch@torch.tensor(x_1) - p1_tch@torch.tensor(x_1)))
            sums1 = torch.mean(torch.maximum(k_tch@torch.tensor(x_r) -torch.tensor(data)@p_tch, k_tch@torch.tensor(x_r) - p_tch@torch.tensor(x_r)))
            return t +y + l*(sums - target), sums1, torch.sum((torch.maximum(torch.maximum(
                torch.maximum(k_tch@x_r -data@p_tch, k_tch@x_r - p_tch@x_r) - t, 
                torch.tensor(0.,requires_grad = True)), 
                torch.maximum(k1_tch@x_1 -data@p1_tch, k1_tch@x_1 - p1_tch@x_1) - y))>= torch.tensor(1.))/Nsample, sums.detach().numpy()


        def gen_demand(n, N, seed=399):
            np.random.seed(seed)
            F = np.random.normal(size = (n,2))
            sig = 0.1*F@(F.T)
            mu = np.random.uniform(-0.2,2,n)
            norms = np.random.multivariate_normal(mu,sig, N)
            d_train = np.exp(norms)
            return mu, sig, d_train

        # Generate data
        mu, sig, data = gen_demand(n,N*2)

        scenarios = {}
        num_scenarios = 5
        for scene in range(num_scenarios):
            np.random.seed(scene+1)
            scenarios[scene] = {}
            scenarios[scene][0] = np.maximum(1,p + np.random.normal(0,2,n))

        np.random.seed(16)
        # data = np.exp(np.random.multivariate_normal(mu,sig, N*2))
        # Formulate uncertainty set
        u = UncertainParameter(n,
                                uncertainty_set=Ellipsoidal(p=2,
                                                            data=data, loss = loss))
        # Formulate the Robust Problem
        x_r = cp.Variable(n)
        x_1 = cp.Variable(n)
        t = cp.Variable()
        y = cp.Variable()
        p = cp.Parameter(n)
        # p1 = cp.Parameter(n)
        p.value = scenarios[0][0]
        # p1.value = scenarios[0][1]
        objective = cp.Minimize(t+y)

        # constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]
        # constraints += [cp.maximum(k1@x_1 - p1@x_1, k1@x_1 - p1@u) <= y]
        constraints = [cp.maximum(k@x_r - p@x_r - t, k@x_r - p@u - t,k1@x_1 - p1@x_1 - y,  k1@x_1 - p1@u - y) <= 0]

        constraints += [x_1 >= 0, x_r >= x_1]

        prob = RobustProblem(objective, constraints)
        target = -0.05
        test_p = 0.5
        s = 5
        train, test = train_test_split(data, test_size=int(data.shape[0]*test_p), random_state=s)
        init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
        # init = np.eye(n)
        init_bval = -init@np.mean(train, axis=0)
        # Train A and b
        result1 = prob.train(lr = 0.000001, step=800, momentum = 0.8, optimizer = "SGD", seed = s, init_A = init, init_b = init_bval, fixb = False,init_lam = 0.1, target_cvar = target, init_alpha = 0., test_percentage = test_p, scenarios = scenarios, num_scenarios = num_scenarios, step_y = 0.01)
        df1 = result1.df
        A_fin = result1.A
        b_fin = result1.b

        # result3 = prob.train(eps = True, lr = 0.00001, step=800, momentum = 0.8, optimizer = "SGD", seed = s, init_A = init, init_b = init_bval, init_mu = 1, init_lam = 0,  target_cvar = target, init_alpha =0.,mu_multiplier = 1.01,test_percentage = test_p,scenarios = scenarios, num_scenarios = num_scenarios)
        # df_r2 = result3.df

        # Grid search epsilon
        result4 = prob.grid(epslst = np.concatenate((np.linspace(0.01, 0.9, 60), np.linspace(0.93, 4.5, 15))), init_A = init, init_b = init_bval, seed = s, init_alpha = 0.,test_percentage = test_p,scenarios = scenarios, num_scenarios = num_scenarios)
        dfgrid = result4.df

        result5 = prob.grid(epslst = np.concatenate((np.linspace(0.01, 0.6, 60), np.linspace(0.61, 4.5, 15))), init_A = A_fin, init_b = b_fin, seed = s, init_alpha = 0.,test_percentage = test_p,scenarios = scenarios, num_scenarios = num_scenarios)
        dfgrid2 = result5.df