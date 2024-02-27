import time
import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy as sc
import torch

# from tests.settings import SOLVER
from sklearn.model_selection import train_test_split

from lropt.parameter import Parameter
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):

        self.n = 4
        self.N = 20
        norms = npr.multivariate_normal(
            np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)
        self.ATOL=ATOL
        self.RTOL=RTOL

    # @unittest.skip('learning not ready')
    def test_simple_learn(self):
        # Setup
        n = self.n
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), self.N)

        # Problem
        # y = np.ones(n)
        y = Parameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = npr.randint(3, 5, n)
        # b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        # z = cp.Variable(n)

        objective = cp.Maximize(a @ x)


        constraints = [x @ (u + y) <= c, cp.norm(x) <= 2*c]

        prob = RobustProblem(objective, constraints)
        prob.train(lr=0.001, num_iter=2, momentum=0.8, optimizer="SGD")
        # prob.solve()


    def test_multidim_learn(self):
        # Setup
        n = self.n
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), self.N)

        # Problem
        # y = np.ones(n)
        y = Parameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        x = cp.Variable(n)

        objective = cp.Maximize(y @ x)

        constraints = [u>=0]
        constraints += [np.ones(n)@u <= 10]

        prob = RobustProblem(objective, constraints)
        prob.train(lr=0.001, num_iter=2, momentum=0.8, optimizer="SGD")
        # prob.solve()

    def test_portfolio_intro(self):
        timestart = time.time()
        n = 2
        kappa = -0.01
        seed = 15
        np.random.seed(seed)
        dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20,
                20, 15, 10, 10, 10, 10, 10, 10])/10)[:n]

        # formulate the family parameter
        y_data = np.random.dirichlet(dist, self.N)
        y = Parameter(n, data=y_data)

        def gen_demand_intro(N, seed):
            np.random.seed(seed)
            sig = np.array([[0.5, -0.3], [-0.3, 0.4]])
            mu = np.array((0.3, 0.3))
            d_train = np.random.multivariate_normal(mu, sig, N)
            return d_train

        # formulate the uncertain parameter
        data = gen_demand_intro(self.N, seed=seed)
        u = UncertainParameter(n,uncertainty_set=Ellipsoidal(p=2,data=data))

        # formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()

        objective = cp.Minimize(t + 0.2*cp.norm(x - y, 1))
        constraints = [-x@u <= t, cp.sum(x) == 1, x >= 0]
        eval_exp = -x @ u + 0.2*cp.norm(x-y, 1)
        prob = RobustProblem(objective, constraints, eval_exp=eval_exp)
        # initialize reshaping parameters
        test_p = 0.1
        train, _ = train_test_split(data, test_size=int(
            data.shape[0]*test_p), random_state=5)
        init = sc.linalg.sqrtm(np.cov(train.T))
        init_bval = np.mean(train, axis=0)

        # Train A and b
        result = prob.train(lr=0.0001, num_iter=100, momentum=0.8,
                            optimizer="SGD",
                            seed=5, init_A=init, init_b=init_bval,
                            init_lam=0.5, init_mu=0.01,
                            mu_multiplier=1.001,
                            init_alpha=0., test_percentage=test_p, kappa=kappa,
                            n_jobs=8, random_init=True,
                            num_random_init=5, parallel = True,
                            position=False)

        timefin = time.time()
        timefin - timestart
        npt.assert_array_less(np.array(
            result.df["Violations_train"])[-1], kappa)

        # print(df)
        # # Grid search epsilon
        # result4 = prob.grid(epslst=np.linspace(0.01, 5, 50), \
        # init_A=init,
        #                     init_b=init_bval, seed=5,
        #                     init_alpha=0., test_percentage=test_p)
        # dfgrid = result4.df

        # result5 = prob.grid(epslst=np.linspace(0.01, 5, 10), \
        # init_A=result.A, init_b=result.b, seed=s,
        #                     init_alpha=0., test_percentage=test_p)
        # dfgrid2 = result5.df
        # print(dfgrid2)

    def test_torch_exp(self):
        # Setup
        n = 3
        num_instances = 5
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

        # Problem
        y = Parameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = np.ones(n)
        c = 5

        x = cp.Variable(n)
        a_tch = torch.tensor(a, dtype=float)
        c_tch = torch.tensor(c, dtype=float)

        objective = cp.Maximize(a @ x)
        constraints = [(u + y) @ x <= c, cp.norm(x) <= 2*c]

        #Variabls and parameters are discovered, from left to right, first at the objective
        #and then at each of the variables.
        #Therefore, in this example, the order of the variables/parameters is:
        #x, u, y
        x_test = torch.arange(n, dtype=float)
        u_test = torch.ones(n, dtype=float)
        y_test = torch.ones(n, dtype=float)
        vars_test = [x_test, u_test, y_test]

        def f_tch(x):
            return a_tch@x

        def g1_tch(x, u, y):
            return (u+y)@x-c_tch

        def g2_tch(x):
            return torch.norm(x)-2*c_tch

        prob = RobustProblem(objective, constraints)

        assert f_tch(x_test) == prob.f(*vars_test)
        assert g1_tch(x_test, u_test, y_test) == prob.g[0](*vars_test)
        assert len(prob.g)==1 #The second constraint is not saved to g since it has no uncertainty
