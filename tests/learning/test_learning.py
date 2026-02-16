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

from cvxro import Trainer, TrainerSettings, max_of_uncertain
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL

ATOL = 1e-5
RTOL = 1e-5


class TestEllipsoidalUncertainty(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.N = 20
        norms = npr.multivariate_normal(np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)
        self.ATOL = ATOL
        self.RTOL = RTOL

    # @unittest.skip('learning not ready')
    def test_simple_learn(self):
        # Setup
        n = self.n
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), self.N)

        # Problem
        # y = np.ones(n)
        y = ContextParameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = npr.randint(3, 5, n)
        # b = np.ones(n)
        c = 5

        x = cp.Variable(n)
        # z = cp.Variable(n)

        objective = cp.Maximize(a @ x)

        constraints = [x @ (u + y) <= c, cp.norm(x) <= 2 * c]

        prob = RobustProblem(objective, constraints)
        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.lr = 0.001
        settings.num_iter = 2
        settings.momentum = 0.8
        settings.optimizer = "SGD"
        settings.parallel = False
        trainer.train(settings=settings)
        # prob.solve()

    def test_multidim_learn(self):
        # Setup
        n = self.n
        np.random.seed(0)
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), self.N)

        # Problem
        # y = np.ones(n)
        y = ContextParameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        x = cp.Variable(n)

        objective = cp.Maximize(y @ x)

        constraints = [u >= -100]
        constraints += [cp.sum(x) == 1, x >= 0]
        constraints += [np.ones(n) @ u <= 100]

        prob = RobustProblem(objective, constraints)
        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.lr = 0.001
        settings.num_iter = 10
        settings.momentum = 0.8
        settings.optimizer = "SGD"
        settings.parallel = False
        settings.num_random_init = 1
        trainer.train(settings=settings)
        # prob.solve()

    def test_portfolio_intro(self):
        timestart = time.time()
        n = 2
        kappa = -0.001
        seed = 15
        np.random.seed(seed)
        dist = (np.array([25, 10, 60, 50, 40, 30, 30, 20, 20, 15, 10, 10, 10, 10, 10, 10]) / 10)[:n]

        # formulate the family parameter
        y_data = np.random.dirichlet(dist, self.N)
        y = ContextParameter(n, data=y_data)

        def gen_demand_intro(N, seed):
            np.random.seed(seed)
            sig = np.array([[0.5, -0.3], [-0.3, 0.4]])
            mu = np.array((0.3, 0.3))
            d_train = np.random.multivariate_normal(mu, sig, N)
            return d_train

        # formulate the uncertain parameter
        data = gen_demand_intro(self.N, seed=seed)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(p=2, data=data))

        # formulate the Robust Problem
        x = cp.Variable(n)
        t = cp.Variable()

        objective = cp.Minimize(t + 0.2 * cp.norm(x - y, 1))
        constraints = [-x @ u <= t, cp.sum(x) == 1, x >= 0]
        eval_exp = -x @ u + 0.2 * cp.norm(x - y, 1)
        prob = RobustProblem(objective, constraints, eval_exp=eval_exp)
        # initialize reshaping parameters
        test_p = 0.1
        train, _ = train_test_split(data, test_size=int(data.shape[0] * test_p), random_state=5)
        init = sc.linalg.sqrtm(np.cov(train.T))
        init_bval = np.mean(train, axis=0)

        # Train A and b
        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.lr = 0.0001
        settings.num_iter = 15
        settings.momentum = 0.8
        settings.optimizer = "SGD"
        settings.seed = 5
        settings.init_A = init
        settings.init_b = init_bval
        settings.init_lam = 0.5
        settings.init_mu = 0.01
        settings.mu_multiplier = 1.001
        settings.init_alpha = 0.0
        settings.test_percentage = test_p
        settings.validate_percentage = 0.01
        settings.kappa = kappa
        settings.n_jobs = 8
        settings.random_init = True
        settings.num_random_init = 2
        settings.parallel = False
        settings.position = False
        result = trainer.train(settings=settings)

        timefin = time.time()
        timefin - timestart
        npt.assert_array_less(np.array(result.df["Violations_train"])[-1], 0.0)

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

    def test_max_learning(self):
        n = 2
        N = 100
        k = np.array([4.0, 5.0])
        p = np.array([5, 6.5])

        def gen_demand_intro(N, seed):
            np.random.seed(seed)
            sig = np.array([[0.6, -0.4], [-0.3, 0.1]])
            mu = np.array((0.9, 0.7))
            norms = np.random.multivariate_normal(mu, sig, N)
            d_train = np.exp(norms)
            return d_train

        # Generate data
        data = gen_demand_intro(N, seed=5)

        num_scenarios = 10
        num_reps = int(N / 10)
        p_data = p + np.random.normal(0, 1, (num_scenarios, n))
        k_data = k + np.random.normal(0, 1, (num_scenarios, n))
        p_data = np.vstack([p_data] * num_reps)
        k_data = np.vstack([k_data] * num_reps)

        # Formulate uncertainty set
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=data))
        # Formulate the Robust Problem
        x_r = cp.Variable(n)
        t = cp.Variable()
        k = ContextParameter(2, data=k_data)
        p = ContextParameter(2, data=p_data)
        p_x = cp.Variable(n)
        objective = cp.Minimize(t)
        constraints = [
            max_of_uncertain(
                [
                    -p[0] * x_r[0] - p[1] * x_r[1],
                    -p[0] * x_r[0] - p_x[1] * u[1],
                    -p_x[0] * u[0] - p[1] * x_r[1],
                    -p_x[0] * u[0] - p_x[1] * u[1],
                ]
            )
            + k @ x_r
            <= t
        ]
        constraints += [p_x == p]
        constraints += [x_r >= 0]

        eval_exp = k @ x_r + cp.maximum(
            -p[0] * x_r[0] - p[1] * x_r[1],
            -p[0] * x_r[0] - p[1] * u[1],
            -p[0] * u[0] - p[1] * x_r[1],
            -p[0] * u[0] - p[1] * u[1],
        )

        prob = RobustProblem(objective, constraints, eval_exp=eval_exp)
        test_p = 0.9
        s = 13
        # setup intial A, b
        train, test = train_test_split(data, test_size=int(data.shape[0] * test_p), random_state=s)
        init = sc.linalg.sqrtm(np.cov(train.T))
        init_bval = np.mean(train, axis=0)
        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.lr = 0.0001
        settings.train_size = False
        settings.num_iter = 3
        settings.optimizer = "SGD"
        settings.seed = 8
        settings.init_A = init
        settings.init_b = init_bval
        settings.init_lam = 1
        settings.init_mu = 1
        settings.mu_multiplier = 1.001
        settings.kappa = 0.0
        settings.init_alpha = 0.0
        settings.test_percentage = test_p
        settings.save_history = True
        settings.quantiles = (0.4, 0.6)
        settings.lr_step_size = 50
        settings.lr_gamma = 0.5
        settings.random_init = False
        settings.num_random_init = 2
        settings.parallel = False
        settings.position = False
        settings.validate_percentage = 0.01
        settings.eta = 0.05
        result = trainer.train(settings=settings)
        npt.assert_array_less(np.array(result.df["Violations_train"])[-1], 0.2)

    @unittest.skip("This test requires some changes. Irina, I need your help.")
    def test_torch_exp(self):
        # Setup
        n = 3
        num_instances = self.N
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

        # Problem
        y = ContextParameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        a = np.ones(n)
        c = 5

        x = cp.Variable(n)
        a_tch = torch.tensor(a, dtype=float)
        c_tch = torch.tensor(c, dtype=float)

        objective = cp.Maximize(a @ x)
        constraints = [(u + y) @ x <= c, cp.norm(x) <= 2 * c]

        # Variabls and parameters are discovered, from left to right, first at the objective
        # and then at each of the variables.
        # Therefore, in this example, the order of the variables/parameters is:
        # x, u, y
        x_test = torch.arange(n, dtype=float)
        u_test = torch.ones(n, dtype=float)
        y_test = torch.ones(n, dtype=float)
        vars_test = [x_test, u_test, y_test]

        def f_tch(x):
            return a_tch @ x

        def g1_tch(x, u, y):
            return (u + y) @ x - c_tch

        def g2_tch(x):
            return torch.norm(x) - 2 * c_tch

        prob = RobustProblem(objective, constraints)

        # TODO: Changes should start here
        prob.remove_uncertainty()
        assert f_tch(x_test) == prob.problem_canon.f(*vars_test)
        assert g1_tch(x_test, u_test, y_test) == prob.problem_canon.g[0](*vars_test)
        assert (
            len(prob.g) == 1
        )  # The second constraint is not saved to g since it has no uncertainty

    def test_news_learning(self):
        n = 2
        N = 100
        k = np.array([4.0, 5.0])
        p = np.array([5, 6.5])

        def gen_demand_intro(N, seed):
            np.random.seed(seed)
            sig = np.array([[0.6, -0.3], [-0.3, 0.1]])
            mu = np.array((1.1, 1.7))
            norms = np.random.multivariate_normal(mu, sig, N)
            d_train = np.exp(norms)
            return d_train

        # Generate data
        data = gen_demand_intro(N, seed=18)

        num_scenarios = N
        num_reps = int(N / num_scenarios)
        k_data = k + np.random.normal(0, 0.5, (num_scenarios, n))
        p_data = k_data + np.maximum(0, np.random.normal(0, 0.5, (num_scenarios, n)))
        p_data = np.vstack([p_data] * num_reps)
        k_data = np.vstack([k_data] * num_reps)

        # Formulate uncertainty set
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=data))
        # Formulate the Robust Problem
        x_r = cp.Variable(n)
        t = cp.Variable()
        k = ContextParameter(2, data=k_data)
        p = ContextParameter(2, data=p_data)
        p_x = cp.Variable(n)
        objective = cp.Minimize(t)
        constraints = [
            max_of_uncertain(
                [
                    -p[0] * x_r[0] - p[1] * x_r[1],
                    -p[0] * x_r[0] - p_x[1] * u[1],
                    -p_x[0] * u[0] - p[1] * x_r[1],
                    -p_x[0] * u[0] - p_x[1] * u[1],
                ]
            )
            + k @ x_r
            <= t
        ]
        constraints += [p_x == p]
        constraints += [x_r >= 0]

        eval_exp = k @ x_r + cp.maximum(
            -p[0] * x_r[0] - p[1] * x_r[1],
            -p[0] * x_r[0] - p[1] * u[1],
            -p[0] * u[0] - p[1] * x_r[1],
            -p[0] * u[0] - p[1] * u[1],
        )

        prob = RobustProblem(objective, constraints, eval_exp=eval_exp)
        test_p = 0.9
        s = 8

        # setup intial A, b
        train, test = train_test_split(data, test_size=int(data.shape[0] * test_p), random_state=s)

        np.random.seed(15)
        initn = sc.linalg.sqrtm(np.cov(train.T))
        init_bvaln = np.mean(train, axis=0)
        # Train A and b
        from cvxro import Trainer

        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.lr = 0.0001
        settings.train_size = False
        settings.num_iter = 10
        settings.optimizer = "SGD"
        settings.seed = 5
        settings.init_A = initn
        settings.init_b = init_bvaln
        settings.init_lam = 0.1
        settings.init_mu = 0.1
        settings.mu_multiplier = 1.001
        settings.kappa = 0.0
        settings.init_alpha = 0.0
        settings.test_percentage = test_p
        settings.validate_percentage = 0.01
        settings.save_history = True
        settings.quantiles = (0.4, 0.6)
        settings.lr_step_size = 50
        settings.lr_gamma = 0.5
        settings.random_init = True
        settings.num_random_init = 2
        settings.parallel = False
        settings.position = False
        settings.eta = 0.3
        settings.contextual = True
        result = trainer.train(settings=settings)
        result.df
        # A_fin = result.A
        # b_fin = result.b

        # trainer.grid(rholst=[0.1,1],init_A=A_fin, init_b=b_fin, seed=s,
        # init_alpha=0., test_percentage=test_p,quantiles = (0.3,0.7),
        # contextual = True, linear = result._linear)
