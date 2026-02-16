import sys
import unittest
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

import cvxro

ATOL = 1e-4
RTOL = 1e-4
SOLVER = cp.CLARABEL
SOLVER_SETTINGS = { "equilibrate_enable": False, "verbose": False }
sys.path.append('..')
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,

    "font.size":24,
    "font.family": "serif"
})

class TestScenario(unittest.TestCase):


    def test_scenario(self):
        # Formulate constants
        n = 2
        N = 500
        k_init = np.array([4.,5.])
        p = np.array([5,6.5])
        def gen_demand_cor(N,seed,x):
            np.random.seed(seed)
            sig = np.eye(2)
            mu = np.array((6,7))
            points_list = []
            for i in range(N):
                mu_shift = -0.4*x[i]
                newpoint = np.random.multivariate_normal(mu+mu_shift,sig)
                points_list.append(newpoint)
            return np.vstack(points_list)

        def calc_eval(x,p,k,u,t):
            val = 0
            vio = 0
            for i in range(u.shape[0]):
                val_cur = k[i]@x + np.max([-p[i][0]*x[0] - p[i][1]*x[1],
                                           -p[i][0]*x[0] - p[i][1]*u[i][1],
                                           -p[i][0]*u[i][0] - p[i][1]*x[1],
                                             -p[i][0]*u[i][0]- p[i][1]*u[i][1]])
                val+= val_cur
                vio += (val_cur >= t)
            return val/u.shape[0], vio/u.shape[0]

        # Generate data
        s = 8
        np.random.seed(s)
        num_scenarios = N
        num_reps = int(N/num_scenarios)
        k_data = np.maximum(0.5,k_init + np.random.normal(0,3,(num_scenarios,n)))
        p_data = k_data + np.maximum(0,np.random.normal(0,3,(num_scenarios,n)))
        p_data = np.vstack([p_data]*num_reps)
        k_data = np.vstack([k_data]*num_reps)

        data = gen_demand_cor(N,seed=5,x=p_data)
        test_p = 0.9
        # setup intial A, b
        test_indices = np.random.choice(N,int(test_p*N), replace=False)
        train_indices = [i for i in range(N) if i not in test_indices]
        train = np.array([data[i] for i in train_indices])
        test = np.array([data[i] for i in test_indices])
        k_train = np.array([k_data[i] for i in train_indices])
        k_test = np.array([k_data[i] for i in test_indices])
        p_train = np.array([p_data[i] for i in train_indices])
        p_test = np.array([p_data[i] for i in test_indices])


        # Using the scenario sets. Both u and p are treated as uncertain params
        u = cvxro.UncertainParameter(n,
                                uncertainty_set=cvxro.Scenario(
                                                            data=train))
        x_r = cp.Variable(n)
        t = cp.Variable()
        # Cartesian set to false, since we do not want to permutate the
        # realizations
        k = cvxro.UncertainParameter(n,
                                uncertainty_set=cvxro.Scenario(
                                                            data=k_train, cartesian = False))
        p = cp.Parameter(n)
        p_x = cp.Variable(n)
        objective = cp.Minimize(t)
        p.value = p_train[0]
        constraints = [cvxro.max_of_uncertain([-p[0]*x_r[0] - p[1]*x_r[1],
                                               -p[0]*x_r[0] - p_x[1]*u[1],
                                               -p_x[0]*u[0] - p[1]*x_r[1],
                                                 -p_x[0]*u[0]- p_x[1]*u[1]])
                                                 + k@x_r <= t]
        constraints += [p_x == p]
        constraints += [x_r >= 0]

        eval_exp = k@x_r + cp.maximum(-p[0]*x_r[0] - p[1]*x_r[1],
                                      -p[0]*x_r[0] - p[1]*u[1], -p[0]*u[0]
                                        - p[1]*x_r[1], -p[0]*u[0]- p[1]*u[1])

        prob_rob = cvxro.RobustProblem(objective, constraints,eval_exp = eval_exp)
        prob_rob.solve()

        eval, prob_vio = calc_eval(x_r.value, p_test, k_test,test,t.value)


        # Without using the scenario set
        x_r1 = cp.Variable(n)
        t = cp.Variable()
        k = cp.Parameter(n)
        p = cp.Parameter(n)
        p_x = cp.Variable(n)
        objective = cp.Minimize(t)
        constraints = []
        for i in range(train.shape[0]):
            constraints += [-p_train[0][0]*x_r1[0] -
                            p_train[0][1]*x_r1[1] + k_train[i]@x_r1  <= t]
            constraints += [-p_train[0][0]*x_r1[0]
                            - p_train[0][1]*train[i][1] + k_train[i]@x_r1  <= t]
            constraints += [ -p_train[0][0]*train[i][0]
                            - p_train[0][1]*x_r1[1]  + k_train[i]@x_r1 <= t]
            constraints+= [-p_train[0][0]*train[i][0]
                           - p_train[0][1]*train[i][1]  + k_train[i]@x_r1 <= t]
        constraints += [x_r1 >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        eval1, prob_vio1 = calc_eval(x_r1.value, p_test, k_test,test,t.value)

        npt.assert_allclose(x_r.value, x_r1.value, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(eval, eval1, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(prob_vio, prob_vio1, rtol=RTOL, atol=ATOL)
