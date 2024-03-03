# import cvxpy as cp
# import torch
# from cvxpylayers.torch import CvxpyLayer

# n, m = 4, 5
# x = cp.Variable(n)
# A = cp.Parameter((n, n))
# b = cp.Parameter(n)
# # A = torch.nn.Parameter(torch.randn(m, n, n))
# # b = torch.nn.Parameter(torch.randn(m, n))
# constraints = [x >= 0]
# objective = cp.Minimize(0.5 * cp.norm(A @ x - b, p = 1))
# problem = cp.Problem(objective, constraints)

# cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
# A_tch = torch.nn.Parameter(torch.randn(m, n, n, requires_grad=True))
# b_tch = torch.nn.Parameter(torch.randn(m, n, requires_grad=True))

# # solve the problem
# solution, = cvxpylayer(A_tch, b_tch)

# print(solution)


# import cvxpy as cp

# # import matplotlib.pyplot as plt
# import numpy as np
# import numpy.random as npr

# # from tests.settings import SOLVER
# from lropt.parameter import Parameter
# from lropt.robust_problem import RobustProblem
# from lropt.uncertain import UncertainParameter
# from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

# # Setup
# n = 4
# norms = npr.multivariate_normal(
#             np.zeros(n), np.eye(n), 100)
# data = np.exp(norms)
# num_instances = 5
# y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

# # Problem
# # y = np.ones(n)
# y = Parameter(n, data=y_data)
# u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=data))

# a = npr.randint(3, 5, n)
# # b = np.ones(n)
# c = 5

# x = cp.Variable(n)
# # z = cp.Variable(n)

# objective = cp.Maximize(a @ x)


# constraints = [x @ (u + y) <= c, cp.norm(x) <= 2*c]

# prob = RobustProblem(objective, constraints)
# res = prob.train(lr=0.001, num_iter=2, momentum=0.8, optimizer="SGD")
# print(res.weights)
# print("DONE")



# import cvxpy as cp

# # import matplotlib.pyplot as plt
# import numpy as np
# import numpy.random as npr

# # from tests.settings import SOLVER
# from lropt.parameter import Parameter
# from lropt.robust_problem import RobustProblem
# from lropt.uncertain import UncertainParameter
# from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

# # import pandas as pd
# # import torch

# n = 3
# num_instances = 5
# # y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)
# y_data = np.array([[-0.7, -0.5, 0.5],
#           [-0.3, 0.2, 0.9],
#           [-0.7, -0.5, 0.5],
#           [-0.3, 0.02, 0.4],
#           [-0.1, 0.1, -1]])
# y = Parameter(n, data=y_data)

# # Problem
# norms = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)
# data = np.exp(norms)
# u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=data))

# a = npr.randint(3, 5, n)
# c = 5

# x = cp.Variable(n)
# objective = cp.Maximize(a @ x)
# constraints = [x @ (u + y) <= c, cp.norm(x) <= 2*c]

# prob = RobustProblem(objective, constraints)
# result = prob.train(lr=0.001, num_iter=2, momentum=0.8, optimizer="SGD", batch_percentage = 1)

# print("done")


import cvxpy as cp
import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split

import lropt

seed = 15
np.random.seed(seed)
kappa = -0.01
n = 2 # number of time periods

# uncertainty distribution
def gen_demand_intro(N, seed):
    np.random.seed(seed)
    theta = 0.025
    # nominal demand (t = 1, 2, 3)
    d_hat = [1000*(1+0.5*np.sin(np.pi*(t-1)/12)) for t in range(1, n+2)]
    d_train_temp = d_hat * np.random.uniform(1-theta, 1+theta, (N, n+1))

    d_train = []
    for i, row in enumerate(d_train_temp):
        if i%2 != 0:
            d_train.append(row[-2:]) # get demand at times t = 2,3 when y = 2
        else:
            d_train.append(row[:2]) # get demand at times t = 1,2 when y = 1
    d_train = np.array(d_train)

    return d_hat, d_train

# initialize problem
cost = np.array([1, 0.87])
P = np.full(n, 567) # max production capacity at each time period
Q = 1100 # max production capacity for all periods
V_min = 500 # minimum inventory
V_max = 2000 # maximum inventory
V_1 = 2000 # initial inventory
d_hat_all, data = gen_demand_intro(100, seed=15) # demand
d_hat = d_hat_all[:2]
print(d_hat)

# simulating inventories at each time beforehand (no uncertainty in demand)
x = cp.Variable(n)
objective = cp.Minimize(x @ cost)
constraints = [cp.sum(x) <= Q, x >=0, x<= P]
for t in range(n):
    b = np.concatenate((np.ones(t+1), np.zeros(n-t-1)))
    constraints.append(x @ b - d_hat @ b >= V_min - V_1)
    constraints.append(x @ b - d_hat @ b <= V_max - V_1)

prob = cp.Problem(objective, constraints)
prob.solve()
opt_obj = prob.value
opt_prod = x.value
x.value

inventories = np.zeros(n)
inventories[0] = V_1
for t in range(1, n):
    inventories[t] = inventories[t-1] + opt_prod[t-1] - d_hat[t-1]

y_data = np.array([[t, inventories[t-1]] for t in range(1, n + 1)])
y_data = np.tile(y_data, (50, 1))

y = lropt.Parameter(n, data = y_data)
u = lropt.UncertainParameter(n, uncertainty_set=lropt.Box(data=data))
x = cp.Variable(n)
objective = cp.Minimize(x @ cost)
constraints = [cp.sum(x) <= Q, x >= 0, x <= P]

for t in range(n): # for n = 2: one constraint for time t, one constrant for time t and t+1
    b = np.concatenate((np.ones(t+1), np.zeros(n-t-1)))
    constraints.append(y @ np.array([0, 1]) + x @ b - u @ b >= V_min)
    constraints.append(y @ np.array([0, 1]) + x @ b - u @ b <= V_max)

prob = lropt.RobustProblem(objective, constraints)

test_p = 0.1
s = 15
train, _ = train_test_split(data, test_size=int(data.shape[0]*test_p), random_state=s)

initn = sc.linalg.sqrtm(np.cov(train.T))
np.random.seed(15)
init_bvaln = np.mean(train, axis=0)

# Train A and b - linear predictor
result = prob.train(lr=0.001, num_iter=20, momentum=0.8,
                    optimizer="SGD", predictor = "CONSTANT",
                    seed=s, init_A=initn, init_b=init_bvaln,
                    init_lam=0.5, init_mu=0.01,
                    mu_multiplier=1.001, init_alpha=0., test_percentage=test_p, kappa=kappa,
                    parallel = False, random_init=True, num_random_init=1,
                    position = True, save_history = True)
print("done")
