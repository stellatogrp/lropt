import time

# import unittest
import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np

# import numpy.random as npr
import numpy.testing as npt
import scipy as sc
from sklearn.model_selection import train_test_split

import lropt

ATOL = 1e-04
RTOL = 1e-04

timestart = time.time()
seed = 15
np.random.seed(seed)
kappa = -0.01
n = 2 # number of time periods

# family parameter
dist = [1-0.5*np.sin(np.pi*(t-1)/12) for t in range(1, n+1)]
sig = np.random.rand(n, n)
sig = np.dot(sig, sig.T) # to ensure symmetric positive definiteness
y_data = np.random.multivariate_normal(dist, sig, 10)
y = lropt.Parameter(n, data=y_data)

# uncertainty distribution
def gen_demand_intro(N, seed):
    np.random.seed(seed)
    theta = 0.2
    d_hat = [1000*(1+0.5*np.sin(np.pi*(t-1)/12)) for t in range(1, n+1)]
    d_train = d_hat * np.random.uniform(1-theta, 1+theta, (N, n))
    # sig = np.array([[1, -0.01], [-0.02, 1.1]]) # np.array([[0.3, -0.4], [-0.5, 0.1]])
    # mu = np.array((1000, 1000)) # np.array((0.3, 0.3))
    # d_train = np.random.multivariate_normal(mu, sig, N)
    # d_train = np.exp(d_train)
    return d_train

# # objective
# def f_tch(x, y, u):
#     # x is a tensor that represents the cp.Variable x.
#     return x @ y

# # constraints with uncertainty
# def g_tch_1(x, y, u):
#     # x,y,u are tensors that represent the cp.Variable x and
#     # cp.Parameter y and u.
#     # The cp.Constant c is converted to a tensor
#     b = torch.tensor([1, 0]).double()
#     return torch.tensor(v_1) - torch.tensor(V_max) + x @ b.T - u @ b.T
# def g_tch_2(x, y, u):
#     b = torch.tensor([1, 0]).double()
#     return torch.tensor(V_min) - torch.tensor(v_1) - x @ b.T + u @ b.T
# def g_tch_3(x, y, u):
#     b = torch.tensor([1, 1]).double()
#     return torch.tensor(v_1) - torch.tensor(V_max) + x @ b.T - u @ b.T
# def g_tch_4(x, y, u):
#     b = torch.tensor([1, 1]).double()
#     return torch.tensor(V_min) - torch.tensor(v_1) - x @ b.T + u @ b.T

# def eval_tch(x, y, u):
#     return x @ y

# initialize problem
P = np.full(n, 567)
Q = 13600
V_min = 500
V_max = 2000
v_1 = 0
data = gen_demand_intro(600, seed=15)
u = lropt.UncertainParameter(n,
                             uncertainty_set=lropt.Box(data=data))
# u = lropt.UncertainParameter(n,
#                        uncertainty_set=lropt.Ellipsoidal(p=2,
#                                                    data=data))

# Formulate the Robust Problem
x = cp.Variable(n)
objective = cp.Minimize(x @ y)
constraints = [cp.sum(x) <= Q, x >= 0, x <= P]
for t in range(n):
    b = np.concatenate((np.ones(t+1), np.zeros(n-t-1)))
    constraints.append(x @ b - u @ b >= V_min - v_1)
    constraints.append(x @ b - u @ b <= V_max - v_1)
prob = lropt.RobustProblem(objective, constraints)
# prob = lropt.RobustProblem(objective, constraints,
#                      objective_torch=f_tch,
#                      constraints_torch=[
#                          g_tch_1, g_tch_2, g_tch_3, g_tch_4], eval_torch=eval_tch)

test_p = 0.1
s = 5
train, _ = train_test_split(data, test_size=int(
    data.shape[0]*test_p), random_state=s)

init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
init_bval= -init@np.mean(train, axis=0)
np.random.seed(15)
initn = np.random.rand(n, n) + 0.1*init + 0.5*np.eye(n)
###### how did we get this initialization?
init_bvaln = -initn@(np.mean(train, axis=0) - 0.3*np.ones(n))
###### how did we get this initialization?

# Train A and b
result = prob.train(lr=0.001, num_iter=100, momentum=0.8, # increase numbe of iterations
                    optimizer="SGD",
                    seed=s, init_A=initn, init_b=init_bvaln,
                    init_lam=0.5, init_mu=0.01,
                    mu_multiplier=1.001, init_alpha=0., test_percentage=test_p, kappa=kappa,
                    parallel = True, random_init=True, num_random_init=2, position = True)
timefin = time.time()
timefin - timestart
df = result.df
df_test = result.df_test
A_fin = result.A
b_fin = result.b

# npt.assert_allclose(np.array(
#     result.df["Violations_train"])[-1], 0.260812,
#     rtol=RTOL, atol=ATOL)
npt.assert_array_less(np.array(df["Violations_train"])[-1], kappa) # Violations_train is cvar

print(df)
print(df["Violations_train"])

print("done")

# def port_prob(A_final, b_final, scene):
#     n = 2
#     u = lropt.UncertainParameter(n,
#                              uncertainty_set=lropt.Box(a = A_final, b = b_final))
#     # Formulate the Robust Problem
#     x = cp.Variable(n)
#     p = cp.Parameter(n)
#     p.value = y_data[scene]
#     objective = cp.Minimize(x @ p)

#     constraints = [cp.sum(x) <= Q, x >= 0, x <= P]
#     for t in range(n):
#         b = np.concatenate((np.ones(t+1), np.zeros(n-t-1)))
#         constraints.append(x @ b - u @ b >= V_min - v_1)
#         constraints.append(x @ b - u @ b <= V_max - v_1)

#     prob = lropt.RobustProblem(objective, constraints)
#     prob.solve()
#     x_opt = x.value
#     return x_opt

# print(port_prob(A_fin, b_fin, 0))

# print("done")
