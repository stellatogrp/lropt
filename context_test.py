import cvxpy as cp
import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split

import lropt

seed = 15
np.random.seed(seed)
n = 2 # number of time periods
theta = 0.01

#initialize problem
cost = np.array([1.0, 100.0])/100000000
P = 567 # max production capacity at each time period
Q = 1100 # max production capacity for all periods
V_min = 500 # minimum inventory
V_max = 2000 # maximum inventory
V_1 = 2000 # initial inventory

# uncertainty distribution
def gen_demand_intro(N, seed):
    np.random.seed(seed)
    # d_hat = [1000*(1+0.5*np.sin(np.pi*(t-1)/12)) for t in range(1, n+2)]
    d_hat = [1000, 1129, 1000] # [1000, 1070, 1000]
    # sig = np.array([[0.5, -0.3, 0.2], [-0.3, 0.4, -0.1], [0.1, 0.3, 0.1]])
    # d_train_temp = np.random.multivariate_normal(d_hat, sig, N)
    d_train_temp = d_hat * np.random.uniform(1-theta, 1+theta, (N, n+1))

    d_train = []
    for i, row in enumerate(d_train_temp):
        # d_train.append(row[0:2])
        d_train.append(row[(i%n):(i%n + 2)]) # get demand time t and t+1
    d_train = np.array(d_train)

    return d_hat, d_train

test_p = 0.2
s = 5
d_hat_all, data = gen_demand_intro(100, seed=15) # demand
print(d_hat_all)
d_hat = d_hat_all[:n]

# d_hat = np.array([1000, 1129]) # nominal demand
# data = d_hat * np.random.uniform(1-theta, 1+theta, (100, n))

train, test = train_test_split(data, test_size=int(
    data.shape[0]*test_p), random_state=s)

# simulating inventories at each time beforehand (no uncertainty in demand)
x = cp.Variable(n)
objective = cp.Minimize(x @ cost)
constraints = [cp.sum(x) <= Q, x >=0, x <= P]
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
print(y_data)
y_data = np.tile(y_data, (int(100/n), 1))

# y_data = np.array([1, V_1])
# y_data = np.tile(y_data, (100, 1))

y = lropt.Parameter(2, data = y_data)
u = lropt.UncertainParameter(2, uncertainty_set=lropt.Ellipsoidal(data=data, rho=1, p=2))
x = cp.Variable(2)
objective = cp.Minimize(x @ cost)
constraints = [cp.sum(x) <= Q, x >= 0, x <= P]
b = {}
for t in range(n): # for n = 2: one constraint for time t, one constrant for time t and t+1
    b[t] = np.concatenate((np.ones(t+1), np.zeros(n-t-1)))
    constraints.append(y @ np.array([0, 1]) + x @ b[t] - u @ b[t] >= V_min)
    constraints.append(y @ np.array([0, 1]) + x @ b[t] - u @ b[t] <= V_max)

prob = lropt.RobustProblem(objective, constraints)

init = sc.linalg.sqrtm(np.cov(train.T))
init_bval = np.mean(train, axis=0)
# Train A and b
result = prob.train(lr=0.00001, train_size = False, num_iter=300, optimizer="SGD",seed=8,
                    init_A=init, init_b=init_bval,
                    init_lam=1, init_mu=1, mu_multiplier=1.001, kappa=0., init_alpha=0.,
                    test_percentage = test_p, save_history = True, quantiles = (0.4,0.6),
                    lr_step_size = 50, lr_gamma = 0.5, random_init = False, num_random_init = 5,
                    parallel = False, position = False, eta=0.05, contextual = True)
df = result.df
A_fin = result.A
b_fin = result.b
