import numpy as np
import cvxpy as cp
import lropt
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(1)
T = 10
F = 5
N = 8
M = 1100
ETA = 100.0
RHO = 0.1

c = np.random.rand(F)
c_dash = np.random.rand(F)
k = np.random.rand(F)
d_ship = np.random.rand(F, N) #Cost of shipping from one location to another
d_star = np.random.rand(N*T)  # Deterministic demand
epsilon = np.random.rand(T)
x = {}
for i in range(F):
    x[i] = cp.Variable((N, T))

d_u = lropt.UncertainParameter(N*T, uncertainty_set = lropt.Ellipsoidal(b = d_star, rho = RHO)) #Flattened Uncertain Parameter - LROPT only supports one dimensional uncertain parameters
# d_u = d_star
#x = cp.Variable((T,F*N), nonneg = True)  #Flattened cp.Variable - LROPT only supports two dimensional variables
p = cp.Variable((F, T))
z = cp.Variable(F)
y = cp.Variable(F, boolean=True) # TODO: change it and call it "y"
theta = cp.Variable()


revenue = cp.sum([((ETA - np.diag(d_ship[i])) @ x[i]).flatten() @ d_u for i in range(F)])
cost_production = cp.sum(c @ p)
fixed_costs = c_dash@z
penalties = k@y

#x_reshaped = cp.reshape(x, (F, N*T)) #Reshape x for easy multiplication with d_u

constraints = [
revenue - cost_production - fixed_costs - penalties >= theta,
z <= M*y,
]

constraints.append(cp.sum([x[i] for i in range(F)]) <= 1)
for i in range(F):
    for t in range(T):
        constraints.append(cp.sum([x[i][j,t] * d_u[j*T + t] for j in range(N)]) <=p[i, t])

for i in range(F):
    constraints.append(x[i]>=0)

for t in range(T):
    constraints.append(p.T[t]<=z)
constraints += [p>=0]

objective = cp.Maximize(theta)
prob = lropt.RobustProblem(objective, constraints)
prob.solve(solver = cp.GUROBI,verbose = True)
