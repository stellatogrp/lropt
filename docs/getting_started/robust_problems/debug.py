import numpy as np
import cvxpy as cp
import lropt
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(1)
T = 10  # Length of the horizon
F = 5   # Number of facilities
N = 8  # Number of candidate locations
M = 1100
NUM_DEC = 3

ETA = 100.0
RHO = 0.3

c = np.random.rand(F)
C = np.random.rand(F)
K = np.random.rand(F)
d = np.random.rand(F, N)
D_value = np.random.rand(N* T)  # Deterministic demand

D = lropt.UncertainParameter(N*T, uncertainty_set = lropt.Ellipsoidal(b = D_value, rho = RHO))

x = cp.Variable((F*T,N), nonneg = True)
P = cp.Variable((F, T), nonneg=True)
P_T = P.T
Z = cp.Variable(F)
I = cp.Variable(F)
epsilon = np.random.rand(T)
theta = cp.Variable()

f1 = np.tile(ETA - d, (T, 1)).flatten()
revenue = cp.sum(cp.reshape(cp.multiply((f1), x.flatten()), (F,N*T)) @ D)
cost_production = cp.sum(c @ P)
fixed_costs = cp.sum(cp.multiply(C, Z))
penalties = cp.sum(cp.multiply(K, I))
x_reshaped = cp.reshape(x, (F, N*T))


constraints = [
revenue - cost_production - fixed_costs - penalties >= theta,
cp.sum(x.flatten()) <= 1,
x.flatten()>=0,
Z <= M*I,
cp.sum(x_reshaped @ D) <=cp.sum(P)
]

for t in range(T):
    constraints.append(P_T[t]<=Z)

objective = cp.Maximize(theta)
prob = lropt.RobustProblem(objective, constraints)
prob.solve()

print(f"The robust optimal value using  is {round(float(theta.value), NUM_DEC)}")
