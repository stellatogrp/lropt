import cvxpy as cp
import scipy as sc
from sklearn import datasets
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import torch
import pandas as pd
import lropt
import matplotlib.pyplot as plt
RTOL = 1e-04
ATOL = 1e-04
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    "font.size":18,
    "font.family": "serif"
})
colors = ["tab:blue", "tab:green", "tab:orange",
          "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive"]
np.random.seed(0)

#restate the ellipsoidal set
np.random.seed(0)
m = 5
data = np.random.normal(0,1,size = (100,m))

ellip_u = lropt.UncertainParameter(m,
                                  uncertainty_set = lropt.Ellipsoidal(p = 2,
                                                                      rho=2., b = np.mean(data, axis = 0)))
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
constraints = [(P@ellip_u + a)@ x_r <= 10]

# formulate Robust Problem
prob_robust = lropt.RobustProblem(objective, constraints)

# solve
prob_robust.solve()
print("LRO objective value: ", prob_robust.objective.value, "\nLRO x: ", x_r.value)
