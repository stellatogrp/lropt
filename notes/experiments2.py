# %%
from pyparsing import dbl_quoted_string
import scipy as sc
from sklearn import datasets
from matplotlib.style import available
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import numpy as np
import cvxpy as cp
import torch
import time
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
import mosek


# %%

synthetic_returns = pd.read_csv('/Users/colebecker/Desktop/colbeck/Bartolomeo Research/mro_experiments/portfolio/sp500_synthetic_returns.csv').to_numpy()[:, 1:]

# %%
def createproblem_port(N, m):
    """Create the problem in cvxpy, minimize CVaR
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    dat = cp.Parameter((N, m))
    eps = cp.Parameter()
    w = cp.Parameter(N)
    a = -5

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    tau = cp.Variable()
    y = cp.Variable()
    # OBJECTIVE #
    objective = tau + y

    # CONSTRAINTS #
    constraints = [cp.multiply(eps, lam) + w@s <= y]
    constraints += [cp.hstack([a*tau]*N) + a*dat@x +
                    cp.hstack([cp.quad_over_lin(-a*x, 4*lam)]*N) <= s]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    # for k in range(2):
    #    constraints += [cp.sum(x[k*np.ceil(m/2):(k+1)*np.ceil(m/2)]) <= 0.50]
    constraints += [lam >= 0, y >=0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, s, tau, y, lam, dat, eps, w


def createproblem_portR(N, m):
    """Create the problem in cvxpy, minimize CVaR
    Parameters
    ----------
    N: int
        Number of data samples
    m: int
        Size of each data sample
    Returns
    -------
    The instance and parameters of the cvxpy problem
    """
    # PARAMETERS #
    datR = cp.Parameter((N, m))
    w = cp.Parameter(N)
    R = cp.Parameter((m,m))
    a = -5

    # VARIABLES #
    # weights, s_i, lambda, tau
    x = cp.Variable(m)
    x1 = cp.Variable(m)
    s = cp.Variable(N)
    lam = cp.Variable()
    tau = cp.Variable()
    y = cp.Variable()
    # OBJECTIVE #
    objective = tau + y

    # CONSTRAINTS #
    constraints = [lam + w@s <= y]
    constraints += [cp.hstack([a*tau]*N) + a*datR@x +
                    cp.hstack([cp.quad_over_lin(-a*x1, 4*lam)]*N) <= s]
    constraints += [R.T@x1 ==x]
    constraints += [cp.sum(x) == 1]
    constraints += [x >= 0, x <= 1]
    # for k in range(2):
    #    constraints += [cp.sum(x[k*np.ceil(m/2):(k+1)*np.ceil(m/2)]) <= 0.50]
    constraints += [lam >= 0, y >=0]
    # PROBLEM #
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, x1, s, tau, y, lam, datR, w, R
# %%
## CONSTANTS

N_test = 5000
N_val = 5000
m = synthetic_returns.shape[1]
K = 1


# %%

def outer_portfolio_problem(Dat_val, tau, x, alpha = 0.2):
    pass

def loss_evalR(test):
    """New outer loss function"""
    x_val, x1_val, s_val, tau_val,y_val, l_val = cvxpylayer(R_torch, w_torch, R_torch,solver_args={'solve_method':'ECOS'})  # check
    lt = torch.matmul(torch.tensor(test, requires_grad=True), x_val)
    lt2 = torch.matmul(torch.tensor(d_eval, requires_grad=True), x_val)
    obj = tau_val + y_val
    return torch.mean(torch.maximum(-5*lt -4*tau_val, tau_val)), obj, torch.mean(torch.maximum(-5*lt2 -4*tau_val, tau_val))

# %%

for r in range(5):
    Dat_exp, Dat_test = train_test_split(synthetic_returns, test_size = N_test)

    ################### DEFINE PROBLEM ################### 
    problem, x, x1, s, tau, y, l, datR,w, R= createproblem_portR(K, m)
    cvxpylayer = CvxpyLayer(problem, parameters=[
                datR,w,R], variables=[x, x1, s,tau,y,l])
    
    
    ################### Initialize R mat #################### 
    R_np = np.identity(m)
    R_torch = torch.tensor(R_np, requires_grad= True)
    
    ################# Define Torch Tracking Variables ############
    variables = [R_torch]
    opt = torch.optim.SGD(variables, lr=.02, momentum=.8)

    steps = 10
    for step in range(steps):
        Dat_train, Dat_val = train_test_split(Dat_exp, test_size = N_val)
        # define the parameters (WILL NEED TO CHANGE IF WE DO CLUSTERING)
        Dat_torch = torch.tensor(Dat_train.mean(axis = 0), requires_grad= True)
        Dat_R_torch = (R_torch @ Dat_torch.T).T

        weights = np.ones(1)
        w_torch = torch.tensor(weights, requires_grad= True)


        outer_loss, obj, evalv = loss_evalR(Dat_val)
        # prob += (evalv.item() <= (outer_loss.item() + 1e-10))


# %%

a = np.array([1,2,3,4])
b = np.array([2,2,2,2])
np.mean((a <= b).astype(int))
# %%
