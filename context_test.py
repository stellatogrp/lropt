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
