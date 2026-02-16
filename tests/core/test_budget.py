import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np

# import numpy.random as npr
import numpy.testing as npt

from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.budget import Budget
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# ATOL = 1e-4
# RTOL = 1e-4
# SOLVER = cp.CLARABEL
# SOLVER_SETTINGS = { "equilibrate_enable": False, "verbose": False }

class TestBudgetUncertainty(unittest.TestCase):

    def test_budget(self):
        m = 5

        budget_u = UncertainParameter(m, uncertainty_set=Budget(rho1=2., rho2=1., a=np.eye(m)))
        n = 4
        # formulate cvxpy variable
        x_r = cp.Variable(n)
        # formulate problem constants
        P = 3. * np.eye(m)[:n, :]
        P1 = 3*np.random.rand(n, m)
        a = 0.1 * np.random.rand(n)
        c = np.random.rand(n)

        # formulate objective
        objective = cp.Minimize(c@x_r)
        # formulate constraints
        constraints = []
        constraints += [(P@budget_u + a) @ x_r <= 10]
        constraints += [P1@budget_u <= x_r]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve()

        x_cvxpy = cp.Variable(n)
        y = cp.Variable(m)
        z = cp.Variable((n, m))

        # formulate objective
        objective = cp.Minimize(c@x_cvxpy)

        # formulate constraints
        constraints = []
        constraints += [a@x_cvxpy + 2 *
                        cp.norm(y, 1) + cp.norm(P.T@x_cvxpy-y, np.inf) <= 10]
        for i in range(n):
            constraints += [-x_cvxpy[i] + 2*cp.norm(z[i], 1) +
                            cp.norm(P1.T@(np.eye(n)[i]) - z[i], np.inf) <= 0]

        # formulate Robust Problem
        prob_cvxpy = cp.Problem(objective, constraints)

        # solve
        prob_cvxpy.solve()

        npt.assert_allclose(x_r.value, x_cvxpy.value, rtol=RTOL, atol=ATOL)
