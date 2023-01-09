import unittest

import cvxpy as cp
# import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from lro.robust_problem import RobustProblem


class TestNonrobustProblems(unittest.TestCase):

    def test_nonrobust_dualize(self):

        n = 4
        x = cp.Variable(n)
        b = 10.
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        c = np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        objective = cp.Minimize(c @ x)
        constraints = [-b_unc @ x + cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_lro = RobustProblem(objective, constraints)
        prob_cvxpy = cp.Problem(objective, constraints)

        problem = prob_lro.dualize_constraints()
        problem.solve()
        prob_cvxpy.solve()

        npt.assert_allclose(prob_lro.value, prob_cvxpy.value)

    def test_nonrobust_solve(self):

        n = 4
        x = cp.Variable(n)
        b = 10.
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        c = np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        objective = cp.Minimize(c @ x)
        constraints = [-b_unc @ x + cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_lro = RobustProblem(objective, constraints)
        prob_cvxpy = cp.Problem(objective, constraints)

        prob_lro.solve()
        prob_cvxpy.solve()

        npt.assert_allclose(prob_lro.value, prob_cvxpy.value)
