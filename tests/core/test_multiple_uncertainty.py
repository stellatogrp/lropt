import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.testing as npt

from cvxro import max_of_uncertain
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.box import Box
from cvxro.uncertainty_sets.budget import Budget
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL
from tests.settings import SOLVER, SOLVER_SETTINGS

ATOL = 5e-4
RTOL = 5e-4

np.random.seed(seed=1234)

class TestMultipleUncertainty(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        c = np.random.rand(self.n)
        self.b = 10.
        self.x = cp.Variable(self.n)
        self.objective = cp.Minimize(c @ self.x)
        # Robust set
        self.rho = 0.2
        self.p = 2

    # @unittest.skip("not currently implementing multiple sets")
    def test_simple_2u_ellipsoidal_box(self):

        # SETUP
        n = 5
        x_cvxro = cp.Variable(n)
        c = npr.rand(n)
        b = 10.
        P = npr.randint(-1, 5, size=(n, n))
        a = np.zeros(n)
        rho_1 = 0.2
        rho_2 = 0.5

        objective_1 = cp.Minimize(c @ x_cvxro)
        # Formulate robust constraints with cvxro
        unc_set_1 = Ellipsoidal(rho=rho_1)
        unc_set_2 = Box(rho=rho_2)
        u_1 = UncertainParameter(n, uncertainty_set=unc_set_1)

        u_2 = UncertainParameter(n, uncertainty_set=unc_set_2)
        constraints_1 = [max_of_uncertain(
            [u_1 @ P @ x_cvxro, a @ x_cvxro]) + u_2 @ x_cvxro <= b]

        prob_robust = RobustProblem(objective_1, constraints_1)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)

        # Robust problem 2

        x_rob2 = cp.Variable(n)
        objective_2 = cp.Minimize(x_rob2 @ c)

        constraints_2 = [u_1 @ P @ x_rob2 + u_2 @ x_rob2  <= b]
        constraints_2 += [a @ x_rob2 + u_2 @ x_rob2 <= b]

        prob_rob2 = RobustProblem(objective_2, constraints_2)
        prob_rob2.solve(solver=SOLVER, **SOLVER_SETTINGS)

        npt.assert_allclose(x_cvxro.value, x_rob2.value, rtol=RTOL, atol=ATOL)


        # Cvxpy problem
        x_cvx = cp.Variable(n)
        objective_3 = cp.Minimize(x_cvx @ c)
        constraints_3 = []
        constraints_3 += [-b + rho_1*cp.norm(P @ x_cvx,2) + rho_2*cp.norm(x_cvx,1) <= 0 ]
        constraints_3 += [-b + rho_2*cp.norm(x_cvx,1) + a @ x_cvx <= 0]
        prob_cvx = cp.Problem(objective_3, constraints_3)
        prob_cvx.solve(solver=SOLVER, **SOLVER_SETTINGS)
        npt.assert_allclose(x_cvxro.value, x_cvx.value, rtol=RTOL, atol=ATOL)


    def test_simple_2u_budget_box(self):
        # SETUP
        n = 5
        x_cvxro = cp.Variable(n)
        c = npr.rand(n)
        b = 10.
        P = npr.randint(-1, 5, size=(n, n))
        rho1 = 0.2
        rho2 = 0.5
        sigma = 0.3

        # Formulate robust constraints with cvxro
        objective_cvxro = cp.Minimize(c@x_cvxro)
        unc_set_1 = Budget(rho1=rho1, rho2=rho2)
        unc_set_2 = Box(rho=sigma)
        u1 = UncertainParameter(n, uncertainty_set=unc_set_1)
        u2 = UncertainParameter(n, uncertainty_set=unc_set_2)
        constraints_cvxro = [u1@P@x_cvxro + u2@x_cvxro <= b]
        prob_robust = RobustProblem(objective_cvxro, constraints_cvxro)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)

        # Formulate the problem with CVXPY
        x_cp = cp.Variable(n)
        r = cp.Variable(n)
        objective_cp = cp.Minimize(c@x_cp)
        constraints_cp = [-b + rho1*cp.norm(r+P@x_cp, 1)+rho2*cp.norm(r, np.inf) + \
                            sigma*cp.norm(x_cp, 1) <= 0]
        prob_cp = cp.Problem(objective_cp, constraints_cp)
        prob_cp.solve(solver=SOLVER, **SOLVER_SETTINGS)
        npt.assert_allclose(x_cvxro.value, x_cp.value, rtol=RTOL, atol=ATOL)

    def test_simple_2u_polyhedral_box(self):
        # SETUP
        n = 5
        x_cvxro = cp.Variable(n)
        c = npr.rand(n)
        b = 10.
        P = npr.randint(-1, 5, size=(n, n))
        a = np.zeros(n)
        rho_1 = 0.2
        rho_2 = 0.5

        objective_1 = cp.Minimize(c @ x_cvxro)
        # Formulate robust constraints with cvxro
        unc_set_1 = Ellipsoidal(rho=rho_1)
        unc_set_2 = Box(rho=rho_2)
        u_1 = UncertainParameter(n, uncertainty_set=unc_set_1)

        u_2 = UncertainParameter(n, uncertainty_set=unc_set_2)
        constraints_1 = [max_of_uncertain(
            [u_1 @ P @ x_cvxro, a @ x_cvxro]) + u_2 @ x_cvxro <= b]

        prob_robust = RobustProblem(objective_1, constraints_1)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)

        # Robust problem 2

        x_rob2 = cp.Variable(n)
        objective_2 = cp.Minimize(x_rob2 @ c)

        constraints_2 = [u_1 @ P @ x_rob2 + u_2 @ x_rob2  <= b]
        constraints_2 += [a @ x_rob2 + u_2 @ x_rob2 <= b]

        prob_rob2 = RobustProblem(objective_2, constraints_2)
        prob_rob2.solve(solver=SOLVER, **SOLVER_SETTINGS)

        npt.assert_allclose(x_cvxro.value, x_rob2.value, rtol=RTOL, atol=ATOL)


        # Cvxpy problem
        x_cvx = cp.Variable(n)
        objective_3 = cp.Minimize(x_cvx @ c)
        constraints_3 = []
        constraints_3 += [-b + rho_1*cp.norm(P @ x_cvx,2) + rho_2*cp.norm(x_cvx,1) <= 0 ]
        constraints_3 += [-b + rho_2*cp.norm(x_cvx,1) + a @ x_cvx <= 0]
        prob_cvx = cp.Problem(objective_3, constraints_3)
        prob_cvx.solve(solver=SOLVER, **SOLVER_SETTINGS)
        npt.assert_allclose(x_cvxro.value, x_cvx.value, rtol=RTOL, atol=ATOL)
