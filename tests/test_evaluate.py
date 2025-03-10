import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.train.parameter import ContextParameter
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

ATOL = 1e-4
RTOL = 1e-4
SOLVER = cp.CLARABEL
SOLVER_SETTINGS = { "equilibrate_enable": False, "verbose": False }

class TestEvaluate(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        # c = np.random.rand(self.n)
        self.b = 1.5
        self.x = cp.Variable(self.n, name="x")
        param_val = cp.Parameter()
        param_val.value = 0.3
        self.context_param = ContextParameter(1, data = np.ones((1,1)))
        self.param_val = param_val
        # Robust set
        self.rho = 0.2
        self.p = 2

    @unittest.SkipTest
    def test_evaluate_no_context(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1]
        objective = cp.Minimize(-a @ self.x + self.param_val)
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        # actually, we should allow the user to input an np array as well (not
        # purely limited to tensor. The code should take care of converting it
        # if needed)
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        a.eval_data = u_data

        # note that even though the objective has param_val, since it is a
        #  cvxpy parameter, the value should already be built-in and does not
        # need an additional input

        eval_value = prob_robust.evaluate()
        actual_value = np.mean(-u_data@x_robust+0.3)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.SkipTest
    def test_evaluate_context(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1]
        objective = cp.Minimize(-a @ self.x + self.param_val + self.context_param)
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        x_data = np.array([[1],
                                 [1.3],
                                 [2]])
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        a.eval_data = u_data
        self.context_param.eval_data = x_data

        eval_value = prob_robust.evaluate()
        actual_value = np.mean(-u_data@x_robust+0.3+x_data)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.SkipTest
    def test_evaluate_multiple_u(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        a1 = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=2*rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1, a1@x -self.context_param <=b]
        objective = cp.Minimize(-a @ self.x + a1@self.x + self.param_val + self.context_param)
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        x_data = np.array([[1],
                                 [1.3],
                                 [2]])
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        u1_data = 0.1*np.ones((3,n))

        a.eval_data = u_data
        self.context_param.eval_data = x_data
        a1.eval_data = u1_data

        eval_value = prob_robust.evaluate()
        actual_value = np.mean(-u_data@x_robust+u1_data@x_robust + 0.3+x_data)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)
