import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

from lropt.robust_problem import RobustProblem
from lropt.train.parameter import ContextParameter
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal

TOLERANCE_DEFAULT = 1e-5
ATOL = 1e-4
RTOL = 1e-4
SOLVER = cp.CLARABEL
SOLVER_SETTINGS = { "equilibrate_enable": False, "verbose": False }

class TestEvaluateSolution(unittest.TestCase):

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

    @unittest.skip
    def test_evaluate_sol_no_context(self):
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

        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        a.eval_sol_data = u_data

        eval_value = np.mean(prob_robust.evaluate_sol())
        actual_value = np.mean(-u_data@x_robust+0.3)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.skip
    def test_evaluate_sol_context(self):
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

        # x_data = np.array([[1],
        #                          [1.3],
        #                          [2]])
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        actual_value = np.mean(-u_data@x_robust+0.3+1)
        a.eval_sol_data = u_data
        # self.context_param.eval_data = x_data
        eval_value = np.mean(prob_robust.evaluate_sol())
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.skip
    def test_evaluate_sol_multiple_u(self):
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

        # x_data = np.array([[1],
        #                          [1.3],
        #                          [2]])
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        u1_data = 0.1*np.ones((3,n))

        a.eval_sol_data = u_data
        # self.context_param.eval_data = x_data
        a1.eval_sol_data = u1_data

        eval_value = np.mean(prob_robust.evaluate_sol())
        actual_value = np.mean(-u_data@x_robust+u1_data@x_robust + 0.3+1)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)


class TestSolutionProbability(unittest.TestCase):

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

    @unittest.skip
    def test_sol_prob_no_context(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        t = cp.Variable()
        constraints = [a @ x <= b, cp.sum(x)==1, -a @ self.x + self.param_val <= t]
        objective = cp.Minimize(t)
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        a.eval_sol_data = u_data

        eval_value = np.mean(prob_robust.evaluate_sol_probability(), axis = 1)
        actual_value = np.mean(np.vstack([u_data@x_robust-b<= TOLERANCE_DEFAULT,
                                          -u_data @x_robust + 0.3 - t.value
                                          <= TOLERANCE_DEFAULT]),axis = 1)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.skip
    def test_sol_prob_context(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        t = cp.Variable()
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1,
                       -a @ self.x + self.param_val + self.context_param <= t]
        objective = cp.Minimize()
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        # x_data = np.array([[1],
        #                          [1.3],
        #                          [2]])
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        actual_value = np.mean(np.vstack([u_data@x_robust-b<= TOLERANCE_DEFAULT,
                           -u_data @x_robust + 0.3 + 1 - t.value
                           <= TOLERANCE_DEFAULT]),axis = 1)

        a.eval_sol_data = u_data
        # self.context_param.eval_data = x_data

        eval_value = np.mean(prob_robust.evaluate_sol_probability(),axis = 1)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.skip
    def test_sol_prob_multiple_u(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        t = cp.Variable()
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        a1 = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=2*rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1, a1@x -self.context_param <=b,
                        -a @ self.x + a1@self.x + self.param_val
                        + self.context_param <=t]
        objective = cp.Minimize(t)
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        # x_data = np.array([[1],
        #                          [1.3],
        #                          [2]])
        u_data = np.array([[0.1,0.2,0.3,0.1,0.1],
                           [0.1,0.5,-0.3,0.1,0.1],
                           [0.4,0.2,0.3,-0.1,0.1]])
        u1_data = 0.1*np.ones((3,n))

        a.eval_sol_data = u_data
        # self.context_param.eval_data = x_data
        a1.eval_sol_data = u1_data

        eval_value = np.mean(prob_robust.evaluate_sol_probability(),axis=1)
        actual_value = np.mean(np.vstack([u_data@x_robust-b<= TOLERANCE_DEFAULT,
                           -u_data @x_robust +u1_data@x_robust + 0.3
                           + 1 - t.value <= TOLERANCE_DEFAULT]),axis = 1)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)


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

    @unittest.skip
    def test_evaluate_context(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p

        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1]
        objective = cp.Minimize(-a @ self.x + self.param_val + self.context_param)
        prob_robust = RobustProblem(objective, constraints)

        x_data = np.array([[1],
                                 [1.3],
                                 [2]])

        # a list of arrays/tensors. Each array corresponds to the data for
        # each context_parameter. batch size need not be the same for
        # different arrays.
        u_data = [np.array([[0.1,0.2,0.3,0.1,0.1],[0.1,0.3,0.4,0.1,0.1]]),
                  np.array([0.1,0.5,-0.3,0.1,0.1]),
                  np.array([0.4,0.2,0.3,-0.1,0.1])]

        eval_vals = []
        # for each context parameter (number of context parameters could be
        # context_parameter.shape[0] if batched, otherwise 1. Should be
        # consistent for all context parameters (share the same batch size))
        for i in range(x_data.shape[0]):
            # set context param value for all context parameters
            self.context_param.value = x_data[i]
            # re-solve the problem with the new parameters
            prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
            # evaluate the solution (can actually call prob.evaluate_sol()
            # instead of calculating it from scratch.
            # will have to set u.eval_sol_data = u_data[i] for all uncertain
            # parameters in that case)
            eval_vals.append(-u_data[i]@x.value + 0.3 + x_data[i])

        actual_value = np.mean(np.hstack(eval_vals))

        a.eval_data = u_data
        self.context_param.eval_data = x_data
        eval_value = np.mean(prob_robust.evaluate())

        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.skip
    def test_evaluate_multiple_u(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p

        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        a1 = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=2*rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1, a1@x -self.context_param <=b]
        objective = cp.Minimize(-a @ self.x + a1@self.x + self.param_val + self.context_param)
        prob_robust = RobustProblem(objective, constraints)

        x_data = np.array([[1],
                                 [1.3],
                                 [2]])
        u_data = [np.array([[0.1,0.2,0.3,0.1,0.1],[0.1,0.3,0.4,0.1,0.1]]),
                  np.array([0.1,0.5,-0.3,0.1,0.1]),
                  np.array([0.4,0.2,0.3,-0.1,0.1])]

        # with multiple uncertain parameters, batch sizes for each corresponding
        # position in the list should be the same
        u1_data = [0.1*np.ones((2,n)), 0.1*np.ones((n)), 0.1*np.ones((n))]

        eval_vals = []
        for i in range(x_data.shape[0]):
            self.context_param.value = x_data[i]
            prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
            eval_vals.append(-u_data[i]@x.value + u1_data[i]@x.value + 0.3 + x_data[i])

        actual_value = np.mean(np.hstack(eval_vals))

        a.eval_data = u_data
        # self.context_param.eval_data = x_data
        a1.eval_data = u1_data
        eval_value = np.mean(prob_robust.evaluate())

        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)


class TestProbability(unittest.TestCase):

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

    @unittest.skip
    def test_prob_context(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        t = cp.Variable()
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        context_2 = ContextParameter(1,data=np.ones((1,1)))
        constraints = [a @ x <= b, cp.sum(x)==1,
                       -a @ self.x + self.param_val + self.context_param + context_2<= t]
        objective = cp.Minimize()
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)

        x_data = np.array([[1],
                                 [1.3],
                                 [2]])
        x2_data = np.array([[2],
                                 [1.3],
                                 [3]])

        u_data = [np.array([[0.1,0.2,0.3,0.1,0.1],[0.1,0.3,0.4,0.1,0.1]]),
                  np.array([0.1,0.5,-0.3,0.1,0.1]),
                  np.array([0.4,0.2,0.3,-0.1,0.1])]

        eval_probs = []
        for i in range(x_data.shape[0]):
            self.context_param.value = x_data[i]
            context_2.value = x2_data[i]
            # re-solve the problem with the new parameters
            prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
            a.eval_sol_data = u_data[i]
            eval_probs.append(prob_robust.evaluate_sol_probability())

        actual_value = np.mean(np.hstack(eval_probs),axis=1)

        a.eval_data = u_data
        # self.context_param.eval_data = x_data

        eval_value = np.mean(prob_robust.evaluate_sol_probability(),axis = 1)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)

    @unittest.skip
    def test_prob_multiple_u(self):
        b, x, n, rho, p = \
            self.b, self.x, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        # Formulate robust constraints with lropt
        t = cp.Variable()
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho, p=p))
        a1 = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=2*rho, p=p))
        constraints = [a @ x <= b, cp.sum(x)==1, a1@x -self.context_param <=b,
                        -a @ self.x + a1@self.x + self.param_val
                        + self.context_param <=t]
        objective = cp.Minimize(t)
        prob_robust = RobustProblem(objective, constraints)

        x_data = np.array([[1],
                                 [1.3],
                                 [2]])
        u_data = [np.array([[0.1,0.2,0.3,0.1,0.1],[0.1,0.3,0.4,0.1,0.1]]),
                  np.array([0.1,0.5,-0.3,0.1,0.1]),
                  np.array([0.4,0.2,0.3,-0.1,0.1])]
        u1_data = [0.1*np.ones((2,n)), 0.1*np.ones((n)), 0.1*np.ones((n))]

        eval_probs = []
        for i in range(x_data.shape[0]):
            self.context_param.value = x_data[i]
            # re-solve the problem with the new parameters
            prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
            a.eval_sol_data = u_data[i]
            a1.eval_sol_data = u1_data[i]
            eval_probs.append(prob_robust.evaluate_sol_probability())

        actual_value = np.mean(np.hstack(eval_probs),axis=1)

        a.eval_data = u_data
        self.context_param.eval_data = x_data
        a1.eval_data = u1_data

        eval_value = np.mean(prob_robust.evaluate_probability(),axis=1)
        npt.assert_allclose(eval_value, actual_value, rtol=RTOL, atol=ATOL)
