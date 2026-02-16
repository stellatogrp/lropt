import unittest

import cvxpy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
from cvxpy.error import DCPError

from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_canon.max_of_uncertain import max_of_uncertain, sum_of_max_of_uncertain
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal
from cvxro.uncertainty_sets.mro import MRO

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL
ATOL = 1e-5
RTOL = 1e-5

class TestMax(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        n = 5
        self.data = np.random.normal(0, 1, size=(100, n))
        # self.data = np.zeros((100,n))
        self.a = npr.uniform(1, 4, n)
        self.d = self.a + npr.uniform(2, 5, n)

    # @unittest.skip("not currently implementing maximum")
    def test_maximum_of_affine(self):
        # formulate uncertainty set
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2, rho=1,
                                                           b=np.mean(self.data, axis=0)))
        # formulate cvxpy variables
        x_r = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        # constraints = [cp.maximum(
        #     self.a@x_r - self.d@x_r, self.a@x_r - self.d@(3*u+3)) <= t]
        #
        constraints = []
        constraints += [max_of_uncertain([ - self.d@x_r,- self.d@(3*u+3)],\
                                         self.a@x_r-t) <=0]
        constraints += [max_of_uncertain([ - self.d@x_r,\
                                                    - self.d@(3*u+3)],\
                                                        self.a@x_r-t)<=0]
        constraints += [sum_of_max_of_uncertain([[- self.d@x_r,\
                                                  - self.d@(3*u+3)],\
                                                    [self.a@x_r-t]])<=0]
        constraints += [sum_of_max_of_uncertain([[- self.d@x_r,\
                                                  - self.d@(3*u+3)]]) + \
                                                    self.a@x_r <= t]
        #constraints = [self.a@x_r - self.d@x_r <= t]
        #constraints += [self.a@x_r - self.d@(3*u+3) <= t]
        constraints += [x_r >= 0]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()

        # formulate using cvxpy
        x_cvxpy = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [self.a@x_cvxpy - self.d@x_cvxpy <= t]
        constraints += [self.a@x_cvxpy - 3*self.d @
                        np.ones(n) + np.mean(self.data, axis=0)@(-3*self.d)
                        + cp.norm(3*self.d, 2) <= t]
        constraints += [x_cvxpy >= 0]

        # formulate problem
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve()

        # assert x values are equal
        npt.assert_allclose(x_r.value, x_cvxpy.value, rtol=RTOL, atol=ATOL)


    def test_maximum_of_affine_additional(self):
        # formulate uncertainty set
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2, rho=1,
                                                           b=np.mean(self.data, axis=0)))
        # formulate cvxpy variables
        x_r = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        # constraints = [cp.maximum(
        #     self.a@x_r - self.d@x_r, self.a@x_r - self.d@(3*u+3)) <= t]
        #
        constraints = []
        constraints += [- self.d@(3*u+3)<= 0]
        constraints += [max_of_uncertain([ - self.d@x_r,- self.d@(3*u+3)],\
                                         self.a@x_r-t) <=0]
        constraints += [max_of_uncertain([ - 2*self.d@x_r,- self.d@(3*u+3)],\
                                         self.a@x_r-t) <=0]
        constraints += [-self.d@(3*u+3) + self.a@x_r-t - 3 <= 0]
        constraints += [x_r >= 0]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()

        # formulate using cvxpy
        x_cvxpy = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [self.a@x_cvxpy - self.d@x_cvxpy <= t]
        constraints += [self.a@x_cvxpy - 2*self.d@x_cvxpy <= t]
        constraints += [- 3*self.d @
                        np.ones(n) + np.mean(self.data, axis=0)@(-3*self.d)
                        + cp.norm(3*self.d, 2) <= 0]
        constraints += [self.a@x_cvxpy - 3*self.d @
                        np.ones(n) + np.mean(self.data, axis=0)@(-3*self.d)
                        + cp.norm(3*self.d, 2) <= t]
        constraints += [self.a@x_cvxpy - 3*self.d @
                        np.ones(n) + np.mean(self.data, axis=0)@(-3*self.d)
                        + cp.norm(3*self.d, 2) -3 <= t]
        constraints += [x_cvxpy >= 0]

        # formulate problem
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve()

        # assert x values are equal
        npt.assert_allclose(x_r.value, x_cvxpy.value, rtol=RTOL, atol=ATOL)

    def test_fail_accept_mult(self):
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2, rho=1,
                                                           b=np.mean(self.data, axis=0)))
        # formulate cvxpy variables
        x_r = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)
        constraints = []
        constraints += [1-max_of_uncertain([ - self.d@x_r,- self.d@(3*u+3)],self.a@x_r-t) <=0]
        constraints += [x_r >= 0]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        try:
            prob_robust.solve()
        except ValueError as e:
            expected_error = "max_of_uncertain must not be"+ \
                " multipled against any term, or" \
                    + " contained in any other expression," + \
                          " or negated. It can only be added" + \
                           " to other terms."
            assert str(e) == expected_error


    def test_fail_accept_contained(self):
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2, rho=1,
                                                           b=np.mean(self.data, axis=0)))
        # formulate cvxpy variables
        x_r = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        constraints = []
        constraints += [cp.sum(-max_of_uncertain([ - self.d@x_r,\
                                                  - self.d@(3*u+3)],\
                                                    self.a@x_r-t)) <=0]
        constraints += [x_r >= 0]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        try:
            prob_robust.solve()
        except ValueError as e:
            expected_error = "max_of_uncertain must not be"+ \
                " multipled against any term, or" \
                    + " contained in any other expression," + \
                          " or negated. It can only be added" + \
                           " to other terms."
            assert str(e) == expected_error

    def test_fail_accept_pwl(self):
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2, rho=1,
                                                           b=np.mean(self.data, axis=0)))
        # formulate cvxpy variables
        x_r = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        constraints = []
        constraints += [max_of_uncertain([ - self.d@x_r**2,\
                                                  - self.d@(3*u+3)],\
                                                    self.a@x_r-t) <=0]
        constraints += [x_r >= 0]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        try:
            prob_robust.solve()
        except DCPError as e:
            assert len(str(e)) >= 1

    @unittest.skip("not currently implementing maximum")
    def test_mro(self):
        # formulate uncertainty set
        n = 5
        u = UncertainParameter(n,
                               uncertainty_set=MRO(K=1, data=self.data,
                                                   p=2,
                                                   rho=1, train=False))
        # formulate cvxpy variables
        x_m = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [cp.maximum(
            self.a@x_m - self.d@x_m, self.a@x_m - self.d@(3*u+3)) <= t]
        constraints += [x_m >= 0]

        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        # solve
        prob_robust.solve()

        # formulate using cvxpy
        x_cvxpy = cp.Variable(n)
        t = cp.Variable()

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [self.a@x_cvxpy - self.d @
                       x_cvxpy + cp.norm(3*self.d, 2) <= t]
        constraints += [self.a@x_cvxpy - 3*self.d @
                        np.ones(n) + np.mean(self.data, axis=0)@(-3*self.d)
                        + cp.norm(3*self.d, 2) <= t]
        constraints += [x_cvxpy >= 0]

        # formulate problem
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve()
        # assert x values are equal
        npt.assert_allclose(x_m.value, x_cvxpy.value, rtol=RTOL, atol=ATOL)
