import unittest

import cvxpy as cp
import numpy as np
# import numpy.testing as npt
import scipy as sc
# import pytest
# import torch
from sklearn import datasets

from lro.robust_problem import RobustProblem
from lro.uncertain import UncertainParameter
from lro.uncertain_atoms.quad_form import quad_form
# from lro.uncertainty_sets.box import Box
# from lro.uncertainty_sets.budget import Budget
from lro.uncertainty_sets.ellipsoidal import Ellipsoidal
# from lro.uncertainty_sets.polyhedral import Polyhedral
from tests.settings import SOLVER

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL


class TestQuad(unittest.TestCase):

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

    def test_quad_simple(self):
        m = 5
        n = 5
        x = cp.Variable(n)
        t = cp.Variable()
        u = UncertainParameter(m,
                               uncertainty_set=Ellipsoidal(p=2,
                                                           rho=2.))
        A = {}
        Ainv = {}
        for i in range(m):
            A[i] = datasets.make_spd_matrix(m, random_state=i)
            Ainv[i] = sc.linalg.sqrtm(np.linalg.inv(A[i]))

        objective = cp.Minimize(t)
        constraints = [cp.sum([-0.5*quad_form(u, A[i]*x[i]) for i in range(m)]) <= t]
        constraints += [x >= 0, x <= 1]
        # import ipdb
        # ipdb.set_trace()
        prob = RobustProblem(objective, constraints)
        newprob = prob.dualize_constraints()
        newprob.solve(solver=SOLVER)
        print(x.value)

    def test_max(self):
        n = 2
        u = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(p=2,
                                                           rho=1))
        # formulate cvxpy variables
        x = cp.Variable(n)
        t = cp.Variable()

        # formulate constants
        a = np.array([2, 3])
        d = np.array([3, 4])

        # formulate objective
        objective = cp.Minimize(t)

        # formulate constraints
        constraints = [cp.maximum(a@x - d@x, a@x - d@u) <= t]
        constraints += [x >= 0]
        # import ipdb
        # ipdb.set_trace()
        # formulate Robust Problem
        prob_robust = RobustProblem(objective, constraints)

        new_prob = prob_robust.dualize_constraints()
        new_prob.solve()
