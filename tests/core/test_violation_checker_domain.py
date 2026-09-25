"""A decision that leaves a variable's DOMAIN by solver dust must not raise.

Interior-point solvers return a nonneg variable at +-1e-13..1e-9 around zero
(measured on lropt_power/multistage, 2026-09-20: minima of -1.2e-9 across seven nonneg
variables, none below -1e-8). cvxpy validates domains at assignment with
GENERAL_PROJECTION_TOL (1e-10), so such a value raised `ValueError: Variable value must
be nonnegative` out of ViolationChecker._set_z_batch and killed a radius search. Variable
domains are not in problem_no_unc.constraints, so that assignment is the only place they
are checked -- it has to classify, at the checker's own tolerance, not raise.
"""

import types
import unittest

import cvxpy as cp
import numpy as np
import torch
from cvxpy.settings import GENERAL_PROJECTION_TOL

from cvxro.parameter import ContextParameter
from cvxro.violation_checker.settings import VIOLATION_TOL
from cvxro.violation_checker.utils import CONSTRAINT_STATUS
from cvxro.violation_checker.violation_checker import ViolationChecker

N, DIM = 4, 2
DUST = -1e-9          # what the solver delivers
GROSS = -1.0          # what a junk decision looks like


class TestDomainClassification(unittest.TestCase):
    def setUp(self):
        self.v = cp.Variable(DIM, nonneg=True)
        layer = types.SimpleNamespace(variables=[self.v])
        # A constraint the projected point always satisfies, so only the domain can
        # flag a sample.
        self.checker = ViolationChecker(layer, [self.v >= -10.0], tol=VIOLATION_TOL)
        self.ctx = ContextParameter(DIM, data=np.zeros((N, DIM)))

    def _check(self, z, per_sample):
        return self.checker.check_constraints(
            z_batch=[torch.as_tensor(z, dtype=torch.double)],
            rho_mult_parameter=[cp.Parameter(nonneg=True)], rho_tch=torch.tensor(1.0),
            cp_parameters=[], cp_param_tch=[],
            x_parameters=[self.ctx], x_batch=[torch.zeros(N, DIM)],
            shape_parameters=[], shape_torches=[], per_sample=per_sample)

    def test_premise_cvxpy_rejects_solver_dust(self):
        # The reason this module exists. If cvxpy ever widens its tolerance past the
        # dust, the projection below becomes redundant and this test says so.
        self.assertLess(GENERAL_PROJECTION_TOL, abs(DUST))
        with self.assertRaises(ValueError):
            cp.Variable(DIM, nonneg=True).value = np.array([1.0, DUST])

    def test_dust_is_projected_and_feasible(self):
        z = np.tile([1.0, DUST], (N, 1))
        self.assertIs(self._check(z, per_sample=True), CONSTRAINT_STATUS.FEASIBLE)
        self.assertTrue(self.checker.last_sample_feasible.all())
        self.assertGreaterEqual(float(self.v.value.min()), 0.0)

    def test_gross_violation_marks_only_that_sample(self):
        z = np.tile([1.0, 1.0], (N, 1))
        z[2, 1] = GROSS
        self.assertIs(self._check(z, per_sample=True), CONSTRAINT_STATUS.INFEASIBLE)
        np.testing.assert_array_equal(self.checker.last_sample_feasible,
                                      [True, True, False, True])

    def test_gross_violation_without_per_sample_is_infeasible(self):
        z = np.tile([1.0, 1.0], (N, 1))
        z[0, 0] = GROSS
        self.assertIs(self._check(z, per_sample=False), CONSTRAINT_STATUS.INFEASIBLE)

    def test_threshold_is_the_checkers_tolerance(self):
        just_inside = np.tile([1.0, -0.5 * VIOLATION_TOL], (N, 1))
        just_outside = np.tile([1.0, -2.0 * VIOLATION_TOL], (N, 1))
        self.assertIs(self._check(just_inside, per_sample=True), CONSTRAINT_STATUS.FEASIBLE)
        self.assertIs(self._check(just_outside, per_sample=True), CONSTRAINT_STATUS.INFEASIBLE)


if __name__ == "__main__":
    unittest.main()
