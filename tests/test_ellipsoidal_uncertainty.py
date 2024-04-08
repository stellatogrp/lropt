import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np

# import numpy.random as npr
import numpy.testing as npt
from cvxpy.constraints.constraint import Constraint
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse._coo import coo_matrix

import lropt
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from tests.settings import SOLVER
from tests.settings import TESTS_ATOL as ATOL
from tests.settings import TESTS_RTOL as RTOL

# import pandas as pd
# import torch


def tensor_reshaper(T_Ab: coo_matrix, n_var: int) -> np.ndarray:
    """
    This function reshapes T_Ab so T_Ab@param_vec gives the constraints row by row instead of 
    column by column. At the moment, it returns a dense matrix instead of a sparse one.
    """
    def _calc_source_row(target_row: int, num_constraints: int) -> int:
        """
        This is a helper function that calculates the index of the source row of T_Ab for the
        reshaped target row.
        """
        constraint_num = target_row%(num_constraints-1)
        var_num = target_row//(num_constraints-1)
        source_row = constraint_num*num_constraints+var_num
        return source_row


    T_Ab = csc_matrix(T_Ab)
    n_var_full = n_var+1 #Includes the free paramter
    num_rows = T_Ab.shape[0]
    num_constraints = num_rows//n_var_full
    T_Ab_res = csr_matrix(T_Ab.shape)
    target_row = 0 #Counter for populating the new row of T_Ab_res
    for target_row in range(num_rows):
        source_row = _calc_source_row(target_row, num_constraints)
        T_Ab_res[target_row, :] = T_Ab[source_row, :]
    return T_Ab_res

def calc_num_constraints(constraints: list[Constraint]) -> int:
    """
    This function calculates the number of constraints from a list of constraints.
    """
    num_constraints = 0
    for constraint in constraints:
        num_constraints += constraint.size
    return num_constraints


class TestEllipsoidalUncertainty(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        np.random.seed(0)
        self.n = 5
        c = np.arange(self.n)
        self.b = 10.
        self.x = cp.Variable(self.n, name="x")
        self.objective = cp.Minimize(c @ self.x)
        # Robust set
        self.rho = 0.2
        self.p = 2

    def test_ellipsoidal(self):
        """Test uncertain variable"""
        u = UncertainParameter(uncertainty_set=Ellipsoidal(rho=3.5))
        assert u.uncertainty_set.dual_norm() == 2.0

    def test_robust_norm_lp(self):
        b, x, objective, n, rho, p = \
            self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=p) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        a = UncertainParameter(n,
                               uncertainty_set=Ellipsoidal(rho=rho))
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_affine_transform(self):
        # import ipdb
        # ipdb.set_trace()
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        constraints = [-b_unc @ x + rho * cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value
        # Formulate robust constraints with lropt
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [1*-(A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_simple_ellipsoidal(self):
        b, x, n, objective, rho, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3. * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust constraints with lropt
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n,
                               uncertainty_set=unc_set)
        constraints = [2 * (A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER)

        # TODO (bart): not sure what we are testing here

    def test_tensor(self):
        SOLVER = "CLARABEL"
        RTOL = 1e-5
        ATOL = 1e-5
        b, x, n, objective, _, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p

        np.random.rand(n, n)
        bar_a = 0.1 * np.random.rand(n)
        lropt.UncertainParameter(n, uncertainty_set=lropt.Ellipsoidal(p=2))

        # Solve with cvxpy
        # prob_cvxpy = cp.Problem(objective, [bar_a @ x + cp.norm(P @ x, p=2) <= b,  # RO
        #                                     cp.sum(x) == 1, x >= 0])
        prob_cvxpy = cp.Problem(objective, [bar_a @ x <= b, cp.sum(x) == 1, x >= 0]) # nominal
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value

        # Solve via tensor reformulation
        a = cp.Parameter(n)
        constraints = [a @ x <= b, cp.sum(x) == 1, x >= 0]
        calc_num_constraints(constraints)
        prob_tensor = cp.Problem(objective, constraints)
        data = prob_tensor.get_problem_data(solver=SOLVER)
        param_prob = data[0]['param_prob']
        n_var = param_prob.reduced_A.var_len
        T_Ab = param_prob.A

        # Tensor mapping (cvxpy works as follows)
        # T_Ab @ (theta, 1) = vec([A | b])
        param_vec = np.hstack([bar_a, 1])
        vecAb = T_Ab @ param_vec
        Ab = vecAb.reshape(-1, n_var + 1, order='F')
        A_rec = -Ab[:, :-1] # note minus sign for different conic form
        b_rec = Ab[:, -1]
        s = cp.Variable(A_rec.shape[0])
        constraints = [A_rec @ x + s == b_rec]
        cones = data[0]['dims']

        if cones.zero > 0:
            constraints.append(cp.Zero(s[:cones.zero]))
        if cones.nonneg > 0:
            constraints.append(cp.NonNeg(s[cones.zero:cones.zero + cones.nonneg]))
        # TODO: Add other cones

        prob_recovered = cp.Problem(objective, constraints)
        prob_recovered.solve(solver=SOLVER)
        x_recovered = x.value


        npt.assert_allclose(x_cvxpy, x_recovered, rtol=RTOL, atol=ATOL)

        # TODO: adapt this example to handle RO formulation
        # from both cvxpy and tensor reformulation

        # TODO: handle parameters in objective as well

    def test_tensor_rows(self):
        b, x, n, objective, _, _ = \
            self.b, self.x, self.n, self.objective, self.rho, self.p

        np.random.rand(n, n)
        bar_a = 0.1 * np.random.rand(n)
        lropt.UncertainParameter(n, uncertainty_set=lropt.Ellipsoidal(p=2))

        # Solve with cvxpy
        # prob_cvxpy = cp.Problem(objective, [bar_a @ x + cp.norm(P @ x, p=2) <= b,  # RO
        #                                     cp.sum(x) == 1, x >= 0])
        prob_cvxpy = cp.Problem(objective, [bar_a @ x <= b, cp.sum(x) == 1, x >= 0]) # nominal
        prob_cvxpy.solve(solver=SOLVER)
        x_cvxpy = x.value

        # Solve via tensor reformulation
        a = cp.Parameter(n)
        constraints = [a @ x <= b, cp.sum(x) == 1, x >= 0]
        num_constraints = calc_num_constraints(constraints)
        prob_tensor = cp.Problem(objective, constraints)
        data = prob_tensor.get_problem_data(solver=SOLVER)
        param_prob = data[0]['param_prob']
        n_var = param_prob.reduced_A.var_len
        T_Ab = param_prob.A
        T_Ab_reshaped = tensor_reshaper(T_Ab, n_var)

        # Tensor mapping (cvxpy works as follows)
        param_vec = np.hstack([bar_a, 1])
        vecAb_reshaped = T_Ab_reshaped@param_vec
        Ab_reshaped = vecAb_reshaped.reshape(num_constraints, n_var + 1, order='C')
        A_rec = -Ab_reshaped[:, :-1] # note minus sign for different conic form
        b_rec = Ab_reshaped[:, -1]
        s = cp.Variable(A_rec.shape[0])
        constraints = [A_rec @ x + s == b_rec]
        cones = data[0]['dims']

        if cones.zero > 0:
            constraints.append(cp.Zero(s[:cones.zero]))
        if cones.nonneg > 0:
            constraints.append(cp.NonNeg(s[cones.zero:cones.zero + cones.nonneg]))
        # TODO: Add other cones

        prob_recovered = cp.Problem(objective, constraints)
        prob_recovered.solve(solver=SOLVER)
        x_recovered = x.value


        npt.assert_allclose(x_cvxpy, x_recovered, rtol=RTOL, atol=ATOL)

        # TODO: adapt this example to handle RO formulation
        # from both cvxpy and tensor reformulation

        # TODO: handle parameters in objective as well