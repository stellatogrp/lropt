import unittest

import cvxpy as cp

# import matplotlib.pyplot as plt
import numpy as np

# import numpy.random as npr
import numpy.testing as npt
import scipy as sc
from cvxpy import SCS, Parameter
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.reduction import Reduction
from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse._coo import coo_matrix

from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal

# from tests.settings import SOLVER, SOLVER_SETTINGS
# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL

ATOL = 1e-4
RTOL = 1e-4
SOLVER = cp.CLARABEL
SOLVER_SETTINGS = {"equilibrate_enable": False, "verbose": False}
# import pandas as pd
# import torch


def _get_tensors(problem: RobustProblem, solver=SCS) -> ndarray:
    """
    This inner function generates A_tensor: the 3D tensor of A. It also generates b,c
    """

    def _gen_param_vec(param_prob: Reduction) -> list:
        """
        This is a helper function that generates the parameters vector.
        This vector will be multiplied by T_Ab to get a vector containing A and b of the
        reformulated conic problem.
        """

        def _select_target(
            is_uncertain: bool,
            param_vec_certain: list,
            param_vec_uncertain: list,
            T_Ab_certain: list,
            T_Ab_uncertain: list,
        ) -> tuple[list, list]:
            """
            This is a helper function that determines whether to add the new parameter and
            columns to the uncertain parameters or the certain parameters.
            """
            if is_uncertain:
                return param_vec_uncertain, T_Ab_uncertain
            else:
                return param_vec_certain, T_Ab_certain

        def _gen_param_value(
            param: Parameter | UncertainParameter, is_uncertain: bool
        ) -> np.ndarray | UncertainParameter:
            """
            This is a helper function that returns the uncertain parameter if the input is
            an uncertain parameter, or the parameter's value for known parameters.
            """
            # TODO: Originanlly was the block below, not sure if it should be it or just
            # param. If just param, this function is redundant.
            # return param
            if is_uncertain:
                return param
            return param.value

        def _safe_np_hstack(vec: list) -> np.ndarray:
            """
            This is a helper function that hstacks the elements of vec or returns None if
            vec is empty.
            """
            # Empty vector - return None
            if not vec:
                return None
            # A vector of uncertain parameters needs cvxpy hstack
            if isinstance(vec[0], UncertainParameter):
                return Hstack(*vec)
            return np.hstack(vec)

        def _safe_gen_vecAb(T_Ab: np.ndarray, param_vec: np.ndarray | Expression | None):
            """
            This function safely generates vecAb = T_Ab @ vec_param, or returns None if
            vec_param is empty.
            """
            if param_vec is None or (isinstance(param_vec, ndarray) and len(param_vec) == 0):
                return None
            return T_Ab @ param_vec

        n_var = param_prob.reduced_A.var_len
        T_Ab = param_prob.A
        T_Ab = tensor_reshaper(T_Ab, n_var)
        param_vec_certain = []
        param_vec_uncertain = []
        T_Ab_certain = []
        T_Ab_uncertain = []
        running_param_size = 0  # This is a running counter that keeps track of the total size
        # of all the parameters seen so far.
        for param in param_prob.parameters:
            param_size = param_prob.param_id_to_size[param.id]
            is_uncertain = isinstance(param, UncertainParameter)
            param_vec_target, T_Ab_target = _select_target(
                is_uncertain, param_vec_certain, param_vec_uncertain, T_Ab_certain, T_Ab_uncertain
            )
            param_val = _gen_param_value(param, is_uncertain)
            param_vec_target.append(param_val)
            T_Ab_target.append(T_Ab[:, running_param_size : running_param_size + param_size])
            running_param_size += param_size

        # Add the parameter-free element:
        # The last element is always 1, represents the free element (not a parameter)
        param_vec_certain.append(1)
        T_Ab_certain.append(T_Ab[:, running_param_size:])

        # Stack all variables. Certain is never empty - always has the free element
        param_vec_uncertain = _safe_np_hstack(param_vec_uncertain)
        param_vec_certain = _safe_np_hstack(param_vec_certain)
        T_Ab_uncertain = _safe_np_hstack(T_Ab_uncertain)
        T_Ab_certain = _safe_np_hstack(T_Ab_certain)
        vec_Ab_certain = _safe_gen_vecAb(T_Ab_certain, param_vec_certain)
        # vec_Ab_uncertain    = _safe_gen_vecAb(T_Ab_uncertain, param_vec_uncertain)

        return vec_Ab_certain, T_Ab_uncertain

    def _finalize_expressions(
        vec_Ab: ndarray | Expression, is_uncertain: bool, n_var: int
    ) -> tuple:
        """
        This is a helper function that generates A, b from vec_Ab.
        """
        if vec_Ab is None:
            return None, None
        Ab_dim = (-1, n_var + 1)  # +1 for the free parameter
        Ab = vec_Ab.reshape(Ab_dim, order="C")
        if not is_uncertain:
            Ab = Ab.tocsr()
        # note minus sign for different conic form in A_rec
        A_rec = -Ab[:, :-1]
        b_rec = None
        if not is_uncertain:
            b_rec = Ab[:, -1]
        return A_rec, b_rec

    def _finalize_expressions_uncertain(T_Ab, n_var):
        """
        This is a helper function that generates A_unc
        """
        if T_Ab is None:
            return None, None
        Ab_dim = (-1, n_var + 1)  # +1 for the free parameter
        A_dic = {}
        for i in range(n_var):
            temp = T_Ab[0][:, i]
            Ab = temp.reshape(Ab_dim, order="C")
            Ab = Ab.tocsr()
            shape = Ab.shape[0]
            A_dic[i] = -Ab[:, :-1]
        return np.vstack(
            [sc.sparse.vstack([A_dic[i][j] for i in range(n_var)]).T for j in range(shape)]
        )

    param_prob = problem.get_problem_data(solver=solver)[0]["param_prob"]
    vec_Ab_certain, T_Ab_uncertain = _gen_param_vec(param_prob)
    n_var = param_prob.reduced_A.var_len
    A_rec_certain, b_rec = _finalize_expressions(vec_Ab_certain, is_uncertain=False, n_var=n_var)
    A_rec_uncertain = _finalize_expressions_uncertain(T_Ab_uncertain, n_var=n_var)
    return A_rec_certain, A_rec_uncertain, b_rec


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
        constraint_num = target_row % (num_constraints - 1)
        var_num = target_row // (num_constraints - 1)
        source_row = constraint_num * num_constraints + var_num
        return source_row

    T_Ab = csc_matrix(T_Ab)
    n_var_full = n_var + 1  # Includes the free paramter
    num_rows = T_Ab.shape[0]
    num_constraints = num_rows // n_var_full
    T_Ab_res = csr_matrix(T_Ab.shape)
    target_row = 0  # Counter for populating the new row of T_Ab_res
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
        c = np.random.rand(self.n)
        self.b = 1.5
        self.x = cp.Variable(self.n, name="x")
        param_val = cp.Parameter()
        param_val.value = 0
        self.param_val = param_val
        self.objective = cp.Minimize(c @ self.x + param_val)
        # Robust set
        self.rho = 0.2
        self.p = 2

    def test_ellipsoidal(self):
        """Test uncertain variable"""
        u = UncertainParameter(uncertainty_set=Ellipsoidal(rho=3.5))
        assert u.uncertainty_set.dual_norm() == 2.0

    def test_robust_norm_lp(self):
        b, x, objective, n, rho, p = self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value
        # Formulate robust constraints with cvxro
        a = UncertainParameter(n, uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_param(self):
        b, x, objective, n, rho, p = self.b, self.x, self.objective, self.n, self.rho, self.p
        # Formulate robust problem explicitly with cvxpy
        constraints = [rho * cp.norm(x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value
        # Formulate robust constraints with cvxro
        a = UncertainParameter(n, uncertainty_set=Ellipsoidal(rho=rho, p=p))
        constraints = [a @ x <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value
        obj_val = prob_robust.objective.value
        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

        self.param_val.value = 1
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        obj_val2 = prob_robust.objective.value
        npt.assert_allclose(obj_val + 1, obj_val2, rtol=RTOL, atol=ATOL)

    def test_robust_norm_lp_affine_transform(self):
        # import ipdb
        # ipdb.set_trace()
        b, x, n, objective, rho, _ = self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3.0 * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust problem explicitly with cvxpy
        constraints = [-b_unc @ x + rho * cp.norm(-A_unc.T @ x, p=2) <= b]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value
        # Formulate robust constraints with cvxro
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n, uncertainty_set=unc_set)
        constraints = [1 * -(A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        npt.assert_allclose(x_cvxpy, x_robust, rtol=RTOL, atol=ATOL)

    def test_simple_ellipsoidal(self):
        b, x, n, objective, rho, _ = self.b, self.x, self.n, self.objective, self.rho, self.p
        # Robust set
        A_unc = 3.0 * np.eye(n)
        b_unc = 0.1 * np.random.rand(n)
        # Formulate robust constraints with cvxro
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n, uncertainty_set=unc_set)
        constraints = [2 * (A_unc @ a + b_unc) @ x * 1 <= b]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)

        # TODO (bart): not sure what we are testing here

    def test_tensor(self):
        b, x, n, objective, _, _ = self.b, self.x, self.n, self.objective, self.rho, self.p

        bar_a = 0.1 * np.random.rand(n)

        # Solve with cvxpy
        # prob_cvxpy = cp.Problem(objective, [bar_a @ x + cp.norm(P @ x, p=2) <= b,  # RO
        #                                     cp.sum(x) == 1, x >= 0])
        prob_cvxpy = cp.Problem(objective, [bar_a @ x <= b, cp.sum(x) == 1, x >= 0])  # nominal
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value

        # Solve via tensor reformulation
        a = cp.Parameter(n)
        constraints = [a @ x <= b, cp.sum(x) == 1, x >= 0]
        prob_tensor = cp.Problem(objective, constraints)
        data = prob_tensor.get_problem_data(solver=SOLVER)
        param_prob = data[0]["param_prob"]
        n_var = param_prob.reduced_A.var_len
        T_Ab = param_prob.A

        # Tensor mapping (cvxpy works as follows)
        # T_Ab @ (theta, 1) = vec([A | b])
        param_vec = np.hstack([0, bar_a, 1])
        vecAb = T_Ab @ param_vec
        Ab = vecAb.reshape(-1, n_var + 1, order="F")
        A_rec = -Ab[:, :-1]  # note minus sign for different conic form
        b_rec = Ab[:, -1]
        s = cp.Variable(A_rec.shape[0])
        constraints = [A_rec @ x + s == b_rec]
        cones = data[0]["dims"]

        if cones.zero > 0:
            constraints.append(cp.Zero(s[: cones.zero]))
        if cones.nonneg > 0:
            constraints.append(cp.NonNeg(s[cones.zero : cones.zero + cones.nonneg]))
        # TODO: Add other cones

        prob_recovered = cp.Problem(objective, constraints)
        prob_recovered.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_recovered = x.value

        npt.assert_allclose(x_cvxpy, x_recovered, rtol=RTOL, atol=ATOL)

        # TODO: adapt this example to handle RO formulation
        # from both cvxpy and tensor reformulation

        # TODO: handle parameters in objective as well

    def test_tensor_rows(self):
        b, x, n, objective, _, _ = self.b, self.x, self.n, self.objective, self.rho, self.p

        bar_a = 0.1 * np.random.rand(n)

        # Solve with cvxpy
        # prob_cvxpy = cp.Problem(objective, [bar_a @ x + cp.norm(P @ x, p=2) <= b,  # RO
        #                                     cp.sum(x) == 1, x >= 0])
        prob_cvxpy = cp.Problem(objective, [bar_a @ x <= b, cp.sum(x) == 1, x >= 0])  # nominal
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value

        # Solve via tensor reformulation
        a = cp.Parameter(n)
        constraints = [a @ x <= b, cp.sum(x) == 1, x >= 0]
        num_constraints = calc_num_constraints(constraints)
        prob_tensor = cp.Problem(objective, constraints)
        data = prob_tensor.get_problem_data(solver=SOLVER)
        param_prob = data[0]["param_prob"]
        n_var = param_prob.reduced_A.var_len
        T_Ab = param_prob.A
        T_Ab_reshaped = tensor_reshaper(T_Ab, n_var)

        # Tensor mapping (cvxpy works as follows)
        param_vec = np.hstack([0, bar_a, 1])
        vecAb_reshaped = T_Ab_reshaped @ param_vec
        Ab_reshaped = vecAb_reshaped.reshape(num_constraints, n_var + 1, order="C")
        A_rec = -Ab_reshaped[:, :-1]  # note minus sign for different conic form
        b_rec = Ab_reshaped[:, -1]
        s = cp.Variable(A_rec.shape[0])
        constraints = [A_rec @ x + s == b_rec]
        cones = data[0]["dims"]

        if cones.zero > 0:
            constraints.append(cp.Zero(s[: cones.zero]))
        if cones.nonneg > 0:
            constraints.append(cp.NonNeg(s[cones.zero : cones.zero + cones.nonneg]))
        # TODO: Add other cones

        prob_recovered = cp.Problem(objective, constraints)
        prob_recovered.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_recovered = x.value
        npt.assert_allclose(x_cvxpy, x_recovered, rtol=RTOL, atol=ATOL)

        # TODO: adapt this example to handle RO formulation
        # from both cvxpy and tensor reformulation

        # TODO: handle parameters in objective as well

    def test_tensor_uncertain(self):
        b, x, n, objective, rho, _ = self.b, self.x, self.n, self.objective, self.rho, self.p

        bar_a = np.array([0.1, 0.1, 0.1, 0.3, 1.2])

        # Solve via tensor reformulation
        a = UncertainParameter(n, uncertainty_set=Ellipsoidal(rho=rho))
        constraints = [((3 * np.eye(n)) @ a + bar_a) @ x <= b, cp.sum(x) == 1, x >= 0]
        # num_constraints = calc_num_constraints(constraints)
        prob_tensor = cp.Problem(objective, constraints)
        data = prob_tensor.get_problem_data(solver=SOLVER)
        cones = data[0]["dims"]

        # A_rec_uncertain as a list of sparse matrices
        A_rec_certain, A_rec_uncertain, b_rec = _get_tensors(prob_tensor)
        # s = cp.Variable(num_constraints)

        newcons = []
        # cannot have equality with norms
        for i in range(cones.zero):
            newcons += [A_rec_certain[i] @ x == b_rec[i]]
        for i in range(cones.zero, cones.zero + cones.nonneg):
            # newcons += [A_rec_certain[i]@x + \
            #         rho*cp.norm(A_rec_uncertain[i][0].T@x) <= b_rec[i] ]
            newcons += [
                A_rec_certain[i] @ x + rho * cp.norm(A_rec_uncertain[i][0].T @ x) - b_rec[i]<=0
            ]

        prob_recovered = cp.Problem(objective, newcons)
        prob_recovered.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_recovered = x.value

        # b_rec = T_Ab_reshaped[:, -1]
        # Tu = T_Ab_reshaped[:, :-1]
        # TODO: isolate columns multiplying y (Ty) and multiplying u (Tu)
        # constraints = [v @ x + cp.norm(Tu.T @ x) + s == b_rec]

        # Compare against current package
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n, uncertainty_set=unc_set)
        constraints = [((3 * np.eye(n)) @ a + bar_a) @ x <= b, cp.sum(x) == 1, x >= 0]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        # Compare against CVXPY
        constraints = [
            bar_a @ x + rho * cp.norm((3 * np.eye(n)).T @ x, p=2) <= b,
            cp.sum(x) == 1,
            x >= 0,
        ]
        prob_cvxpy = cp.Problem(objective, constraints)
        prob_cvxpy.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_cvxpy = x.value

        npt.assert_allclose(x_robust, x_recovered, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(x_cvxpy, x_recovered, rtol=RTOL, atol=ATOL)

        # TODO: adapt this example to handle RO formulation
        # from both cvxpy and tensor reformulation

        # TODO: handle parameters in objective as well

    def test_tensor_uncertain_concat(self):
        b, x, n, objective, rho, _ = self.b, self.x, self.n, self.objective, self.rho, self.p

        bar_a = np.array([0.1, 0.1, 0.1, 0.3, 1.2])

        # Solve via tensor reformulation
        a = UncertainParameter(n, uncertainty_set=Ellipsoidal(rho=rho))
        constraints = [((3 * np.eye(n)) @ a + bar_a) @ x <= b, cp.sum(x) == 1, x >= 0]
        # num_constraints = calc_num_constraints(constraints)
        prob_tensor = cp.Problem(objective, constraints)
        data = prob_tensor.get_problem_data(solver=SOLVER)
        cones = data[0]["dims"]

        # A_rec_uncertain as a list of sparse matrices
        A_rec_certain, A_rec_uncertain, b_rec = _get_tensors(prob_tensor)
        # s = cp.Variable(num_constraints)

        newcons = []
        # cannot have equality with norms
        for i in range(cones.zero):
            newcons += [A_rec_certain[i] @ x == b_rec[i]]
        for i in range(cones.zero, cones.zero + cones.nonneg):
            newcons += [
                A_rec_certain[i] @ x + rho * cp.norm(A_rec_uncertain[i][0].T @ x) <= b_rec[i]
            ]

        prob_recovered = cp.Problem(objective, newcons)
        prob_recovered.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_recovered = x.value

        # turn new problem into Robust problem and solve
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n, uncertainty_set=unc_set)
        newcons = []
        for i in range(cones.zero):
            newcons += [A_rec_certain[i].toarray() @ x == b_rec[i].toarray()]
        for i in range(cones.zero, cones.zero + cones.nonneg):
            newcons += [
                A_rec_certain[i].toarray() @ x + (A_rec_uncertain[i][0].toarray() @ a).T @ x
                <= b_rec[i].toarray()
            ]

        prob_concat = RobustProblem(objective, newcons)
        prob_concat.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_concat = x.value

        # Compare against current package
        unc_set = Ellipsoidal(rho=rho)
        a = UncertainParameter(n, uncertainty_set=unc_set)
        constraints = [((3 * np.eye(n)) @ a + bar_a) @ x <= b, cp.sum(x) == 1, x >= 0]
        prob_robust = RobustProblem(objective, constraints)
        prob_robust.solve(solver=SOLVER, **SOLVER_SETTINGS)
        x_robust = x.value

        npt.assert_allclose(x_robust, x_recovered, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(x_concat, x_recovered, rtol=RTOL, atol=ATOL)
