import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse._coo import coo_matrix


def standard_invert(solution: Solution, inverse_data: InverseData) -> Solution:
    """
    This function maps the primal/dual variables of the solution to the ones of original problem.
    Should be used as an invert function of a Reduction object.
    """
    pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                if vid in solution.primal_vars}
    dvars = {orig_id: solution.dual_vars[vid]
                for orig_id, vid in inverse_data.cons_id_map.items()
                if vid in solution.dual_vars}

    return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)

def tensor_reshaper(T_Ab: coo_matrix, n_var: int) -> np.ndarray:
    """
    This function reshapes T_Ab so T_Ab@param_vec gives the constraints row by row instead of
    column by column. At the moment, it returns a dense matrix instead of a sparse one.
    """
    def _calc_source_row(target_row: int, num_constraints: int, n_var: int) -> int:
        """
        This is a helper function that calculates the index of the source row of T_Ab for the
        reshaped target row.
        """
        constraint_num = 0 if n_var == 0 else target_row%n_var
        var_num = target_row if n_var == 0 else target_row//n_var
        source_row = constraint_num*num_constraints+var_num
        return source_row


    T_Ab = csc_matrix(T_Ab)
    n_var_full = n_var+1 #Includes the free paramter
    num_rows = T_Ab.shape[0]
    num_constraints = num_rows//n_var_full
    T_Ab_res = csr_matrix(T_Ab.shape)
    for target_row in range(num_rows): #Counter for populating the new row of T_Ab_res
        source_row = _calc_source_row(target_row, num_constraints, n_var_full)
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

def promote_expr(expr: Expression | int) -> Expression | int:
    """
    This function unsqueezes an expression, converting it from a 1D expression to a 2D expression.
    """
    if not isinstance(expr, Expression):
        return expr
    ndim = expr.ndim
    if ndim > 1:
        return expr
    
    return cp.reshape(expr, shape=(expr.size, 1))

def unique_list(duplicates_list):
    """
    Return unique list preserving the order.
    https://stackoverflow.com/a/480227
    """
    used = set()
    unique = [x for x in duplicates_list if not (x in used or used.add(x))]

    return unique