from typing import Union

import numpy as np
from cvxpy import SCS, Parameter, Variable
from cvxpy import hstack as cp_hstack
from cvxpy import reshape as cp_reshape
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.atoms.affine.hstack import Hstack
from numpy import ndarray
from scipy.sparse import csc_matrix, csr_matrix, vstack
from scipy.sparse._coo import coo_matrix

from lropt import Parameter as LroptParameter
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.utils import standard_invert

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

def unsqueeze_expression(expr: Expression | int) -> Expression | int:
    """
    This function unsqueezes an expression, converting it from a 1D expression to a 2D expression.
    """
    if not isinstance(expr, Expression):
        return expr
    ndim = expr.ndim
    if ndim > 1:
        return expr
    
    return cp_reshape(expr, shape=(expr.size, 1))