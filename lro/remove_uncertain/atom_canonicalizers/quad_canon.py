import numpy as np
import scipy as sc
from cvxpy.atoms.quad_over_lin import quad_over_lin

from lro.uncertain import UncertainParameter


def quad_canon(expr, args, var, cons):
    # import ipdb
    # ipdb.set_trace()
    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, matrix = args
        if len(matrix.variables()) == 0:
            raise ValueError("You must multiply the matrix by a variable")
        P = matrix.args[0]
        x = matrix.args[1].args[0]
        P_invsqrt = sc.linalg.sqrtm(np.linalg.inv(P))
        new_expr = quad_over_lin(var@P_invsqrt, (4*x)*cons)

        return new_expr, []
    else:
        return expr, []
