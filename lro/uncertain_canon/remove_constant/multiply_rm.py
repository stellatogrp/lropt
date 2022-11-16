from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression


def multiply_rm(unc_canon, expr, constant):
    if expr.args[0].is_constant() and not unc_canon.has_unc_param(expr.args[0]):
        if expr.args[0].is_scalar():
            constant = (constant*expr.args[0]).value
        elif isinstance(expr.args[0], Promote):
            constant = (constant*expr.args[0].args[0]).value
        return unc_canon.remove_constant(expr.args[1], constant)
    if expr.args[1].is_constant() and not unc_canon.has_unc_param(expr.args[1]):
        if expr.args[1].is_scalar():
            constant = (constant*expr.args[1]).value
        elif isinstance(expr.args[1], Promote):
            constant = (constant*expr.args[1].args[0]).value
        return unc_canon.remove_constant(expr.args[0], constant)
    if isinstance(expr.args[0], NegExpression) and isinstance(expr.args[1], NegExpression):
        return unc_canon.remove_constant(expr.args[0].args[0]*expr.args[1].args[0], constant)
    elif isinstance(expr.args[0], NegExpression):
        constant = (-1*constant).value
        return unc_canon.remove_constant(expr.args[0].args[0], constant)
    elif isinstance(expr.args[1], NegExpression):
        constant = (-1*constant).value
        return unc_canon.remove_constant(expr.args[1].args[0], constant)
    else:
        expr1, constant = unc_canon.remove_constant(expr.args[0], constant)
        expr2, constant = unc_canon.remove_constant(expr.args[1], constant)
        return expr1*expr2, constant
