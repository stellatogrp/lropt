from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression


def multiply_rm(unc_canon, expr, cons):
    if expr.args[0].is_constant() and not unc_canon.has_unc_param(expr.args[0]):
        if expr.args[0].is_scalar():
            cons = (cons*expr.args[0]).value
        elif isinstance(expr.args[0], Promote):
            cons = (cons*expr.args[0].args[0]).value
        return unc_canon.remove_const(expr.args[1], cons)
    if expr.args[1].is_constant() and not unc_canon.has_unc_param(expr.args[1]):
        if expr.args[1].is_scalar():
            cons = (cons*expr.args[1]).value
        elif isinstance(expr.args[1], Promote):
            cons = (cons*expr.args[1].args[0]).value
        return unc_canon.remove_const(expr.args[0], cons)
    if isinstance(expr.args[0], NegExpression) and isinstance(expr.args[1], NegExpression):
        return unc_canon.remove_const(expr.args[0].args[0]*expr.args[1].args[0], cons)
    elif isinstance(expr.args[0], NegExpression):
        cons = (-1*cons).value
        return unc_canon.remove_const(expr.args[0].args[0], cons)
    elif isinstance(expr.args[1], NegExpression):
        cons = (-1*cons).value
        return unc_canon.remove_const(expr.args[1].args[0], cons)
    else:
        expr1, cons = unc_canon.remove_const(expr.args[0], cons)
        expr2, cons = unc_canon.remove_const(expr.args[1], cons)
        return expr1*expr2, cons
