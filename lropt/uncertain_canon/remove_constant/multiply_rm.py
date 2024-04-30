from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.diag import diag
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression


def modify(constant):
    while not (isinstance(constant,int) or isinstance(constant,float)):
        constant = constant[0]
    return constant

def multiply_rm(unc_canon, expr, constant):
    if expr.args[0].is_constant() and not unc_canon.has_unc_param(expr.args[0]):
        if expr.args[0].is_scalar():
            constant = modify((constant*expr.args[0]).value)
        elif isinstance(expr.args[0], Promote):
            constant = modify((constant*expr.args[0].args[0]).value)
        else:
            return diag(expr.args[0])@expr.args[1], constant
        return unc_canon.remove_constant(expr.args[1], constant)
    if expr.args[1].is_constant() and not unc_canon.has_unc_param(expr.args[1]):
        if expr.args[1].is_scalar():
            constant = modify((constant*expr.args[1]).value)
        elif isinstance(expr.args[1], Promote):
            constant = modify((constant*expr.args[1].args[0]).value)
        else:
            return diag(expr.args[1])@expr.args[0], constant
        return unc_canon.remove_constant(expr.args[0], constant)
    if isinstance(expr.args[0], NegExpression) and isinstance(expr.args[1], NegExpression):
        return unc_canon.remove_constant(expr.args[0].args[0]*expr.args[1].args[0], constant)
    elif isinstance(expr.args[0], NegExpression):
        constant = modify((-1*constant).value)
        return unc_canon.remove_constant(expr.args[0].args[0], constant)
    elif isinstance(expr.args[1], NegExpression):
        constant = modify((-1*constant).value)
        return unc_canon.remove_constant(expr.args[1].args[0], constant)
    else:
        expr1, constant = unc_canon.remove_constant(expr.args[0], constant)
        expr2, constant = unc_canon.remove_constant(expr.args[1], constant)
        return multiply(expr1,expr2), constant
