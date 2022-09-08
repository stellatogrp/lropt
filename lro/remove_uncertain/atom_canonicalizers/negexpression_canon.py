from cvxpy.atoms.affine.unary_operators import NegExpression


def negexpression_canon(expr, args):

    if isinstance(expr, NegExpression) and \
            isinstance(args[0], NegExpression):
        return args[0].args[0], []
    else:
        return expr, []
