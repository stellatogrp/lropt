#  from cvxpy.atoms.affine.promote import promote
#  from cvxpy.expressions.variable import Variable
from rcvx.uncertain import UncertainParameter
#  from cvxpy.atoms.affine.unary_operators import NegExpression


def matmul_canon(expr, args):

    # Check of negative expressions
    #  import ipdb; ipdb.set_trace()
    #  if isinstance(args[0], NegExpression) and \
    #          isinstance(args[0].args[0], UncertainParameter):
    #      u = args[0].args[0]
    #      x = args[1]
    #      new_expr, new_con = matmul_canon(u * x, [u, x])
    #      return -new_expr, new_con
    #
    #  if isinstance(args[1], NegExpression) and \
    #          isinstance(args[1].args[0], UncertainParameter):
    #      u = args[1].args[0]
    #      x = args[0]
    #      new_expr, new_con = matmul_canon(u * x, [u, x])
    #      return -new_expr, new_con
    #
    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args
    else:
        # No uncertain variables
        return expr, []

    return u.canonicalize(x)

    # TODO: Need to check if constraints have proper uncertainty sets

    # Return expression by copying arguments
    # Add constraints
    # return expression_with_changed args, new constraints

    #  x = args[0]
    #  shape = expr.shape
    #  t = Variable(shape)
    #
    #  promoted_t = promote(t, x.shape)
    #  return t, [x <= promoted_t, x + promoted_t >= 0]
