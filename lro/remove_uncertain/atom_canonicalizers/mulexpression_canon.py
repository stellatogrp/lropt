import numpy as np
from cvxpy.atoms.affine.unary_operators import NegExpression

from lro.uncertain import UncertainParameter


def mulexpression_canon(expr, args, var):

    # import ipdb
    # ipdb.set_trace()
    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args
    # check for negative parameter and negate affine transform
    elif isinstance(args[0], NegExpression):
        if isinstance(args[0].args[0], UncertainParameter):
            u = args[0].args[0]
            x = args[1]
            u = mulexpression_canon_transform(u, -np.eye(u.shape[0]))
    elif isinstance(args[1], NegExpression):
        if isinstance(args[1].args[0], UncertainParameter):
            u = args[1].args[0]
            x = args[0]
            u = mulexpression_canon_transform(u, -np.eye(u.shape[0]))
    else:
        # No uncertain variables
        return args[0]@args[1], []

    # adjust affine transform when multiplied by a constant matrix
    if (x.is_constant()):
        u = mulexpression_canon_transform(u, x)
        return u, []
    return u.canonicalize(x, var)


def mulexpression_canon_transform(u, P):
    # adjust affine transform
    uset = u.uncertainty_set
    trans = uset.affine_transform_temp
    if trans:
        trans['b'] = P@trans['b']
        trans['A'] = P@trans['A']
    else:
        trans['b'] = 0
        trans['A'] = P
    return u
