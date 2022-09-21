import numpy as np
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression

from lro.uncertain import UncertainParameter


def mul_canon(expr, args):

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
            u = mul_canon_transform(u, -1)
    elif isinstance(args[1], NegExpression):
        if isinstance(args[1].args[0], UncertainParameter):
            u = args[1].args[0]
            x = args[0]
            u = mul_canon_transform(u, -1)
    else:
        # No uncertain variables
        return args[0]*args[1], []

    # adjust affine transform when multiplied by a constant
    if (x.is_constant()):
        u = mul_canon_transform(u, x)
        return u, []
    return u.canonicalize(x)


def mul_canon_transform(u, c):
    # adjust affine transform
    uset = u.uncertainty_set
    trans = uset.affine_transform
    if isinstance(c, Promote):
        c = c.value[0]
    if trans:
        trans['b'] = c*trans['b']
        trans['A'] = c*trans['A']
    else:
        trans['b'] = 0
        trans['A'] = c*np.eye(u.shape[0])
    return u
