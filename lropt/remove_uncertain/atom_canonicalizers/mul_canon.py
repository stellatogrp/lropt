import numpy as np
from cvxpy.atoms.affine.promote import Promote

from lropt.uncertain import UncertainParameter

# from cvxpy.atoms.affine.unary_operators import NegExpression


def mul_canon(expr, args, var, cons):

    # import ipdb
    # ipdb.set_trace()
    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args
    # check for negative parameter and negate affine transform
    # elif isinstance(args[0], NegExpression):
    #     if isinstance(args[0].args[0], UncertainParameter):
    #         u = args[0].args[0]
    #         x = args[1]
    #         u = mul_canon_transform(u, -1)
    # elif isinstance(args[1], NegExpression):
    #     if isinstance(args[1].args[0], UncertainParameter):
    #         u = args[1].args[0]
    #         x = args[0]
    #         u = mul_canon_transform(u, -1)
    else:
        # No uncertain variables
        return args[0]*args[1], []

    # adjust affine transform when multiplied by a constant
    # if (x.is_constant()):
    #     u = mul_canon_transform(u, x)
    #     return u, []
    u = mul_canon_transform(u, cons)
    return u.canonicalize(x, var)


def mul_canon_transform(u, c):
    # adjust affine transform
    # import ipdb
    # ipdb.set_trace()
    uset = u.uncertainty_set
    if isinstance(c, Promote):
        c = c.value[0]
    if uset.affine_transform_temp:
        uset.affine_transform_temp['b'] = c*uset.affine_transform_temp['b']
        uset.affine_transform_temp['A'] = c*uset.affine_transform_temp['A']
    else:
        if len(u.shape) == 0:
            uset.affine_transform_temp = {'A': c*np.eye(1), 'b': 0}
        else:
            uset.affine_transform_temp = {'A': c*np.eye(u.shape[0]), 'b': np.zeros(u.shape[0])}

    return u


def mul_convert(u, c):
    # adjust affine transform
    # import ipdb
    # ipdb.set_trace()
    uset = u.uncertainty_set
    if uset.affine_transform_temp:
        uset.affine_transform_temp['A'] = c*uset.affine_transform_temp['A']
    else:
        if len(u.shape) == 0:
            uset.affine_transform_temp = {'A': c*np.eye(1), 'b': 0}
        else:
            uset.affine_transform_temp = {'A': c*np.eye(u.shape[0]), 'b': np.zeros(u.shape[0])}

    return u
