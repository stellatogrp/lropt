import numpy as np

# from cvxpy.atoms.affine.unary_operators import NegExpression
from lropt.remove_uncertain.atom_canonicalizers.mul_canon import \
    mul_canon_transform
from lropt.uncertain import UncertainParameter


def mulexpression_canon(expr, args, var, cons):

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
    #         u = mulexpression_canon_transform(u, -np.eye(u.shape[0]))
    # elif isinstance(args[1], NegExpression):
    #     if isinstance(args[1].args[0], UncertainParameter):
    #         u = args[1].args[0]
    #         x = args[0]
    #         u = mulexpression_canon_transform(u, -np.eye(u.shape[0]))
    else:
        # No uncertain variables
        return args[0]@args[1], []

    # adjust affine transform when multiplied by a constant matrix
    if (x.is_constant()):
        u = mulexpression_canon_transform(u, x)
        return u, []
    u = mul_canon_transform(u, cons)
    return u.canonicalize(x, var)


def mulexpression_canon_transform(u, P):
    # import ipdb
    # ipdb.set_trace()
    # adjust affine transform
    uset = u.uncertainty_set
    if uset.affine_transform_temp:
        uset.affine_transform_temp['b'] = P@uset.affine_transform_temp['b']
        uset.affine_transform_temp['A'] = P@uset.affine_transform_temp['A']
    else:
        uset.affine_transform_temp = {'A': P, 'b': np.zeros(np.shape(P)[0])}
    return u
