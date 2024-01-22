import numpy as np

# from cvxpy.atoms.affine.unary_operators import NegExpression
from lropt.remove_uncertain.atom_canonicalizers.mul_canon import mul_canon_transform
from lropt.uncertain import UncertainParameter


def mulexpression_canon(expr, args, var, cons):


    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args
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
    if len(P.shape) == 1:
        P = np.reshape(P,(1,P.shape[0]))
    uset = u.uncertainty_set
    if uset.affine_transform_temp:
        uset.affine_transform_temp['b'] = P@uset.affine_transform_temp['b']
        uset.affine_transform_temp['A'] = P@uset.affine_transform_temp['A']
    else:
        uset.affine_transform_temp = {'A': P, 'b': np.zeros(np.shape(P)[0])}
    return u
