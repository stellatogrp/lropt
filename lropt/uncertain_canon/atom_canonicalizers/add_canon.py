import numpy as np

from lropt.uncertain import UncertainParameter

# from cvxpy.atoms.affine.unary_operators import NegExpression

# from lropt.uncertain_canon.atom_canonicalizers.mul_canon import \
# mul_canon_transform
# from lropt.uncertain_canon.atom_canonicalizers.mulexpression_canon import \
# mulexpression_canon_transform


def add_canon(expr, args, var, cons):

    # import ipdb
    # ipdb.set_trace()
    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args

    else:
        # No uncertain variables
        return args[0] + args[1], []

    # adjust affine transform when adding constant vector
    if (x.is_constant()):
        u = add_canon_transform(u, x)
        return u, []

    raise ValueError("You must multiply uncertainty by a variable before adding by a variable.")


def add_canon_transform(u, c):
    # adjust affine transform
    uset = u.uncertainty_set
    if uset.affine_transform_temp:
        uset.affine_transform_temp['b'] = c + uset.affine_transform_temp['b']
    else:
        uset.affine_transform_temp = {'A': np.eye(u.shape[0]), 'b': c}
    return u
