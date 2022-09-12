from lro.uncertain import UncertainParameter


def mulexpression_canon(expr, args):

    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args
    else:
        # No uncertain variables
        return expr, []

    return u.canonicalize(x)
