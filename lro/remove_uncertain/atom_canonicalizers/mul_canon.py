from lro.uncertain import UncertainParameter


def mul_canon(expr, args):

    import ipdb
    ipdb.set_trace()
    # Check for direct parameter usage
    if isinstance(args[0], UncertainParameter):
        u, x = args
    elif isinstance(args[1], UncertainParameter):
        x, u = args
    else:
        # No uncertain variables
        return expr, []
    # if (x.is_scalar()):
    #    for element in range(u.shape[0]):
    #        u
    return u.canonicalize(x)
