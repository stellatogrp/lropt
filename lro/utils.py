from lro.uncertain import UncertainParameter


class UncertaintyError(Exception):
    """Error thrown if the uncertain problem has not been formulated correctly.
    """
    pass


def check_affine_transform(affine_transform):
    assert 'b' in affine_transform
    assert 'A' in affine_transform


def unique_list(duplicates_list):
    """
    Return unique list preserving the order.
    https://stackoverflow.com/a/480227
    """
    used = set()
    unique = [x for x in duplicates_list if not (x in used or used.add(x))]
    return unique


def has_uncertain_param(expr):
    # does this expression have an uncertain parameter?
    has_uncertain = False
    if len(expr.args) == 0:
        return isinstance(expr, UncertainParameter)

    else:
        for args in expr.args:
            if has_uncertain_param(args):
                return True

        return has_uncertain


# def consolidate_constants(expr):
#     # take a single expression, and consolidate all of the constants
#     c = []
#     if len(expr.args) is 0:
#         if isinstance(expr, UncertainParameter):
#             has_uncertain = True
#             u = expr

#     else:
#         for args in expr.args:
#             if has_uncertain_param(args):
#                 return True

#     return has_uncertain, u
