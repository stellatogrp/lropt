
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
