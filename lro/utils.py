class UncertaintyError(Exception):
    """Error thrown if the uncertain problem has not been formulated correctly.
    """
    pass


def check_affine_transform(affine_transform):
    assert 'b' in affine_transform
    assert 'A' in affine_transform
