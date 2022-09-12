import numpy as np
import scipy.sparse as spa

from lro.uncertainty_sets.ellipsoidal import Ellipsoidal


class Box(Ellipsoidal):
    """
    Box uncertainty set where :math:`u` satisfies
    :math:`\\{\\Pi(u) | \\| u \\|_{\\infty} \\le \\rho\\}`

    where :math:`\\Pi(u)` is an identity by default but can be
    an arbitrary affine transformation :math:`A u + b` explicitly
    passed by the user or an affine transformation to match the parameters
    center and length.

    Parameters
    ----------
    rho : float, optional
        Box scaling. Default 1.0.
    center : np.array, optional
        An array of dimension n, n the dimension of the uncertanty.
        Translation of the center of the box. If lengths is passed but
        not center it defaults to an array of zeros.
    side : np.array, optional
        An array of dimension n, n the dimension of the uncertanty.
        Length of each side of the box. If center is passed but side not
        it defaults to an array of 2.
    affine_transform : dict, optional
        Affine transformation dictionary with keys 'A' and 'b'.

    Returns
    -------
    Box
        Box uncertainty set.
    """

    def __init__(self, rho=1.,
                 center=None, side=None,
                 affine_transform=None):

        if center is not None:
            center = np.atleast_1d(center)
        if side is not None:
            side = np.atleast_1d(side)

        if (center is not None) and (side is None):
            side = 2. * np.ones(len(center))

        if (center is None) and (side is not None):
            center = np.zeros(len(side))

        if (center is not None) and (side is not None):
            if affine_transform:
                raise ValueError("You must provide either center/side"
                                 "or an affine transform.")

            if len(center) != len(side):
                raise ValueError("Center and side must have the same size.")

            A = spa.diags(.5 * side)
            b = center
            affine_transform = {'A': A, 'b': b}

        if rho <= 0:
            raise ValueError("Rho value must be positive.")

        super(Box, self).__init__(
            p=np.inf,
            rho=rho,
            affine_transform=affine_transform)