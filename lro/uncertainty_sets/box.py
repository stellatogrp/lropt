import numpy as np

from lro.uncertainty_sets.ellipsoidal import Ellipsoidal

# import scipy.sparse as spa


class Box(Ellipsoidal):
    r"""
    Box uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{box}} = \{u \ | \ \| Au + b \|_\infty \le \rho\}

    Parameters
    ----------
    rho : float, optional
        Box scaling. Default 1.0.
    center : np.array, optional
        An array of dimension n, n the dimension of the uncertanty.
        Translation of the center of the box. If lengths is passed but
        not center it defaults to an array of zeros.
    A : np.array, optional
        matrix defining :math:`A` in uncertainty set definition. By default :math:`A = I`
    b : np.array, optional
        vector defining :math:`b` in uncertainty set definition. By default :math:`b = 0`
    side : np.array, optional
        An array of dimension n, n the dimension of the uncertanty.
        Length of each side of the box. If center is passed but side not
        it defaults to an array of 2.
    data: np.array, optional
        An array of uncertainty realizations, where each row is one realization. Required if the uncertainty should
        be trained, or if `loss` function passed.
    loss: function, optional
        The loss function used to train the uncertainty set. Required if uncertainty set parameters should be trained
        or if `data` is passed. function must use torch tensors, and arguments to loss function must be given in the
        same order as cvxpy variables defined in problem.

    Returns
    -------
    Box
        Box uncertainty set.
    """

    def __init__(self, rho=1.,
                 A=None, b=None, center=None, side=None,
                 affine_transform=None, data=None, loss=None):

        # import ipdb
        # ipdb.set_trace()
        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        # if data is None and center is None and side is None and affine_transform is None:
        #     raise ValueError("You must provide either data "
        #                      "or an affine transform "
        #                      "or a center/side.")
        if data is not None and affine_transform:
            raise ValueError("You must provide either data "
                             "or an affine transform "
                             "or a center/side.")
        if data is not None and (center is not None or side is not None):
            raise ValueError("You must provide either data "
                             "or an affine transform "
                             "or a center/side.")
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
                raise ValueError("You must provide either center/side "
                                 "or an affine transform.")

            if len(center) != len(side):
                raise ValueError("Center and side must have the same size.")

            A_init = np.diag(.5 * side)
            b_init = center
            affine_transform = {'A': A_init, 'b': b_init}

        if rho <= 0:
            raise ValueError("Rho value must be positive.")

        if data is not None:
            super(Box, self).__init__(
                p=np.inf,
                rho=rho,
                affine_transform=None, data=data, loss=loss, A=A, b=b)
        else:
            super(Box, self).__init__(
                p=np.inf,
                rho=rho,
                affine_transform=affine_transform, data=None, loss=None, A=A, b=b)
