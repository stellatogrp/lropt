import numpy as np

from lropt.uncertainty_sets.norm import Norm

# import scipy.sparse as spa


class Box(Norm):
    r"""
    Box uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{box}} = \{u \ | \ \| Au + b \|_\infty \le \rho\}

    Parameters
    ----------
    rho : float, optional
        Box scaling. Default 1.0.
    A : np.array, optional
        matrix defining :math:`A` in uncertainty set definition. By default :math:`A = I`
    b : np.array, optional
        vector defining :math:`b` in uncertainty set definition. By default :math:`b = 0`
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
                 A=None, b=None,
                 data=None, loss=None):

        super(Box, self).__init__(
            p=np.inf,
            rho=rho,
            A=A, b=b, data=data, loss=loss)
