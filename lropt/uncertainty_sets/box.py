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
        An array of uncertainty realizations, where each row is one realization.
        Required if the uncertainty should be trained, or if `loss` function passed.
    loss: function, optional
        The loss function used to train the uncertainty set.
        Required if uncertainty set parameters should be trained or if `data` is passed.
        Function must use torch tensors, and arguments to loss function must be given in the
        same order as cvxpy variables defined in problem.
    c: np.array, optional
        matrix defining the lhs of the polyhedral support: :math: `cu \le d`. By default None.
    d: np.array, optional
        vector defining the rhs of the polyhedral support: :math: `cu \le d`. By default None.
    ub: np.array | float, optional
        vector or float defining the upper bound of the support. If scalar, broadcast to a vector. 
        By default None.
    lb: np.array | float, optional
        vector or float defining the lower bound of the support. If scalar, broadcast to a vector. 
        By default None.
    sum_eq: np.array | float, optinal
        vector or float defining an equality constraint for the uncertain vector. By default None.

    Returns
    -------
    Box
        Box uncertainty set.
    """

    def __init__(self, rho=1.,
                 a=None, b=None, c=None, d=None,
                 data=None, loss=None, ub=None, lb=None,sum_eq=None):

        super(Box, self).__init__(
            p=np.inf,
            rho=rho,
            a=a, b=b, c=c, d=d, data=data, loss=loss, ub=ub, lb=lb, sum_eq=sum_eq)
