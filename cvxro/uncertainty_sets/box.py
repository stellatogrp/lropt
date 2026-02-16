import numpy as np

from cvxro.uncertainty_sets.norm import Norm


class Box(Norm):
    r"""
    Box uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{box}} = \{Az+b \ | \ \|z\|_\infty \le \rho\}

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
        Required if the uncertainty should be trained.
    c: np.array, optional
        matrix defining the lhs of the polyhedral support: :math:`cu \le d`. By default None.
    d: np.array, optional
        vector defining the rhs of the polyhedral support: :math:`cu \le d`. By default None.
    ub: np.array | float, optional
        vector or float defining the upper bound of the support. If scalar, broadcast to a vector.
        By default None.
    lb: np.array | float, optional
        vector or float defining the lower bound of the support. If scalar, broadcast to a vector.
        By default None.
    sum_eq: np.array | float, optional
        vector or float defining an equality constraint for the uncertain vector. By default None.
    indices_dict: dict, optional
        Optional mapping with keys 'train', 'test', and 'validate' that specify
        integer index arrays or boolean masks to split the provided `data` into
        training, testing, and validation subsets. If provided, the indices are
        validated against `data.shape[0]` and normalized to integer index arrays.

    Returns
    -------
    Box
        Box uncertainty set.
    """

    def __init__(self, dimension = None, rho=1.,
                 a=None, b=None, c=None, d=None,
                 data=None, ub=None, lb=None, sum_eq=None, eval_data=None, indices_dict=None):

        super(Box, self).__init__(
            dimension =dimension,
            p=np.inf,
            rho=rho,
            a=a, b=b, c=c, d=d, data=data,  ub=ub,
            lb=lb, sum_eq=sum_eq, eval_data=eval_data,
              indices_dict=indices_dict)
