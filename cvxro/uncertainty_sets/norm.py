import numpy as np
from cvxpy import Variable, norm

from cvxro.parameter import ShapeParameter, SizeParameter
from cvxro.uncertainty_sets.uncertainty_set import UncertaintySet
from cvxro.uncertainty_sets.utils import check_indices_dict


class Norm(UncertaintySet):
    r"""
    Norm uncertainty set, defined as

    .. math::
        \mathcal{U}_{\text{Norm}} = \{Az+b \ | \ \|z\|_p \le \rho\}

    when :math:`p = 2` this is an ellipsoidal set, and when :math:`p = \infty` this is a box set

    Parameters
    ----------
    rho : float, optional
        Default 1.0.
    p : integer, optional
        Order of the norm. Default 2.
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
    Norm
        Norm uncertainty set.
    """

    def __init__(self, dimension = None, p=2, rho=1.,
                 a=None, b=None, c=None, d=None, data=None,
                 ub=None, lb=None, sum_eq=None, eval_data = None, indices_dict=None):
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")

        if data is not None:
            dat_shape = data.shape[1]
            if dimension is None:
                dimension = dat_shape
            a = ShapeParameter((dat_shape, dimension))
            a.value = np.eye(dat_shape, dimension)
            b = ShapeParameter(dat_shape)
            b.value = np.mean(data, axis=0)

        if dimension is not None:
            if a is not None:
                if a.shape[1] != dimension:
                    raise ValueError("Mismatching dimension for A.")
            if a is None:
                raise ValueError("You must provide A if you provide a dimension without data.")

        self.affine_transform_temp = None
        self.affine_transform = None
        self.eval_data = eval_data
        self._dimension = dimension
        self._p = p
        self._rho = rho
        self._data = data
        self._a = a
        self._b = b
        # validate and normalize indices_dict if provided
        if indices_dict is not None:
            normalized = check_indices_dict(indices_dict, None if data is None else data.shape[0])
            self.indices_dict = normalized
        else:
            self.indices_dict = None
        self._trained = False
        self._c = c
        self._d = d
        self._define_support = False
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq
        self._rho_mult = SizeParameter(value=1.)


    @property
    def p(self):
        return self._p

    @property
    def rho_mult(self):
        return self._rho_mult

    @property
    def dimension(self):
        return self._dimension

    @property
    def rho(self):
        return self._rho

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    @property
    def data(self):
        return self._data

    @property
    def ub(self):
        return self._ub

    @property
    def lb(self):
        return self._lb

    @property
    def sum_eq(self):
        return self._sum_eq

    @property
    def trained(self):
        return self._trained

    def dual_norm(self):
        if self.p == 1:
            return np.inf
        return 1. + 1. / (self.p - 1.)

    def conjugate(self, var, supp_var, k_ind=0):
        if not self._define_support:
            if self._c is None:
                if not isinstance(var, Variable):
                    self._c = np.zeros((var, var))
                else:
                    self._c = np.zeros((supp_var.shape[0], supp_var.shape[0]))
            if self._d is None:
                if not isinstance(var, Variable):
                    self._d = np.zeros(var)
                else:
                    self._d = np.zeros(supp_var.shape[0])
            self._define_support = True
        if not isinstance(var, Variable):
            lmbda = Variable()
            constr = [lmbda >= 0]
            return self.rho_mult*self.rho*lmbda, constr, lmbda, None
        else:
            lmbda = Variable()
            supp_newvar = Variable(len(self._d))
            constr = [norm(var, p=self.dual_norm()) <= lmbda]
            constr += [self._c.T@supp_newvar == supp_var]
            constr += [supp_newvar >= 0]
            return self.rho_mult*self.rho * lmbda + self._d@supp_newvar, constr, lmbda, None
