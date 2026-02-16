import numpy as np
from cvxpy import Variable, norm

from cvxro.parameter import ShapeParameter, SizeParameter
from cvxro.uncertainty_sets.uncertainty_set import UncertaintySet


class Budget(UncertaintySet):
    r"""
    Budget uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{budget}} = \{Az+b \ | \ \|z \|_\infty \le \rho_1,
        \|z \|_1 \leq \rho_2\}

    Parameters
    ----------
    rho1 : float, optional
        Box scaling. Default 1.0.
    rho2 : float, optional
        1-norm scaling. Default 1.0.
    a1 : np.array, optional
        matrix defining :math:`A_1` in uncertainty set definition. By default :math:`A_1 = I`
    a2 : np.array, optional
        matrix defining :math:`A_2` in uncertainty set definition. By default :math:`A_2 = I`
    b1 : np.array, optional
        vector defining :math:`b_1` in uncertainty set definition. By default :math:`b_1 = 0`
    b2 : np.array, optional
        vector defining :math:`b_2` in uncertainty set definition. By default :math:`b_2 = 0`
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
    Budget
        Budget uncertainty set.
    """

    def __init__(self, dimension=None, rho1=1., rho2=1.,
                 a=None, b=None, c=None, d=None, data=None,
                  ub=None, lb=None, sum_eq=None, eval_data=None, indices_dict=None):
        if rho2 <= 0 or rho1 <= 0:
            raise ValueError("Rho values must be positive.")

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
        self.eval_data = eval_data
        self.affine_transform = None
        self._dimension = dimension
        self._rho1 = rho1
        self._rho2 = rho2
        self._data = data
        self._a = a
        self._b = b
        self._trained = False
        self._c = c
        self._d = d
        self._define_support = False
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq
        self._rho_mult = SizeParameter(value=1.)
        from cvxro.uncertainty_sets.utils import check_indices_dict
        if indices_dict is not None:
            self.indices_dict = check_indices_dict(
                indices_dict, None if data is None else data.shape[0])
        else:
            self.indices_dict = None


    @property
    def rho1(self):
        return self._rho1

    @property
    def rho2(self):
        return self._rho2

    @property
    def rho_mult(self):
        return self._rho_mult

    @property
    def dimension(self):
        return self._dimension

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def data(self):
        return self._data

    @property
    def trained(self):
        return self._trained

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
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

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
        if isinstance(var, Variable):
            ushape = var.shape
            newvar1 = Variable(ushape)
            newvar2 = Variable(ushape)
            constr = [newvar1 + newvar2 == var]
        else:
            ushape = var
            newvar1 = Variable(ushape)
            newvar2 = Variable(ushape)
            constr = [newvar1 + newvar2 == 0]
        lmbda1 = Variable()
        lmbda2 = Variable()
        supp_newvar = Variable(len(self._d))
        constr += [norm(newvar1, 1) <= lmbda1]
        constr += [norm(newvar2, np.inf) <= lmbda2]
        constr += [self._c.T@supp_newvar == supp_var]
        constr += [supp_newvar >= 0]
        return self.rho_mult*self.rho1*lmbda1 + self._d@supp_newvar +\
                self.rho_mult*self.rho2*lmbda2, \
            constr, (lmbda1, lmbda2), None
