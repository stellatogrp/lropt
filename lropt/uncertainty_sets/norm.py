import numpy as np
from cvxpy import Variable, norm

from lropt.train.parameter import EpsParameter, ShapeParameter
from lropt.uncertainty_sets.uncertainty_set import UncertaintySet


class Norm(UncertaintySet):
    r"""
    Norm uncertainty set, defined as

    .. math::
        \mathcal{U}_{\text{Norm}} = \{Az+b \ | \ z\| \|_p \le \rho\}

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
    Norm
        Norm uncertainty set.
    """

    def __init__(self, dimension = None, p=2, rho=1.,
                 a=None, b=None, c=None, d=None, data=None, loss=None,
                 ub=None, lb=None, sum_eq=None):
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")

        if data is not None:
            dat_shape = data.shape[1]
            if dimension is None:
                dimension = dat_shape
            a = ShapeParameter((dat_shape, dimension))
            b = ShapeParameter(dat_shape)

        if dimension is not None:
            if a is not None:
                if a.shape[1] != dimension:
                    raise ValueError("Mismatching dimension for A.")
            if a is None:
                raise ValueError("You must provide A if you provide a dimension without data.")

        self.affine_transform_temp = None
        self.affine_transform = None

        self._dimension = dimension
        self._p = p
        self._rho = rho
        self._data = data
        self._a = a
        self._b = b
        self._trained = False
        self._loss = loss
        self._c = c
        self._d = d
        self._define_support = False
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq
        self._rho_mult = EpsParameter(value=1.)


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
    def loss(self):
        return self._loss

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
            # ushape = var.shape[1]  # shape of uncertainty
            # if self.b is None:
            #     self._b = np.zeros(ushape)
            lmbda = Variable()
            supp_newvar = Variable(len(self._d))
            constr = [norm(var, p=self.dual_norm()) <= lmbda]
            constr += [lmbda >= 0]
            constr += [self._c.T@supp_newvar == supp_var]
            constr += [supp_newvar >= 0]
            return self.rho_mult*self.rho * lmbda + self._d@supp_newvar, constr, lmbda, None
