import numpy as np
from cvxpy import Variable, norm

from lropt.shape_parameter import ShapeParameter
from lropt.uncertainty_sets.uncertainty_set import UncertaintySet


class Norm(UncertaintySet):
    r"""
    Norm uncertainty set, defined as

    .. math::
        \mathcal{U}_{\text{Norm}} = \{u \ | \ \| Au + b\|_p \le \rho\}

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

    def __init__(self, p=2, rho=1.,
                 a=None, b=None, c=None, d=None, data=None, loss=None,
                 ub=None, lb=None, sum_eq=None):
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")

        if data is not None:
            dat_shape = data.shape[1]
            a = ShapeParameter((dat_shape, dat_shape))
            b = ShapeParameter(dat_shape)

        self.affine_transform_temp = None
        self.affine_transform = None

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


    @property
    def p(self):
        return self._p

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

    def canonicalize(self, x, var):
        # import ipdb
        # ipdb.set_trace()
        trans = self.affine_transform_temp
        if x.is_scalar():
            if trans:
                new_expr = trans['b'] * x
                new_constraints = [var == -trans['A'] * x]
            else:
                new_expr = 0
                new_constraints = [var == -x]

        else:
            if trans:
                new_expr = trans['b'] @ x
                new_constraints = [var == -trans['A'].T @ x]
            else:
                new_expr = 0
                new_constraints = [var == -x]

        if self.affine_transform:
            self.affine_transform_temp = self.affine_transform.copy()
        else:
            self.affine_transform_temp = None

        return new_expr, new_constraints

    def isolated_unc(self, i, var, num_constr):
        trans = self.affine_transform_temp
        new_expr = 0
        if i == 0:
            if trans:
                new_expr += trans['b']
        e = np.eye(num_constr)[i]
        if len(trans['A'].shape) == 1:
            if var.is_scalar():
                if trans:
                    new_constraints = [var == -trans['A'] * e]
                else:
                    new_constraints = [var == - e]
            else:
                if trans:
                    new_constraints = [var == -trans['A']]
                else:
                    new_constraints = [var == - e]
        else:
            if var.is_scalar():
                if trans:
                    new_constraints = [var == -trans['A'] * e]
                else:
                    new_constraints = [var == - e]
            else:
                if trans:
                    new_constraints = [var == -trans['A'].T @ e]
                else:
                    new_constraints = [var == - e]
        if i == (num_constr - 1):
            if self.affine_transform:
                self.affine_transform_temp = self.affine_transform.copy()
            else:
                self.affine_transform_temp = None
        return new_expr, new_constraints

    def conjugate(self, var, supp_var, shape, k_ind=0):
        if not self._define_support:
            if self._c is None:
                if not isinstance(var, Variable):
                    self._c = np.zeros((var, var))
                else:
                    self._c = np.zeros((var.shape[1], var.shape[1]))
            if self._d is None:
                if not isinstance(var, Variable):
                    self._d = np.zeros(var)
                else:
                    self._d = np.zeros(var.shape[1])
            self._define_support = True
        if not isinstance(var, Variable):
            lmbda = Variable(shape)
            constr = [lmbda >= 0]
            return self.rho*lmbda, constr, lmbda
        else:
            ushape = var.shape[1]  # shape of uncertainty
            if self.b is None:
                self._b = np.zeros(ushape)
            if self.data is not None or self.a is not None:
                if shape == 1:
                    newvar = Variable(ushape)  # gamma aux variable
                    lmbda = Variable()
                    supp_newvar = Variable(len(self._d))
                    constr = [norm(newvar, p=self.dual_norm()) <= lmbda]
                    constr += [self.a.T@newvar == var[0]]
                    constr += [lmbda >= 0]
                    constr += [self._c.T@supp_newvar == supp_var[0]]
                    constr += [supp_newvar >= 0]
                    return self.rho * lmbda + self._d@supp_newvar - newvar@self.b, constr, lmbda
                else:
                    constr = []
                    lmbda = Variable(shape)
                    newvar = Variable((shape, ushape))
                    constr += [lmbda >= 0]
                    supp_newvar = Variable((shape, len(self._d)))
                    constr += [supp_newvar >= 0]
                    for ind in range(shape):
                        constr += [norm(newvar[ind],
                                        p=self.dual_norm()) <= lmbda[ind]]
                        constr += [self.a.T@newvar[ind] == var[ind]]
                        constr += [self._c.T@supp_newvar[ind] == supp_var[ind]]
                    return self.rho * lmbda + supp_newvar@self._d - newvar@self.b, constr, lmbda
            else:
                if shape == 1:
                    lmbda = Variable()
                    supp_newvar = Variable(len(self._d))
                    constr = [norm(var[0], p=self.dual_norm()) <= lmbda]
                    constr += [lmbda >= 0]
                    constr += [self._c.T@supp_newvar == supp_var[0]]
                    constr += [supp_newvar >= 0]
                    return self.rho * lmbda + self._d@supp_newvar - var[0]@self.b, constr, lmbda
                else:
                    constr = []
                    lmbda = Variable(shape)
                    constr += [lmbda >= 0]
                    supp_newvar = Variable((shape, len(self._d)))
                    constr += [supp_newvar >= 0]
                    for ind in range(shape):
                        constr += [norm(var[ind], p=self.dual_norm())
                                   <= lmbda[ind]]
                        constr += [self._c.T@supp_newvar[ind] == supp_var[ind]]
                    return self.rho * lmbda + supp_newvar@self._d - var@self.b, constr, lmbda
