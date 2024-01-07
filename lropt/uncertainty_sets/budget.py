import numpy as np
from cvxpy import Variable, norm

from lropt.shape_parameter import ShapeParameter
from lropt.uncertainty_sets.uncertainty_set import UncertaintySet


class Budget(UncertaintySet):
    r"""
    Budget uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{budget}} = \{Au + b \ | \ \| u \|_\infty \le \rho_1,
        \| u\|_1 \leq \rho_2\}

    Parameters
    ----------
    rho1 : float, optional
        Box scaling. Default 1.0.
    rho2 : float, optional
        1-norm scaling. Default 1.0.
    a : np.array, optional
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
    eq: np.array | float, optinal
        vector or float defining an equality constraint for the uncertain vector. By default None.

    Returns
    -------
    Budget
        Budget uncertainty set.
    """

    def __init__(self, dimension = None, rho1=1., rho2=1.,
                 a=None, b=None, c=None, d=None, data=None, loss=None,
                 ub=None, lb=None, eq=None):
        if rho2 <= 0 or rho1 <= 0:
            raise ValueError("Rho values must be positive.")

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            dat_shape = data.shape[1]
            if dimension is None:
                dimension = dat_shape
            a = ShapeParameter((dat_shape, dimension))
            b = ShapeParameter(dat_shape)

        self.affine_transform_temp = None
        self.affine_transform = None
        self._rho1 = rho1
        self._rho2 = rho2
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
        self._eq = eq

    @property
    def rho1(self):
        return self._rho1

    @property
    def rho2(self):
        return self._rho2

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
    def loss(self):
        return self._loss

    @property
    def trained(self):
        return self._trained

    @property
    def train_box(self):
        return self._train_box

    @property
    def ub(self):
        return self._ub

    @property
    def lb(self):
        return self._lb

    @property
    def eq(self):
        return self._eq

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

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
        # TODO: Make A and b parameters

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
        if isinstance(var, Variable):
            ushape = var.shape[1]
            newvar1 = Variable(var.shape)
            newvar2 = Variable(var.shape)
            constr = [newvar1 + newvar2 == var]
        else:
            ushape = var
            newvar1 = Variable(ushape)
            newvar2 = Variable(ushape)
            constr = [newvar1 + newvar2 == 0]
        if shape == 1:
            lmbda1 = Variable()
            lmbda2 = Variable()
            supp_newvar = Variable(len(self._d))
            constr += [norm(newvar1[0], 1) <= lmbda1]
            constr += [norm(newvar2[0], np.inf) <= lmbda2]
            constr += [lmbda1 >= 0, lmbda2 >= 0]
            constr += [self._c.T@supp_newvar == supp_var[0]]
            constr += [supp_newvar >= 0]
            return self.rho1 * lmbda1 + self._d@supp_newvar +\
                  self.rho2 * lmbda2, constr, (lmbda1, lmbda2)
        else:
            lmbda1 = Variable(shape)
            lmbda2 = Variable(shape)
            supp_newvar = Variable((shape, len(self._d)))
            constr += [lmbda1 >= 0, lmbda2 >= 0]
            constr += [supp_newvar >= 0]
            for ind in range(shape):
                constr += [norm(newvar1[ind], p=1) <= lmbda1[ind]]
                constr += [norm(newvar2[ind], p=np.inf) <= lmbda2[ind]]
                constr += [self._c.T@supp_newvar[ind] == supp_var[ind]]
            return self.rho1 * lmbda1 + supp_newvar@self._d \
                + self.rho2 * lmbda2, constr, (lmbda1, lmbda2)
