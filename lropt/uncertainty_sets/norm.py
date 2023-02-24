import numpy as np
from cvxpy import Parameter, Variable, norm

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
        An array of uncertainty realizations, where each row is one realization. Required if the uncertainty should
        be trained, or if `loss` function passed.
    loss: function, optional
        The loss function used to train the uncertainty set. Required if uncertainty set parameters should be trained
        or if `data` is passed. function must use torch tensors, and arguments to loss function must be given in the
        same order as cvxpy variables defined in problem.

    Returns
    -------
    Norm
        Norm uncertainty set.
    """

    def __init__(self, p=2, rho=1.,
                 A=None, b=None, data=None, loss=None):
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            dat_shape = data.shape[1]
            paramT = Parameter((dat_shape, dat_shape))
            paramb = Parameter(dat_shape)

        else:
            if A is not None:
                paramT = A
            else:
                paramT = None
            if b is not None:
                paramb = b
            else:
                paramb = None

        self.affine_transform_temp = None
        self.affine_transform = None

        self._p = p
        self._rho = rho
        self._data = data
        self._paramT = paramT
        self._paramb = paramb
        self._trained = False
        self._loss = loss

    @property
    def p(self):
        return self._p

    @property
    def rho(self):
        return self._rho

    @property
    def paramT(self):
        return self._paramT

    @property
    def paramb(self):
        return self._paramb

    @property
    def data(self):
        return self._data

    @property
    def loss(self):
        return self._loss

    @property
    def trained(self):
        return self._trained

    def dual_norm(self):
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
        # TODO: Make A and b parameters

        return new_expr, new_constraints

    def isolated_unc(self, i, var, num_constr):
        # import ipdb
        # ipdb.set_trace()
        trans = self.affine_transform_temp
        new_expr = 0
        if i == 0:
            if trans:
                new_expr += trans['b']
        e = np.eye(num_constr)[i]
        if len(trans['A'].shape) == 1:
            newA = np.reshape(trans['A'].value, (1, trans['A'].shape[0]))
        else:
            newA = trans['A']
        if var.is_scalar():
            if trans:
                new_constraints = [var == -newA * e]
            else:
                new_constraints = [var == - e]
        else:
            if trans:
                new_constraints = [var == -newA.T @ e]
            else:
                new_constraints = [var == - e]
        if i == (num_constr - 1):
            if self.affine_transform:
                self.affine_transform_temp = self.affine_transform.copy()
            else:
                self.affine_transform_temp = None
        return new_expr, new_constraints

    def conjugate(self, var, shape, k_ind=0):
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(var, Variable):
            lmbda = Variable(shape)
            constr = [lmbda >= 0]
            return self.rho*lmbda, constr, lmbda
        else:
            ushape = var.shape[1]  # shape of uncertainty
            if self.paramb is None:
                self._paramb = np.zeros(ushape)
            if self.data is not None or self.paramT is not None:
                if shape == 1:
                    newvar = Variable(ushape)  # gamma aux variable
                    lmbda = Variable()
                    constr = [norm(newvar, p=self.dual_norm()) <= lmbda]
                    constr += [self.paramT.T@newvar == var[0]]
                    constr += [lmbda >= 0]
                    return self.rho * lmbda - newvar*self.paramb, constr, lmbda
                else:
                    constr = []
                    lmbda = Variable(shape)
                    newvar = Variable((shape, ushape))
                    constr += [lmbda >= 0]
                    for ind in range(shape):
                        constr += [norm(newvar[ind], p=self.dual_norm()) <= lmbda[ind]]
                        constr += [self.paramT.T@newvar[ind] == var[ind]]

                    return self.rho * lmbda - newvar@self.paramb, constr, lmbda
            else:
                if shape == 1:
                    lmbda = Variable()
                    constr = [norm(var[0], p=self.dual_norm()) <= lmbda]
                    constr += [lmbda >= 0]
                    return self.rho * lmbda - var[0]*self.paramb, constr, lmbda
                else:
                    constr = []
                    lmbda = Variable(shape)
                    constr += [lmbda >= 0]
                    for ind in range(shape):
                        constr += [norm(var[ind], p=self.dual_norm()) <= lmbda[ind]]
                    return self.rho * lmbda - var@self.paramb, constr, lmbda
