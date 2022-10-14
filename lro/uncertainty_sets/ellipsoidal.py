import numpy as np
from cvxpy import Parameter, Variable, norm

from lro.uncertainty_sets.uncertainty_set import UncertaintySet
from lro.utils import check_affine_transform


class Ellipsoidal(UncertaintySet):
    """
    Uncertainty set where the norm is constrained as
    :math:`\\{\\Pi(u) | \\| u \\|_p \\le \\rho\\}`

    where :math:`\\Pi(u)` is an identity by default but can be
    an affine transformation :math:`A u + b`.
    """

    def __init__(self, p=2, rho=1.,
                 affine_transform=None, data=None, loss=None):
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            if affine_transform:
                raise ValueError("You must provide either data"
                                 "or an affine transform."
                                 )
            else:
                dat_shape = data.shape[1]
                paramT = Parameter((dat_shape, dat_shape))
                paramb = Parameter(dat_shape)
                self.affine_transform_temp = None
        else:
            paramT = None
            paramb = None
            if affine_transform:
                check_affine_transform(affine_transform)
                affine_transform['A'] = np.array(affine_transform['A'])
                affine_transform['b'] = np.array(affine_transform['b'])
                self.affine_transform_temp = affine_transform.copy()
            else:
                self.affine_transform_temp = None

        self.affine_transform = affine_transform

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
        trans = self.affine_transform_temp
        new_expr = 0
        if i == 0:
            if trans:
                new_expr += trans['b']
        e = np.eye(num_constr)[i]
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
        if self.affine_transform:
            self.affine_transform_temp = self.affine_transform.copy()
        else:
            self.affine_transform_temp = None
        return new_expr, new_constraints

    def conjugate(self, var, shape):
        if self.data is not None:
            newvar = Variable(self.data.shape[1])
            if shape == 1:
                lmbda = Variable()
                constr = [norm(newvar, p=self.dual_norm()) <= lmbda]
                constr += [self.paramT.T@newvar == var[0]]
                constr += [lmbda >= 0]
                return self.rho * lmbda + var[0]*self.paramb, constr
            else:
                constr = []
                lmbda = Variable(shape)
                newvar = Variable((shape, self.data.shape[1]))
                constr += [lmbda >= 0]
                for ind in range(shape):
                    constr += [norm(var[ind], p=self.dual_norm()) <= lmbda[ind]]
                    constr += [self.paramT.T@newvar[ind] == var[ind]]

                return self.rho * lmbda + var@self.paramb, constr
        else:
            if shape == 1:
                lmbda = Variable()
                constr = [norm(var[0], p=self.dual_norm()) <= lmbda]
                constr += [lmbda >= 0]
                return self.rho * lmbda, constr
            else:
                constr = []
                lmbda = Variable(shape)
                constr += [lmbda >= 0]
                for ind in range(shape):
                    constr += [norm(var[ind], p=self.dual_norm()) <= lmbda[ind]]
                return self.rho * lmbda, constr
