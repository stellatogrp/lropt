import numpy as np
from cvxpy import Parameter, Variable, norm

from lro.uncertainty_sets.uncertainty_set import UncertaintySet
from lro.utils import check_affine_transform


class Budget(UncertaintySet):
    """
    Uncertainty set where the norm is constrained as
    :math:`\\{\\Pi(u) | \\| u \\|_infty \\le \\rho_1, \\| u \\|_1 \\le \\rho_2,\\}`

    where :math:`\\Pi(u)` is an identity by default but can be
    an affine transformation :math:`A u + b`.
    """

    def __init__(self, rho1=1., rho2=1.,
                 affine_transform=None, data=None, loss=None, train_box=True):
        if rho2 <= 0 or rho1 <= 0:
            raise ValueError("Rho values must be positive.")

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            if affine_transform:
                raise ValueError("You must provide either data"
                                 "or an affine transform, not both"
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

        self._rho1 = rho1
        self._rho2 = rho2
        self._data = data
        self._paramT = paramT
        self._paramb = paramb
        self._trained = False
        self._loss = loss
        self._train_box = train_box

    @property
    def rho1(self):
        return self._rho1

    @property
    def rho2(self):
        return self._rho2

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

    @property
    def train_box(self):
        return self._train_box

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
        newvar1 = Variable(var.shape)
        newvar2 = Variable(var.shape)
        constr = [newvar1 + newvar2 == var]
        if self.data is not None and self.train_box:
            if shape == 1:
                newvar = Variable(self.data.shape[1])  # z conjugate variables
                lmbda1 = Variable()
                lmbda2 = Variable()
                constr += [norm(newvar, 1) <= lmbda1]
                constr += [norm(newvar2[0], np.inf) <= lmbda2]
                constr += [self.paramT.T@newvar == newvar1[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 + newvar1[0]*self.paramb, constr
            else:
                constr = []
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar = Variable((shape, self.data.shape[1]))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                for ind in range(shape):
                    constr += [norm(newvar[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar2[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self.paramT.T@newvar[ind] == newvar1[ind]]
                return self.rho * lmbda1 + newvar1@self.paramb, constr
        elif self.data is not None and not self.train_box:
            if shape == 1:
                newvar = Variable(self.data.shape[1])  # z conjugate variables
                lmbda1 = Variable()
                lmbda2 = Variable()
                constr += [norm(newvar1, 1) <= lmbda1]
                constr += [norm(newvar[0], np.inf) <= lmbda2]
                constr += [self.paramT.T@newvar == newvar2[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 + newvar2[0]*self.paramb, constr
            else:
                constr = []
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar = Variable((shape, self.data.shape[1]))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                for ind in range(shape):
                    constr += [norm(newvar1[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self.paramT.T@newvar[ind] == newvar2[ind]]
                return self.rho * lmbda1 + newvar2@self.paramb, constr
        else:
            if shape == 1:
                lmbda1 = Variable()
                lmbda2 = Variable()
                constr += [norm(newvar1[0], p=1) <= lmbda1]
                constr += [norm(newvar2[0], p=np.inf) <= lmbda2]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2, constr
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                for ind in range(shape):
                    constr += [norm(newvar1[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar2[ind], p=np.inf) <= lmbda2[ind]]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2, constr