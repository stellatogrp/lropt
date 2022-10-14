import numpy as np
from cvxpy import Parameter, Variable

from lro.uncertainty_sets.uncertainty_set import UncertaintySet
from lro.utils import check_affine_transform


class Polyhedral(UncertaintySet):
    """
    Uncertainty set where the parameter is constrained to lie in a
    polyhedron of the form
    :math:`\\{D\\Pi(u) \\le d\\}`

    where :math:`\\Pi(u)` is an identity by default but can be
    an affine transformation :math:`A u + b`.
    """

    def __init__(self, d, D,
                 affine_transform=None, data=None, loss=None):

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            if affine_transform:
                raise ValueError("You must provide either data"
                                 "or an affine transform."
                                 )
            else:
                paramT = Parameter((data.shape[1], data.shape[1]))
                paramb = Parameter(data.shape[1])
                assert (data.shape[1] == D.shape[1])
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

        self._d = d
        self._D = D
        self._data = data
        self._paramT = paramT
        self._paramb = paramb
        self._loss = loss
        self._trained = False

    @property
    def d(self):
        return self._d

    @property
    def D(self):
        return self._D

    @property
    def paramT(self):
        return self._paramT

    @property
    def data(self):
        return self._data

    @property
    def loss(self):
        return self._loss

    @property
    def trained(self):
        return self._trained

    def canonicalize(self, x, var):
        trans = self.affine_transform_temp

        if x.is_scalar():
            new_expr = 0
            if trans:
                new_expr += trans['b'] * x
                new_constraints = [var == -trans['A'] * x]
            else:
                new_constraints = [var == -x]

        else:
            new_expr = 0
            if trans:
                new_expr += trans['b'] @ x
                new_constraints = [var == -trans['A'].T @ x]
            else:
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
            if shape == 1:
                newvar = Variable(self.D.shape[1])
                lmbda = Variable(len(self.d))
                constr = [lmbda >= 0]
                if len(self.d) == 1:
                    constr += [newvar == lmbda*self.D]
                    constr += [self.paramT.T@newvar == var[0]]
                    return lmbda*self.d, constr
                else:
                    constr += [newvar == lmbda@self.D]
                    constr += [self.paramT.T@newvar == var[0]]
                    return lmbda@self.d, constr
            else:
                lmbda = Variable((shape, len(self.d)))
                constr = [lmbda >= 0]
                newvar = Variable((shape, self.D.shape[1]))
                for ind in range(shape):
                    constr += [newvar[ind] == lmbda[ind]@self.D]
                    constr += [self.paramT.T@newvar[ind] == var[ind]]
                return lmbda@self.d, constr
        else:
            if shape == 1:
                lmbda = Variable(len(self.d))
                constr = [lmbda >= 0]
                if len(self.d) == 1:
                    constr += [var[0] == lmbda*self.D]
                    return lmbda*self.d, constr
                else:
                    constr += [var[0] == lmbda@self.D]
                    return lmbda@self.d, constr
            else:
                lmbda = Variable((shape, len(self.d)))
                constr = [lmbda >= 0]
                for ind in range(shape):
                    constr += [var[ind] == lmbda[ind]@self.D]
                return lmbda@self.d, constr
