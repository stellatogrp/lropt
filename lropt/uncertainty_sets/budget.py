import numpy as np
from cvxpy import Parameter, Variable, norm

from lropt.uncertainty_sets.uncertainty_set import UncertaintySet


class Budget(UncertaintySet):
    r"""
    Budget uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{budget}} = \{u \ | \ \| A_1 u + b_1 \|_\infty \le \rho_1, \| A_2 u + b_2 \|_1 \leq \rho_2\}

    Parameters
    ----------
    rho1 : float, optional
        Box scaling. Default 1.0.
    rho2 : float, optional
        1-norm scaling. Default 1.0.
    A1 : np.array, optional
        matrix defining :math:`A_1` in uncertainty set definition. By default :math:`A_1 = I`
    A2 : np.array, optional
        matrix defining :math:`A_2` in uncertainty set definition. By default :math:`A_2 = I`
    b1 : np.array, optional
        vector defining :math:`b_1` in uncertainty set definition. By default :math:`b_1 = 0`
    b2 : np.array, optional
        vector defining :math:`b_2` in uncertainty set definition. By default :math:`b_2 = 0`
    data: np.array, optional
        An array of uncertainty realizations, where each row is one realization. Required if the uncertainty should
        be trained, or if `loss` function passed.
    loss: function, optional
        The loss function used to train the uncertainty set. Required if uncertainty set parameters should be trained
        or if `data` is passed. function must use torch tensors, and arguments to loss function must be given in the
        same order as cvxpy variables defined in problem.

    Returns
    -------
    Budget
        Budget uncertainty set.
    """

    def __init__(self, rho1=1., rho2=1.,
                 A1=None, A2=None, b1=None, b2=None, data=None, loss=None, train_box=True):
        if rho2 <= 0 or rho1 <= 0:
            raise ValueError("Rho values must be positive.")

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            dat_shape = data.shape[1]
            paramT = Parameter((dat_shape, dat_shape))
            paramb = Parameter(dat_shape)

        else:
            paramT = None
            paramb = None

        self.affine_transform_temp = None
        self.affine_transform = None
        self._A1 = A1
        self._A2 = A2
        self._b1 = b1
        self._b2 = b2
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
        # import ipdb
        # ipdb.set_trace()
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
        if i == (num_constr-1):
            if self.affine_transform:
                self.affine_transform_temp = self.affine_transform.copy()
            else:
                self.affine_transform_temp = None
        return new_expr, new_constraints

    def conjugate(self, var, shape, k_ind=0):
        # import ipdb
        # ipdb.set_trace()
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
        if self._A1 is None:
            self._A1 = np.eye(ushape)
        if self._A2 is None:
            self._A2 = np.eye(ushape)
        if self._b1 is None:
            self._b1 = np.zeros(ushape)
        if self._b2 is None:
            self._b2 = np.zeros(ushape)
        if self.data is not None and self.train_box:
            if shape == 1:
                newvar = Variable(ushape)  # z conjugate variables
                lmbda1 = Variable()
                lmbda2 = Variable()
                constr += [norm(newvar, 1) <= lmbda1]
                constr += [norm(newvar2[0], np.inf) <= lmbda2]
                constr += [self.paramT.T@newvar == newvar1[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 - newvar*self.paramb, constr, (lmbda1, lmbda2)
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar = Variable((shape, ushape))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                for ind in range(shape):
                    constr += [norm(newvar[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar2[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self.paramT.T@newvar[ind] == newvar1[ind]]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 - newvar@self.paramb, constr, (lmbda1, lmbda2)
        elif self.data is not None and not self.train_box:
            if shape == 1:
                newvar = Variable(ushape)  # z conjugate variables
                lmbda1 = Variable()
                lmbda2 = Variable()
                constr += [norm(newvar1, 1) <= lmbda1]
                constr += [norm(newvar[0], np.inf) <= lmbda2]
                constr += [self.paramT.T@newvar == newvar2[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 - newvar*self.paramb, constr, (lmbda1, lmbda2)
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar = Variable((shape, ushape))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                for ind in range(shape):
                    constr += [norm(newvar1[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self.paramT.T@newvar[ind] == newvar2[ind]]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 - newvar@self.paramb, constr, (lmbda1, lmbda2)
        # else:
        #     if shape == 1:
        #         lmbda1 = Variable()
        #         lmbda2 = Variable()
        #         constr += [norm(newvar1[0], p=1) <= lmbda1]
        #         constr += [norm(newvar2[0], p=np.inf) <= lmbda2]
        #         constr += [lmbda1 >= 0, lmbda2 >= 0]
        #         return self.rho1 * lmbda1 + self.rho2 * lmbda2, constr
        #     else:
        #         lmbda1 = Variable(shape)
        #         lmbda2 = Variable(shape)
        #         constr += [lmbda1 >= 0, lmbda2 >= 0]
        #         for ind in range(shape):
        #             constr += [norm(newvar1[ind], p=1) <= lmbda1[ind]]
        #             constr += [norm(newvar2[ind], p=np.inf) <= lmbda2[ind]]
        #         return self.rho1 * lmbda1 + self.rho2 * lmbda2, constr
        else:
            if shape == 1:
                newvar_1 = Variable(ushape)  # z conjugate variables
                newvar_2 = Variable(ushape)
                lmbda1 = Variable()
                lmbda2 = Variable()
                constr += [norm(newvar_1, 1) <= lmbda1]
                constr += [norm(newvar_2, np.inf) <= lmbda2]
                constr += [self._A1.T@newvar_1 == newvar1[0]]
                constr += [self._A2.T@newvar_2 == newvar2[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 \
                    - newvar_1*self._b1 - newvar_2*self._b2,\
                    constr, (lmbda1, lmbda2)
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar_1 = Variable((shape, ushape))
                newvar_2 = Variable((shape, ushape))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                for ind in range(shape):
                    constr += [norm(newvar_1[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar_2[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self._A1.T@newvar_1[ind] == newvar1[ind]]
                    constr += [self._A2.T@newvar_2[ind] == newvar2[ind]]
                return self.rho1 * lmbda1 + self.rho2 * lmbda2 \
                    - newvar_1@self._b1 - newvar_2@self._b2, constr,\
                    (lmbda1, lmbda2)
