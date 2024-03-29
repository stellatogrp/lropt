import numpy as np
from cvxpy import Parameter, Variable, norm

from lropt.uncertainty_sets.uncertainty_set import UncertaintySet


class Budget(UncertaintySet):
    r"""
    Budget uncertainty set defined as

    .. math::
        \mathcal{U}_{\text{budget}} = \{u \ | \ \| A_1 u + b_1 \|_\infty \le \rho_1,
        \| A_2 u + b_2 \|_1 \leq \rho_2\}

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
    Budget
        Budget uncertainty set.
    """

    def __init__(self, rho1=1., rho2=1.,
                 a1=None, a2=None, b1=None, b2=None, c=None, d=None, data=None, loss=None,
                 train_box=True, ub=None, lb=None, sum_eq=None):
        if rho2 <= 0 or rho1 <= 0:
            raise ValueError("Rho values must be positive.")

        if data is not None and loss is None:
            raise ValueError("You must provide a loss function")

        if data is not None:
            dat_shape = data.shape[1]
            a = Parameter((dat_shape, dat_shape))
            b = Parameter(dat_shape)
        else:
            a = None
            b = None
        self.affine_transform_temp = None
        self.affine_transform = None
        self._a1 = a1
        self._a2 = a2
        self._b1 = b1
        self._b2 = b2
        self._rho1 = rho1
        self._rho2 = rho2
        self._data = data
        self._a = a
        self._b = b
        self._trained = False
        self._loss = loss
        self._train_box = train_box
        self._c = c
        self._d = d
        self._define_support = False
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq

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
    def sum_eq(self):
        return self._sum_eq
    
    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

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
        if self._a1 is None:
            self._a1 = np.eye(ushape)
        if self._a2 is None:
            self._a2 = np.eye(ushape)
        if self._b1 is None:
            self._b1 = np.zeros(ushape)
        if self._b2 is None:
            self._b2 = np.zeros(ushape)

        if self.data is not None and self.train_box:
            if shape == 1:
                newvar = Variable(ushape)  # z conjugate variables
                lmbda1 = Variable()
                lmbda2 = Variable()
                supp_newvar = Variable(len(self._d))
                constr += [norm(newvar, 1) <= lmbda1]
                constr += [norm(newvar2[0], np.inf) <= lmbda2]
                constr += [self.a.T@newvar == newvar1[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                constr += [self._c.T@supp_newvar == supp_var[0]]
                constr += [supp_newvar >= 0]
                return self.rho1*lmbda1 + self._d@supp_newvar + self.rho2*lmbda2 - newvar*self.b, \
                    constr, (lmbda1, lmbda2)
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar = Variable((shape, ushape))
                supp_newvar = Variable((shape, len(self._d)))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                constr += [supp_newvar >= 0]
                for ind in range(shape):
                    constr += [norm(newvar[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar2[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self.a.T@newvar[ind] == newvar1[ind]]
                    constr += [self._c.T@supp_newvar[ind] == supp_var[ind]]
                return self.rho1*lmbda1 + supp_newvar@self._d + self.rho2*lmbda2 - newvar@self.b, \
                    constr, (lmbda1, lmbda2)
        elif self.data is not None and not self.train_box:
            if shape == 1:
                newvar = Variable(ushape)  # z conjugate variables
                lmbda1 = Variable()
                lmbda2 = Variable()
                supp_newvar = Variable(len(self._d))
                constr += [norm(newvar1, 1) <= lmbda1]
                constr += [norm(newvar[0], np.inf) <= lmbda2]
                constr += [self.a.T@newvar == newvar2[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                constr += [self._c.T@supp_newvar == supp_var[0]]
                constr += [supp_newvar >= 0]
                return self.rho1*lmbda1 + self._d@supp_newvar + self.rho2*lmbda2 - newvar*self.b, \
                    constr, (lmbda1, lmbda2)
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar = Variable((shape, ushape))
                supp_newvar = Variable((shape, len(self._d)))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                constr += [supp_newvar >= 0]
                for ind in range(shape):
                    constr += [norm(newvar1[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self.a.T@newvar[ind] == newvar2[ind]]
                    constr += [self._c.T@supp_newvar[ind] == supp_var[ind]]
                return self.rho1*lmbda1 + supp_newvar@self._d + self.rho2*lmbda2 - newvar@self.b,\
                    constr, (lmbda1, lmbda2)
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
                supp_newvar = Variable(len(self._d))
                constr += [norm(newvar_1, 1) <= lmbda1]
                constr += [norm(newvar_2, np.inf) <= lmbda2]
                constr += [self._a1.T@newvar_1 == newvar1[0]]
                constr += [self._a2.T@newvar_2 == newvar2[0]]
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                constr += [self._c.T@supp_newvar == supp_var[0]]
                constr += [supp_newvar >= 0]
                return self.rho1 * lmbda1 + self._d@supp_newvar + self.rho2 * lmbda2 \
                    - newvar_1@self._b1 - newvar_2@self._b2,\
                    constr, (lmbda1, lmbda2)
            else:
                lmbda1 = Variable(shape)
                lmbda2 = Variable(shape)
                newvar_1 = Variable((shape, ushape))
                newvar_2 = Variable((shape, ushape))
                supp_newvar = Variable((shape, len(self._d)))
                constr += [lmbda1 >= 0, lmbda2 >= 0]
                constr += [supp_newvar >= 0]
                for ind in range(shape):
                    constr += [norm(newvar_1[ind], p=1) <= lmbda1[ind]]
                    constr += [norm(newvar_2[ind], p=np.inf) <= lmbda2[ind]]
                    constr += [self._a1.T@newvar_1[ind] == newvar1[ind]]
                    constr += [self._a2.T@newvar_2[ind] == newvar2[ind]]
                    constr += [self._c.T@supp_newvar[ind] == supp_var[ind]]
                return self.rho1 * lmbda1 + supp_newvar@self._d + self.rho2 * lmbda2 \
                    - newvar_1@self._b1 - newvar_2@self._b2, constr,\
                    (lmbda1, lmbda2)
