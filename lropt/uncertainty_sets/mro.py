import numpy as np
import scipy as sc
from cvxpy import Parameter, Variable, norm
from sklearn.cluster import KMeans

from lropt.uncertainty_sets.uncertainty_set import UncertaintySet

# from lropt.utils import check_affine_transform


class MRO(UncertaintySet):
    r"""
    Uncertainty set where the parameter is constrained to lie in a
    Wasserstein ball of the form
    .. math::
        \{ \sum( w_k||u_k - d_k ||^\text{power}_p)\leq \rho\\}
    """

    def __init__(self, K=1, rho=1, data=None, power=1, p=2,
                 A=None, train=True, loss=None, uniqueA=False):

        if train and loss is None:
            raise ValueError("You must provide a loss function")
        if data is None:
            raise ValueError("You must provide data")
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")
        if power < 0:
            raise ValueError("Power must be a nonnegative integer.")

        kmeans = KMeans(n_clusters=K).fit(data)
        self.affine_transform_temp = None
        self.affine_transform = None
        self._data = data
        self._N = data.shape[0]
        self._loss = loss
        self._K = K
        self._power = power
        self._p = p
        self._Dbar = kmeans.cluster_centers_
        self._w = np.bincount(kmeans.labels_) / data.shape[0]
        self._rho = rho
        self._s = None
        self._train = train
        self._uniqueA = uniqueA
        self._m = data.shape[1]

        if train:
            if self._uniqueA:
                paramT = Parameter((K*self._m, self._m))
                dat = data[kmeans.labels_ == 0]
                if dat.shape[0] <= 2:
                    initnew = sc.linalg.sqrtm(sc.linalg.inv(np.cov(data.T)))
                else:
                    initnew = sc.linalg.sqrtm(sc.linalg.inv(np.cov(dat.T)))
                for k_ind in range(1, K):
                    dat = data[kmeans.labels_ == k_ind]
                    if dat.shape[0] <= 2:
                        initnew = np.vstack((initnew, sc.linalg.sqrtm(sc.linalg.inv(np.cov(data.T)))))
                    else:
                        initnew = np.vstack((initnew, sc.linalg.sqrtm(sc.linalg.inv(np.cov(dat.T)))))
                self._initA = initnew
            else:
                paramT = Parameter((self._m, self._m))
        else:
            if A is not None:
                paramT = A
                if self._uniqueA and A.shape[0] != (K*self._m):
                    raise ValueError("A must be of dimension (K*m, m)")
            else:
                paramT = None
        self._paramT = paramT
        self._lam = None

    @property
    def rho(self):
        return self._rho

    @property
    def p(self):
        return self._p

    @property
    def power_val(self):
        return self._power

    @property
    def loss(self):
        return self._loss

    @property
    def paramT(self):
        return self._paramT

    @property
    def N(self):
        return self._N

    @property
    def data(self):
        return self._data

    @property
    def K(self):
        return self._K

    @property
    def w(self):
        return self._w

    @property
    def Dbar(self):
        return self._Dbar

    def q(self):
        return 1. + 1. / (self._power - 1.)

    def dual_norm(self):
        return 1. + 1. / (self._p - 1.)

    def s(self):
        return self._s

    def phi(self):
        if self._power == float("inf"):
            return 1
        else:
            return (self.q() - 1.)**(self.q() - 1.)/(self.q()**self.q())

    # def canonicalize(self, x):

    #     # TODO: DEFINE k
    #     k = 10  # put it now to fix linting errors

    #     trans = self.affine_transform
    #     s = Variable(self._K)
    #     lam = Variable()

    #     new_expr = -s[k]
    #     new_constraints = [lam >= 0]
    #     new_constraints += [lam * self._eps + self._w * s <= 0]
    #     if trans:
    #         new_expr += trans['b'] * x + (trans['A'].T * x).T * self._Dbar[k]
    #         if self._p == 1:
    #             new_constraints += [norm(trans['A'].T * x,
    #                                      self.dual_norm()) <= lam]
    #         else:
    #             new_expr += self.phi() * lam**(-(self.q() - 1)) * \
    #                 norm(trans['A'].T * x, self.dual_norm())**self.q()
    #     else:
    #         new_expr += x.T * self._Dbar[k]
    #         if self._p == 1:
    #             new_constraints += [norm(x, self.dual_norm()) <= lam]
    #         else:
    #             new_expr += self.phi() * lam**(-(self.q() - 1)) * \
    #                 norm(x, self.dual_norm())**self.q()

    #     return new_expr, new_constraints

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

    def conjugate(self, var, shape, k_ind):
        if shape == 1 and k_ind == 0:
            lmbda = Variable()
            self._lam = lmbda
            s = Variable(self._K)
            self._s = s
        elif shape != 1 and k_ind == 0:
            lmbda = Variable(shape)
            self._lam = lmbda
            s = Variable(self._K, shape)
            self._s = s
        lmbda = self._lam
        sval = self._s
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(var, Variable):
            constr = [lmbda >= 0]
            return -sval[k_ind], constr, lmbda, sval
        else:
            ushape = var.shape[1]  # shape of uncertainty
            if (self._train or (self.paramT is not None)) and not self._uniqueA:
                if shape == 1:
                    newvar = Variable(ushape)  # gamma aux variable
                    constr = [norm(newvar, p=self.dual_norm()) <= lmbda]
                    constr += [self.paramT.T@newvar == var[0]]
                    constr += [lmbda >= 0]
                    return newvar*(self.paramT@self.Dbar[k_ind])-sval[k_ind], constr, lmbda, sval
                else:
                    constr = []
                    newvar = Variable((shape, ushape))
                    constr += [lmbda >= 0]
                    for ind in range(shape):
                        constr += [norm(newvar[ind], p=self.dual_norm()) <= lmbda[ind]]
                        constr += [self.paramT.T@newvar[ind] == var[ind]]

                    return newvar@(self.paramT@self.Dbar[k_ind])-sval[k_ind], constr, lmbda, sval
            elif self._train or (self.paramT is not None):
                if shape == 1:
                    newvar = Variable(ushape)  # gamma aux variable
                    constr = [norm(newvar, p=self.dual_norm()) <= lmbda]
                    constr += [self.paramT[k_ind*self._m:(k_ind+1)*self._m, 0:self._m].T@newvar == var[0]]
                    constr += [lmbda >= 0]
                    return newvar*(self.paramT[k_ind*self._m:(k_ind+1) *
                                               self._m, 0:self._m]@self.Dbar[k_ind]) -\
                        sval[k_ind], constr, lmbda, sval
                else:
                    constr = []
                    newvar = Variable((shape, ushape))
                    constr += [lmbda >= 0]
                    for ind in range(shape):
                        constr += [norm(newvar[ind], p=self.dual_norm()) <= lmbda[ind]]
                        constr += [self.paramT[k_ind*self._m:(k_ind+1) *
                                               self._m, 0:self._m].T@newvar[ind] == var[ind]]

                    return newvar@(self.paramT[k_ind*self._m:(k_ind+1) *
                                               self._m, 0:self._m]@self.Dbar[k_ind]) -\
                        sval[k_ind], constr, lmbda, sval
            else:
                if shape == 1:
                    constr = [norm(var[0], p=self.dual_norm()) <= lmbda]
                    constr += [lmbda >= 0]
                    return var[0]*self.Dbar[k_ind]-sval[k_ind], constr, lmbda, sval
                else:
                    constr = []
                    constr += [lmbda >= 0]
                    for ind in range(shape):
                        constr += [norm(var[ind], p=self.dual_norm()) <= lmbda[ind]]
                    return var@self.Dbar[k_ind]-sval[k_ind], constr, lmbda, sval
