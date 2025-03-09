import numpy as np
from cvxpy import Variable, norm
from sklearn.cluster import KMeans

from lropt.train.parameter import ShapeParameter, SizeParameter
from lropt.uncertainty_sets.uncertainty_set import UncertaintySet


class MRO(UncertaintySet):
    r"""
    Uncertainty set where the parameter is constrained to lie in a
    Wasserstein ball of the form
    .. math::
        \{ \sum( w_k||u_k - d_k ||^\text{power}_p)\leq \rho\\}
    """

    def __init__(self, K=1, rho=1, data=None, power=1, p=2,
                 a=None, b=None, train=False, c=None, d=None,
                    ub=None, lb=None, sum_eq=None):

        if data is None:
            raise ValueError("You must provide data")

        self._dimension = data.shape[1]
        if train:
            a = ShapeParameter((self._dimension, self._dimension))
            b = ShapeParameter(self._dimension)

        if (not train) and (a is not None):
            if a.shape[1] != self._dimension:
                raise ValueError("Mismatching dimension for A.")
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")
        if power < 0:
            raise ValueError("Power must be a nonnegative integer.")

        kmeans = KMeans(n_clusters=K, n_init='auto').fit(data)
        self.affine_transform_temp = None
        self.affine_transform = None
        self._data = data
        self._N = data.shape[0]
        self._K = K
        self._power = power
        self._p = p
        self._Dbar = kmeans.cluster_centers_
        self._w = np.bincount(kmeans.labels_) / data.shape[0]
        self._rho = rho
        self._s = None
        self._train = train
        self._m = data.shape[1]
        self._c = c
        self._d = d
        self._define_support = False
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq
        self._b = b
        self._a = a
        self._rho_mult = SizeParameter(value=1.)


        # if train:
        #     if self._uniqueA:
        #         a = Parameter((K*self._m, self._m))
        #         dat = data[kmeans.labels_ == 0]
        #         if dat.shape[0] <= 2:
        #             initnew = sc.linalg.sqrtm(sc.linalg.inv(np.cov(data.T)))
        #         else:
        #             initnew = sc.linalg.sqrtm(sc.linalg.inv(np.cov(dat.T)))
        #         for k_ind in range(1, K):
        #             dat = data[kmeans.labels_ == k_ind]
        #             if dat.shape[0] <= 2:
        #                 initnew = np.vstack((initnew,
        #                                      sc.linalg.sqrtm(sc.linalg.inv(np.cov(data.T)))))
        #             else:
        #                 initnew = np.vstack((initnew,
        #                                      sc.linalg.sqrtm(sc.linalg.inv(np.cov(dat.T)))))
        #         self._initA = initnew
        #     else:
        #         a = Parameter((self._m, self._m))
        # else:
        #     if a is not None:
        #         if self._uniqueA and a.shape[0] != (K*self._m):
        #             raise ValueError("a must be of dimension (K*m, m)")

        self._lam = None

    @property
    def rho(self):
        return self._rho

    @property
    def p(self):
        return self._p

    @property
    def rho_mult(self):
        return self._rho_mult

    @property
    def power_val(self):
        return self._power

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def dimension(self):
        return self._dimension

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

    def q(self):
        return 1. + 1. / (self._power - 1.)

    def dual_norm(self):
        if self.p == 1:
            return np.inf
        return 1. + 1. / (self._p - 1.)

    def s(self):
        return self._s

    def phi(self):
        if self._power == float("inf"):
            return 1
        else:
            return (self.q() - 1.)**(self.q() - 1.)/(self.q()**self.q())

    def conjugate(self, var, supp_var,k_ind):
        if not self._define_support:
            if self._c is None:
                if not isinstance(var, Variable):
                    self._c = np.zeros((var, var))
                else:
                    self._c = np.zeros((var.shape[0], var.shape[0]))
            if self._d is None:
                if not isinstance(var, Variable):
                    self._d = np.zeros(var)
                else:
                    self._d = np.zeros(var.shape[0])
            self._define_support = True
        if k_ind == 0:
            lmbda = Variable()
            self._lam = lmbda
            s = Variable(self._K)
            self._s = s
        lmbda = self._lam
        sval = self._s
        if not isinstance(var, Variable):
            constr = []
            return -sval[k_ind], constr, lmbda, sval
        else:
            supp_newvar = Variable(len(self._d))
            constr = [norm(var, p=self.dual_norm()) <= lmbda]
            constr += [lmbda >= 0]
            constr += [self._c.T@supp_newvar == supp_var]
            constr += [supp_newvar >= 0]
            return var@self.Dbar[k_ind] + \
                self._d@supp_newvar-sval[k_ind], constr, lmbda, sval
