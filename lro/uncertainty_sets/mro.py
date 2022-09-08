from lro.uncertainty_sets.uncertainty_set import UncertaintySet
from lro.utils import check_affine_transform
from cvxpy import Variable
from sklearn.cluster import KMeans
import numpy as np


class MRO(UncertaintySet):
    """
    Uncertainty set where the parameter is constrained to lie in a
    Wasserstein ball of the form
    :math:`\\{ Sum( w_k||\\Pi(u_k) - d_k ||^p)\\le eps\\}`

    where :math:`\\Pi(u)` is an identity by default but can be
    an affine transformation :math:`A u + b`.
    """

    def __init__(self, K, eps, D_in, p, norm=2,
                 affine_transform=None):

        if affine_transform:
            check_affine_transform(affine_transform)

        kmeans = KMeans(n_clusters=K).fit(D_in)
        self.affine_transform = affine_transform
        self._D = D_in
        self._N = D_in.shape[0]
        self._K = K
        self._norm = norm
        self._p = p
        self._Dbar = kmeans.cluster_centers_
        self._w = np.bincount(kmeans.labels_) / D_in.shape[0]
        self._eps = eps

    @property
    def eps(self):
        return self._eps

    @property
    def p(self):
        return self._p

    @property
    def norm_val(self):
        return self._norm

    @property
    def N(self):
        return self._N

    @property
    def D_in(self):
        return self._D_in

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
        return 1. + 1. / (self._p - 1.)

    def dual_norm(self):
        return 1. + 1. / (self.norm_val - 1.)

    def phi(self):
        if self._p == float("inf"):
            return 1
        else:
            return (self.q() - 1.)**(self.q() - 1.)/(self.q()**self.q())

    def canonicalize(self, x):
        trans = self.affine_transform
        s = Variable(self._K)
        lam = Variable()

        new_expr = -s[k]
        new_constraints = [lam >= 0]
        new_constraints += [lam * self._eps + self._w * s <= 0]
        if trans:
            new_expr += trans['b'] * x + (trans['A'].T * x).T * self._Dbar[k]
            if self._p == 1:
                new_constraints += [norm(trans['A'].T * x,
                                         self.dual_norm()) <= lam]
            else:
                new_expr += self.phi() * lam**(-(self.q() - 1)) * \
                    norm(trans['A'].T * x, self.dual_norm())**self.q()
        else:
            new_expr += x.T * self._Dbar[k]
            if self._p == 1:
                new_constraints += [norm(x, self.dual_norm()) <= lam]
            else:
                new_expr += self.phi() * lam**(-(self.q() - 1)) * \
                    norm(x, self.dual_norm())**self.q()

        return new_expr, new_constraints
