from cvxpy import norm

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
                 affine_transform=None):
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")

        if affine_transform:
            check_affine_transform(affine_transform)
            self.affine_transform_temp = affine_transform.copy()
        else:
            self.affine_transform_temp = None

        self.affine_transform = affine_transform
        self._p = p
        self._rho = rho

    @property
    def p(self):
        return self._p

    @property
    def rho(self):
        return self._rho

    def dual_norm(self):
        return 1. + 1. / (self.p - 1.)

    def canonicalize(self, x):
        trans = self.affine_transform_temp
        if x.is_scalar():
            if trans:
                new_expr = trans['b'] * x
                new_expr += self.rho * norm(trans['A'] * x,
                                            p=self.dual_norm())
            else:
                new_expr = self.rho * norm(x, p=self.dual_norm())

        else:
            if trans:
                new_expr = trans['b'] @ x
                new_expr += self.rho * norm(trans['A'].T @ x,
                                            p=self.dual_norm())
            else:
                new_expr = self.rho * norm(x, p=self.dual_norm())

        if self.affine_transform:
            self.affine_transform_temp = self.affine_transform.copy()
        else:
            self.affine_transform_temp = None
        # TODO: Make A and b parameters

        return new_expr, []
