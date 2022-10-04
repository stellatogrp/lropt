from cvxpy import Variable, norm

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

    def conjugate(self, var):
        lmbda = Variable()
        constr = [norm(var, p=self.dual_norm()) <= lmbda]
        constr += [lmbda >= 0]
        return self.rho * lmbda, constr
