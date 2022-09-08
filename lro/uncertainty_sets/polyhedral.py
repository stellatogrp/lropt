from cvxpy import Variable

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
                 affine_transform=None):

        if affine_transform:
            check_affine_transform(affine_transform)

        self.affine_transform = affine_transform

        self._d = d
        self._D = D

    @property
    def d(self):
        return self._d

    @property
    def D(self):
        return self._D

    def canonicalize(self, x, minimize=False):
        trans = self.affine_transform
        n_hyper = len(self.d)
        p = Variable(n_hyper)

        D = self.D if not minimize else -self.D

        new_expr = p * self.d
        new_constraints = [p >= 0]
        if trans:
            new_expr += trans['b'] * x
            new_constraints += [p.T * D == trans['A'].T * x]
        else:
            new_constraints += [p.T * D == x]

        return new_expr, new_constraints
