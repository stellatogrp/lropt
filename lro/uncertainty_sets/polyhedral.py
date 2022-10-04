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
            self.affine_transform_temp = affine_transform.copy()
        else:
            self.affine_transform_temp = None
        self.affine_transform = affine_transform

        self._d = d
        self._D = D

    @property
    def d(self):
        return self._d

    @property
    def D(self):
        return self._D

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

    def conjugate(self, var):
        lmbda = Variable(len(self.d))
        constr = [lmbda >= 0]
        if len(self.d) == 1:
            constr += [var == lmbda*self.D]
            return lmbda*self.d, constr
        else:
            constr += [var == lmbda@self.D]
            return lmbda.T@self.d, constr
