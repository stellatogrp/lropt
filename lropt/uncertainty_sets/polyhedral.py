import numpy as np
from cvxpy import Variable

from lropt.uncertainty_sets.uncertainty_set import UncertaintySet
from lropt.utils import check_affine_transform


class Polyhedral(UncertaintySet):
    r"""
    Polyhedral uncertainty set, defined as

    .. math::
        \mathcal{U}_{\text{poly}} = \{c u \leq d\}

    Parameters
    ----------
    c: 2 dimentional np.array, optional
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
    Polyhedral
        Polyhedral uncertainty set.
    """

    def __init__(self, c, d,
                 affine_transform=None, ub=None, lb=None,sum_eq=None):

        data, loss = None, None

        if data is not None or loss is not None:
            raise ValueError("You cannot train a polyhedral set")

        if affine_transform:
            check_affine_transform(affine_transform)
            affine_transform['A'] = np.array(affine_transform['A'])
            affine_transform['b'] = np.array(affine_transform['b'])
            self.affine_transform_temp = affine_transform.copy()
        else:
            self.affine_transform_temp = None
        self.affine_transform = affine_transform

        self._d = d
        self._c = c
        self._trained = False
        self._data = data
        self._loss = loss
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq


    @property
    def d(self):
        return self._d

    @property
    def trained(self):
        return self._trained

    @property
    def data(self):
        return self._data
    
    @property
    def c(self):
        return self._c
    
    @property
    def ub(self):
        return self._ub

    @property
    def lb(self):
        return self._lb

    @property
    def sum_eq(self):
        return self._sum_eq

    def _safe_mul(self, lhs, rhs):
        """
        This function uses either @ or *, depending on the size of rhs
        """
        def _is_scalar(rhs):
            """
            Helper function that determines if rhs is a scalar (CVXPY or else e.g. numpy scalar)
            """
            if hasattr(rhs, "is_scalar"):
                return rhs.is_scalar()
            return not len(rhs)>1

        if _is_scalar(rhs):
            return lhs*rhs
        return lhs@rhs

    def canonicalize(self, x, var):
        trans = self.affine_transform_temp
        new_expr = 0
        if trans:
            new_expr += self._safe_mul(trans['b'], x)
            lhs = -trans['A']
            if not x.is_scalar():
                lhs = lhs.T
            new_constraints = [var == self._safe_mul(lhs, x)]
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
        if i == 0 and trans:
            new_expr += trans['b']
        
        e = np.eye(num_constr)[i]
        if trans:
            lhs = -trans['A']
            if var.is_scalar():
                lhs = lhs.T
            new_constraints = [var == self._safe_mul(lhs, e)]
        else:
            new_constraints = [var == - e]

        if i == (num_constr - 1):
            if self.affine_transform:
                self.affine_transform_temp = self.affine_transform.copy()
            else:
                self.affine_transform_temp = None
        return new_expr, new_constraints

    def conjugate(self, var, supp_var, shape, k_ind=0):
        constr = [supp_var == 0]
        if isinstance(var, Variable):
            if shape == 1:
                lmbda = Variable(len(self.d))
                constr += [lmbda >= 0]
                if len(self.d) == 1:
                    constr += [var[0] == lmbda*self.c]
                    return lmbda*self.d, constr, lmbda
                else:
                    constr += [var[0] == lmbda@self.c]
                    return lmbda@self.d, constr, lmbda
            else:
                lmbda = Variable((shape, len(self.d)))
                constr += [lmbda >= 0]
                for ind in range(shape):
                    constr += [var[ind] == lmbda[ind]@self.c]
                return lmbda@self.d, constr, lmbda
        else:
            return 0, [], 0
