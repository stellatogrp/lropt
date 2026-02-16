import numpy as np
from cvxpy import Variable

from cvxro.parameter import ShapeParameter, SizeParameter
from cvxro.uncertainty_sets.uncertainty_set import UncertaintySet
from cvxro.uncertainty_sets.utils import check_affine_transform


class Polyhedral(UncertaintySet):
    r"""
    Polyhedral uncertainty set, defined as

    .. math::
        \mathcal{U}_{\text{poly}} = \{Az+b \ | \ lhs z \leq rhs\}

    Parameters
    ----------
    lhs: 2 dimentional np.array, optional
        matrix defining the lhs of the polyhedral support: :math:`cu \le d`. By default None.
    rhs: np.array, optional
        vector defining the rhs of the polyhedral support: :math:`cu \le d`. By default None.
    ub: np.array | float, optional
        vector or float defining the upper bound of the support. If scalar, broadcast to a vector.
        By default None.
    lb: np.array | float, optional
        vector or float defining the lower bound of the support. If scalar, broadcast to a vector.
        By default None.
    sum_eq: np.array | float, optional
        vector or float defining an equality constraint for the uncertain vector. By default None.
    indices_dict: dict, optional
        Optional mapping with keys 'train', 'test', and 'validate' that specify
        integer index arrays or boolean masks to split the provided `data` into
        training, testing, and validation subsets. If provided, the indices are
        validated against `data.shape[0]` and normalized to integer index arrays.

    Returns
    -------
    Polyhedral
        Polyhedral uncertainty set.
    """

    def __init__(self, lhs, rhs, c=None, d=None, dimension = None, a = None, b=None,
                 affine_transform=None, data = None, ub=None,
                 lb=None, sum_eq=None, eval_data=None, indices_dict=None):

        # if data is not None:
        #     raise ValueError("You cannot train a polyhedral set")

        if affine_transform:
            check_affine_transform(affine_transform)
            affine_transform['A'] = np.array(affine_transform['A'])
            affine_transform['b'] = np.array(affine_transform['b'])
            self.affine_transform_temp = affine_transform.copy()
        else:
            self.affine_transform_temp = None
        self.affine_transform = affine_transform

        if data is not None:
            dat_shape = data.shape[1]
            if dimension is None:
                dimension = dat_shape
            a = ShapeParameter((dat_shape, dimension))
            a.value = np.eye(dat_shape, dimension)
            b = ShapeParameter(dat_shape)
            b.value = np.zeros(dat_shape)

        if dimension is not None:
            if a is not None:
                if a.shape[1] != dimension:
                    raise ValueError("Mismatching dimension for A.")
            if a is None:
                raise ValueError("You must provide A if you provide a dimension.")
        self.eval_data = eval_data
        self._a = a
        self._b = b
        self._dimension = dimension
        self._d = d
        self._c = c
        self._lhs = lhs
        self._rhs = rhs
        self._trained = False
        self._data = data
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq
        self._define_support = False
        self._rho_mult = SizeParameter(value=1.)
        from cvxro.uncertainty_sets.utils import check_indices_dict
        if indices_dict is not None:
            # polyhedral sets may not have data; require data for validation
            self.indices_dict = check_indices_dict(indices_dict,
                     None if self._data is None else self._data.shape[0])
        else:
            self.indices_dict = None



    @property
    def d(self):
        return self._d

    @property
    def rho_mult(self):
        return self._rho_mult

    @property
    def rhs(self):
        return self._rhs

    @property
    def lhs(self):
        return self._lhs

    @property
    def trained(self):
        return self._trained

    @property
    def data(self):
        return self._data

    @property
    def dimension(self):
        return self._dimension

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

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


    def conjugate(self, var, supp_var, k_ind=0):
        constr = []
        if not self._define_support:
            if self._c is None:
                if not isinstance(var, Variable):
                    self._c = np.zeros((var, var))
                else:
                    self._c = np.zeros((supp_var.shape[0], supp_var.shape[0]))
            if self._d is None:
                if not isinstance(var, Variable):
                    self._d = np.zeros(var)
                else:
                    self._d = np.zeros(supp_var.shape[0])
            self._define_support = True
        if isinstance(var, Variable):
            supp_newvar = Variable(len(self._d))
            lmbda = Variable(len(self.rhs))
            constr += [lmbda >= 0]
            constr += [self._c.T@supp_newvar == supp_var]
            constr += [supp_newvar >= 0]
            if len(self.rhs) == 1:
                constr += [var == lmbda*self.lhs]
                return self.rho_mult*lmbda*self.rhs + self._d@supp_newvar, constr, lmbda, None
            else:
                constr += [var == lmbda@self.lhs]
                return self.rho_mult*lmbda@self.rhs, constr, lmbda, None
        else:
            return 0, [], 0, None
