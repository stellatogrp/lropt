import numpy as np

from cvxro.parameter import SizeParameter
from cvxro.uncertainty_sets.uncertainty_set import UncertaintySet


class Scenario(UncertaintySet):
    r"""
    Scenario approach, where the constraints must be satisfied
    for all realizations of the uncertainty set.

    Parameters
    ----------
    data
        The dataset of uncertainty realizations to use
    cartesian
        Whether or not the dataset should be permutated with the other uncertain
        datasets.

    Returns
    -------
    Scenario
        Scenario "uncertainty set"
    """

    def __init__(self, data, eval_data = None, cartesian = True):

        if data is None:
            raise ValueError("You need a dataset for the scenario approach")

        assert isinstance(data, np.ndarray),"data must be a numpy array"

        self.affine_transform = None
        self.affine_transform_temp = None

        self.eval_data = eval_data
        self._a = None
        self._b = None
        self._dimension = None
        self._d = None
        self._c = None
        self._lhs = None
        self._rhs = None
        self._trained = False
        self._train = False
        self._data = data
        self._ub = None
        self._lb = None
        self._sum_eq = None
        self._define_support = False
        self._rho_mult = SizeParameter(value=1.)
        self._cartesian = cartesian
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
        return None
