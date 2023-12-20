import cvxpy as cp
import numpy as np

from lropt.uncertainty_sets.mro import MRO
from lropt.uncertainty_sets.polyhedral import Polyhedral
from lropt.uncertainty_sets.uncertainty_set import SUPPORT_TYPE


class UncertainParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):

        uncertainty_set = kwargs.pop('uncertainty_set', None)

        if uncertainty_set is None:
            raise ValueError("You must specify an uncertainty set.")

        super(UncertainParameter, self).__init__(*args, **kwargs)
        self.value = np.zeros(self.shape)
        self.uncertainty_set = uncertainty_set
        n = 1 if len(self.shape)==0 else self.shape[0]
        self.uncertainty_set.add_support_type(self.uncertainty_set.ub,
                                              n, SUPPORT_TYPE.UPPER_BOUND)
        self.uncertainty_set.add_support_type(self.uncertainty_set.lb,
                                              n, SUPPORT_TYPE.LOWER_BOUND)
        self.uncertainty_set.add_support_type(self.uncertainty_set.eq,
                                              n, SUPPORT_TYPE.EQUALITY)

        if not (type(self.uncertainty_set) == MRO or type(self.uncertainty_set) == Polyhedral):
            if self.uncertainty_set._b is None:
                self.uncertainty_set._b = np.zeros(n)
            if self.uncertainty_set._a is None:
                self.uncertainty_set._a = np.eye(n)

            self.uncertainty_set.affine_transform_temp = {'A':
            self.uncertainty_set._a, 'b': self.uncertainty_set._b}
            self.uncertainty_set.affine_transform = {'A':
                self.uncertainty_set._a , 'b': self.uncertainty_set._b}

    def canonicalize(self, x, var):
        """Reformulate uncertain parameter"""
        return self.uncertainty_set.canonicalize(x, var)

    def isolated_unc(self, i, var, num_constr):
        """Remove isolated uncertainty"""
        return self.uncertainty_set.isolated_unc(i, var, num_constr)

    def conjugate(self, var, supp_var, shape, k_ind):
        """Reformulate uncertainty set"""
        return self.uncertainty_set.conjugate(var, supp_var, shape, k_ind)

    def __repr__(self):
        """String to recreate the object.
        """
        attr_str = self._get_attr_str()
        if len(attr_str) > 0:
            return "UncertainParameter(%s%s)" % (self.shape, attr_str)
        else:
            return "UncertainParameter(%s)" % (self.shape,)
