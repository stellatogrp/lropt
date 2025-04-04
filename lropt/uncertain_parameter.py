import cvxpy as cp
import numpy as np

from lropt.uncertainty_sets.uncertainty_set import SUPPORT_TYPE


class UncertainParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):

        uncertainty_set = kwargs.pop('uncertainty_set', None)
        eval_data = kwargs.pop('eval_data', None)
        self.eval_data = eval_data

        if uncertainty_set is None:
            raise ValueError("You must specify an uncertainty set.")

        super(UncertainParameter, self).__init__(*args, **kwargs)
        self.value = np.zeros(self.shape)
        self.uncertainty_set = uncertainty_set
        n = 1 if len(self.shape)==0 else self.shape[0]
        if self.uncertainty_set._dimension is None:
            self.uncertainty_set._dimension = n
        self.uncertainty_set.add_support_type(self.uncertainty_set.ub,
                                              n, SUPPORT_TYPE.UPPER_BOUND)
        self.uncertainty_set.add_support_type(self.uncertainty_set.lb,
                                              n, SUPPORT_TYPE.LOWER_BOUND)
        self.uncertainty_set.add_support_type(self.uncertainty_set.sum_eq,
                                              n, SUPPORT_TYPE.SUM_EQUALITY)



    def remove_uncertain(self, x, var):
        """Reformulate uncertain parameter"""
        return self.uncertainty_set.remove_uncertain(x, var)

    def isolated_unc(self,var):
        """Remove isolated uncertainty"""
        return self.uncertainty_set.isolated_unc(var)

    def conjugate(self, var, supp_var, k_ind):
        """Reformulate uncertainty set"""
        return self.uncertainty_set.conjugate(var, supp_var, k_ind)

    def __repr__(self):
        """String to recreate the object.
        """
        attr_str = self._get_attr_str()
        if len(attr_str) > 0:
            return "UncertainParameter(%s%s)" % (self.shape, attr_str)
        else:
            return "UncertainParameter(%s)" % (self.shape,)
