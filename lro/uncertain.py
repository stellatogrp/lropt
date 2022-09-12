import cvxpy as cp
import numpy as np


class UncertainParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):

        uncertainty_set = kwargs.pop('uncertainty_set', None)

        if uncertainty_set is None:
            raise ValueError("You must specify an uncertainty set.")

        super(UncertainParameter, self).__init__(*args, **kwargs)
        self.value = np.zeros(self.shape)
        self.uncertainty_set = uncertainty_set

    def canonicalize(self, x):
        """Reformulate uncertain parameter"""
        return self.uncertainty_set.canonicalize(x)

    def __repr__(self):
        """String to recreate the object.
        """
        attr_str = self._get_attr_str()
        if len(attr_str) > 0:
            return "UncertainParameter(%s%s)" % (self.shape, attr_str)
        else:
            return "UncertainParameter(%s)" % (self.shape,)
