import cvxpy as cp
import numpy as np


class FamilyParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        instances = kwargs.pop('instances', None)

        if instances is None:
            raise ValueError("You must provide instances")
        assert isinstance(instances, np.ndarray), "not a numpy array"

        super(FamilyParameter, self).__init__(*args, **kwargs)

        assert (instances.shape[1:] == self.shape)

        self.instances = instances
