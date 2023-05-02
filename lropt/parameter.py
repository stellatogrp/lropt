import cvxpy as cp
import numpy as np


class Parameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        instances = kwargs.pop('instances', None)

        if instances is None:
            raise ValueError("You must provide instances")
        assert isinstance(instances, np.ndarray), "not a numpy array"

        super(Parameter, self).__init__(*args, **kwargs)

        assert (instances.shape[1:] == self.shape)

        self.instances = instances
