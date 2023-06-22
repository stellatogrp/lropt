import cvxpy as cp
import numpy as np


class Parameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        data = kwargs.pop('data', None)

        if data is None:
            raise ValueError("You must provide data")
        assert isinstance(data, np.ndarray), "not a numpy array"

        super(Parameter, self).__init__(*args, **kwargs)

        assert (data.shape[1:] == self.shape)
        assert (data.shape[0] > 0)

        self.data = data
