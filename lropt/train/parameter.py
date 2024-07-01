import cvxpy as cp
import numpy as np


class Parameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        data = kwargs.pop('data', None)

        if data is None:
            raise ValueError("You must provide data")
        assert (isinstance(data, np.ndarray) or isinstance(data, list)), "not a numpy array or list"
        #Convert to a 2D np.array
        if isinstance(data, list):
            for j in range(len(data)):
                if len(data[j].shape)==1:
                    data[j] = np.expand_dims(data[j], axis=0)
            super(Parameter, self).__init__(*args, **kwargs)

        else:
            if len(data.shape)==1:
                data = np.expand_dims(data, axis=0)
            super(Parameter, self).__init__(*args, **kwargs)

            assert (data.shape[1:] == self.shape)
            assert (data.shape[0] > 0)

        self.data = data
        self.value = data[0] if not isinstance(data, list) else data[0][0]


class ShapeParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        super(ShapeParameter, self).__init__(*args, **kwargs)

class EpsParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        super(EpsParameter, self).__init__(*args, **kwargs)
