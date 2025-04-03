import cvxpy as cp
import numpy as np


class ContextParameter(cp.Parameter):
    # Context parameter with data
    def __init__(self, *args, **kwargs):
        data = kwargs.pop('data', None)
        eval_data = kwargs.pop('eval_data', None)

        if data is None:
            raise ValueError("You must provide data")
        assert isinstance(data, np.ndarray),"data must be a numpy array"
        #Convert to a 2D np.array
        if len(data.shape)==1:
            data = np.expand_dims(data, axis=0)
        super(ContextParameter, self).__init__(*args, **kwargs)

        assert (data.shape[1:] == self.shape)
        assert (data.shape[0] > 0)

        self.data = data
        self.value = data[0]
        self.eval_data = eval_data

class Parameter(cp.Parameter):
    # CVXPY parameter
    def __init__(self, *args, **kwargs):
        super(Parameter, self).__init__(*args, **kwargs)

class ShapeParameter(cp.Parameter):
    # A and b to be trained
    def __init__(self, *args, **kwargs):
        super(ShapeParameter, self).__init__(*args, **kwargs)

class SizeParameter(cp.Parameter):
    # Size parameter to be trained
    def __init__(self, *args, **kwargs):
        super(SizeParameter, self).__init__(*args, **kwargs)
