import cvxpy as cp


class ShapeParameter(cp.Parameter):
    def __init__(self, *args, **kwargs):
        super(ShapeParameter, self).__init__(*args, **kwargs)
