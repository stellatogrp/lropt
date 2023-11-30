import torch


class Linear(torch.nn.Module):

    def __init__(self, w=None, a=None, b=None, data=None, loss=None):
        super().__init__()

        # data is u
        if data is not None:
            dat_shape = data.shape[1]
            w = torch.nn.Parameter(torch.randn(dat_shape^2+dat_shape, dat_shape))

        self._data = data
        self._w = w
        self._loss = loss

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def w(self):
        return self._w

    @property
    def data(self):
        return self._data

    @property
    def loss(self):
        return self._loss

    def forward(self, y):
        return self.w * y
