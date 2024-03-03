import numpy as np
import torch

import lropt.settings as settings
from lropt.set_predictors.set_predictor import SetPredictor


class Constant(SetPredictor):

    def __init__(self, size, init_A, init_b):
        super().__init__()

        # self._w_a = torch.nn.Parameter(torch.randn(size, size, dtype=settings.DTYPE))
        # self._w_b = torch.nn.Parameter(torch.randn(size, dtype=settings.DTYPE))

        self._weights_a = torch.nn.Parameter(torch.randn(size, size, dtype = settings.DTYPE))
        # self._w_a = self._weights_a.repeat(batch_size, 1, 1)
        self._weights_b = torch.nn.Parameter(torch.randn(size, 1, dtype = settings.DTYPE))
        # self._w_b = self._weights_b.repeat(batch_size, 1, 1)

        if init_A is not None:
            # self._w_a = torch.nn.Parameter(torch.tensor(init_A, dtype=settings.DTYPE))
            self._weights_a = torch.nn.Parameter(torch.tensor(init_A, dtype = settings.DTYPE))
            # self._w_a = self._weights_a.repeat(batch_size, 1, 1)
        if init_b is not None:
            # self._w_b = torch.nn.Parameter(torch.tensor(init_b, dtype=settings.DTYPE))
            self._weights_b = torch.nn.Parameter(torch.tensor(init_b,
                                                              dtype = settings.DTYPE).view(-1, 1))
            # self._w_b = self._weights_b.repeat(batch_size, 1, 1)

    # @property
    # def w_a(self):
    #     return self._w_a

    # @property
    # def w_b(self):
    #     return self._w_b

    @property
    def weights_a(self):
        return self._weights_a

    @property
    def weights_b(self):
        return self._weights_b

    def forward(self, y):
        torch.tensor(np.repeat(np.eye(y.size()[1])[np.newaxis], len(y), axis=0),
                                 dtype = settings.DTYPE)
        w_a = self._weights_a.repeat(len(y), 1, 1)
        w_b = self._weights_b.repeat(len(y), 1, 1)
        return w_a, w_b.squeeze()
        # return torch.bmm(identity, w_a), torch.bmm(identity, w_b).squeeze()
