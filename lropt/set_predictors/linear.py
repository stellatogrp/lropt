import numpy as np
import torch

import lropt.settings as settings
from lropt.set_predictors.set_predictor import SetPredictor


class Linear(SetPredictor):

    def __init__(self, size, init_A, init_b):
        super().__init__()
        self._size = size

        self._weights_a = torch.nn.Parameter(torch.randn(size*size, size, dtype=settings.DTYPE))
        self._intercept_a = torch.nn.Parameter(torch.randn(size, size, dtype=settings.DTYPE))

        self._weights_b = torch.nn.Parameter(torch.randn(size, size, dtype=settings.DTYPE))
        self._intercept_b = torch.nn.Parameter(torch.randn(size, 1, dtype=settings.DTYPE))

        if init_A is not None:
            combined_A = np.tile(init_A, (size, 1))
            self._weights_a = torch.nn.Parameter(torch.tensor(combined_A, dtype=settings.DTYPE))
        if init_b is not None:
            combined_b = np.tile(init_b.reshape(size, 1), (1, size))
            self._weights_b = torch.nn.Parameter(torch.tensor(combined_b, dtype=settings.DTYPE))

    @property
    def weights_a(self):
        return self._weights_a

    @property
    def intercept_a(self):
        return self._intercept_a

    @property
    def weights_b(self):
        return self._weights_b

    @property
    def intercept_b(self):
        return self._intercept_b

    def forward(self, y):
        w_a = self._weights_a.repeat(len(y), 1, 1) # flattened a
        intcpt_a = self._intercept_a.repeat(len(y), 1, 1)

        w_b = self._weights_b.repeat(len(y), 1, 1)
        intcpt_b = self._intercept_b.repeat(len(y), 1, 1)

        return (torch.bmm(w_a, y[:,:,np.newaxis]).view(len(y), self._size, self._size) + intcpt_a,
                (torch.bmm(w_b, y[:,:,np.newaxis]) + intcpt_b).squeeze())
