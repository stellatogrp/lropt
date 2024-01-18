import numpy as np
import torch

import lropt.settings as settings
from lropt.set_predictors.set_predictor import SetPredictor


class Linear(SetPredictor):

    def __init__(self, size, init_A, init_b):
        super().__init__()

        self._w_a = torch.nn.Parameter(torch.randn(size*size, size, dtype=settings.DTYPE))
        self._intcpt_a = torch.nn.Parameter(torch.randn(size*size, dtype=settings.DTYPE))
        self._w_b = torch.nn.Parameter(torch.randn(size, size, dtype=settings.DTYPE))
        self._intcpt_b = torch.nn.Parameter(torch.randn(size, dtype=settings.DTYPE))

        if init_A is not None:
            combined_A = np.tile(init_A, (size, 1))
            self._w_a = torch.nn.Parameter(torch.tensor(combined_A, dtype=settings.DTYPE))
        if init_b is not None:
            combined_b = np.tile(init_b.reshape(size, 1), (1, size))
            self._w_b = torch.nn.Parameter(torch.tensor(combined_b, dtype=settings.DTYPE))

    @property
    def w_a(self):
        return self._w_a

    @property
    def intcpt_a(self):
        return self._intcpt_a

    @property
    def w_b(self):
        return self._w_b

    @property
    def intcpt_b(self):
        return self._intcpt_b

    def forward(self, y):
        return (torch.matmul(self._w_a, y).view(-1) + self._intcpt_a,
                torch.matmul(self._w_b, y).view(-1) + self._intcpt_b)
