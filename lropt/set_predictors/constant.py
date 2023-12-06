import torch

import lropt.settings as settings
from lropt.set_predictors.set_predictor import SetPredictor


class Constant(SetPredictor):

    def __init__(self, size):
        super().__init__()

        self._w_a = torch.nn.Parameter(torch.randn(size, size, dtype=settings.DTYPE))
        self._w_b = torch.nn.Parameter(torch.randn(size, dtype=settings.DTYPE))

    @property
    def w_a(self):
        return self._w_a

    @property
    def w_b(self):
        return self._w_b

    def forward(self, y):
        return self._w_a, self._w_b
