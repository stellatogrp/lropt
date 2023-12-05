import torch

from lropt.set_predictors.set_predictor import SetPredictor


class Constant(SetPredictor):

    def __init__(self, size):
        self.w_a = torch.nn.Parameter(torch.randn(size^2, size))
        self.w_b = torch.nn.Parameter(torch.randn(size, size))

    @property
    def w_a(self):
        return self.w_a

    @property
    def w_b(self):
        return self.w_b

    def forward(self, y):
        return self.w_a, self.w_b
