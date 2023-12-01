import torch


class Linear(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.w_a = torch.nn.Parameter(torch.randn(size^2, size))
        self.w_b = torch.nn.Parameter(torch.randn(size, size))

    @property
    def w_a(self):
        return self._w_a

    @property
    def w_b(self):
        return self._w_b

    def forward(self, y):
        return self.w_a * y, self.w_b * y
