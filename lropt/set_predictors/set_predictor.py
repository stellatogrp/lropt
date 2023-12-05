from abc import ABC, abstractmethod

import torch


class SetPredictor(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, x, var):
        return NotImplemented
