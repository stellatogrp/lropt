from abc import ABC, abstractmethod


class SetPredictor(ABC):

    @abstractmethod
    def forward(self, x, var):
        return NotImplemented
