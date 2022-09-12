from abc import ABC, abstractmethod


class UncertaintySet(ABC):

    @abstractmethod
    def canonicalize(self, x, minimize=False):
        return NotImplemented
