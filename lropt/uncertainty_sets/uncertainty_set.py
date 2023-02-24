from abc import ABC, abstractmethod


class UncertaintySet(ABC):

    @abstractmethod
    def canonicalize(self, x, var):
        return NotImplemented

    @abstractmethod
    def conjugate(self, var):
        return NotImplemented
