from future.utils import with_metaclass
import abc


class UncertaintySet(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def canonicalize(self, x, minimize=False):
        return NotImplemented
