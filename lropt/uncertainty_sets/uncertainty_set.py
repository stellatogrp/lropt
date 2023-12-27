from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

SUPPORT_TYPE = Enum("SUPPORT_TYPE", "UPPER_BOUND LOWER_BOUND SUM_EQUALITY")

class UncertaintySet(ABC):

    @abstractmethod
    def canonicalize(self, x, var):
        return NotImplemented

    @abstractmethod
    def conjugate(self, var):
        return NotImplemented


    def add_support_type(self, value: np.ndarray | float | None, n: int,
                         support_type: SUPPORT_TYPE):
        """
        
        """
        if value is None:
            return
        value = np.array(value)
        
        self._update_c(value=value, n=n, support_type=support_type)
        self._update_d(value=value, n=n, support_type=support_type)

    def _safe_add_to_list(self, value: np.ndarray | float | None, target: np.ndarray | None, 
                          func: callable):
        """
        This function returns a new matrix with value added to target if target is not None, or
        creates a new array with value if it doesn't exist.

        Args:
            value:
                ub/lb/sum_eq values
            target:
                Existing self._c or self.d
            func:
                np.vstack for self._c or np.hstack for self._d

        """
        if value is None:
            return target
        value = np.array(value)
        if target is None:
            return value
        return func([target, value])
    
    def _update_c(self, value: np.ndarray | float | None, n: int, support_type: SUPPORT_TYPE):
        """
        This function adds value to self._c and updates it.
        """
        
        if support_type == SUPPORT_TYPE.UPPER_BOUND:
            value = np.eye(n)
        elif support_type == SUPPORT_TYPE.LOWER_BOUND:
            value = -np.eye(n)
        elif support_type == SUPPORT_TYPE.SUM_EQUALITY:
            if value.size > 1:
                value = np.eye(n)
            else:
                value = np.ones(n)

        self._c = self._safe_add_to_list(value, self.c, np.vstack)
        if support_type == SUPPORT_TYPE.SUM_EQUALITY:
            self._c = self._safe_add_to_list(-value, self.c, np.vstack)

    def _update_d(self, value: np.ndarray | float | None, n: int, support_type: SUPPORT_TYPE):
        """
        This function adds value to self._d and updates it.
        """
        #If the given value is a number, broadcast it to a vector.
        if value.size==1 and (support_type == SUPPORT_TYPE.UPPER_BOUND or \
                        support_type == SUPPORT_TYPE.LOWER_BOUND):
            value = value*np.ones(n)

        positive_negative = [True, True] #Add value/-value if the 0/1 element is True
        if support_type == SUPPORT_TYPE.UPPER_BOUND:
            positive_negative[1] = False
        elif support_type == SUPPORT_TYPE.LOWER_BOUND:
            positive_negative[0] = False

        if positive_negative[0]:
            self._d = self._safe_add_to_list(value, self._d, np.hstack)
        if positive_negative[1]:
            self._d = self._safe_add_to_list(-value, self._d, np.hstack)