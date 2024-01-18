from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

SUPPORT_TYPE = Enum("SUPPORT_TYPE", "UPPER_BOUND LOWER_BOUND SUM_EQUALITY")

class UncertaintySet(ABC):

    # @abstractmethod
    # def canonicalize(self, x, var):
    #     return NotImplemented

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

    def _safe_mul(self, lhs, rhs):
        """
        This function uses either @ or *, depending on the size of rhs
        """
        def _is_scalar(rhs):
            """
            Helper function that determines if rhs is a scalar (CVXPY or else e.g. numpy scalar)
            """
            if hasattr(rhs, "is_scalar"):
                return rhs.is_scalar()
            return not len(rhs)>1

        if _is_scalar(rhs):
            return lhs*rhs
        return lhs@rhs

    def canonicalize(self, x, var):
        trans = self.affine_transform_temp
        new_expr = 0
        if trans:
            new_expr += self._safe_mul(trans['b'], x)
            lhs = -trans['A']
            if not x.is_scalar():
                lhs = lhs.T
            new_constraints = [var == self._safe_mul(lhs, x)]
        else:
            new_constraints = [var == -x]

        if self.affine_transform:
            self.affine_transform_temp = self.affine_transform.copy()
        else:
            self.affine_transform_temp = None
        return new_expr, new_constraints

    def isolated_unc(self, i, var, num_constr):
        trans = self.affine_transform_temp
        new_expr = 0
        if i == 0 and trans:
            new_expr += trans['b']

        if self.b is not None:
            new_expr = new_expr + self._safe_mul(-self.b,var)

        e = np.eye(num_constr)[i]
        if trans:
            lhs = -trans['A']
            if not var.is_scalar():
                lhs = lhs.T
            new_constraints = [var == self._safe_mul(lhs, e)]
        else:
            new_constraints = [var == - e]

        if i == (num_constr - 1):
            if self.affine_transform:
                self.affine_transform_temp = self.affine_transform.copy()
            else:
                self.affine_transform_temp = None
        return new_expr, new_constraints
