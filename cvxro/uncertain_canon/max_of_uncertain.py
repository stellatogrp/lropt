import itertools
import sys
from typing import Tuple

import cvxpy.utilities as u
import numpy as np
import torch
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.atom import Atom

# from cvxpy.atoms.elementwise.maximum import maximum
# from cvxpy.constraints.constraint import Constraint

if sys.version_info >= (3, 0):
    from functools import reduce

# SumOfMaxOfUncertainConstraint
class sum_of_max_of_uncertain(Atom):
    r"""Represents the sum of the maximums of uncertain atoms,
        where the terms in each maximum expression is in a list, and the
        input is a list of lists. An additional expression not in the sums can
        be passed as a second input.
        For example, $\max\{a(u),b(u)\} + \max\{c(u),d(u)\} + e(u)$
        can be passed as sum_of_max_of_uncertain([[a(u),b(u)],[c(u),d(u)]],
        e(u))

        epigraph_index marks which term is the epigraph-objective row
        f(z,u,x) - z_{n+1}, so that feasibility can be judged on the remaining
        terms alone. Leave it None when the distinction is not needed. It is
        only accepted for a single list of terms; see max_of_uncertain."""
    def __init__(self, lst, term_not_in_max=0, epigraph_index=None) -> None:
        if not isinstance(lst,list):
            raise ValueError("Input must contain a list of lists")
        max_num_elements = 0
        for ind_lst in lst:
            if (not isinstance(ind_lst,list)):
                raise ValueError("Input must contain a list of lists")
            num_elements = len(ind_lst)
            if num_elements ==0:
                raise ValueError("Must not contain empty lists")
            max_num_elements = np.maximum(max_num_elements,num_elements)
        if max_num_elements <= 1:
                raise ValueError("At least one list must have two or more terms")
        if epigraph_index is not None and len(lst) > 1:
            raise ValueError(
                "epigraph_index is only meaningful for a single list of terms: with "
                "several lists the arguments are their cross product, so no argument "
                "corresponds to one input term. Use max_of_uncertain instead."
            )
        combined_list = [[term_not_in_max],*lst]
        permutations= list(itertools.product(*combined_list))
        combined_permutation_list = [AddExpression([Atom.cast_to_const(arg) \
                                for arg in lst]) for lst in permutations]
        super(sum_of_max_of_uncertain,\
               self).__init__(*combined_permutation_list)
        self.epigraph_index = epigraph_index

    def shape_from_args(self) -> Tuple[int, ...]:
        """Shape is the same as the sum of the arguments.
        """
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def numeric(self, values):
        """Returns the elementwise maximum.
        """
        # values = [arg.numeric() for arg in self.args]
        return reduce(np.maximum, values)

    def torch_numeric(self, values):
        # values = [arg.torch_numeric() for arg in self.args]
        return reduce(torch.maximum, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Reduces the list of argument signs according to the following rules:
        #     POSITIVE, ANYTHING = POSITIVE
        #     ZERO, UNKNOWN = POSITIVE
        #     ZERO, ZERO = ZERO
        #     ZERO, NEGATIVE = ZERO
        #     UNKNOWN, NEGATIVE = UNKNOWN
        #     NEGATIVE, NEGATIVE = NEGATIVE
        is_pos = any(arg.is_nonneg() for arg in self.args)
        is_neg = all(arg.is_nonpos() for arg in self.args)
        return (is_pos, is_neg)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return all(arg.is_affine() for arg in self.args)

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return all(arg.is_pwl() for arg in self.args)

    def _grad(self, values) -> None:
        return None

    # def torch_numeric(self, expr: Expression, values: list[Tensor]) -> Tensor:
    #     return torch.max(*values)

    # def is_dcp(self, dpp: bool = False) -> bool:
    #     """An max constraint is DCP if its argument is affine."""
    #     if dpp:
    #         with scopes.dpp_scope():
    #             for arg in self.args:
    #                 return_bool = arg.is_affine()
    #                 if not return_bool:
    #                     return False
    #             return return_bool
    #     for arg in self.args:
    #         return_bool = arg.is_affine()
    #         if not return_bool:
    #             return False
    #     return return_bool

    # def gen_torch_exp(self):
    #     # initarg = self.args[0].gen_torch_exp()
    #     # vals = []
    #     # for arg in self.args[1:]:
    #     #     vals.append(arg.gen_torch_exp()+initarg)
    #     # return torch.maximum(torch.stack(vals))
    #     # return super().gen_torch_exp()
    #     # vals = []
    #     # for arg in self.args:
    #     #     vals.append(arg + self.args[0])
    #     expr = maximum(*self.args)
    #     return expr.gen_torch_exp()


class max_of_uncertain(sum_of_max_of_uncertain):
    r"""Represents the maximum of uncertain atoms passed as a list.

    epigraph_index marks which entry of lst is the epigraph-objective row
    f(z,u,x) - z_{n+1}. Marking it lets the evaluation report feasibility and
    feasibility-filtered statistics over the remaining rows only, matching the
    distinction between the probability of certificate violation (all rows) and
    infeasibility (all rows but the epigraph). Leave it None when not needed.
    """
    def __init__(self, lst, term_not_in_max=0, epigraph_index=None) -> None:
        if not isinstance(lst,list):
            raise ValueError("Input must contain a list")
        if len(lst) <= 1:
            raise ValueError("There must be at least two terms in the list")
        if epigraph_index is not None and not 0 <= epigraph_index < len(lst):
            raise ValueError(
                f"epigraph_index {epigraph_index} is out of range for {len(lst)} terms"
            )
        super(max_of_uncertain, self).__init__([lst],term_not_in_max, epigraph_index)
